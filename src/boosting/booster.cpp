#include "../../include/shimaenaga/booster.h"
#include "block_trainer.h"
#include "../../src/objective/objective.h"
#include "../../src/io/serializer.h"
#include "../../src/util/log.h"
#include "../../src/util/simd.h"
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace shimaenaga {


Booster::Booster(const Config& cfg, std::shared_ptr<Dataset> train)
    : cfg_(cfg), train_(std::move(train)) {
  cfg_.Validate();
}

Booster::~Booster() = default;

void Booster::ApplyThreadConfig() const {
#ifdef _OPENMP
  if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
#endif
}

void Booster::AddValidData(std::shared_ptr<Dataset> valid) {
  valid_ = std::move(valid);
}

void Booster::Train() {
  ApplyThreadConfig();
  auto obj = Objective::Create(cfg_);
  obj->Init(*train_);

  data_size_t N = train_->NumData();
  int C = cfg_.num_class > 1 ? cfg_.num_class : 1;
  model_.C = C;

  // F_0 (E.50-E.52)
  F_train_ = obj->InitScore(*train_);
  model_.F0.assign(F_train_.begin(), F_train_.begin() + C);

  if (valid_) {
    data_size_t Nv = valid_->NumData();
    F_valid_.resize((size_t)Nv * C, 0.0);
    for (data_size_t i = 0; i < Nv; ++i)
      for (int k = 0; k < C; ++k)
        F_valid_[(size_t)i * C + k] = model_.F0[k];
  }

  BlockTrainer trainer(cfg_, *train_, *obj);

  double best_valid  = 1e18;
  int    rounds_no_improve = 0;

  for (int m = 0; m < cfg_.num_iterations; ++m) {
    // Train one block
    AttentiveBlock blk = trainer.TrainOneIter(m + 1, F_train_, beta_prev_);

    // Evaluate training metric
    double train_metric = obj->EvalMetric(F_train_.data(), N, train_->Labels(),
                                          &train_->GroupBoundaries());

    // Valid evaluation: incrementally apply newest block, then evaluate metric
    // with the VALID set's own group boundaries (not the train ones).
    double valid_metric = 0.0;
    if (valid_) {
      UpdateValidScores(blk);
      valid_metric = obj->EvalMetric(F_valid_.data(), valid_->NumData(),
                                     valid_->Labels(), &valid_->GroupBoundaries());
    }

    if ((m + 1) % std::max(1, cfg_.num_iterations / 10) == 0 || m == 0)
      LogProgress(m + 1, train_metric, valid_metric);

    // Non-finite score check (§15: fail loudly, never silently) — full scan.
    // O(NC) via fixed-chunk parallel reduction; negligible next to training.
    {
      bool finite = true;
      const size_t total = (size_t)N * C;
      #pragma omp parallel for schedule(static) reduction(&& : finite) if (total >= 65536)
      for (int64_t t = 0; t < (int64_t)total; ++t)
        finite = finite && std::isfinite(F_train_[t]);
      if (!finite)
        throw TrainError("Non-finite score detected at iteration " + std::to_string(m + 1));
    }

    model_.blocks.push_back(std::move(blk));

    // Track best iteration by valid metric (fallback to train metric if no valid).
    double monitor = valid_ ? valid_metric : train_metric;
    if (monitor < best_valid - 1e-8) {
      best_valid = monitor;
      best_iter_ = m + 1;
      rounds_no_improve = 0;
    } else {
      best_iter_ = best_iter_ ? best_iter_ : m + 1;
      rounds_no_improve++;
      if (cfg_.early_stopping_rounds > 0 && valid_ &&
          rounds_no_improve >= cfg_.early_stopping_rounds) {
        SHIMAENAGA_LOG_INFO("Early stopping at iteration %d (best=%d)", m + 1, best_iter_);
        break;
      }
    }

    // Update beta_prev for next Phase A with the actual learned β from this iter
    beta_prev_ = trainer.GetBeta();
  }

  // Keep only blocks up to best_iteration when early stopping was active.
  if (cfg_.early_stopping_rounds > 0 && valid_ && best_iter_ > 0 &&
      (int)model_.blocks.size() > best_iter_)
    model_.blocks.resize(best_iter_);

  model_.train_cfg = cfg_;
  model_.bin_mappers = train_->GetBinMappers();
  pred_cache_.reset();  // model changed → prediction LUTs are stale
  SHIMAENAGA_LOG_INFO("Training complete. Best iteration: %d", best_iter_);
}

void Booster::LogProgress(int iter, double train_metric, double valid_metric) const {
  if (valid_)
    SHIMAENAGA_LOG_INFO("[%4d] train %s=%.6g  valid %s=%.6g",
               iter, "metric", train_metric, "metric", valid_metric);
  else
    SHIMAENAGA_LOG_INFO("[%4d] train %s=%.6g", iter, "metric", train_metric);
}

// ─── Prediction LUTs (葉ペア分解 M9 applied at inference) ───

void Booster::BuildBlockLUT(const AttentiveBlock& blk, BlockLUT& out,
                            bool build_selfattn) {
  int P = blk.P, H = blk.H, d_a = blk.d_a, Lg = blk.gate_num_leaves;
  int lp_max = 1;
  for (int p = 0; p < P; ++p) lp_max = std::max(lp_max, blk.v_lsize[p]);
  out.lp_max = lp_max;
  const float inv_sqrt = 1.0f / std::sqrt((float)d_a);

  // readout[h][p][lg][lp]
  out.readout.assign((size_t)H * P * Lg * lp_max, 0.0f);
  std::vector<size_t> k_base(P);
  {
    size_t off = 0;
    for (int p = 0; p < P; ++p) { k_base[p] = off; off += (size_t)blk.v_lsize[p] * H * d_a; }
  }
  for (int h = 0; h < H; ++h)
    for (int p = 0; p < P; ++p) {
      float* dst = out.readout.data() + ((size_t)h * P + p) * Lg * lp_max;
      for (int lg = 0; lg < Lg; ++lg) {
        for (int lp = 0; lp < blk.v_lsize[p]; ++lp) {
          float s;
          if (blk.attention_mode == "score_tree") {
            s = blk.z_or_q[(size_t)lg * H * P + h * P + p] + blk.b[h * P + p];
          } else {
            const float* q  = blk.z_or_q.data() + (size_t)lg * H * d_a + h * d_a;
            const float* kv = blk.k.data() + k_base[p] + (size_t)lp * H * d_a + h * d_a;
            s = simd::dot(q, kv, d_a) * inv_sqrt + blk.b[h * P + p];
          }
          dst[(size_t)lg * lp_max + lp] = s;
        }
      }
    }

  // selfattn[h][p][r][lp][lr] (tier-2 only, subject to the memory budget)
  if (build_selfattn && blk.tier >= 2 && !blk.qA.empty()) {
    out.selfattn.assign((size_t)H * P * P * lp_max * lp_max, 0.0f);
    for (int h = 0; h < H; ++h)
      for (int p = 0; p < P; ++p)
        for (int r = 0; r < P; ++r) {
          float* dst = out.selfattn.data() +
                       (((size_t)h * P + p) * P + r) * lp_max * lp_max;
          for (int lp = 0; lp < blk.v_lsize[p]; ++lp) {
            const float* qp = blk.qA.data() + k_base[p] + (size_t)lp * H * d_a + h * d_a;
            for (int lr = 0; lr < blk.v_lsize[r]; ++lr) {
              const float* kr = blk.kA.data() + k_base[r] + (size_t)lr * H * d_a + h * d_a;
              dst[(size_t)lp * lp_max + lr] =
                  simd::dot(qp, kr, d_a) * inv_sqrt + blk.bA[h * P * P + p * P + r];
            }
          }
        }
  }
}

void Booster::EnsurePredictCache() const {
  if (pred_cache_ && pred_cache_->luts.size() == model_.blocks.size()) return;
  auto cache = std::make_unique<PredictCache>();
  cache->luts.resize(model_.blocks.size());

  // Predict builds ONLY the readout LUT (E.45 applied to inference). The
  // self-attn LUT is ~15× larger than the raw qA/kA params (H·P²·L² vs
  // P·L·H·d_a) and, scanned across every block per row, thrashes the cache:
  // measured 838ms (dots) vs 1111ms (LUT) for 20k rows × 100 tier-2 blocks.
  // It pays off only when ONE block is applied repeatedly — see
  // UpdateValidScores, which does build it.
  #pragma omp parallel for schedule(dynamic)
  for (int64_t b = 0; b < (int64_t)model_.blocks.size(); ++b)
    BuildBlockLUT(model_.blocks[b], cache->luts[b], /*build_selfattn=*/false);

  pred_cache_ = std::move(cache);
}

std::vector<score_t> Booster::Predict(const float* X, data_size_t n, int num_features) const {
  ApplyThreadConfig();
  // Use model's saved mappers (portable after save/load), fallback to train dataset
  const auto& mappers = !model_.bin_mappers.empty()
                            ? model_.bin_mappers
                            : train_->GetBinMappers();
  int Fm = (int)mappers.size();
  if (num_features != Fm)
    throw DataError("Predict: num_features=" + std::to_string(num_features) +
                    " does not match the " + std::to_string(Fm) +
                    " features the model was trained on");
  EnsurePredictCache();
  const auto& luts = pred_cache_->luts;

  int C = model_.C;
  std::vector<score_t> result((size_t)n * C, 0.0);

  #pragma omp parallel
  {
    // Per-thread scratch (allocated once per thread, not per row/block).
    std::vector<bin_t> row_bins(Fm);
    std::vector<score_t> out(C);
    std::vector<leaf_t> lp(kMaxTokens);

    #pragma omp for schedule(static)
    for (data_size_t i = 0; i < n; ++i) {
      const float* row = X + (size_t)i * num_features;
      // Bin the row ONCE; tree traversal then compares uint8 bins instead of
      // re-running the mapper's binary search at every node of every tree.
      for (int f = 0; f < Fm; ++f) row_bins[f] = mappers[f].MapValue(row[f]);

      for (int k = 0; k < C; ++k) out[k] = (k < (int)model_.F0.size()) ? model_.F0[k] : 0.0;

      for (size_t b = 0; b < model_.blocks.size(); ++b) {
        const auto& blk = model_.blocks[b];
        leaf_t lg = blk.gate_tree.GetLeafBinned(row_bins.data());
        for (int p = 0; p < blk.P; ++p)
          lp[p] = blk.token_trees[p].GetLeafBinned(row_bins.data());
        ApplyBlock(blk, lp.data(), lg, out.data(), nullptr, &luts[b]);
      }

      for (int k = 0; k < C; ++k)
        result[(size_t)i * C + k] = out[k];
    }
  }
  return result;
}

void Booster::ApplyBlock(const AttentiveBlock& blk, const leaf_t* lp,
                         leaf_t lg, score_t* out,
                         float* beta_acc, const BlockLUT* lut) const {
  int P = blk.P, C = blk.C, H = blk.H, d_a = blk.d_a;
  double inv_sqrt = 1.0 / std::sqrt((double)d_a);
  const bool has_rlut = lut && !lut->readout.empty();
  const bool has_slut = lut && !lut->selfattn.empty();
  const int  lp_max   = lut ? lut->lp_max : 0;
  const int  Lg       = blk.gate_num_leaves;

  // Fixed-bound stack scratch (P ≤ kMaxTokens, H ≤ kMaxHeads by Validate());
  // C is dynamic → per-thread reusable buffers. This runs once per row per
  // block, so nested-vector allocations here would dominate Predict.
  float s[kMaxTokens];
  float alpha[kMaxHeads * kMaxTokens];
  float beta[kMaxTokens];
  size_t k_base[kMaxTokens];
  static thread_local std::vector<float> y_buf;
  if ((int)y_buf.size() < 2 * P * C) y_buf.resize((size_t)2 * P * C);
  float* y0 = y_buf.data();
  float* y1 = y_buf.data() + (size_t)P * C;

  // y0[p][k] = v[p][lp[p]][k]; k_base[p] = offset of token p's [L][H][d_a] slab
  size_t v_off = 0, k_off = 0;
  for (int p = 0; p < P; ++p) {
    const float* vp = blk.v.data() + v_off + (size_t)lp[p] * C;
    for (int k = 0; k < C; ++k) y0[(size_t)p * C + k] = vp[k];
    v_off += (size_t)blk.v_lsize[p] * C;
    k_base[p] = k_off;
    k_off += (size_t)blk.v_lsize[p] * H * d_a;
  }

  // Readout scores → alpha (LUT lookup when cached, QK dot otherwise)
  if (has_rlut) {
    for (int h = 0; h < H; ++h) {
      const float* base = lut->readout.data() + (size_t)h * P * Lg * lp_max;
      for (int p = 0; p < P; ++p)
        s[p] = base[(size_t)p * Lg * lp_max + (size_t)lg * lp_max + lp[p]];
      simd::softmax(s, alpha + h * P, P);
    }
  } else if (blk.attention_mode == "score_tree") {
    for (int h = 0; h < H; ++h) {
      for (int p = 0; p < P; ++p)
        s[p] = blk.z_or_q[(size_t)lg * H * P + h * P + p] + blk.b[h * P + p];
      simd::softmax(s, alpha + h * P, P);
    }
  } else {
    for (int h = 0; h < H; ++h) {
      const float* q = blk.z_or_q.data() + (size_t)lg * H * d_a + h * d_a;
      for (int p = 0; p < P; ++p) {
        const float* kv = blk.k.data() + k_base[p] + (size_t)lp[p] * H * d_a + h * d_a;
        s[p] = simd::dot(q, kv, d_a) * (float)inv_sqrt + blk.b[h * P + p];
      }
      simd::softmax(s, alpha + h * P, P);
    }
  }

  for (int p = 0; p < P; ++p) {
    float bp = 0.0f;
    for (int h = 0; h < H; ++h) bp += blk.rho[h] * alpha[h * P + p];
    beta[p] = bp;
    if (beta_acc) beta_acc[p] += bp;
  }

  // Tier-2 self-attention carrier mixing (E.20-E.22). Mirrors AttentionEngine::Forward.
  const float* yc = y0;
  if (blk.tier >= 2 && !blk.qA.empty()) {
    double eta = cfg_.eta_attn;
    const bool masked = !blk.attn_mask.empty();
    for (size_t t = 0; t < (size_t)P * C; ++t) y1[t] = (float)(1.0 - eta) * y0[t];
    for (int h = 0; h < H; ++h)
      for (int p = 0; p < P; ++p) {
        const uint32_t mrow = masked ? blk.attn_mask[p] : ~0u;
        if (has_slut) {
          const float* base = lut->selfattn.data() +
                              (((size_t)h * P + p) * P) * lp_max * lp_max +
                              (size_t)lp[p] * lp_max;
          for (int r = 0; r < P; ++r)
            s[r] = ((mrow >> r) & 1u)
                       ? base[(size_t)r * lp_max * lp_max + lp[r]]
                       : -1e30f;
        } else {
          const float* qp = blk.qA.data() + k_base[p] + (size_t)lp[p] * H * d_a + h * d_a;
          for (int r = 0; r < P; ++r) {
            if (!((mrow >> r) & 1u)) { s[r] = -1e30f; continue; }
            const float* kr = blk.kA.data() + k_base[r] + (size_t)lp[r] * H * d_a + h * d_a;
            s[r] = simd::dot(qp, kr, d_a) * (float)inv_sqrt + blk.bA[h * P * P + p * P + r];
          }
        }
        simd::softmax(s, s, P);  // A[h][p][:] in place
        for (int r = 0; r < P; ++r) {
          float coeff = (float)(eta * blk.rhoA[h]) * s[r];
          const float* y0r = y0 + (size_t)r * C;
          float* y1p = y1 + (size_t)p * C;
          for (int k = 0; k < C; ++k) y1p[k] += coeff * y0r[k];
        }
      }
    yc = y1;
  }

  for (int p = 0; p < P; ++p)
    for (int k = 0; k < C; ++k)
      out[k] += cfg_.learning_rate * beta[p] * yc[(size_t)p * C + k];
}

void Booster::UpdateValidScores(const AttentiveBlock& blk) {
  if (!valid_) return;
  data_size_t Nv = valid_->NumData();
  int C = model_.C;
  int P = blk.P;
  const auto& vbins = valid_->AllBins();  // shared, no copy

  // The freshly trained block is applied Nv times — build its LUT once.
  BlockLUT lut;
  BuildBlockLUT(blk, lut, /*build_selfattn=*/true);

  #pragma omp parallel
  {
    std::vector<leaf_t> lp(P);
    #pragma omp for schedule(static)
    for (data_size_t i = 0; i < Nv; ++i) {
      for (int p = 0; p < P; ++p) lp[p] = blk.token_trees[p].GetLeaf(vbins, i);
      leaf_t lg = blk.gate_tree.GetLeaf(vbins, i);
      ApplyBlock(blk, lp.data(), lg, F_valid_.data() + (size_t)i * C, nullptr, &lut);
    }
  }
}

std::vector<score_t> Booster::PredictContrib(
    const float* X, data_size_t n, int num_features,
    std::vector<float>* beta_out) const {
  ApplyThreadConfig();
  const auto& mappers = !model_.bin_mappers.empty()
                            ? model_.bin_mappers
                            : train_->GetBinMappers();
  int Fm = (int)mappers.size();
  if (num_features != Fm)
    throw DataError("PredictContrib: num_features=" + std::to_string(num_features) +
                    " does not match the " + std::to_string(Fm) +
                    " features the model was trained on");
  EnsurePredictCache();
  const auto& luts = pred_cache_->luts;

  int C = model_.C;
  const int P = model_.blocks.empty() ? 0 : model_.blocks[0].P;
  const size_t B = model_.blocks.size();
  std::vector<score_t> result((size_t)n * C, 0.0);
  if (beta_out) beta_out->assign((size_t)n * P, 0.0f);

  #pragma omp parallel
  {
    std::vector<bin_t> row_bins(Fm);
    std::vector<score_t> out(C);
    std::vector<leaf_t> lp(kMaxTokens);
    std::vector<float> beta_acc(P > 0 ? P : 1);

    #pragma omp for schedule(static)
    for (data_size_t i = 0; i < n; ++i) {
      const float* row = X + (size_t)i * num_features;
      for (int f = 0; f < Fm; ++f) row_bins[f] = mappers[f].MapValue(row[f]);
      for (int k = 0; k < C; ++k) out[k] = (k < (int)model_.F0.size()) ? model_.F0[k] : 0.0;
      std::fill(beta_acc.begin(), beta_acc.end(), 0.0f);

      for (size_t b = 0; b < B; ++b) {
        const auto& blk = model_.blocks[b];
        leaf_t lg = blk.gate_tree.GetLeafBinned(row_bins.data());
        for (int p = 0; p < blk.P; ++p)
          lp[p] = blk.token_trees[p].GetLeafBinned(row_bins.data());
        ApplyBlock(blk, lp.data(), lg, out.data(),
                   beta_out ? beta_acc.data() : nullptr, &luts[b]);
      }

      for (int k = 0; k < C; ++k) result[(size_t)i * C + k] = out[k];
      if (beta_out && B > 0)
        for (int p = 0; p < P; ++p)
          (*beta_out)[(size_t)i * P + p] = beta_acc[p] / (float)B;  // mean over blocks
    }
  }
  return result;
}

void Booster::SaveModel(const std::string& path) const {
  Serializer::Save(model_, path);
}

void Booster::LoadModel(const std::string& path) {
  model_ = Serializer::Load(path);
  cfg_ = model_.train_cfg;
  pred_cache_.reset();  // model changed → prediction LUTs are stale
}

} // namespace shimaenaga
