#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include "config.h"
#include "export.h"
#include "tree.h"
#include "bin_mapper.h"

namespace shimaenaga {
struct TokenPlan;

// Leaf parameters for one boosting block (詳細設計書 §10.1)
struct AttentiveBlock {
  // Tree structures (P token trees + 1 gate tree)
  std::vector<Tree> token_trees;
  Tree gate_tree;

  int P, H, C, d_a;
  std::string attention_mode;  // score_tree / qk_leaf
  int tier;

  // value  v[p][l][k]  shape: P * L_p * C
  // Flattened as v[p * L_max * C + l * C + k]
  std::vector<param_t> v;        // [P][L_p][C]
  std::vector<int>     v_lsize;  // L_p per token p

  // Gate: score_tree: z[L_g][H][P], qk_leaf: q[L_g][H][d_a]
  std::vector<param_t> z_or_q;  // gate leaf params
  int gate_num_leaves;

  // Key: k[p][l][h][d_a]
  std::vector<param_t> k;        // [P][L_p][H][d_a]

  // Tier-2 self-attention QA, KA: [p][l][h][d_a]
  std::vector<param_t> qA, kA;

  // Biases: b[h][p] (readout), bA[h][p][p'] (self-attn, flattened)
  std::vector<param_t> b;   // H * P
  std::vector<param_t> bA;  // H * P * P

  // Head weights: rho[h] (readout), rhoA[h] (self-attn)
  std::array<float, kMaxHeads> rho  = {};
  std::array<float, kMaxHeads> rhoA = {};

  // Self-attention mask rows: bit r of attn_mask[p] = token p may attend to r.
  // Empty = full attention (backward compatible with format v2 models).
  std::vector<uint32_t> attn_mask;  // size P when set

  // Offsets into v for each token p
  size_t VOffset(int p) const {
    size_t off = 0;
    for (int pp = 0; pp < p; ++pp) off += v_lsize[pp] * C;
    return off;
  }
  size_t KOffset(int p) const {
    size_t off = 0;
    for (int pp = 0; pp < p; ++pp) off += v_lsize[pp] * H * d_a;
    return off;
  }
  size_t QAOffset(int p) const { return KOffset(p); }  // same shape
  size_t KAOffset(int p) const { return KOffset(p); }
};

// Full model (詳細設計書 §10.1)
struct SHIMAENAGA_EXPORT Model {
  int C = 1;
  std::vector<score_t> F0;  // initial prediction, size C
  std::vector<AttentiveBlock> blocks;
  Config train_cfg;
  // BinMappers saved at training time for portable prediction
  std::vector<BinMapper> bin_mappers;
};

} // namespace shimaenaga
