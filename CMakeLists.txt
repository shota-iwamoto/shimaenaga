cmake_minimum_required(VERSION 3.20)
project(shimaenaga VERSION 1.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build options ───
option(SHIMAENAGA_USE_AVX2      "Enable AVX2 optimizations"   ON)
option(SHIMAENAGA_BUILD_PYTHON  "Build Python bindings"        ON)
option(SHIMAENAGA_BUILD_TESTS   "Build unit tests"             ON)

# ─── Compiler flags ───
if(MSVC)
  add_compile_options(/W4 /O2)
  if(SHIMAENAGA_USE_AVX2)
    add_compile_options(/arch:AVX2)
  endif()
else()
  add_compile_options(-O3 -fno-math-errno -Wall -Wextra -Wno-unused-parameter)
  if(SHIMAENAGA_USE_AVX2)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2 -mfma" HAS_AVX2)
    if(HAS_AVX2)
      add_compile_options(-mavx2 -mfma)
      add_compile_definitions(SHIMAENAGA_AVX2)
    endif()
  endif()
endif()

# ─── OpenMP ───
find_package(OpenMP)

# ─── Core library sources ───
set(SHIMAENAGA_SOURCES
  src/config.cpp
  src/tree/tree.cpp
  src/tree/tree_learner.cpp
  src/data/bin_mapper.cpp
  src/data/token_planner.cpp
  src/data/dataset.cpp
  src/objective/regression.cpp
  src/objective/binary.cpp
  src/objective/multiclass.cpp
  src/objective/lambdarank.cpp
  src/attention/attention_engine.cpp
  src/boosting/refitter.cpp
  src/boosting/line_search.cpp
  src/boosting/block_trainer.cpp
  src/boosting/booster.cpp
  src/io/serializer.cpp
  src/c_api.cpp
)

add_library(shimaenaga_core SHARED ${SHIMAENAGA_SOURCES})
target_compile_definitions(shimaenaga_core PRIVATE SHIMAENAGA_BUILD_DLL)
target_include_directories(shimaenaga_core PUBLIC include src)

if(OpenMP_CXX_FOUND)
  target_link_libraries(shimaenaga_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Static version for tests (no DLL export complexity)
add_library(shimaenaga_static STATIC ${SHIMAENAGA_SOURCES})
target_include_directories(shimaenaga_static PUBLIC include src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(shimaenaga_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─── Python bindings ───
if(SHIMAENAGA_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development QUIET)
  if(Python3_FOUND)
    # Try to find pybind11 via pip install pybind11[global] or conda
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE PYBIND11_NOT_FOUND
    )
    if(NOT PYBIND11_NOT_FOUND)
      list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
      find_package(pybind11 CONFIG QUIET)
    endif()
  endif()

  if(pybind11_FOUND)
    pybind11_add_module(_shimaenaga python/_shimaenaga.cpp)
    target_link_libraries(_shimaenaga PRIVATE shimaenaga_static)
    target_include_directories(_shimaenaga PRIVATE include src)
    # Output to python/shimaenaga/ so it can be imported directly
    set_target_properties(_shimaenaga PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    )
    message(STATUS "pybind11 found. Python module will be built.")
  else()
    message(WARNING "pybind11 not found. Install with: pip install pybind11")
  endif()
endif()

# ─── Tests ───
if(SHIMAENAGA_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    # GTest-based unit tests, referenced by their actual source filenames.
    # (Their core properties — degenerate identity, attention normalisation —
    #  are also covered standalone by review_tests so CI still validates them
    #  even when GoogleTest is unavailable.)
    set(GTEST_TESTS t01_degenerate t02_attention_norm)
    foreach(tid ${GTEST_TESTS})
      if(EXISTS ${CMAKE_SOURCE_DIR}/tests/${tid}.cpp)
        add_executable(${tid} tests/${tid}.cpp)
        target_link_libraries(${tid} PRIVATE shimaenaga_static GTest::gtest_main)
        target_include_directories(${tid} PRIVATE include src)
        if(OpenMP_CXX_FOUND)
          target_link_libraries(${tid} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${tid} COMMAND ${tid})
      endif()
    endforeach()
    message(STATUS "GoogleTest found. Unit tests will be built.")
  else()
    message(STATUS "GoogleTest not found. GTest unit tests skipped "
                   "(equivalent checks remain in review_tests).")
  endif()
endif()

# ─── Standalone tests (no GTest dependency) ───
if(SHIMAENAGA_BUILD_TESTS)
  enable_testing()
  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/smoke_test.cpp)
    add_executable(smoke_test tests/smoke_test.cpp)
    target_link_libraries(smoke_test PRIVATE shimaenaga_static)
    target_include_directories(smoke_test PRIVATE include src)
    add_test(NAME smoke_test COMMAND smoke_test)
  endif()
  # review_tests: design-traceability + coverage suite (T1-T10, ranking, C API,
  # subsampling, weights). Needs OpenMP since it links the static core directly.
  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/review_tests.cpp)
    add_executable(review_tests tests/review_tests.cpp)
    target_link_libraries(review_tests PRIVATE shimaenaga_static)
    target_include_directories(review_tests PRIVATE include src)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(review_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME review_tests COMMAND review_tests)
  endif()
endif()

# ─── Python tests (sklearn integration, API, ST/UAT) ───
# Registered with CTest so `ctest` runs the Python suite too. Requires the
# shared core (libshimaenaga_core) which the ctypes backend loads at runtime,
# plus pytest. Skipped gracefully if either is missing.
# Use the interpreter on PATH (e.g. the active pyenv/venv) for running tests,
# which may differ from the framework Python found for building the bindings.
find_program(SHIMAENAGA_TEST_PYTHON NAMES python3 python)
if(SHIMAENAGA_BUILD_TESTS AND SHIMAENAGA_TEST_PYTHON
   AND EXISTS ${CMAKE_SOURCE_DIR}/tests/python)
  execute_process(
    COMMAND ${SHIMAENAGA_TEST_PYTHON} -c "import pytest, sklearn"
    RESULT_VARIABLE PYTEST_MISSING
  )
  if(PYTEST_MISSING EQUAL 0)
    add_test(
      NAME python_tests
      COMMAND ${SHIMAENAGA_TEST_PYTHON} -m pytest ${CMAKE_SOURCE_DIR}/tests/python -q
    )
    # The ctypes backend loads libshimaenaga_core from <repo>/build at runtime,
    # so build the project before invoking `ctest`.
    message(STATUS "pytest + sklearn found. Python tests will run via CTest.")
  else()
    message(STATUS "pytest/sklearn not found. Python tests skipped in CTest.")
  endif()
endif()

# ─── Install ───
install(TARGETS shimaenaga_core
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/shimaenaga DESTINATION include)
