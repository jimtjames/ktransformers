#ifndef AMX_RAW_KERNELS_HPP
#define AMX_RAW_KERNELS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>

#include "amx_config.hpp"
#include "amx_raw_buffers.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"

namespace amx {

struct GemmKernel224BF16 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 2;
  static constexpr int TILE_M = 16;
  static constexpr int TILE_K = 32;
  static constexpr int TILE_N = 16;
  static constexpr int VNNI_BLK = 2;

  static constexpr int M_STEP = TILE_M * 2;
  static constexpr int N_STEP = TILE_N * 2;
  static constexpr int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 1792;
  static std::string name() { return "BF16"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
#ifdef HAVE_AMX
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 32
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 16 x 32
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  static void load_a(dt* a, size_t lda) {
#ifdef HAVE_AMX
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
#else
    (void)a;
    (void)lda;
#endif
  }

  static void load_b(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void clean_c() {
#ifdef HAVE_AMX
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
#endif
  }

  static void load_c(output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
#else
    (void)c;
    (void)ldc;
#endif
  }

  static void store_c(output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
#else
    (void)c;
    (void)ldc;
#endif
  }

  static void run_tile() {
#ifdef HAVE_AMX
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
#endif
  }
  using BufferA = BufferABF16Impl<GemmKernel224BF16>;
  using BufferB = BufferBBF16Impl<GemmKernel224BF16>;
  using BufferC = BufferCFP32Impl<GemmKernel224BF16>;

  // Basic AVX kernel for BF16: process entire K_BLOCK
  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator at the start of k_block
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    // Process entire K_BLOCK
    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      int32_t* a32 = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512bh* b512 = (__m512bh*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512bh ma = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma, b512[k_i]);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma, b512[16 + k_i]);
        }
      }
    }
  }

  // Optimized AVX kernel: process 4 k_i at once, unroll m rows by 2
  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                           BufferB* bb) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator at the start of k_block
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    // Process entire K_BLOCK
    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      int32_t* a32 = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512bh* b512 = (__m512bh*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      // Process 4 k_i at once - load B vectors and reuse across all m rows
      for (int k_i = 0; k_i < 16; k_i += 4) {
        // Load 4 B vector pairs (lo and hi for each k_i)
        __m512bh b0_lo = b512[k_i];
        __m512bh b0_hi = b512[16 + k_i];
        __m512bh b1_lo = b512[k_i + 1];
        __m512bh b1_hi = b512[16 + k_i + 1];
        __m512bh b2_lo = b512[k_i + 2];
        __m512bh b2_hi = b512[16 + k_i + 2];
        __m512bh b3_lo = b512[k_i + 3];
        __m512bh b3_hi = b512[16 + k_i + 3];

        // Process m rows - unroll by 2 for better ILP
        int m_i = 0;
        for (; m_i + 1 < m_block_end; m_i += 2) {
          // Load A values for 2 rows, 4 k_i each
          __m512bh ma0_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          __m512bh ma1_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 1]);
          __m512bh ma2_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 2]);
          __m512bh ma3_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 3]);
          __m512bh ma0_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i]);
          __m512bh ma1_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i + 1]);
          __m512bh ma2_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i + 2]);
          __m512bh ma3_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i + 3]);

          // Process row 0
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0_0, b0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0_0, b0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1_0, b1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1_0, b1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2_0, b2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2_0, b2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3_0, b3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3_0, b3_hi);

          // Process row 1
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma0_1, b0_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma0_1, b0_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma1_1, b1_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma1_1, b1_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma2_1, b2_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma2_1, b2_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma3_1, b3_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma3_1, b3_hi);
        }
        // Handle remaining row
        for (; m_i < m_block_end; m_i++) {
          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 1]);
          __m512bh ma2 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 2]);
          __m512bh ma3 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 3]);

          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0, b0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0, b0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1, b1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1, b1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2, b2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2, b2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3, b3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3, b3_hi);
        }
      }
    }
  }

  // AMX kernel for BF16: process entire K_BLOCK using AMX tiles
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    if (k_block_begin == 0) {
      clean_c();
    } else {
      load_c(c, N_STEP * sizeof(float));
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K_STEP * sizeof(ggml_bf16_t));
      load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K_STEP * sizeof(ggml_bf16_t));
      run_tile();
    }

    store_c(c, N_STEP * sizeof(float));
  }
};

// FP8 (e4m3) AMX kernel that mirrors the GemmKernel224BF16 interface.
struct GemmKernel224FP8 {
  using fp8_t = uint8_t;
  using output_t = float;

  static constexpr double ELEMENT_SIZE = 1.0;
  static constexpr int TILE_M = 16;
  static constexpr int TILE_K = 32;
  static constexpr int TILE_N = 16;
  static constexpr int VNNI_BLK = 2;

  static constexpr int M_STEP = TILE_M * 2;
  static constexpr int N_STEP = TILE_N * 2;
  static constexpr int K_STEP = TILE_K;

  static inline const int BLOCK_SIZE = 128;  // 128 x 128 block quantization
  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "FP8"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {}

  // FP8->BF16 conversion lookup tables (public for reuse by GemmKernel224FP8PerChannel)
  alignas(64) static constexpr uint8_t bf16_hi_0_val[64] = {
      0x00, 0x3b, 0x3b, 0x3b, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c,
      0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d,
      0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e,
      0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
  };
  alignas(64) static constexpr uint8_t bf16_hi_1_val[64] = {
      0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
      0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
      0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
      0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43,
  };
  alignas(64) static constexpr uint8_t bf16_lo_0_val[64] = {
      0x00, 0x00, 0x80, 0xc0, 0x00, 0x20, 0x40, 0x60, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
  };
  alignas(64) static constexpr uint8_t bf16_lo_1_val[64] = {
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
  };
  // _mm512_set1_epi8 is not constexpr; keep it as a static cached value
  alignas(64) static const __m512i sign_mask_val;
  static inline __m512i bf16_hi_0_mask() { return _mm512_load_si512((__m512i const*)bf16_hi_0_val); }
  static inline __m512i bf16_hi_1_mask() { return _mm512_load_si512((__m512i const*)bf16_hi_1_val); }
  static inline __m512i bf16_lo_0_mask() { return _mm512_load_si512((__m512i const*)bf16_lo_0_val); }
  static inline __m512i bf16_lo_1_mask() { return _mm512_load_si512((__m512i const*)bf16_lo_1_val); }
  static inline __m512i sign_mask() { return _mm512_set1_epi8(0x80); }
  using BufferA = BufferABF16Impl<GemmKernel224FP8>;
  using BufferB = BufferBFP8Impl<GemmKernel224FP8>;
  using BufferC = BufferCFP32ReduceImpl<GemmKernel224FP8>;

  alignas(64) static inline const uint32_t fp8_to_fp32_lut_int[256] = {
      0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000, 0x3c800000,
      0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000, 0x3d000000, 0x3d100000,
      0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000, 0x3d800000, 0x3d900000, 0x3da00000,
      0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000, 0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000,
      0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000, 0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000,
      0x3ed00000, 0x3ee00000, 0x3ef00000, 0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000,
      0x3f600000, 0x3f700000, 0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000,
      0x3ff00000, 0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
      0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000, 0x41000000,
      0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000, 0x41800000, 0x41900000,
      0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000, 0x42000000, 0x42100000, 0x42200000,
      0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000, 0x42800000, 0x42900000, 0x42a00000, 0x42b00000,
      0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000, 0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000,
      0x43500000, 0x43600000, 0x43700000, 0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000,
      0x43e00000, 0x43f00000, 0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000,
      0xbc600000, 0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
      0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000, 0xbd800000,
      0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000, 0xbe000000, 0xbe100000,
      0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000, 0xbe800000, 0xbe900000, 0xbea00000,
      0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000, 0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000,
      0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000, 0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000,
      0xbfd00000, 0xbfe00000, 0xbff00000, 0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000,
      0xc0600000, 0xc0700000, 0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000,
      0xc0f00000, 0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
      0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000, 0xc2000000,
      0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000, 0xc2800000, 0xc2900000,
      0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000, 0xc3000000, 0xc3100000, 0xc3200000,
      0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000, 0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000,
      0xc3c00000, 0xc3d00000, 0xc3e00000, 0xc3f00000,
  };

  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
#endif

    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, _mm512_set1_epi32(0xFFFF0000)));
#endif
    }
  };

  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d_lo;
    __m512bh d_hi;
#else
    __m512 w_lo_even;
    __m512 w_lo_odd;
    __m512 w_hi_even;
    __m512 w_hi_odd;
#endif

    __attribute__((always_inline)) DequantizedWeight(__m512i bfp8_512) {
#if defined(__AVX512BF16__)
      __m512i b_hi = _mm512_permutex2var_epi8(bf16_hi_0_mask(), bfp8_512, bf16_hi_1_mask());
      __m512i b_lo = _mm512_permutex2var_epi8(bf16_lo_0_mask(), bfp8_512, bf16_lo_1_mask());
      b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask(), bfp8_512), b_hi);
      d_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
      d_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);
#else
      __m512i even_bytes = _mm512_and_si512(bfp8_512, _mm512_set1_epi16(0x00FF));
      __m512i odd_bytes = _mm512_srli_epi16(bfp8_512, 8);

      __m512i lo_even_idx = _mm512_unpacklo_epi16(even_bytes, _mm512_setzero_si512());
      __m512i hi_even_idx = _mm512_unpackhi_epi16(even_bytes, _mm512_setzero_si512());
      __m512i lo_odd_idx = _mm512_unpacklo_epi16(odd_bytes, _mm512_setzero_si512());
      __m512i hi_odd_idx = _mm512_unpackhi_epi16(odd_bytes, _mm512_setzero_si512());

      w_lo_even = _mm512_i32gather_ps(lo_even_idx, (const float*)fp8_to_fp32_lut_int, 4);
      w_hi_even = _mm512_i32gather_ps(hi_even_idx, (const float*)fp8_to_fp32_lut_int, 4);
      w_lo_odd = _mm512_i32gather_ps(lo_odd_idx, (const float*)fp8_to_fp32_lut_int, 4);
      w_hi_odd = _mm512_i32gather_ps(hi_odd_idx, (const float*)fp8_to_fp32_lut_int, 4);
#endif
    }
  };

  __attribute__((always_inline)) static inline void fp8_dot_bf16(__m512& acc_lo, __m512& acc_hi,
                                                                 const DequantizedWeight& w,
                                                                 const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    acc_lo = _mm512_dpbf16_ps(acc_lo, act.a, w.d_lo);
    acc_hi = _mm512_dpbf16_ps(acc_hi, act.a, w.d_hi);
#else
    acc_lo = _mm512_fmadd_ps(act.a_even, w.w_lo_even, acc_lo);
    acc_lo = _mm512_fmadd_ps(act.a_odd, w.w_lo_odd, acc_lo);
    acc_hi = _mm512_fmadd_ps(act.a_even, w.w_hi_even, acc_hi);
    acc_hi = _mm512_fmadd_ps(act.a_odd, w.w_hi_odd, acc_hi);
#endif
  }

  // Optimized AVX kernel: process entire k_group_size
  // Load all data first, then convert all, then compute all
  // This gives compiler more freedom to schedule instructions
  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_group_begin, float* c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    for (int m_i = 0; m_i < m_block_end; m_i++) {
      c512[m_i * 2] = _mm512_setzero_ps();
      c512[m_i * 2 + 1] = _mm512_setzero_ps();
    }

    for (int k_begin = 0; k_begin < k_group_size && k_group_begin + k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* abf16 = (ggml_bf16_t*)ba->get_submat(m, k, m_begin, k_group_begin + k_begin);
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_group_begin + k_begin);

      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i += 2) {
          ActivationBF16 a0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]));
          ActivationBF16 a1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]));

          DequantizedWeight d0(bfp8_512[k_i]);
          DequantizedWeight d1(bfp8_512[k_i + 1]);

          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d0, a0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d1, a1);
        }
      }
    }
  }

  // Optimized AVX kernel: process 4 k_i at once, convert B once and reuse for all m rows
  // This version achieved ~493 GB/s - restoring as baseline for further optimization

  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                           BufferB* bb, int k_group_size) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* abf16 = (ggml_bf16_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      for (int k_i = 0; k_i < 16; k_i += 4) {
        DequantizedWeight d0(bfp8_512[k_i]);
        DequantizedWeight d1(bfp8_512[k_i + 1]);
        DequantizedWeight d2(bfp8_512[k_i + 2]);
        DequantizedWeight d3(bfp8_512[k_i + 3]);

        int m_i = 0;
        for (; m_i + 1 < m_block_end; m_i += 2) {
          ActivationBF16 a0_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]));
          ActivationBF16 a1_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]));
          ActivationBF16 a2_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]));
          ActivationBF16 a3_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]));
          ActivationBF16 a0_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + k_i * 2]));
          ActivationBF16 a1_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 1) * 2]));
          ActivationBF16 a2_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 2) * 2]));
          ActivationBF16 a3_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 3) * 2]));

          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d0, a0_0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d1, a1_0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d2, a2_0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d3, a3_0);

          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d0, a0_1);
          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d1, a1_1);
          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d2, a2_1);
          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d3, a3_1);
        }
        for (; m_i < m_block_end; m_i++) {
          ActivationBF16 a0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]));
          ActivationBF16 a1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]));
          ActivationBF16 a2((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]));
          ActivationBF16 a3((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]));

          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d0, a0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d1, a1);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d2, a2);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d3, a3);
        }
      }
    }
  }

  static void apply_scale_kgroup(int m, int n, int m_begin, int n_begin, int k_block_begin, float* c, float* reduce_c,
                                 BufferA* ba, BufferB* bb, int k, int k_group_size) {
    using K = GemmKernel224FP8;
    int to = std::min(m - m_begin, K::M_STEP);

    for (int i = 0; i < to; i++) {
      // Get scale for this k_group
      __m512 bs = _mm512_set1_ps(*bb->get_scale(n, n_begin, k, k_block_begin));
      __m512 now = _mm512_load_ps(reduce_c + i * K::N_STEP);
      __m512 result = _mm512_mul_ps(now, bs);
      __m512 existing = _mm512_load_ps(c + i * K::N_STEP);
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps(c + i * K::N_STEP, result);

      now = _mm512_load_ps(reduce_c + i * K::N_STEP + K::TILE_N);
      result = _mm512_mul_ps(now, bs);
      existing = _mm512_load_ps(c + i * K::N_STEP + K::TILE_N);
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps(c + i * K::N_STEP + K::TILE_N, result);
    }
  }
};

// all step = 32
template <typename K, bool amx_or_avx = false>
void float_mat_vec_kgroup(int m, int n, int k, int k_group_size, typename K::BufferA* ba, typename K::BufferB* bb,
                          typename K::BufferC* bc, int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % k_group_size == 0);
  assert(k_group_size % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  // Process by k_groups
  for (int k_group_begin = 0; k_group_begin < k; k_group_begin += k_group_size) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);
        float* reduce_c = bc->get_reduce_submat(m, n, m_begin, n_begin);

        if (k_group_begin == 0) {
          for (int i = 0; i < K::M_STEP && m_begin + i < m; i++) {
            for (int j = 0; j < K::N_STEP; j++) {
              c[i * K::N_STEP + j] = 0.0f;
            }
          }
        }

        // avx_kernel_4 now processes entire k_group_size internally (like INT8's avx_kernel)
        if constexpr (amx_or_avx && AMX_AVAILABLE) {
          for (int k_begin = k_group_begin; k_begin < std::min(k, k_group_begin + k_group_size); k_begin += K::K_STEP) {
            K::amx_kernel(m, n, k, m_begin, n_begin, k_begin, reduce_c, ba, bb, k_group_size);
          }
        } else {
          // Single call processes entire k_group
          K::avx_kernel(m, n, k, m_begin, n_begin, k_group_begin, reduce_c, ba, bb, k_group_size);
        }
        K::apply_scale_kgroup(m, n, m_begin, n_begin, k_group_begin, c, reduce_c, ba, bb, k, k_group_size);
      }
    }
  }
}

// ============================================================================
// GemmKernel224BF16 vec_mul/mat_mul
// ============================================================================

// Template function for BF16 mat_mul/vec_mul with AMX or AVX backend
template <typename K, bool amx_or_avx = true>
void float_mat_vec(int m, int n, int k, typename K::BufferA* ba, typename K::BufferB* bb, typename K::BufferC* bc,
                   int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);

        if constexpr (amx_or_avx && AMX_AVAILABLE) {
          K::amx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        } else {
          K::avx_kernel_4(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        }
      }
    }
  }
}

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF16::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF16::BufferB> bb, std::shared_ptr<GemmKernel224BF16::BufferC> bc,
                    int ith, int nth) {
  float_mat_vec<GemmKernel224BF16, true>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF16::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF16::BufferB> bb, std::shared_ptr<GemmKernel224BF16::BufferC> bc,
                    int ith, int nth) {
  float_mat_vec<GemmKernel224BF16, false>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void vec_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                           std::shared_ptr<GemmKernel224FP8::BufferB> bb, std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                           int ith, int nth) {
  float_mat_vec_kgroup<GemmKernel224FP8, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                           std::shared_ptr<GemmKernel224FP8::BufferB> bb, std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                           int ith, int nth) {
  float_mat_vec_kgroup<GemmKernel224FP8, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

// ============================================================================
// Per-Channel FP8 GEMM (for GLM-4.7-FP8 style quantization)
// ============================================================================

/**
 * @brief FP8 Per-Channel Kernel
 *
 * Similar to GemmKernel224FP8 but with per-channel scaling instead of block-wise scaling.
 * - Block-wise: scale shape = [n/128, k/128], one scale per 128x128 block
 * - Per-channel: scale shape = [n], one scale per output row
 */
struct GemmKernel224FP8PerChannel {
  using fp8_t = uint8_t;
  using output_t = float;

  static constexpr double ELEMENT_SIZE = 1.0;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  // Use smaller N_BLOCK for per-channel to allow efficient scale application
  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "FP8PerChannel"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {}

  using BufferA = BufferABF16Impl<GemmKernel224FP8PerChannel>;
  using BufferB = BufferBFP8PerChannelImpl<GemmKernel224FP8PerChannel>;
  using BufferC = BufferCFP32Impl<GemmKernel224FP8PerChannel>;

  // Reuse FP8->BF16 conversion from GemmKernel224FP8
  alignas(64) static inline const uint32_t fp8_to_fp32_lut_int[256] = {
      0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000, 0x3c800000,
      0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000, 0x3d000000, 0x3d100000,
      0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000, 0x3d800000, 0x3d900000, 0x3da00000,
      0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000, 0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000,
      0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000, 0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000,
      0x3ed00000, 0x3ee00000, 0x3ef00000, 0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000,
      0x3f600000, 0x3f700000, 0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000,
      0x3ff00000, 0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
      0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000, 0x41000000,
      0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000, 0x41800000, 0x41900000,
      0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000, 0x42000000, 0x42100000, 0x42200000,
      0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000, 0x42800000, 0x42900000, 0x42a00000, 0x42b00000,
      0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000, 0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000,
      0x43500000, 0x43600000, 0x43700000, 0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000,
      0x43e00000, 0x43f00000, 0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000,
      0xbc600000, 0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
      0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000, 0xbd800000,
      0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000, 0xbe000000, 0xbe100000,
      0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000, 0xbe800000, 0xbe900000, 0xbea00000,
      0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000, 0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000,
      0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000, 0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000,
      0xbfd00000, 0xbfe00000, 0xbff00000, 0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000,
      0xc0600000, 0xc0700000, 0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000,
      0xc0f00000, 0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
      0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000, 0xc2000000,
      0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000, 0xc2800000, 0xc2900000,
      0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000, 0xc3000000, 0xc3100000, 0xc3200000,
      0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000, 0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000,
      0xc3c00000, 0xc3d00000, 0xc3e00000, 0xc3f00000,
  };

  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
#endif

    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, _mm512_set1_epi32(0xFFFF0000)));
#endif
    }
  };

  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d_lo;
    __m512bh d_hi;
#else
    __m512 w_lo_even;
    __m512 w_lo_odd;
    __m512 w_hi_even;
    __m512 w_hi_odd;
#endif

    __attribute__((always_inline)) DequantizedWeight(__m512i bfp8_512) {
#if defined(__AVX512BF16__)
      __m512i b_hi = _mm512_permutex2var_epi8(bf16_hi_0_mask(), bfp8_512, bf16_hi_1_mask());
      __m512i b_lo = _mm512_permutex2var_epi8(bf16_lo_0_mask(), bfp8_512, bf16_lo_1_mask());
      b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask(), bfp8_512), b_hi);
      d_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
      d_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);
#else
      __m512i even_bytes = _mm512_and_si512(bfp8_512, _mm512_set1_epi16(0x00FF));
      __m512i odd_bytes = _mm512_srli_epi16(bfp8_512, 8);

      __m512i lo_even_idx = _mm512_unpacklo_epi16(even_bytes, _mm512_setzero_si512());
      __m512i hi_even_idx = _mm512_unpackhi_epi16(even_bytes, _mm512_setzero_si512());
      __m512i lo_odd_idx = _mm512_unpacklo_epi16(odd_bytes, _mm512_setzero_si512());
      __m512i hi_odd_idx = _mm512_unpackhi_epi16(odd_bytes, _mm512_setzero_si512());

      w_lo_even = _mm512_i32gather_ps(lo_even_idx, (const float*)fp8_to_fp32_lut_int, 4);
      w_hi_even = _mm512_i32gather_ps(hi_even_idx, (const float*)fp8_to_fp32_lut_int, 4);
      w_lo_odd = _mm512_i32gather_ps(lo_odd_idx, (const float*)fp8_to_fp32_lut_int, 4);
      w_hi_odd = _mm512_i32gather_ps(hi_odd_idx, (const float*)fp8_to_fp32_lut_int, 4);
#endif
    }
  };

  __attribute__((always_inline)) static inline void fp8_dot_bf16(__m512& acc_lo, __m512& acc_hi,
                                                                 const DequantizedWeight& w,
                                                                 const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    acc_lo = _mm512_dpbf16_ps(acc_lo, act.a, w.d_lo);
    acc_hi = _mm512_dpbf16_ps(acc_hi, act.a, w.d_hi);
#else
    acc_lo = _mm512_fmadd_ps(act.a_even, w.w_lo_even, acc_lo);
    acc_lo = _mm512_fmadd_ps(act.a_odd, w.w_lo_odd, acc_lo);
    acc_hi = _mm512_fmadd_ps(act.a_even, w.w_hi_even, acc_hi);
    acc_hi = _mm512_fmadd_ps(act.a_odd, w.w_hi_odd, acc_hi);
#endif
  }

  /**
   * @brief Apply per-channel scale to result
   *
   * Unlike block-wise scaling, per-channel scaling applies a different scale to each column
   * of the result (each output channel).
   *
   * @param m Total rows
   * @param n Total columns
   * @param m_begin Starting row
   * @param n_begin Starting column
   * @param c Output buffer (M_STEP x N_STEP)
   * @param bb BufferB containing per-channel scales
   */
  static void apply_scale_perchannel(int m, [[maybe_unused]] int n, int m_begin, int n_begin, float* c, BufferB* bb) {
    int to = std::min(m - m_begin, M_STEP);

    // Load N_STEP per-channel scales (32 floats)
    __m512 bs_lo = _mm512_loadu_ps(bb->get_scale(n_begin));           // scale[n_begin..n_begin+15]
    __m512 bs_hi = _mm512_loadu_ps(bb->get_scale(n_begin + TILE_N));  // scale[n_begin+16..n_begin+31]

    for (int i = 0; i < to; i++) {
      // Each row gets multiplied by the same set of per-channel scales
      __m512 c_lo = _mm512_load_ps(c + i * N_STEP);
      __m512 c_hi = _mm512_load_ps(c + i * N_STEP + TILE_N);
      _mm512_store_ps(c + i * N_STEP, _mm512_mul_ps(c_lo, bs_lo));
      _mm512_store_ps(c + i * N_STEP + TILE_N, _mm512_mul_ps(c_hi, bs_hi));
    }
  }

  // AVX kernel for per-channel FP8 GEMM - processes entire K dimension
  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                           BufferB* bb) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* abf16 = (ggml_bf16_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      for (int k_i = 0; k_i < 16; k_i += 4) {
        DequantizedWeight d0(bfp8_512[k_i]);
        DequantizedWeight d1(bfp8_512[k_i + 1]);
        DequantizedWeight d2(bfp8_512[k_i + 2]);
        DequantizedWeight d3(bfp8_512[k_i + 3]);

        int m_i = 0;
        for (; m_i + 1 < m_block_end; m_i += 2) {
          ActivationBF16 a0_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]));
          ActivationBF16 a1_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]));
          ActivationBF16 a2_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]));
          ActivationBF16 a3_0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]));
          ActivationBF16 a0_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + k_i * 2]));
          ActivationBF16 a1_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 1) * 2]));
          ActivationBF16 a2_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 2) * 2]));
          ActivationBF16 a3_1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 3) * 2]));

          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d0, a0_0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d1, a1_0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d2, a2_0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d3, a3_0);

          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d0, a0_1);
          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d1, a1_1);
          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d2, a2_1);
          fp8_dot_bf16(c512[(m_i + 1) * 2], c512[(m_i + 1) * 2 + 1], d3, a3_1);
        }
        for (; m_i < m_block_end; m_i++) {
          ActivationBF16 a0((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]));
          ActivationBF16 a1((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]));
          ActivationBF16 a2((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]));
          ActivationBF16 a3((__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]));

          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d0, a0);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d1, a1);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d2, a2);
          fp8_dot_bf16(c512[m_i * 2], c512[m_i * 2 + 1], d3, a3);
        }
      }
    }
  }
};

/**
 * @brief Per-channel FP8 GEMM function
 *
 * Unlike block-wise FP8 which applies scale per 128x128 block during computation,
 * per-channel FP8 processes entire K dimension first, then applies per-channel scale at the end.
 */
template <typename K>
void float_mat_vec_perchannel(int m, int n, int k, typename K::BufferA* ba, typename K::BufferB* bb,
                              typename K::BufferC* bc, int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
    for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
      float* c = bc->get_submat(m, n, m_begin, n_begin);

      // Process entire K dimension with K_BLOCKs
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
        K::avx_kernel_4(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
      }

      // Apply per-channel scale once after all K is processed
      K::apply_scale_perchannel(m, n, m_begin, n_begin, c, bb);
    }
  }
}

inline void vec_mul_perchannel(int m, int n, int k, std::shared_ptr<GemmKernel224FP8PerChannel::BufferA> ba,
                               std::shared_ptr<GemmKernel224FP8PerChannel::BufferB> bb,
                               std::shared_ptr<GemmKernel224FP8PerChannel::BufferC> bc, int ith, int nth) {
  float_mat_vec_perchannel<GemmKernel224FP8PerChannel>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

#endif  // AMX_RAW_KERNELS_HPP
