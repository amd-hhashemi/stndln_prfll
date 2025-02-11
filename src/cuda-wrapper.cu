#include "cuda-wrapper.h"
#include <cooperative_groups.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cp_async.cuh"
#include "fastdiv.cuh"
#include "frag_layout_swizzle.cuh"
#include "math.cuh"
//#include "mma.cuh"
#include "page.cuh"
#include "permuted_smem.cuh"
#include "pos_enc.cuh"
//#include "utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"

namespace cg = cooperative_groups;
constexpr uint32_t WARP_SIZE = 32;

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}



/**
 * CUDA kernel that computes reciprocal values for a given vector
 */


template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillPagedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  uint8_t* maybe_custom_mask;
  IdType* q_indptr;
  IdType* maybe_mask_indptr;
  IdType* maybe_q_rope_offset;  // maybe_q_rope_offset is only used for fused-rope attention
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  bool* block_valid_mask;
  IdType* kv_chunk_size_ptr;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ BatchPrefillPagedParams()
      : q(nullptr),
        paged_kv(),
        maybe_custom_mask(nullptr),
        q_indptr(nullptr),
        maybe_mask_indptr(nullptr),
        maybe_q_rope_offset(nullptr),
        o(nullptr),
        lse(nullptr),
        maybe_alibi_slopes(nullptr),
        group_size(),
        num_qo_heads(0),
        q_stride_n(0),
        q_stride_h(0),
        window_left(0),
        logits_soft_cap(0.0f),
        sm_scale(0.0f),
        rope_rcp_scale(0.0f),
        rope_rcp_theta(0.0f),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        block_valid_mask(nullptr),
        kv_chunk_size_ptr(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false) {}

  __host__ BatchPrefillPagedParams(DTypeQ* q, paged_kv_t<DTypeKV, IdType> paged_kv,
                                   uint8_t* maybe_custom_mask, IdType* q_indptr,
                                   IdType* maybe_mask_indptr, IdType* maybe_q_rope_offset,
                                   DTypeO* o, float* lse, float* maybe_alibi_slopes,
                                   uint32_t num_qo_heads, IdType q_stride_n, IdType q_stride_h,
                                   int32_t window_left, float logits_soft_cap, float sm_scale,
                                   float rope_scale, float rope_theta)
      : q(q),
        paged_kv(paged_kv),
        maybe_custom_mask(maybe_custom_mask),
        q_indptr(q_indptr),
        maybe_mask_indptr(maybe_mask_indptr),
        maybe_q_rope_offset(maybe_q_rope_offset),
        o(o),
        lse(lse),
        maybe_alibi_slopes(maybe_alibi_slopes),
        group_size(num_qo_heads / paged_kv.num_heads),
        num_qo_heads(num_qo_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        block_valid_mask(nullptr),
        kv_chunk_size_ptr(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

template <uint32_t NUM_WARPS_KV, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem[CTA_TILE_Q * HEAD_DIM_QK];
      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];
    };
    struct {  // NOTE(Zihao): synchronize attention states across warps
      alignas(
          16) std::conditional_t<NUM_WARPS_KV == 1, float[1],
                                 float[NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM_VO]> cta_sync_o_smem;
      alignas(16) std::conditional_t<NUM_WARPS_KV == 1, float2[1],
                                     float2[NUM_WARPS_KV * CTA_TILE_Q]> cta_sync_md_smem;
    };
    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];
  };
};

template <MaskMode MASK_MODE_, uint32_t CTA_TILE_Q_, uint32_t NUM_MMA_Q_, uint32_t NUM_MMA_KV_,
          uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_, uint32_t NUM_WARPS_Q_,
          uint32_t NUM_WARPS_KV_, PosEncodingMode POS_ENCODING_MODE_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename DTypeQKAccum_, typename IdType_,
          typename AttentionVariant_>
struct KernelTraits {
  static constexpr MaskMode MASK_MODE = MASK_MODE_;
  static constexpr uint32_t NUM_MMA_Q = NUM_MMA_Q_;
  static constexpr uint32_t NUM_MMA_KV = NUM_MMA_KV_;
  static constexpr uint32_t NUM_MMA_D_QK = NUM_MMA_D_QK_;
  static constexpr uint32_t NUM_MMA_D_VO = NUM_MMA_D_VO_;
  static constexpr uint32_t NUM_WARPS_Q = NUM_WARPS_Q_;
  static constexpr uint32_t NUM_WARPS_KV = NUM_WARPS_KV_;
  static constexpr uint32_t NUM_THREADS = NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE;
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV;
  static constexpr uint32_t HEAD_DIM_QK = NUM_MMA_D_QK * 16;
  static constexpr uint32_t HEAD_DIM_VO = NUM_MMA_D_VO * 16;
  static constexpr uint32_t UPCAST_HEAD_DIM_Q = HEAD_DIM_QK / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_HEAD_DIM_K = HEAD_DIM_QK / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_HEAD_DIM_V = HEAD_DIM_VO / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_HEAD_DIM_O = HEAD_DIM_VO / upcast_size<DTypeO_>();
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16;

  static constexpr SwizzleMode SWIZZLE_MODE_Q = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KV =
      (sizeof(DTypeKV_) == 1 && HEAD_DIM_VO == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  static constexpr uint32_t KV_THR_LAYOUT_ROW = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 8;
  static constexpr uint32_t KV_THR_LAYOUT_COL = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 8 : 4;
  static constexpr PosEncodingMode POS_ENCODING_MODE = POS_ENCODING_MODE_;
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using DTypeQKAccum = DTypeQKAccum_;
  using IdType = IdType_;
  using AttentionVariant = AttentionVariant_;

  static constexpr bool IsInvalid() {
    return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
            (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && NUM_MMA_D_VO > 4 &&
             NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
            (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
            (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
            (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama));
  }

  using SharedStorage = SharedStorageQKVO<NUM_WARPS_KV, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_QK,
                                          HEAD_DIM_VO, DTypeQ, DTypeKV, DTypeO>;

  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? DTypeQKAccum(-math::inf) : DTypeQKAccum(0.f);
};

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_q() {
  if constexpr (KTraits::NUM_WARPS_Q == 1) {
    return 0;
  } else {
    return threadIdx.y;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_kv() {
  if constexpr (KTraits::NUM_WARPS_KV == 1) {
    return 0;
  } else {
    return threadIdx.z;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx() {
  return get_warp_idx_kv<KTraits>() * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>();
}

template <typename KTraits, typename Params, typename DTypeQKAccum>
__device__ __forceinline__ void logits_transform(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint_fastdiv group_size,
    DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8]) {
  const uint32_t lane_idx = threadIdx.x, kv_head_idx = blockIdx.z;
  uint32_t q[KTraits::NUM_MMA_Q][2], r[KTraits::NUM_MMA_Q][2];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        s_frag[mma_q][mma_kv][reg_id] =
            variant.LogitsTransform(params, s_frag[mma_q][mma_kv][reg_id], batch_idx, q_idx, kv_idx,
                                    qo_head_idx, kv_head_idx);
      }
    }
  }
}

template <typename T>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset) {
  static_assert(sizeof(T) == 2);
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7
    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}



template <typename KTraits>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4], const float rope_rcp_scale,
                                               const float rope_rcp_theta) {
  constexpr uint32_t HEAD_DIM = KTraits::NUM_MMA_D_QK * 16;
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 2; ++mma_d) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      rope_freq[mma_d][j] =
          rope_rcp_scale *
          __powf(rope_rcp_theta,
                 float(2 * ((mma_d * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) %
                            (HEAD_DIM / 2))) /
                     float(HEAD_DIM));
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_states(typename KTraits::AttentionVariant variant,
                                            float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                            typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2]) {
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[mma_q][mma_d][reg_id] = 0.f;
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m[mma_q][j] = typename KTraits::DTypeQKAccum(-math::inf);
        d[mma_q][j] = 1.f;
      }
    }
  }
}

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};


template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F16_M16N8K16_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
#elif defined(FLASHINFER_MMA_F16F16F16_M16N8K8_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[2]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[2]), "r"(A[3]), "r"(B[3]), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[2]), "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[2]), "r"(A[3]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}




template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    if constexpr (std::is_same_v<T, half>) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    }
  } else {
    if constexpr (std::is_same_v<T, half>) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    }
  }
#elif defined(FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED)
  if constexpr (std::is_same_v<T, half>) {
    if constexpr (mma_mode == MMAMode::kInit) {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[2]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[2]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}




template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, uint32_t* q_smem_offset_r,
    smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8]) {
  constexpr uint32_t HEAD_DIM = KTraits::NUM_MMA_D_QK * 16;
  constexpr uint32_t UPCAST_HEAD_DIM_Q = HEAD_DIM / upcast_size<typename KTraits::DTypeQ>();
  constexpr uint32_t UPCAST_HEAD_DIM_K = HEAD_DIM / upcast_size<typename KTraits::DTypeKV>();
  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK; ++mma_d) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[mma_q]);
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, UPCAST_HEAD_DIM_Q>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, mma_d) -
                       KTraits::NUM_MMA_Q * 16 * UPCAST_HEAD_DIM_Q;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
          k_smem->ldmatrix_m8n8x4_left_half(*k_smem_offset_r, b_frag_f8);
        } else {
          k_smem->ldmatrix_m8n8x4_right_half(*k_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(
            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_f8);
      } else {
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      }
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, UPCAST_HEAD_DIM_K>(*k_smem_offset_r);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          if (mma_d == 0) {
            mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          }
        } else if (std::is_same_v<typename KTraits::DTypeQKAccum, half>) {
          if (mma_d == 0) {
            mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
                (uint32_t*)s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)s_frag[mma_q][mma_kv],
                                                      a_frag[mma_q], b_frag);
          }
        }
      }
    }
    if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
      if (mma_d % 2 == 1) {
        *k_smem_offset_r =
            k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d / 2);
      }
      *k_smem_offset_r -= KTraits::NUM_MMA_KV * 16 * UPCAST_HEAD_DIM_K;
    } else {
      *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d) -
                         KTraits::NUM_MMA_KV * 16 * UPCAST_HEAD_DIM_K;
    }
  }
  *q_smem_offset_r -= KTraits::NUM_MMA_D_QK * 2;
  *k_smem_offset_r -= KTraits::NUM_MMA_D_QK * sizeof(typename KTraits::DTypeKV);
}

template <typename KTraits>
__device__ __forceinline__ void load_q_global_smem(
    uint32_t packed_offset, const uint32_t qo_upper_bound, typename KTraits::DTypeQ* q_ptr_base,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_HEAD_DIM_Q = KTraits::UPCAST_HEAD_DIM_Q;
  const uint32_t lane_idx = threadIdx.x, warp_idx_x = get_warp_idx_q<KTraits>();

  if (get_warp_idx_kv<KTraits>() == 0) {
    uint32_t q_smem_offset_w = q_smem->get_permuted_offset<UPCAST_HEAD_DIM_Q>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2 * 2; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h;
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
          // load q fragment from gmem to smem
          q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                              q_idx < qo_upper_bound);
          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, mma_do);
          q_ptr += 8 * upcast_size<DTypeQ>();
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<4, UPCAST_HEAD_DIM_Q>(q_smem_offset_w) -
            2 * KTraits::NUM_MMA_D_QK;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    float (*rope_freq)[4]) {
  using DTypeKV = typename KTraits::DTypeKV;
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t UPCAST_HEAD_DIM_K = KTraits::UPCAST_HEAD_DIM_K;
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = threadIdx.x;
  if constexpr (KTraits::NUM_MMA_D_QK == 4 && KTraits::NUM_WARPS_Q == 4) {
    static_assert(KTraits::NUM_WARPS_KV == 1);
    const uint32_t warp_idx = get_warp_idx_q<KTraits>();
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(KTraits::NUM_MMA_KV % 2 == 0,
                  "when NUM_MMA_D_QK == 4, NUM_MMA_KV must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * UPCAST_HEAD_DIM_K;
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV / 2; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t mma_di = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                       rope_freq[mma_di], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * UPCAST_HEAD_DIM_K;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + KTraits::NUM_MMA_KV) * 16 * UPCAST_HEAD_DIM_K;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(), warp_idx_z = get_warp_idx_kv<KTraits>();
    static_assert(KTraits::NUM_MMA_D_QK % (2 * KTraits::NUM_WARPS_Q) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
    // | 1-16*NUM_MMA_KV                | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
    // | 16*NUM_MMA_KV-32*NUM_MMA_KV    | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
    // ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * KTraits::NUM_MMA_KV * 16) + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_MMA_D_QK / (2 * KTraits::NUM_WARPS_Q); ++j) {
        uint32_t mma_di = warp_idx_x + j * KTraits::NUM_WARPS_Q;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                         rope_freq[mma_di], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->template advance_offset_by_column<2 * KTraits::NUM_WARPS_Q>(
                k_smem_offset_r_first_half, mma_di);
      }
      *k_smem_offset_r += 16 * UPCAST_HEAD_DIM_K;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - KTraits::NUM_MMA_KV * 16 * UPCAST_HEAD_DIM_K;
  }
}



template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t kv_idx_base, const size_t* thr_local_kv_offset, const uint32_t kv_len) {
  // NOTE: for fp8, this function doesn't work for head_dim = 64 at the moment
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_HEAD_DIM =
      produce_v ? KTraits::UPCAST_HEAD_DIM_V : KTraits::UPCAST_HEAD_DIM_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;
  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * upcast_size<DType>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_HEAD_DIM>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_HEAD_DIM;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_HEAD_DIM>(*smem_offset);
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_HEAD_DIM;
  }
}



template <typename KTraits, typename Params>
__device__ __forceinline__ void q_smem_inplace_transform(const Params& params,
                                                         typename KTraits::AttentionVariant variant,
                                                         smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem) {
  using DTypeQ = typename KTraits::DTypeQ;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;
  constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
#pragma unroll
  for (uint32_t i = 0; i < KTraits::CTA_TILE_Q * HEAD_DIM_QK / (NUM_WARPS * 256); ++i) {
    vec_t<DTypeQ, 8> tmp;
    tmp.load((DTypeQ*)(q_smem->base) + (i * NUM_WARPS + warp_idx) * 256 + lane_idx * 8);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp[reg_id] = variant.QueryTransform(params, tmp[reg_id]);
    }
    tmp.store((DTypeQ*)(q_smem->base) + (i * NUM_WARPS + warp_idx) * 256 + lane_idx * 8);
  }
}



template <typename T, typename IdType>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                                 const float* rope_freq,
                                                                 const uint32_t qo_packed_offset,
                                                                 const uint_fastdiv group_size,
                                                                 const IdType* q_rope_offset) {
  float pos[2] = {static_cast<float>(q_rope_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_rope_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}



template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos(
    const uint32_t q_packed_idx_base, const typename KTraits::IdType* q_rope_offset,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, const uint_fastdiv group_size,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4]) {
  if (get_warp_idx_kv<KTraits>() == 0) {
    constexpr uint32_t UPCAST_HEAD_DIM_Q = KTraits::UPCAST_HEAD_DIM_Q;
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<typename KTraits::DTypeQ, typename KTraits::IdType>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di], q_packed_idx_base + mma_q * 16 + lane_idx / 4, group_size,
            q_rope_offset);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_HEAD_DIM_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_HEAD_DIM_Q;
  }
}

template <typename T>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(float((qo_packed_offset + 8 * i) / group_size) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}




template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4]) {
  if (get_warp_idx_kv<KTraits>() == 0) {
    constexpr uint32_t UPCAST_HEAD_DIM_Q = KTraits::UPCAST_HEAD_DIM_Q;
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<typename KTraits::DTypeQ>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di],
            q_packed_idx + kv_len * group_size - qo_len * group_size + mma_q * 16 + lane_idx / 4,
            group_size);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_HEAD_DIM_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_HEAD_DIM_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], smem_t<KTraits::SWIZZLE_MODE_Q>* o_smem,
    typename KTraits::DTypeO* o_ptr_base, const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound, const uint32_t o_stride_n, const uint32_t o_stride_h,
    const uint_fastdiv group_size) {
  constexpr uint32_t UPCAST_HEAD_DIM_O = KTraits::UPCAST_HEAD_DIM_O;
  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>();
  const uint32_t lane_idx = threadIdx.x;

  if (get_warp_idx_kv<KTraits>() == 0) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        uint32_t o_frag_f16[8 / 2];
        vec_cast<typename KTraits::DTypeO, float>::cast<8>((typename KTraits::DTypeO*)o_frag_f16,
                                                           o_frag[mma_q][mma_d]);

#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
        uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_HEAD_DIM_O>(
            (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx % 16,
            mma_d * 2 + lane_idx / 16);
        o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
        uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_HEAD_DIM_O>(
            (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx / 4, mma_d * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * UPCAST_HEAD_DIM_O))[lane_idx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * UPCAST_HEAD_DIM_O))[lane_idx % 4] = o_frag_f16[3];
#endif
      }
    }

    uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_HEAD_DIM_O>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2 * 2; ++j) {
        uint32_t q, r;
        group_size.divmod(o_packed_idx_base + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
        const uint32_t o_idx = q;
        typename KTraits::DTypeO* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h;
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_VO / 4; ++mma_do) {
          if (o_idx < qo_upper_bound) {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
          o_ptr += 8 * upcast_size<typename KTraits::DTypeO>();
          o_smem_offset_w = o_smem->template advance_offset_by_column<8>(o_smem_offset_w, mma_do);
        }
        o_smem_offset_w =
            o_smem->template advance_offset_by_row<4, UPCAST_HEAD_DIM_O>(o_smem_offset_w) -
            2 * KTraits::NUM_MMA_D_VO;
      }
    }
  }
}



template <typename KTraits>
__device__ __forceinline__ void normalize_d(float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                            typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2]) {
  using AttentionVariant = typename KTraits::AttentionVariant;
  if constexpr (AttentionVariant::use_softmax) {
    float d_rcp[KTraits::NUM_MMA_Q][2];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        d_rcp[mma_q][j] = (m[mma_q][j] != typename KTraits::DTypeQKAccum(-math::inf))
                              ? math::ptx_rcp(d[mma_q][j])
                              : 0.f;
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[mma_q][mma_d][reg_id] =
              o_frag[mma_q][mma_d][reg_id] * d_rcp[mma_q][(reg_id % 4) / 2];
        }
      }
    }
  }
}



template <typename KTraits>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::SharedStorage* smem_storage,
    typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2], const uint32_t warp_idx,
    const uint32_t lane_idx) {
  // only necessary when blockDim.z > 1
  if constexpr (KTraits::NUM_WARPS_KV > 1) {
    float* smem_o = smem_storage->cta_sync_o_smem;
    float2* smem_md = smem_storage->cta_sync_md_smem;
    // o: [num_warps, NUM_MMA_Q, NUM_MMA_D_VO, WARP_SIZE(32), 8]
    // md: [num_warps, NUM_MMA_Q, 16, 2 (m/d)]
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        vec_t<float, 8>::memcpy(
            smem_o + (((warp_idx * KTraits::NUM_MMA_Q + mma_q) * KTraits::NUM_MMA_D_VO + mma_d) *
                          WARP_SIZE +
                      lane_idx) *
                         8,
            o_frag[mma_q][mma_d]);
      }
    }

    if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          smem_md[((warp_idx * KTraits::NUM_MMA_Q + mma_q) * 2 + j) * 8 + lane_idx / 4] =
              make_float2(float(m[mma_q][j]), d[mma_q][j]);
        }
      }

      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        float o_scale[2][KTraits::NUM_WARPS_KV];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_new = -math::inf, d_new = 1.f;
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>()) *
                                      KTraits::NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    8 +
                                lane_idx / 4];
            float m_prev = m_new, d_prev = d_new;
            m_new = max(m_new, md.x);
            d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
          }

#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>()) *
                                      KTraits::NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    8 +
                                lane_idx / 4];
            float mi = md.x;
            o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
          }
          m[mma_q][j] = typename KTraits::DTypeQKAccum(m_new);
          d[mma_q][j] = d_new;
        }

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_o +
                    ((((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>()) * KTraits::NUM_MMA_Q +
                       mma_q) *
                          KTraits::NUM_MMA_D_VO +
                      mma_d) *
                         WARP_SIZE +
                     lane_idx) *
                        8);

#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
              o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    } else {
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_o +
                    ((((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>()) * KTraits::NUM_MMA_Q +
                       mma_q) *
                          KTraits::NUM_MMA_D_VO +
                      mma_d) *
                         WARP_SIZE +
                     lane_idx) *
                        8);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
              o_new[reg_id] += oi[reg_id];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    }
  }
}



template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], float (*d)[2]) {
  constexpr uint32_t UPCAST_HEAD_DIM_V = KTraits::UPCAST_HEAD_DIM_V;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];
  if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv],
                                                           s_frag[mma_q][mma_kv]);
      }
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);
        } else {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag[mma_q][mma_kv]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
      uint32_t b_frag[4];
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
          v_smem->ldmatrix_m8n8x4_trans_left_half(*v_smem_offset_r, b_frag_f8);
        } else {
          v_smem->ldmatrix_m8n8x4_trans_right_half(*v_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(
            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_f8);
        swap(b_frag[1], b_frag[2]);
      } else {
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
      }
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);
        }
      }
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        if (mma_d % 2 == 1) {
          *v_smem_offset_r =
              v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d / 2);
        }
      } else {
        *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d);
      }
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, UPCAST_HEAD_DIM_V>(*v_smem_offset_r) -
        sizeof(typename KTraits::DTypeKV) * KTraits::NUM_MMA_D_VO;
  }
  *v_smem_offset_r -= 16 * KTraits::NUM_MMA_KV * UPCAST_HEAD_DIM_V;
}



template <typename KTraits>
__device__ __forceinline__ void update_mdo_states(
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::DTypeQKAccum (*m)[2],
    float (*d)[2]) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  constexpr bool use_softmax = AttentionVariant::use_softmax;

  if constexpr (use_softmax) {
    if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_prev = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            float m_local =
                max(max(s_frag[mma_q][mma_kv][j * 2 + 0], s_frag[mma_q][mma_kv][j * 2 + 1]),
                    max(s_frag[mma_q][mma_kv][j * 2 + 4], s_frag[mma_q][mma_kv][j * 2 + 5]));
            m[mma_q][j] = max(m[mma_q][j], m_local);
          }
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x2));
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x1));

          float o_scale = math::ptx_exp2(m_prev - m[mma_q][j]);
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            s_frag[mma_q][mma_kv][j * 2 + 0] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 0] - m[mma_q][j]);
            s_frag[mma_q][mma_kv][j * 2 + 1] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 1] - m[mma_q][j]);
            s_frag[mma_q][mma_kv][j * 2 + 4] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 4] - m[mma_q][j]);
            s_frag[mma_q][mma_kv][j * 2 + 5] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 5] - m[mma_q][j]);
          }
        }
      }
    } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        half m_prev[2];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          m_prev[j] = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            half2 m_local = __hmax2(*(half2*)&s_frag[mma_q][mma_kv][j * 2],
                                    *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4]);
            m[mma_q][j] = __hmax(m[mma_q][j], __hmax(m_local.x, m_local.y));
          }
        }
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x2));
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x1));
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float o_scale = math::ptx_exp2(float(m_prev[j] - m[mma_q][j]));
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
          half2 m2 = make_half2(m[mma_q][j], m[mma_q][j]);
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            *(half2*)&s_frag[mma_q][mma_kv][j * 2] =
                math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2] - m2);
            *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] =
                math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] - m2);
          }
        }
      }
    }
  }
}



template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t chunk_end, const uint_fastdiv group_size,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8]) {
  const uint32_t lane_idx = threadIdx.x, kv_head_idx = blockIdx.z;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  uint32_t q[NUM_MMA_Q][2], r[NUM_MMA_Q][2];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        const bool mask =
            (!(MASK_MODE == MaskMode::kCausal
                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end)) &&
            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
        s_frag[mma_q][mma_kv][reg_id] =
            (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}



template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
    printf("\nin kernel!\n");
   using DTypeQ = typename Params::DTypeQ;
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using IdType = typename Params::IdType;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_HEAD_DIM_Q = KTraits::UPCAST_HEAD_DIM_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_HEAD_DIM_K = KTraits::UPCAST_HEAD_DIM_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_HEAD_DIM_V = KTraits::UPCAST_HEAD_DIM_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_HEAD_DIM_O = KTraits::UPCAST_HEAD_DIM_O;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    IdType* request_indices = params.request_indices;
    IdType* qo_tile_indices = params.qo_tile_indices;
    IdType* kv_tile_indices = params.kv_tile_indices;
    DTypeQ* q = params.q;
    IdType* q_indptr = params.q_indptr;
    IdType* o_indptr = params.o_indptr;
    DTypeO* o = params.o;
    float* lse = params.lse;
    bool* block_valid_mask = params.block_valid_mask;
    const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
    const bool partition_kv = params.partition_kv;
    const int32_t maybe_window_left = params.window_left;
    const uint_fastdiv& group_size = params.group_size;



    static_assert(sizeof(DTypeQ) == 2);
    static_assert(sizeof(DTypeO) == 2);
    auto block = cg::this_thread_block();
    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

    const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x, warp_idx = get_warp_idx<KTraits>(),
                   kv_head_idx = blockIdx.z;
    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
    const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                   kv_tile_idx = kv_tile_indices[bx];
    extern __shared__ uint8_t smem[];
    auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
    AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
    const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                   window_left = variant.window_left;
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    const uint32_t qo_packed_idx_base =
        (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>()) * NUM_MMA_Q * 16;
    const uint32_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;
    DTypeQ* q_ptr_base = q + q_indptr[request_idx] * q_stride_n +
                         (kv_head_idx * group_size) * q_stride_h +
                         (lane_idx % 8) * upcast_size<DTypeQ>();
    DTypeO* o_ptr_base =
        partition_kv
            ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                  (kv_head_idx * group_size) * o_stride_h + (lane_idx % 8) * upcast_size<DTypeO>()
            : o + o_indptr[request_idx] * o_stride_n + (kv_head_idx * group_size) * o_stride_h +
                  (lane_idx % 8) * upcast_size<DTypeO>();

    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<UPCAST_HEAD_DIM_Q>(
        get_warp_idx_q<KTraits>() * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                                q_stride_h, group_size, &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      IdType* q_rope_offset = nullptr;
      //TODO: couldn't find this
      //if constexpr (has_maybe_q_rope_offset_v<Params>) {
      //  q_rope_offset = params.maybe_q_rope_offset;
      //}
      if (q_rope_offset == nullptr) {
        q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size,
                                             &qo_smem, &q_smem_offset_r, rope_freq);
      } else {
        q_smem_inplace_apply_rotary_with_pos<KTraits>(
            qo_packed_idx_base, q_rope_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq);
      }
      block.sync();
    }
    q_smem_inplace_transform<KTraits>(params, variant, &qo_smem);

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
    size_t thr_local_kv_offset[NUM_MMA_KV * KV_THR_LAYOUT_COL / 2 / NUM_WARPS_Q];

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_HEAD_DIM_K>(
                 get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_HEAD_DIM_V>(
                 get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_HEAD_DIM_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_HEAD_DIM_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);
    const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t packed_page_iter_base =
        paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
    for (uint32_t i = 0;
         i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                    lane_idx / KV_THR_LAYOUT_COL +
                                    KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx,
          (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>(), last_indptr);
    }
    page_produce_kv<false, KTraits>(k_smem, &k_smem_offset_w, paged_kv, 0, thr_local_kv_offset,
                                    chunk_size);
    cp_async::commit_group();
    page_produce_kv<true, KTraits>(v_smem, &v_smem_offset_w, paged_kv, 0, thr_local_kv_offset,
                                   chunk_size);
    cp_async::commit_group();

    const uint32_t num_iterations = ceil_div(
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size, sub_if_greater_or_zero(
                                   kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size,
                                   chunk_start))
             : chunk_size),
        CTA_TILE_KV);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                        qo_len + window_left + chunk_start),
                 CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                          chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      packed_page_iter_base += CTA_TILE_KV;
#pragma unroll
      for (uint32_t i = 0;
           i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                      lane_idx / KV_THR_LAYOUT_COL +
                                      KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                  page_iter, entry_idx);
        thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
            page_iter, kv_head_idx, entry_idx,
            (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>(), last_indptr);
      }
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<KTraits>(
            (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
                chunk_start + iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<KTraits>(
            params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag);
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(s_frag, o_frag, m, d);

      block.sync();
      page_produce_kv<false, KTraits>(k_smem, &k_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                                      thr_local_kv_offset, chunk_size);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      page_produce_kv<true, KTraits>(v_smem, &v_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                                     thr_local_kv_offset, chunk_size);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d<KTraits>(o_frag, m, d);

    const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

    // write_back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr) {
        if (get_warp_idx_kv<KTraits>() == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_upper_bound) {
                if (partition_kv) {
                  lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) *
                          num_qo_heads +
                      qo_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
}




template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM_QK * sizeof(DTypeQ)) - NUM_MMA_Q * NUM_WARPS_Q) /
      (2 * NUM_WARPS_KV);

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flash-infer-ai/flash-infer/issues)"
                 " and report the issue to the developers.";
      //FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);

      //std::cout << "Params:" << typeid(Params).name() << "\n";


      auto kernel = BatchPrefillWithPagedKVCacheKernel<KTraits, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&params};
	printf("dispatching with %d,%d,%d", nthrs, nblks, smem_size); 
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM_VO, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.merge_indptr, o, params.max_total_num_rows,
                                         params.total_num_rows, num_qo_heads, HEAD_DIM_VO, stream));
        }
      }
    }
  });
  return cudaSuccess;
}


struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t total_num_rows_offset;
  int64_t cta_tile_q;
  int64_t request_indices_offset;
  int64_t qo_tile_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t merge_indptr_offset;
  int64_t o_indptr_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t v_offset;
  int64_t s_offset;
  int64_t block_valid_mask_offset;
  bool enable_cuda_graph;
  bool split_kv;

  PrefillPlanInfo()
      : padded_batch_size(0),
        total_num_rows(0),
        total_num_rows_offset(0),
        cta_tile_q(0),
        request_indices_offset(0),
        qo_tile_indices_offset(0),
        kv_tile_indices_offset(0),
        merge_indptr_offset(0),
        o_indptr_offset(0),
        kv_chunk_size_ptr_offset(0),
        v_offset(0),
        s_offset(0),
        block_valid_mask_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert PrefillPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            total_num_rows,
            total_num_rows_offset,
            cta_tile_q,
            request_indices_offset,
            qo_tile_indices_offset,
            kv_tile_indices_offset,
            merge_indptr_offset,
            o_indptr_offset,
            kv_chunk_size_ptr_offset,
            v_offset,
            s_offset,
            block_valid_mask_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to PrefillPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 15) {
      std::ostringstream err_msg;
      std::cout << "PrefillPlanInfo::FromVector: vec.size() should be 14, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    total_num_rows_offset = vec[2];
    cta_tile_q = vec[3];
    request_indices_offset = vec[4];
    qo_tile_indices_offset = vec[5];
    kv_tile_indices_offset = vec[6];
    merge_indptr_offset = vec[7];
    o_indptr_offset = vec[8];
    kv_chunk_size_ptr_offset = vec[9];
    v_offset = vec[10];
    s_offset = vec[11];
    block_valid_mask_offset = vec[12];
    enable_cuda_graph = vec[13];
    split_kv = vec[14];
  }
};

#include <iostream>
#include <fstream>
void* loadfromfile(std::string filename) {
    //std::string filename = "float_workspace_buffer.bin";
    std::vector<int> data_array;
    char temp_data;
    std::ifstream input_file(filename, std::ios::binary);
    //printf("%s", filename);
    if (!input_file.is_open()) {
        std::cout << "\nError opening file: " << filename << std::endl;
        return nullptr;
    }
    while (input_file.read(reinterpret_cast<char*>(&temp_data), sizeof(temp_data))) {
        data_array.push_back(temp_data);
    }
    if (input_file.eof()) {
        //std::cout << "End of file reached." << std::endl;
    } else if (input_file.fail()) {
        std::cout << "\nError reading file." << std::endl;
    }
    input_file.close();
    return (void*)(&data_array[0]);
}

template <typename T>
T* GetPtrFromBaseOffset(void* base_ptr, int64_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(base_ptr) + offset);
}

#include "batch_prefill_config.inc"
#if 0

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using IdType = int32_t;

#define ADDITIONAL_FUNC_PARAMS BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER BATCH_PREFILL_ADDITIONAL_PARAMS_SETTER

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,    \
                         POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP,              \
                         USE_FP16_QK_REDUCTION, AttentionVariant, RaggedParams, PagedParams, ...) \
  {                                                                                               \
    DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {                                                \
      return DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(                                                 \
          q_scalar_type, kv_scalar_type, DTypeQ, DTypeKV, [&] {                                   \
            using DTypeO = DTypeQ;                                                                \
            using RaggedParams = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;       \
            using PagedParams = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;         \
            constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;                            \
            constexpr bool USE_FP16_QK_REDUCTION = false;                                         \
            constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;                      \
            return DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, [&] {                              \
              [[maybe_unused]] constexpr int HEAD_DIM_VO = HEAD_DIM_QK;                                            \
              return DISPATCH_BOOL(window_left > -1, USE_SLIDING_WINDOW, [&] {                    \
                return DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, [&] {            \
                  using AttentionVariant =                                                        \
                      DefaultAttention</*use_custom_mask=*/use_custom_mask, USE_SLIDING_WINDOW,   \
                                       USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;            \
                  __VA_ARGS__();                                                                  \
                  return true;                                                                    \
                });                                                                               \
              });                                                                                 \
            });                                                                                   \
          });                                                                                     \
    });                                                                                           \
  }
#endif

#if 1
#include <torch/torch.h>
void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::optional<at::Tensor> maybe_lse, unsigned int mask_mode_code, unsigned int layout,
    int32_t window_left ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream) {

using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int32_t;

  PrefillPlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  
  auto device = q.device();
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

printf("\n--- Entering Launch ---\n");

/*  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {*/
//        PagedParams params;

using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int32_t;
int HEAD_DIM_VO = 128; //from test

using Params = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
Params params;
using AttentionVariant1 = DefaultAttention<true, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;
//using AttentionVariant1 = DefaultAttention</*use_custom_mask=*/use_custom_mask, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());

        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        //ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;


        BatchPrefillWithPagedKVCacheDispatched<128, 128, 128, PosEncodingMode::kNone, 0, MaskMode::kCustom, AttentionVariant1, Params>(
           params,
           //half* tmp_v, float* tmp_s,
           tmp_v, tmp_s, //no kv_partition for now
           stream);

        DTypeO* ref_output = (DTypeO*)loadfromfile("after_o.bin");

        if (ref_output[0] != params.o[0])
                printf("missmatch!!!!");


#if 0
        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, stream);
        });
        TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
#endif
//      });
}

#endif 

//#define LOD_FROM_PT
#ifdef LOD_FROM_PT
#include <torch/torch.h>

#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>

void lodVec(const char* filename, std::vector<int64_t>* plan_info_vec) { 
    FILE* file = fopen(filename, "r");
    if (file == nullptr) {
        printf("Error opening file\n");
        exit(EXIT_FAILURE);
    }
    int num;
    while (fscanf(file, "%d", &num) == 1) {
        plan_info_vec->push_back(num);
    }
    if (ferror(file)) {
        printf("Error reading file\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
}


void lodFromPt(std::string filename, torch::Tensor* retTens) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Could not open file" << std::endl;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
         std::cerr << "Error while reading the file" << std::endl;
    }

    torch::IValue data = torch::pickle_load(buffer);

    if (data.isTensor()) {
        *retTens = data.toTensor();
        //std::cout << "Tensor loaded successfully with shape: " << tensor.sizes() << std::endl;
    } else {
        std::cerr << "Loaded data is not a tensor." << filename << std::endl;
    }

}


torch::Tensor lodFromPt(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Could not open file" << std::endl;
        return torch::empty({0});
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
         std::cerr << "Error while reading the file" << std::endl;
	 return torch::empty({0});
    }

    torch::IValue data = torch::pickle_load(buffer);

    if (data.isTensor()) {
        return data.toTensor();
        //std::cout << "Tensor loaded successfully with shape: " << tensor.sizes() << std::endl;
    } else {
        std::cerr << "Loaded data is not a tensor." << filename << std::endl;
    }

}


void call_it() {
 printf("\nLoad Tensors and Run Prefill here...\n");

//from flashinfer run...
using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int32_t;
torch::Tensor float_workspace_buffer; lodFromPt("float_workspace_buffer.pt", &float_workspace_buffer); float_workspace_buffer = float_workspace_buffer.to(torch::kChar, 0);
torch::Tensor int_workspace_buffer; lodFromPt("int_workspace_buffer.pt", &int_workspace_buffer); int_workspace_buffer = int_workspace_buffer.to(torch::kChar, 0);
torch::Tensor q; lodFromPt("q.pt", &q); q = q.to(torch::kChar, 0);
torch::Tensor paged_k_cache;  lodFromPt("paged_k_cache.pt", &paged_k_cache); paged_k_cache = paged_k_cache.to(torch::kChar, 0);
torch::Tensor paged_v_cache; lodFromPt("paged_v_cache.pt", &paged_v_cache); paged_v_cache = paged_v_cache.to(torch::kChar, 0);
torch::Tensor qo_indptr; lodFromPt("qo_indptr.pt", &qo_indptr); qo_indptr = qo_indptr.to(torch::kChar, 0);
torch::Tensor paged_kv_indptr; lodFromPt("paged_kv_indptr.pt", &paged_kv_indptr); paged_kv_indptr = paged_kv_indptr.to(torch::kChar, 0);
torch::Tensor paged_kv_indices; lodFromPt("paged_kv_indices.pt", &paged_kv_indices); paged_kv_indices = paged_kv_indices.to(torch::kChar, 0);
torch::Tensor paged_kv_last_page_len; lodFromPt("paged_kv_last_page_len.pt", &paged_kv_last_page_len); paged_kv_last_page_len = paged_kv_last_page_len.to(torch::kChar, 0);
torch::Tensor o; lodFromPt("o.pt", &o); o = o.to(torch::kChar, 0);
torch::Tensor maybe_lse; lodFromPt("maybe_lse.pt", &maybe_lse); 
//maybe_lse = (maybe_lse->defined()) ? std::optional<torch::Tensor>(maybe_lse->to(torch::kChar, 0)) : std::nullopt;
torch::Tensor maybe_custom_mask; lodFromPt("maybe_custom_mask.pt", &maybe_custom_mask); 
//maybe_custom_mask =  (maybe_custom_mask->defined()) ? std::optional<torch::Tensor>(maybe_custom_mask->to(torch::kChar, 0)) : std::nullopt;
torch::Tensor maybe_mask_indptr; lodFromPt("maybe_mask_indptr.pt", &maybe_mask_indptr); 
//maybe_mask_indptr = (maybe_mask_indptr->defined()) ? std::optional<torch::Tensor>(maybe_mask_indptr->to(torch::kChar, 0)) : std::nullopt;
torch::Tensor maybe_alibi_slopes; lodFromPt("maybe_alibi_slopes.pt", &maybe_alibi_slopes); 
//maybe_alibi_slopes = (maybe_alibi_slopes->defined()) ? std::optional<torch::Tensor>(maybe_alibi_slopes->to(torch::kChar, 0)) : std::nullopt;

int mask_mode_code = 0;
int layout= 1;
int window_left = -1;
float logits_soft_cap= 0.0;
float sm_scale =  0.08838834764831843;
float rope_scale =  1.0;
float rope_theta = 10000.0;
torch::Tensor after_o; lodFromPt("after_o.pt", &after_o);

std::vector<int64_t> plan_info_vec;
lodVec("plan_info_vec.txt", &plan_info_vec);
std::cout << "plan value 0" << plan_info_vec[0];

cudaStream_t myStream;
cudaStreamCreate(&myStream);

printf("\n--- Done loading ---\n");

//from flashinfer run..
/*                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            paged_k_cache,
                            paged_v_cache,
                            qo_indptr,
                            paged_kv_indptr,
                            paged_kv_indices,
                            paged_kv_last_page_len,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            maybe_custom_mask,
                            maybe_mask_indptr,
                            maybe_alibi_slopes,
                            logits_soft_cap,
                            sm_scale,
                            1.0 / rope_scale,  # rope_rcp_scale
                            1.0 / rope_theta,  # rope_rcp_theta
                            get_cuda_stream(device),*/
 
BatchPrefillWithPagedKVCacheRun(float_workspace_buffer, int_workspace_buffer,
    plan_info_vec, q, paged_k_cache,
    paged_v_cache, qo_indptr, paged_kv_indptr,
    paged_kv_indices, paged_kv_last_page_len, o,
    maybe_lse, mask_mode_code, layout,
    window_left, /*ADDITIONAL_FUNC_PARAMS,*/ std::nullopt, std::nullopt, maybe_alibi_slopes, /*maybe_custom_mask, maybe_mask_indptr, maybe_alibi_slopes,*/ logits_soft_cap, sm_scale, 1.0 / rope_scale, 1.0 / rope_theta, reinterpret_cast<int64_t>(myStream)); 

printf("\n--- Done running ---\n");

}

#else

void call_it() {
 printf("\nLoad Tensors and Run Prefill here...\n");

//TestBatchPagedPrefillKernelOneHotCorrectness<half>(false);
using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int32_t;

using Params = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
Params params;

int mask_mode = 0;
int layout= 1;
int window_left = -1;
float logits_soft_cap= 0.0;
float sm_scale =  0.08838834764831843;
float rope_scale =  1.0;
float rope_theta = 10000.0;

//params.padded_batch_size = 57;
//params.num_qo_heads = 4;
//params.paged_kv.num_heads = 4;

std::vector<int64_t> plan_info_vec;
    const char* filename = "plan_info_vec.txt";
    FILE* file = fopen(filename, "r");
    if (file == nullptr) {
        printf("Error opening file\n");
        exit(EXIT_FAILURE);
    }
    int num;
    while (fscanf(file, "%d", &num) == 1) {
        plan_info_vec.push_back(num);
    }
    if (ferror(file)) {
        printf("Error reading file\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);

int q_stride_0_ = 512;
int q_stride_1_ = 128;
int q_size_0_ = 444;
int q_size_1_ = 4;
int q_size_2_ = 128;
int paged_kv_indptr_size_0_= 13;
int paged_k_cache_size_1_ = 4;
int paged_k_cache_size_2_ = 5;
int lse_size_0_ = 444;
int lse_size_1 = 4;
int HEAD_DIM_VO = 128; //from test


using AttentionVariant1 = DefaultAttention<true, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;
//AttentionVariant1 A10();

  PrefillPlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  //auto device = q.device();
  int64_t batch_size = paged_kv_indptr_size_0_ - 1;
  int64_t num_qo_heads = q_size_1_;
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q_size_2_;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache_size_1_;
    page_size = paged_k_cache_size_2_;
  } else {
    page_size = paged_k_cache_size_1_;
    num_kv_heads = paged_k_cache_size_2_;
  }

  bool maybe_lse = true; //have lse
  if (maybe_lse) {
    //const auto& lse = loadfromfile("maybe_lse.bin");//*maybe_lse;
    //TORCH_CHECK(lse_size_0_ == q_size_0_, lse_size_0_, q_size_0_);
    //TORCH_CHECK(lse_size_1_ == q_size_1_, lse_size_1_, q_size_1_);
  }

//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>


//#include <torch/torch.h>
  //void* float_buffer_ptr = static_cast<void*>(torch::load("float_workspace_buffer.pt").data_ptr());
  void* float_buffer_ptr = static_cast<void*>(loadfromfile("float_workspace_buffer.bin"));// float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(loadfromfile("int_workspace_buffer.bin"));// int_workspace_buffer.data_ptr());

  //const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  //auto q_scalar_type = Half;//q.scalar_type();
  //auto kv_scalar_type = Half;//paged_k_cache.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q_stride_0_;
  const auto q_stride_h = q_stride_1_;

  // get kv_cache_strides
  const int64_t kv_cache_strides[] = {5120, 640, 128, 1}; //from dump //nullptr;
  //auto k_strides = paged_k_cache.strides();
  //auto v_strides = paged_v_cache.strides();
  //TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  //kv_cache_strides = k_strides.data();
  //kv_cache_strides = int64_t[5120, 640, 128, 1];// (const int64_t*)loadfromfile("k_strides.bin");

cudaStream_t myStream;
    cudaStreamCreate(&myStream);

        params.q = static_cast<DTypeQ*>(loadfromfile("q.bin")); //q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(loadfromfile("paged_k_cache.bin")), //paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(loadfromfile("paged_v_cache.bin")), kv_cache_strides, //paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(loadfromfile("paged_kv_indices.bin")), // paged_kv_indices.data_ptr()),
            static_cast<IdType*>(loadfromfile("paged_kv_indptr.bin")), // paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(loadfromfile("paged_kv_last_page_len.bin"))); // paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(loadfromfile("qo_indptr.bin"));// qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(loadfromfile("o.bin")); // o.data_ptr());

        params.lse = !maybe_lse ? nullptr : static_cast<float*>(loadfromfile("maybe_lse.bin")); // maybe_lse->data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        //ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        BatchPrefillWithPagedKVCacheDispatched<128, 128, 128, PosEncodingMode::kNone, 0, MaskMode::kCustom, AttentionVariant1, Params>(
           params, 
           //half* tmp_v, float* tmp_s,
           tmp_v, tmp_s, //no kv_partition for now
           myStream);

// Synchronize and check for errors
auto the_error = cudaDeviceSynchronize();  // Or cudaStreamSynchronize(stream) if using streams
if (the_error != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(the_error) << std::endl;
    // Handle error
}


	DTypeO* ref_output = (DTypeO*)loadfromfile("after_o.bin");

	if (ref_output[0] != params.o[0])
		printf("missmatch!!!!");

#if 0
        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, stream);
        });
#endif


#if 0 
BatchPrefillWithPagedKVCacheDispatched<128, 128, 128, PosEncodingMode::kNone, 0, MaskMode::kCustom, AttentionVariant1, Params>(
    params, 
    //half* tmp_v, float* tmp_s,
    nullptr, nullptr, //no kv_partition for now
    myStream);
#endif
 printf("\nCompleted...\n");

}
#endif



