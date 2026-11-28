#include "utils.cuh"
#include "reduction_utils.cuh"

namespace vllm {

template<typename T>
__device__ __forceinline__ T silu(const T &x) {
    // x * sigmoid(x)
    return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template<typename scalar_t>
__global__ void silu_and_mul_kernel(scalar_t *__restrict__ out,         // [..., d]
                                    const scalar_t *__restrict__ input, // [..., 2 * d]
                                    const int d) {

    const int token_idx        = blockIdx.x;
    const int64_t token_idx_d  = token_idx * int64_t(d);
    const int64_t token_idx_2d = token_idx_d * 2;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
        const scalar_t x       = __ldg(&input[token_idx_2d + idx]);
        const scalar_t y       = __ldg(&input[token_idx_2d + d + idx]);
        out[token_idx_d + idx] = silu(x) * y;
    }
}

// dequant int32 input, apply silu and mul, then per token quant to int8
template<typename scale_type, bool use_per_token_quant>
__global__ void dequant_silu_and_mul_quant_kernel(int8_t *__restrict__ out,          // [..., d]
                                                  const int32_t *__restrict__ input, // [..., 2 * d]
                                                  const int d,
                                                  const float scale_gate,
                                                  const float scale_up,
                                                  scale_type scale_out,             // [num_tokens]
                                                  float *__restrict__ tmp = nullptr // [num_tokens, d]
) {
    const int token_idx = blockIdx.x;
    if constexpr (use_per_token_quant) {
        float amax_val   = 0.0f;
        const float zero = 0.0f;

        for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const float x            = (float)__ldg(&input[token_idx * 2 * d + idx]) * scale_gate;
            const float y            = (float)__ldg(&input[token_idx * 2 * d + d + idx]) * scale_up;
            float t                  = silu(x) * y;
            tmp[token_idx * d + idx] = t;
            t                        = t > zero ? t : -t;
            if (t > amax_val)
                amax_val = t;
        }

        __shared__ float s_amax;
        const float block_amax_val = blockReduceMax(amax_val);
        if (threadIdx.x == 0) {
            s_amax               = block_amax_val;
            scale_out[token_idx] = block_amax_val / 127.0f;
        }
        __syncthreads();

        float tmp_scale = 127.0f / s_amax;
        for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
            out[token_idx * d + idx] = float_to_int8_rn(tmp_scale * tmp[token_idx * d + idx]);
        }
    } else {
        for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const float x            = (float)__ldg(&input[token_idx * 2 * d + idx]) * scale_gate;
            const float y            = (float)__ldg(&input[token_idx * 2 * d + d + idx]) * scale_up;
            out[token_idx * d + idx] = float_to_int8_rn(silu(x) * y / scale_out);
        }
    }
}
} // namespace vllm

namespace vllm {

// Element-wise activation kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &)>
__global__ void activation_kernel(scalar_t *__restrict__ out,         // [..., d]
                                  const scalar_t *__restrict__ input, // [..., d]
                                  const int d) {
    const int token_idx = blockIdx.x;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
        const scalar_t x         = __ldg(&input[token_idx * d + idx]);
        out[token_idx * d + idx] = ACT_FN(x);
    }
}

} // namespace vllm

namespace vllm {

template<typename T>
__device__ __forceinline__ T gelu_new_kernel(const T &x) {
    const float x3 = (float)(x * x * x);
    const T t      = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
    return ((T)0.5) * x * (((T)1.0) + t);
}

template<typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T &x) {
    const float f = (float)x;
    const T t     = (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
    return ((T)0.5) * x * (((T)1.0) + t);
}

template<typename T>
struct SituVecTraits;

template<>
struct SituVecTraits<half> {
    using vec_t = half2;
    __device__ __forceinline__ static float2 to_float2(vec_t v) { return __half22float2(v); }
    __device__ __forceinline__ static vec_t from_float2(float2 v) { return __floats2half2_rn(v.x, v.y); }
};

template<>
struct SituVecTraits<__nv_bfloat16> {
    using vec_t = __nv_bfloat162;
    __device__ __forceinline__ static float2 to_float2(vec_t v) { return __bfloat1622float2(v); }
    __device__ __forceinline__ static vec_t from_float2(float2 v) { return __floats2bfloat162_rn(v.x, v.y); }
};

template<>
struct SituVecTraits<float> {
    using vec_t = float2;
    __device__ __forceinline__ static float2 to_float2(vec_t v) { return v; }
    __device__ __forceinline__ static vec_t from_float2(float2 v) { return v; }
};

__device__ __forceinline__ float situ_tanh(float x) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float situ_sigmoid(float x) {
    return fmaf(0.5f, situ_tanh(0.5f * x), 0.5f);
}

__device__ __forceinline__ float2 situ_pair(float2 hidden, float2 gate) {
    const float g0 = situ_tanh(gate.x * 0.25f) * situ_sigmoid(gate.x);
    const float g1 = situ_tanh(gate.y * 0.25f) * situ_sigmoid(gate.y);
    const float h0 = situ_tanh(hidden.x * 0.04f);
    const float h1 = situ_tanh(hidden.y * 0.04f);
    return make_float2((4.0f * g0) * (25.0f * h0), (4.0f * g1) * (25.0f * h1));
}

template<typename scalar_t>
__global__ void situ_and_mul_kernel(scalar_t *__restrict__ out,         // [..., d]
                                    const scalar_t *__restrict__ input, // [..., 2 * d]
                                    const int d) {
    using traits = SituVecTraits<scalar_t>;
    using vec_t  = typename traits::vec_t;

    const int token_idx = blockIdx.x;
    const scalar_t *up  = input + int64_t(token_idx) * 2 * d;
    const scalar_t *gate = up + d;
    scalar_t *dst       = out + int64_t(token_idx) * d;

    for (int idx = threadIdx.x * 2; idx < d; idx += blockDim.x * 2) {
        const float2 hidden = traits::to_float2(*reinterpret_cast<const vec_t *>(up + idx));
        const float2 gate_v = traits::to_float2(*reinterpret_cast<const vec_t *>(gate + idx));
        *reinterpret_cast<vec_t *>(dst + idx) = traits::from_float2(situ_pair(hidden, gate_v));
    }
}

} // namespace vllm
