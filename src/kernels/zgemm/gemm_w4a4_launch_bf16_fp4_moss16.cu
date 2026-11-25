#include "gemm_w4a4_launch_impl.cuh"

namespace nunchaku::kernels {

template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16_MOSS16, true>;
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16_MOSS16, false>;
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16_8W_STREAM, true>;
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16_8W_STREAM, false>;

}; // namespace nunchaku::kernels
