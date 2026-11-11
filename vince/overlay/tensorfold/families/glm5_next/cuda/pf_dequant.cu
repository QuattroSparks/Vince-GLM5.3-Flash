// Prefill work (2026-11-08): a Q4 weight (qmm.Q4: int32 words [n/64][k/gs][8][32][gs/32], bf16 scales and biases
// [k/gs][npad]) as a plain bf16 matrix W [npad, K] (row n = output column n, K contiguous), each value made by exactly
// the ops qmm_prefill.cu's prefill_kernel feeds its mma: bf16(fma.rn.bf16x2(pair(word, shift), (s, s), (b, b))).
// A GEMM that runs the same m16n8k16 bf16 mma over k16 steps 0, 16, 32, ... into one fp32 chain then gives the
// prefill kernel's bits.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t pair(uint32_t w, int s) {     // qmm_frag::pair (sm_90+ form)
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x43004300u;
    uint32_t r;
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
    return r;
}

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {   // qmm_prefill.cu's fma2
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// One thread a (tile t, group g, n8 block j, lane): its GS/32 words hold, for each k16 step kt of the group, the
// B fragment (b0, b1) of column n = t*64 + j*8 + lane/4 at rows k = g*GS + kt*16 + 2*(lane%4) + {0, 1} (b0) and
// + 8 + {0, 1} (b1), as the mma's B fragment layout puts them.
template <int GS>
__global__ void dequant_kernel(const uint32_t* __restrict__ w, const uint16_t* __restrict__ scales,
                               const uint16_t* __restrict__ biases, uint16_t* __restrict__ out, int K, int npad,
                               int tiles) {
    const int KG = K / GS;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;      // over tiles * KG * 8 * 32
    const long long total = (long long)tiles * KG * 8 * 32;
    if (idx >= total) return;
    const int lane = (int)(idx & 31);
    const int j = (int)((idx >> 5) & 7);
    const long long tg = idx >> 8;                 // t * KG + g
    const int g = (int)(tg % KG);
    const int t = (int)(tg / KG);
    const uint32_t* src = w + idx * (GS / 32);     // [t][g][j][lane][GS/32]
    const int n = t * 64 + j * 8 + (lane >> 2);
    const uint32_t s = (uint32_t)scales[(size_t)g * npad + n] * 0x10001u;
    const uint32_t b = (uint32_t)biases[(size_t)g * npad + n] * 0x10001u;
    uint16_t* row = out + (size_t)n * K + (size_t)g * GS + 2 * (lane & 3);
#pragma unroll
    for (int kt = 0; kt < GS / 16; ++kt) {
        const uint32_t word = src[kt / 2];
        const uint32_t b0 = fma2(pair(word, (kt & 1) * 8), s, b);
        const uint32_t b1 = fma2(pair(word, (kt & 1) * 8 + 4), s, b);
        *reinterpret_cast<uint32_t*>(row + kt * 16) = b0;          // k, k + 1 (low half first)
        *reinterpret_cast<uint32_t*>(row + kt * 16 + 8) = b1;      // k + 8, k + 9
    }
}

}  // namespace

void pf_dequant_q4_cuda(const at::Tensor& weight, const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out,
                        int64_t K, int64_t gs) {
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kInt && weight.is_contiguous(), "pf_dequant: words");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(1) == K, "pf_dequant: out [npad, K]");
    const int npad = (int)scales.size(1);
    TORCH_CHECK(out.size(0) >= npad && npad % 64 == 0, "pf_dequant: out rows");
    const int tiles = npad / 64;
    const long long total = (long long)tiles * (K / gs) * 8 * 32;
    const int threads = 256;
    const unsigned blocks = (unsigned)((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto w = reinterpret_cast<const uint32_t*>(weight.data_ptr());
    auto s = reinterpret_cast<const uint16_t*>(scales.data_ptr());
    auto b = reinterpret_cast<const uint16_t*>(biases.data_ptr());
    auto o = reinterpret_cast<uint16_t*>(out.data_ptr());
    if (gs == 64)
        dequant_kernel<64><<<blocks, threads, 0, stream>>>(w, s, b, o, (int)K, npad, tiles);
    else if (gs == 32)
        dequant_kernel<32><<<blocks, threads, 0, stream>>>(w, s, b, o, (int)K, npad, tiles);
    else
        TORCH_CHECK(false, "pf_dequant: group size 32 or 64");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
