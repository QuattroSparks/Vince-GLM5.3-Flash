#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void pf_xchg_cuda(const at::Tensor&, int64_t, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

// One RoCE ring exchange (pf_xchg.cu): block p of ``in`` (at p * in_stride 16-byte packs) to rank p, and every peer's
// block for this rank into ``out`` at p * out_stride packs; block_packs 16-byte packs a block.
void exchange(const at::Tensor& in, int64_t in_stride, at::Tensor out, int64_t out_stride, int64_t block_packs,
              int64_t recv_base, int64_t flag_base, int64_t send_base, int64_t ctrl, int64_t slot_bytes,
              int64_t send_slot_bytes, int64_t counters, int64_t spin_limit, int64_t world, int64_t rank, int64_t slots,
              int64_t flag_stride, int64_t n_flag, int64_t grid, int64_t threads) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "pf exchange: CUDA tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "pf exchange: 16-byte aligned tensors");
    TORCH_CHECK(block_packs >= 0 && world * n_flag <= 32, "pf exchange: geometry");   // 0: exchange_close (the epoch only)
    const at::cuda::CUDAGuard guard(in.device());
    pf_xchg_cuda(in, in_stride, out, out_stride, block_packs, recv_base, flag_base, send_base, ctrl, slot_bytes,
                 send_slot_bytes, counters, spin_limit, world, rank, slots, flag_stride, n_flag, grid, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exchange", &exchange, "RoCE ring exchange (all-to-all), b12x's RoCEnante protocol in exchange mode");
}
