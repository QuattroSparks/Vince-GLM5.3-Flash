#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void pf_dequant_q4_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t);

// A Q4 weight as bf16 [npad, K], the prefill kernel's own B values (pf_dequant.cu).
void dequant_q4(const at::Tensor& weight, const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out,
                int64_t K, int64_t gs) {
    c10::cuda::CUDAGuard guard(weight.device());
    pf_dequant_q4_cuda(weight, scales, biases, out, K, gs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_q4", &dequant_q4);
}
