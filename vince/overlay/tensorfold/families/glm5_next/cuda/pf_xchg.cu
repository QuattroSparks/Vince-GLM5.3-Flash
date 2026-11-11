// Prefill work (2026-11-08): the GPU side of the RoCE ring exchange (pf_xchg_proxy.c's exchange mode). After b12x's
// RoCEnante one-shot protocol (https://github.com/local-inference-lab/b12x, Apache-2.0) as TensorFold's roce.cu runs
// it (stages, flags and memory orderings), as three launches so that only one block ever spins:
//   stage: every block copies its share of the destinations' blocks (block p from in + p * in_stride) into the send
//          slot (seq & 1); the last block to finish publishes nbytes and rings ctrl.seq after a system fence;
//   wait:  one block, a thread a (peer, stripe), spins on that stripe's flag with system-scope acquire loads (a limit
//          poisons the runtime);
//   copy:  every block copies each peer's block from its NIC-written receive slot to out + peer * out_stride (its own
//          block is left alone); the last block advances the device epoch.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

constexpr int C_SEQ = 0, C_NBYTES = 1, C_ERROR = 2, C_PEER = 3, C_SLOT_NBYTES = 4, C_HCA = 6;
// device counters: [0] epoch, [1] stage arrivals, [2] copy arrivals, [3] poison

__global__ void stage_kernel(const uint4* __restrict__ in, long long in_stride, int block_packs, uint32_t nbytes,
                             char* send_base, long long send_slot_bytes, uint32_t* ctrl, uint32_t* counters,
                             int world, int rank) {
    if (ld_relaxed_gpu(counters + 3) != 0u) return;
    const uint32_t seq = ld_relaxed_gpu(counters) + 1u, slot = seq & 1u;
    uint4* send = reinterpret_cast<uint4*>(send_base + (long long)slot * send_slot_bytes);
    // in_stride < 0 (TF_GLM_PF_XCHG_BCAST): one block for every peer, staged once at the slot's start; bit 0 of the
    // slot's byte count tells the proxy
    // in_stride == -2 (TF_GLM_PF_SI): the blocks are in the send slot already (the fills wrote them there): nothing to
    // copy, only the doorbell
    const bool bcast = in_stride == -1, staged = in_stride == -2;
    const long long total = staged ? 0 : bcast ? (long long)block_packs : (long long)world * block_packs;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        if (bcast) {
            send[i] = in[i];
            continue;
        }
        const int p = (int)(i / block_packs);
        if (p == rank) continue;
        const long long j = i - (long long)p * block_packs;
        send[i] = in[(long long)p * in_stride + j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        fence_sc_sys();
        const uint32_t prior = atom_add_relaxed_gpu(counters + 1, 1u);
        if (prior + 1u == gridDim.x) {
            counters[1] = 0u;                                  // the next op counts from zero (ops are stream-ordered)
            st_relaxed_sys(ctrl + C_NBYTES, nbytes);
            st_relaxed_sys(ctrl + C_SLOT_NBYTES + slot, nbytes | (bcast ? 1u : 0u));
            fence_sc_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
}

__global__ void wait_kernel(char* flag_base, uint32_t* ctrl, uint32_t* counters, uint32_t spin_limit, int world,
                            int rank, int slots, int flag_stride, int n_flag) {
    if (ld_relaxed_gpu(counters + 3) != 0u) return;
    const uint32_t seq = ld_relaxed_gpu(counters) + 1u, slot = seq & 1u;
    const int t = threadIdx.x;
    if (t < world * n_flag) {
        const int peer = t / n_flag, s = t - peer * n_flag;
        if (peer != rank) {
            const uint32_t* flag = reinterpret_cast<const uint32_t*>(
                flag_base + (((long long)peer * slots + slot) * n_flag + s) * flag_stride);
            uint32_t polls = 0;
            while (ld_acquire_sys(flag) != seq) {
                if (++polls >= spin_limit) {
                    st_relaxed_sys(ctrl + C_PEER, (uint32_t)peer);
                    st_relaxed_sys(ctrl + C_HCA, (uint32_t)s);
                    st_relaxed_sys(ctrl + C_ERROR, seq);
                    st_release_gpu(counters + 3, seq);
                    break;
                }
            }
        }
    }
    fence_sc_sys();
}

__global__ void copy_kernel(uint4* __restrict__ out, long long out_stride, int block_packs, char* recv_base,
                            long long slot_bytes, uint32_t* ctrl, uint32_t* counters, int world, int rank, int slots) {
    const bool ok = ld_relaxed_gpu(counters + 3) == 0u;
    const uint32_t seq = ld_relaxed_gpu(counters) + 1u, slot = seq & 1u;
    if (ok) {
        const long long total = (long long)world * block_packs;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
             i += (long long)gridDim.x * blockDim.x) {
            const int p = (int)(i / block_packs);
            if (p == rank) continue;
            const long long j = i - (long long)p * block_packs;
            const uint4* src = reinterpret_cast<const uint4*>(recv_base + ((long long)p * slots + slot) * slot_bytes);
            out[(long long)p * out_stride + j] = ld_relaxed_sys_v4(src + j);
        }
    }
    fence_sc_gpu();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(counters + 2, 1u);
        if (prior + 1u == gridDim.x) {
            counters[2] = 0u;
            fence_sc_gpu();
            if (ok && ld_relaxed_sys(ctrl + C_ERROR) == 0u) st_release_gpu(counters, seq);
        }
    }
}

}  // namespace

void pf_xchg_cuda(const at::Tensor& in, int64_t in_stride, at::Tensor& out, int64_t out_stride, int64_t block_packs,
                  int64_t recv_base, int64_t flag_base, int64_t send_base, int64_t ctrl, int64_t slot_bytes,
                  int64_t send_slot_bytes, int64_t counters, int64_t spin_limit, int64_t world, int64_t rank,
                  int64_t slots, int64_t flag_stride, int64_t n_flag, int64_t grid, int64_t threads) {
    // grid < 0 (timing only): launch just kernel -grid (1 stage, 2 wait, 3 copy) with |threads|
    const int only = grid < 0 ? (int)-grid : 0;
    if (only) grid = 32;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* cnt = reinterpret_cast<uint32_t*>(counters);
    auto* c = reinterpret_cast<uint32_t*>(ctrl);
    if (!only || only == 1) stage_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(in.data_ptr()), (long long)in_stride, (int)block_packs,
        (uint32_t)(block_packs * 16), reinterpret_cast<char*>(send_base), (long long)send_slot_bytes, c, cnt,
        (int)world, (int)rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!only || only == 2) wait_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<char*>(flag_base), c, cnt, (uint32_t)spin_limit, (int)world,
                                      (int)rank, (int)slots, (int)flag_stride, (int)n_flag);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!only || only == 3) copy_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(
        reinterpret_cast<uint4*>(out.data_ptr()), (long long)out_stride, (int)block_packs,
        reinterpret_cast<char*>(recv_base), (long long)slot_bytes, c, cnt, (int)world, (int)rank, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
