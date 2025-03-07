#include "cooperative_groups.h"
#include "kittens.cuh"
// #include <torch/torch.h>
#include <math_constants.h>
// #include <c10/util/BFloat16.h>

#define ASSERT(cond) if (!(cond)) { printf("Assertion failed: %s\n", #cond); return 1; }
#define CUDA_ASSERT(cond, tidx) if (!(cond)) { if (tidx == -1 || threadIdx.x == tidx && tp == 0) printf("Kernel assert failed: %s\n", #cond); return; }
#define STATIC_PRINT(num) template <int> struct static_print; static_assert(static_print<num>::x, "");

using namespace kittens;
using wg = kittens::warpgroup;

constexpr int G = WARPGROUP_WARPS;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int CONSUMER_WARPGROUPS = 1;
constexpr int NUM_WORKERS = (PRODUCER_WARPGROUPS+CONSUMER_WARPGROUPS)*G;

template<int TP, ducks::st::all ST>
__device__ __forceinline__ void square_all_reduce(ST &tile, ST &tile_other, int tp) {
    if constexpr (TP == 1) {
        tma::cluster::arrive_aligned();
    } else {
        static_assert(TP == 4, "TP must be 4 for this square_all_reduce implementation");
        __shared__ semaphore dsmem_semaphore[2];

        if (wg::warpid() == 0) {
            init_semaphore(dsmem_semaphore[0], 0, 1);
            tma::expect_bytes(dsmem_semaphore[0], sizeof(tile_other));
            init_semaphore(dsmem_semaphore[1], 0, 1);
            tma::expect_bytes(dsmem_semaphore[1], sizeof(tile_other));
        }
        tma::cluster::sync();

        for(int stage = 0; stage < 2; stage++) {
            if (wg::warpid() == 0) {
                tma::cluster::store_async(tile_other, tile, tp ^ (1 << stage), dsmem_semaphore[stage]);
                wait(dsmem_semaphore[stage], 0);
            }
            wg::sync(1);
            wg::add(tile, tile, tile_other);
        }
    }
}

template<int B, int H, int NC, int N, int F, int K, int TP>
__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__cluster_dims__(TP)
__global__ void ttt_tp_forward_ker(
    // TODO: make B and H runtime dimensions by instantiating templates with -1 args
    const __grid_constant__ gl<bf16, B, H, NC*N, F, st_bf<N, F>> XQ_gl,
    const __grid_constant__ gl<bf16, B, H, NC*N, F, st_bf<N, F>> XK_gl,
    const __grid_constant__ gl<bf16, B, H, NC*N, F, st_bf<N, F>> XV_gl,
    const __grid_constant__ gl<bf16, B, H, F, F*K, st_bf<F, F*K/TP>> W1_gl,
    const __grid_constant__ gl<bf16, B, H, F*K, F, st_bf<F*K/TP, F>> W2_gl,
    const __grid_constant__ gl<bf16, B, H, NC*N, F, st_bf<N, F>> out_gl,
    bool *signal
) {
    int b = blockIdx.y;
    int h = blockIdx.z;
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    int tp = cluster.block_rank();
    CUDA_ASSERT(tp == blockIdx.x, 0);

    // Define shared memory tiles
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf<F, F*K/TP> &W1 = al.allocate<typeof(W1)>();
    st_bf<F*K/TP, F> &W2 = al.allocate<typeof(W2)>();
    st_bf<N, F> &XQ = al.allocate<typeof(XQ)>();
    st_bf<N, F> &XK = al.allocate<typeof(XK)>();
    st_bf<N, F> &XV = al.allocate<typeof(XV)>();

    st_bf<N, F*K/TP> &Z1 = al.allocate<typeof(Z1)>();
    st_bf<N, F> &Z2 = al.allocate<typeof(Z2)>();
    st_bf<N, F> &reduction_buffer = al.allocate<typeof(reduction_buffer)>();

    st_bf<F, N> &grad_l_wrt_Z2 = Z2;
    st_bf<N, F*K/TP> &grad_l_wrt_Z1 = al.allocate<typeof(grad_l_wrt_Z1)>();
    st_bf<F*K/TP, N> &Z1_bar = al.allocate<typeof(Z1_bar)>();
    st_bf<F*K/TP, N> &Z2_bar = al.allocate<typeof(Z2_bar)>();

    constexpr size_t total_size = sizeof(XQ) + sizeof(XK) + sizeof(XV) + 
                                 sizeof(W1) + sizeof(W2) + sizeof(Z1) + 
                                 sizeof(Z2) + sizeof(reduction_buffer) +
                                 sizeof(grad_l_wrt_Z1) + sizeof(Z1_bar) + 
                                 sizeof(Z2_bar);
    static_assert(total_size <= MAX_SHARED_MEMORY-8192, 
                 "Total shared memory allocation exceeds available space");

    // Define semaphores and preload data
    __shared__ semaphore w1_sem, w2_sem, q_sem, k_sem, v_sem, minibatch_first_reduction_done, minibatch_done;
    if (wg::groupid() == 0 && wg::warpid() == 0) {
        init_semaphore(w1_sem, 0, 1);
        init_semaphore(w2_sem, 0, 1);
        init_semaphore(q_sem, 0, 1);
        init_semaphore(k_sem, 0, 1);
        init_semaphore(v_sem, 0, 1);
        init_semaphore(minibatch_first_reduction_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(minibatch_done, CONSUMER_WARPGROUPS, 0);
        
        tma::expect_bytes(w1_sem, sizeof(W1));
        tma::load_async(W1, W1_gl, {b, h, 0, tp}, w1_sem);
        tma::expect_bytes(w2_sem, sizeof(W2));
        tma::load_async(W2, W2_gl, {b, h, tp, 0}, w2_sem);
        
        tma::expect_bytes(k_sem, sizeof(XK));
        tma::load_async(XK, XK_gl, {b, h, 0, 0}, k_sem);
        tma::expect_bytes(v_sem, sizeof(XV));
        tma::load_async(XV, XV_gl, {b, h, 0, 0}, v_sem);
        tma::expect_bytes(q_sem, sizeof(XQ));
        tma::load_async(XQ, XQ_gl, {b, h, 0, 0}, q_sem);
    }
    __syncthreads();

    if (wg::groupid() == CONSUMER_WARPGROUPS) {
        wg::decrease_registers<32>();
        tma::cluster::arrive_aligned();
        wait(minibatch_first_reduction_done, 0);

        for (int i = 1; i < NC; i++) {
            wait(minibatch_done, (i+1)%2); // Wait for the previous minibatch to complete
            if (wg::warpid() == 0) {
                // TODO: Prefetch the next minibatch into L2 cache
                tma::expect_bytes(k_sem, sizeof(XK));
                tma::load_async(XK, XK_gl, {b, h, i, 0}, k_sem);
                tma::expect_bytes(q_sem, sizeof(XQ));
                tma::load_async(XQ, XQ_gl, {b, h, i, 0}, q_sem);
                tma::expect_bytes(v_sem, sizeof(XV));
                tma::load_async(XV, XV_gl, {b, h, i, 0}, v_sem);
            }
            wg::sync(1); // Sync all warps in producer warpgroup
            tma::cluster::arrive_aligned();
            wait(minibatch_first_reduction_done, i%2);
        }
    } else {
        warpgroup::increase_registers<120>();
        rt_fl<N/G, F*K/TP> cs_tp_reg;
        rt_fl<N/G, N> cs_cs_fl_reg;
        rt_bf<N/G, N> cs_cs_bf_reg;

        for (int i = 0; i < NC; i++) {
            // Hidden state forward
            wait(k_sem, i%2);
            if (i == 0) wait(w1_sem, 0);
            wg::mm_AB(cs_tp_reg, XK, W1);
            wg::mma_async_wait();
            wg::store(Z1, cs_tp_reg);

            if (i == 0) wait(w2_sem, 0);
            wg::mm_AB(cs_tp_reg, Z1, W2);
            wg::mma_async_wait();
            wg::store(Z2, cs_tp_reg);

            // All reduce across SM cluster
            square_all_reduce<TP>(Z2, reduction_buffer, tp);
            if (wg::laneid() == 0) arrive(minibatch_first_reduction_done, 1);

            // Calculate (negative) grad_l_wrt_Z2 / grad_l_wrt_Z1 (store into SMEM)
            // We use negative gradients to use the WGMMA accumulator
            wait(v_sem, i%2);
            wg::sub(grad_l_wrt_Z2, Z2, XV);
            wg::mm_ABt(cs_tp_reg, grad_l_wrt_Z2, W2);
            wg::mma_async_wait();
            wg::store(grad_l_wrt_Z1, cs_tp_reg);

            // Compute Attn1 and Z1_bar partial (on registers)
            wait(q_sem, i%2);
            wg::mm_ABt(cs_cs_fl_reg, XQ, XK);
            wg::mm_AB(cs_tp_reg, XQ, W1);

            // Compute Z1_bar using Z1_bar partial (on registers)
            copy(cs_cs_bf_reg, cs_cs_fl_reg);
            make_causal(cs_cs_bf_reg, cs_cs_bf_reg, base_types::constants<bf16>::zero());
            wg::mma_AB(cs_tp_reg, cs_cs_bf_reg, grad_l_wrt_Z1);
            wg::mma_async_wait();
            wg::store(Z1_bar, cs_tp_reg);

            // Compute Attn2 and Z2_bar partial (on registers)
            wg::mm_ABt(cs_cs_fl_reg, Z1_bar, Z1);
            wg::mm_AB(cs_tp_reg, Z1_bar, W2);
            wg::mma_async_wait();

            // Compute Z2_bar using Z2_bar partial (on registers)
            copy(cs_cs_bf_reg, cs_cs_fl_reg);
            make_causal(cs_cs_bf_reg, cs_cs_bf_reg, base_types::constants<bf16>::zero());
            wg::mma_AB(cs_tp_reg, cs_cs_bf_reg, grad_l_wrt_Z2);
            wg::mma_async_wait();

            // TODO: perhaps make Z2_bar share memory with Z1_bar. Move store_async_read_wait to immediately above wg::store(Z1_bar, ...)
            if (i != 0) tma::store_async_read_wait(); // Wait until clear to start editing Z2_bar
            wg::store(Z2_bar, cs_tp_reg);
            if (wg::warpid() == 0) {
                tma::store_add_async(out_gl, Z2_bar, {b, h, i, 0});
                tma::store_commit_group();
            }

            // Update hidden states (TODO: Is there a more efficient way to do this?)
            wg::load(cs_cs_fl_reg, W1);
            wg::mma_AtB(cs_cs_fl_reg, XK, grad_l_wrt_Z1);
            wg::mma_async_wait();
            wg::store(W1, cs_cs_fl_reg);

            wg::load(cs_cs_fl_reg, W2);
            wg::mma_AtB(cs_cs_fl_reg, Z1, grad_l_wrt_Z2);
            wg::mma_async_wait();
            wg::store(W2, cs_cs_fl_reg);

            if (wg::laneid() == 0) arrive(minibatch_done, 1);
        }
        tma::store_async_wait();
    }

    __syncthreads();
    *signal = true;
}

int main() {
    constexpr int B = 1, H = 1, NC = 1, N = 64, F = 64, K = 4, TP = 4;

    bf16 *h_XQ, *h_XK, *h_XV, *h_W1, *h_W2, *h_out;

    // Allocate host memory
    h_XQ = (bf16*)malloc(B*H*NC*N*F*sizeof(bf16));
    h_XK = (bf16*)malloc(B*H*NC*N*F*sizeof(bf16));
    h_XV = (bf16*)malloc(B*H*NC*N*F*sizeof(bf16));
    h_W1 = (bf16*)malloc(B*H*F*F*K*sizeof(bf16));
    h_W2 = (bf16*)malloc(B*H*F*K*F*sizeof(bf16));
    h_out = (bf16*)malloc(B*H*NC*N*F*sizeof(bf16));

    // Initialize host arrays
    for (int i = 0; i < B*H*NC*N*F; i++) {
        h_XQ[i] = __int2bfloat16_rn(i);
        h_XK[i] = __int2bfloat16_rn(i);
        h_XV[i] = __int2bfloat16_rn(i);
        h_out[i] = __int2bfloat16_rn(0);
    }
    for (int i = 0; i < B*H*F*F*K; i++) {
        h_W1[i] = __int2bfloat16_rn(i);
        h_W2[i] = __int2bfloat16_rn(i);
    }

    bf16 *XQ, *XK, *XV, *W1, *W2, *out;

    // Allocate device memory
    cudaMalloc(&XQ, B*H*NC*N*F*sizeof(bf16));
    cudaMalloc(&XK, B*H*NC*N*F*sizeof(bf16));
    cudaMalloc(&XV, B*H*NC*N*F*sizeof(bf16));
    cudaMalloc(&W1, B*H*F*F*K*sizeof(bf16));
    cudaMalloc(&W2, B*H*F*K*F*sizeof(bf16));
    cudaMalloc(&out, B*H*NC*N*F*sizeof(bf16));

    // Copy data from host to device
    cudaMemcpy(XQ, h_XQ, B*H*NC*N*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(XK, h_XK, B*H*NC*N*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(XV, h_XV, B*H*NC*N*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(W1, h_W1, B*H*F*F*K*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, B*H*F*K*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(out, h_out, B*H*NC*N*F*sizeof(bf16), cudaMemcpyHostToDevice);

    bool *h_signal = (bool *)malloc(sizeof(bool)), *signal;
    cudaMalloc(&signal, sizeof(bool));
    *h_signal = false;
    cudaMemcpy(signal, h_signal, sizeof(bool), cudaMemcpyHostToDevice);

    printf("Launching kernel\n");

    cudaFuncSetAttribute(ttt_tp_forward_ker<B, H, NC, N, F, K, TP>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);

    ttt_tp_forward_ker<B, H, NC, N, F, K, TP><<<dim3(TP, B, H), NUM_WORKERS*kittens::WARP_THREADS, MAX_SHARED_MEMORY>>>(
        gl<bf16, B, H, NC*N, F, st_bf<N, F>>{XQ, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, NC*N, F, st_bf<N, F>>{XK, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, NC*N, F, st_bf<N, F>>{XV, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, F, F*K, st_bf<F, F*K/TP>>{W1, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, F*K, F, st_bf<F*K/TP, F>>{W2, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, NC*N, F, st_bf<N, F>>{out, nullptr, nullptr, nullptr, nullptr},
        signal
    );

    cudaMemcpy(h_signal, signal, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT(*h_signal);

    printf("Ran successfully\n");
}
