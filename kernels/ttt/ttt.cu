#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>
#include <cstdio>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_MLP_FORWARD_TP
#endif

constexpr int TP = (2);
constexpr int CONSUMER_WARPGROUPS = (2);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

using namespace kittens;

template <int head_dim> struct fwd_ttt_mlp_ker_tile_dims {};
template <> struct fwd_ttt_mlp_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int tile_height = (64);
    constexpr static int stages = (2);
};

template <int head_dim> struct fwd_globals {
    using tile_type = st_bf<fwd_ttt_mlp_ker_tile_dims<head_dim>::tile_height, fwd_ttt_mlp_ker_tile_dims<head_dim>::tile_width>;
    using vec_type = sv_bf<fwd_ttt_mlp_ker_tile_dims<head_dim>::tile_height>;

    // Global memory layout
    using q_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using o_gl = gl<bf16, -1, -1, -1, -1, tile_type>;

    using w1_init_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b1_init_gl = gl<bf16, -1, -1, -1, -1, vec_type>;
    using w2_init_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b2_init_gl = gl<bf16, -1, -1, -1, -1, vec_type>;

    // Remat checkpoints
    using w1_checkpoints_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b1_checkpoints_gl = gl<bf16, -1, -1, -1, -1, vec_type>;
    using w2_checkpoints_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b2_checkpoints_gl = gl<bf16, -1, -1, -1, -1, vec_type>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    w1_init_gl w1;
    b1_init_gl b1;
    w2_init_gl w2;
    b2_init_gl b2;

    w1_checkpoints_gl w1_checkpoints;
    b1_checkpoints_gl b1_checkpoints;
    w2_checkpoints_gl w2_checkpoints;
    b2_checkpoints_gl b2_checkpoints;

    const int seq_len;
    const int num_checkpoints;
    const int checkpoint_group_size;
};

template <int head_dim>
__cluster_dims__(TP)
__global__ __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_ttt_mlp_ker(const __grid_constant__ fwd_globals<head_dim> g) {
    using K = fwd_ttt_mlp_ker_tile_dims<head_dim>;
    using tile_type = st_bf<K::tile_height, K::tile_width>;
    using vec_type = sv_bf<K::tile_height>;

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    int tp = cluster.block_rank();

    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int n_minibatch = g.seq_len / (K::tile_height);
    const int n_remat_groups = g.num_checkpoints;
    const int checkpoint_group_size = g.checkpoint_group_size;

    int warpid = kittens::warpid(); // Global warp ID
    int wg_warpid = warpid % kittens::WARPGROUP_WARPS; // Warp ID within Warpgroup
    int warpgroupid = warpid / kittens::WARPGROUP_WARPS; // Warpgroup ID
    int cluster_wgid = warpgroupid + CONSUMER_WARPGROUPS * tp; // Cluster Warpgroup ID
    const int tp_shard_rank = cluster_wgid;

    // if (warpgroupid < 2)
    // {
    //     printf("Warpid=%d, wg_warpid=%d, warpgroupid=%d, cluster_wgid=%d, tp=%d\n", warpid, wg_warpid, warpgroupid, cluster_wgid, tp);
    // }

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    // Shared memory for hidden states
    tile_type(&w1_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    vec_type(&b1_smem)[CONSUMER_WARPGROUPS] = al.allocate<vec_type, CONSUMER_WARPGROUPS>();
    tile_type(&w2_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    vec_type(&b2_smem) = al.allocate<vec_type>();
    
    // Shared memory for inputs (staged)
    tile_type(&q_smem)[K::stages] = al.allocate<tile_type, K::stages>();
    tile_type(&k_smem)[K::stages] = al.allocate<tile_type, K::stages>();
    tile_type(&v_smem)[K::stages] = al.allocate<tile_type, K::stages>();
    
    // Shared memory for intermediates
    tile_type(&z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    tile_type(&x2_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    tile_type(&grad_l_z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();

    // tile_type(&attn1_smem) = al.allocate<tile_type>();
    tile_type(&reduction_buffer) = al.allocate<tile_type>();

    // Reinterpretations for intermediates
    auto(*z2_smem) = reinterpret_cast<tile_type(*)>(grad_l_z1_smem);
    auto(*grad_l_z2_smem) = reinterpret_cast<tile_type(*)>(v_smem);
    auto(*x2_bar_smem) = reinterpret_cast<tile_type(*)>(z1_smem);
    // auto(*attn2_smem) = reinterpret_cast<tile_type(*)>(grad_l_z1_smem);
    auto(*z2_bar_smem) = reinterpret_cast<tile_type(*)>(z1_smem);

    __shared__ kittens::semaphore 
        w1_arrived,
        w2_arrived,
        b1_arrived,
        b2_arrived,
        dsmem_semaphore,
        reduction_done,
        q_sem_arrived[K::stages],
        k_sem_arrived[K::stages], 
        v_sem_arrived[K::stages],
        compute_done[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(w1_arrived, 0, 1);
        init_semaphore(b1_arrived, 0, 1);
        init_semaphore(w2_arrived, 0, 1);
        init_semaphore(b2_arrived, 0, 1);
        init_semaphore(dsmem_semaphore, 0, 1);
        init_semaphore(reduction_done, CONSUMER_WARPGROUPS, 0);
        for (int i = 0; i < K::stages; i++) {
            init_semaphore(q_sem_arrived[i], 0, 1);
            init_semaphore(k_sem_arrived[i], 0, 1);
            init_semaphore(v_sem_arrived[i], 0, 1);
            init_semaphore(compute_done[i], CONSUMER_WARPGROUPS, 0);
        }

        // Load hidden states across consumer warpgroups
        tma::expect_bytes(w1_arrived, sizeof(w1_smem));
        tma::expect_bytes(b1_arrived, sizeof(b1_smem));
        tma::expect_bytes(w2_arrived, sizeof(w2_smem));
        tma::expect_bytes(b2_arrived, sizeof(b2_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            tma::load_async(w1_smem[wg], g.w1, {batch_idx, head_idx, 0, wg + CONSUMER_WARPGROUPS * tp}, w1_arrived);
            tma::load_async(b1_smem[wg], g.b1, {batch_idx, head_idx, 0, wg + CONSUMER_WARPGROUPS * tp}, b1_arrived);
            tma::load_async(w2_smem[wg], g.w2, {batch_idx, head_idx, wg + CONSUMER_WARPGROUPS * tp, 0}, w2_arrived);
        }
        tma::load_async(b2_smem, g.b2, {batch_idx, head_idx, 0, 0}, b2_arrived);

        // Preload minibatches
        for (int j = 0; j < K::stages - 1; j++) {
            int4 tile_idx = {batch_idx, head_idx, j, 0};
            tma::expect_bytes(k_sem_arrived[j], sizeof(tile_type));
            tma::load_async(k_smem[j], g.k, tile_idx, k_sem_arrived[j]);
            tma::expect_bytes(v_sem_arrived[j], sizeof(tile_type));
            tma::load_async(v_smem[j], g.v, tile_idx, v_sem_arrived[j]);
            tma::expect_bytes(q_sem_arrived[j], sizeof(tile_type));
            tma::load_async(q_smem[j], g.q, tile_idx, q_sem_arrived[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    // First warp in last warpgroup is the producer
    if (warpgroupid == NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();
        tma::cluster::arrive_aligned();

        int iters = n_minibatch - 1;
        if (warpid == NUM_WORKERS - 4) {
            for (auto idx = pipe_idx - 1; idx < iters; idx++) {
                int4 tile_idx = {batch_idx, head_idx, idx + 1, 0};

                tma::expect_bytes(k_sem_arrived[(idx + 1) % K::stages], sizeof(tile_type));
                tma::load_async(k_smem[(idx + 1) % K::stages], g.k, tile_idx, k_sem_arrived[(idx + 1) % K::stages]);
                tma::expect_bytes(v_sem_arrived[(idx + 1) % K::stages], sizeof(tile_type));
                tma::load_async(v_smem[(idx + 1) % K::stages], g.v, tile_idx, v_sem_arrived[(idx + 1) % K::stages]);
                tma::expect_bytes(q_sem_arrived[(idx + 1) % K::stages], sizeof(tile_type));
                tma::load_async(q_smem[(idx + 1) % K::stages], g.q, tile_idx, q_sem_arrived[(idx + 1) % K::stages]);

                // Wait on previous stage to finish computation
                kittens::wait(compute_done[(idx+2) % K::stages], ((idx+2) / K::stages) % 2);
                tma::cluster::arrive_aligned();
            }
        }
    } else {
        warpgroup::increase_registers<240>();

        rt_fl<16, K::tile_height> cs_cs_fl_reg;
        typeof(cs_cs_fl_reg)::row_vec cs_row_fl_reg;

        kittens::wait(w1_arrived, 0);
        kittens::wait(b1_arrived, 0);
        kittens::wait(w2_arrived, 0);
        kittens::wait(b2_arrived, 0);

        for (auto idx = 0; idx < n_minibatch; idx++) {
            // Save hidden state checkpoint (W1)
            if (wg_warpid == 0 && idx % checkpoint_group_size == 0) {
                const int curr_checkpoint_idx = idx / checkpoint_group_size;
                int4 curr_checkpoint = {batch_idx, head_idx, curr_checkpoint_idx, tp_shard_rank};
                tma::store_async(g.w1_checkpoints, w1_smem[warpgroupid], curr_checkpoint);

                curr_checkpoint = {batch_idx, head_idx, curr_checkpoint_idx, tp_shard_rank};
                tma::store_async(g.b1_checkpoints, b1_smem[warpgroupid], curr_checkpoint);

                const int sharded_checkpoint_offset = curr_checkpoint_idx * TP * CONSUMER_WARPGROUPS + tp_shard_rank;
                curr_checkpoint = {batch_idx, head_idx, sharded_checkpoint_offset, 0};
                tma::store_async(g.w2_checkpoints, w2_smem[warpgroupid], curr_checkpoint);

                curr_checkpoint = {batch_idx, head_idx, curr_checkpoint_idx, 0};
                tma::store_async(g.b2_checkpoints, b2_smem, curr_checkpoint);

                tma::store_commit_group();
            }

            // Hidden state forward
            kittens::wait(k_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            zero(cs_cs_fl_reg);
            warpgroup::mm_AB(cs_cs_fl_reg, k_smem[idx % K::stages], w1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            load(cs_row_fl_reg, b1_smem[warpgroupid]);
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);

            zero(cs_cs_fl_reg);
            warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], w2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            // Only add b2 to one of the sharded Z2
            // Else, post reduction it will result in adding 4*b2
            if (tp_shard_rank == 0) {
                load(cs_row_fl_reg, b2_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            }
            warpgroup::store(z2_smem[warpgroupid], cs_cs_fl_reg);

            // Warpgroup 0 will perform reduction
            if (warpgroupid == 0) {
                // Reduce intra CTA Z2 before inter CTA reduce
                warpgroup::add(z2_smem[0], z2_smem[0], z2_smem[1]); // Ideally we can do an atomic store into shared memory
                if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore, sizeof(reduction_buffer));
                tma::cluster::sync();
                if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, z2_smem[0], tp ^ 1, dsmem_semaphore);
                // This operation is flipped so that it can be accumulated without having to flip signs
                warpgroup::sub(grad_l_z2_smem[idx % K::stages], z2_smem[0], v_smem[idx % K::stages]);
                kittens::wait(dsmem_semaphore, idx % 2);
                warpgroup::sync(1);
                warpgroup::add(grad_l_z2_smem[idx % K::stages], reduction_buffer, grad_l_z2_smem[idx % K::stages]);
            }
            else {
                tma::cluster::arrive_aligned();
            }

            // Wait on each other to complete
            if (warpgroup::laneid() == 0) arrive(reduction_done, 1);
            kittens::wait(reduction_done, idx % 2);

            // Calculate grad_l_wrt_Z1
            zero(cs_cs_fl_reg);
            warpgroup::mm_ABt(cs_cs_fl_reg, grad_l_z2_smem[idx % K::stages], w2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(grad_l_z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::load(cs_cs_fl_reg, z1_smem[warpgroupid]);
            gelu_bwd(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::mul(grad_l_z1_smem[warpgroupid], grad_l_z1_smem[warpgroupid], z1_smem[warpgroupid]);

            // Update hidden state (W2)
            warpgroup::load(cs_cs_fl_reg, w2_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem[warpgroupid], grad_l_z2_smem[idx % K::stages]);
            warpgroup::mma_async_wait();
            warpgroup::store(w2_smem[warpgroupid], cs_cs_fl_reg);

            // Update b2
            if (warpgroupid == 0)
            {
                col_sum(b2_smem, grad_l_z2_smem[idx % K::stages]);
            }


            // Update hidden states (W1)
            warpgroup::load(cs_cs_fl_reg, w1_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[idx % K::stages], grad_l_z1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(w1_smem[warpgroupid], cs_cs_fl_reg);

            // Update b1
            col_sum(b1_smem[warpgroupid], grad_l_z1_smem[warpgroupid]);

            // Compute output
            zero(cs_cs_fl_reg);
            kittens::wait(q_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            warpgroup::mm_AB(cs_cs_fl_reg, q_smem[idx % K::stages], w1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            load(cs_row_fl_reg, b1_smem[warpgroupid]);
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);

            warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], w2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            // Only add b2 to one of the sharded Z2
            // Else, post reduction it will result in adding 4*b2
            if (tp_shard_rank == 0) {
                load(cs_row_fl_reg, b2_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            }
            warpgroup::store(z2_smem[warpgroupid], cs_cs_fl_reg);

            // Store output to global
            if (wg_warpid == 0) {
                tma::store_add_async(g.o, z2_smem[warpgroupid], {batch_idx, head_idx, idx, 0});
            }

            if (warpgroup::laneid() == 0) arrive(compute_done[idx % K::stages], 1);
        }
    }
}

#if TORCH_COMPILE

#include "common/pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

torch::Tensor ttt_forward(
    const torch::Tensor XQ,
    const torch::Tensor XK,
    const torch::Tensor XV,
    const torch::Tensor W1,
    const torch::Tensor b1,
    const torch::Tensor W2,
    const torch::Tensor b2,
    const torch::Tensor W1_checkpoints,
    const torch::Tensor b1_checkpoints,
    const torch::Tensor W2_checkpoints,
    const torch::Tensor b2_checkpoints,
    const torch::Tensor Out
) {
    constexpr int F = 64;
    constexpr int K = 4;
    unsigned long B = XQ.size(0);
    unsigned long H = XQ.size(1);
    unsigned long T = XQ.size(2) * XQ.size(3); // seq len
    const int NC = XQ.size(2);
    const int CS = XQ.size(3);
    unsigned long num_checkpoints = W1_checkpoints.size(2);
    const int checkpoint_group_size = NC / num_checkpoints;

    TORCH_CHECK(NC % num_checkpoints == 0, "N % R == 0");
    
    TORCH_CHECK(XQ.device().is_cuda() && XQ.is_contiguous() && XQ.dim() == 5 && XQ.size(4) == F, "Invalid dims for XQ");
    TORCH_CHECK(XK.device().is_cuda() && XK.is_contiguous() && XK.dim() == 5 && XK.size(4) == F, "Invalid dims for XK");
    TORCH_CHECK(XV.device().is_cuda() && XV.is_contiguous() && XV.dim() == 5 && XV.size(4) == F, "Invalid dims for XV");
    TORCH_CHECK(W1.device().is_cuda() && W1.is_contiguous() && W1.dim() == 4 && W1.size(0) == B && W1.size(1) == H && W1.size(2) == F && W1.size(3) == F*K, "Invalid dims for W1");
    TORCH_CHECK(W2.device().is_cuda() && W2.is_contiguous() && W2.dim() == 4 && W2.size(0) == B && W2.size(1) == H && W2.size(2) == F*K && W2.size(3) == F, "Invalid dims for W2");
    TORCH_CHECK(W1_checkpoints.device().is_cuda() && W1_checkpoints.is_contiguous() && W1_checkpoints.dim() == 5 && W1_checkpoints.size(0) == B && W1_checkpoints.size(1) == H && W1_checkpoints.size(2) == num_checkpoints && W1_checkpoints.size(3) == F && W1_checkpoints.size(4) == F*K, "Invalid dims for W1_checkpoints");
    TORCH_CHECK(W2_checkpoints.device().is_cuda() && W2_checkpoints.is_contiguous() && W2_checkpoints.dim() == 5 && W2_checkpoints.size(0) == B && W2_checkpoints.size(1) == H && W2_checkpoints.size(2) == num_checkpoints && W2_checkpoints.size(3) == F*K && W2_checkpoints.size(4) == F, "Invalid dims for W2_checkpoints");
    TORCH_CHECK(Out.device().is_cuda() && Out.is_contiguous() && Out.dim() == 5 && Out.size(4) == F, "Invalid dims for Out");

    using tile_type = st_bf<fwd_ttt_mlp_ker_tile_dims<F>::tile_height, fwd_ttt_mlp_ker_tile_dims<F>::tile_width>;
    using tile_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using vec_type = sv_bf<fwd_ttt_mlp_ker_tile_dims<F>::tile_height>;
    using vec_gl = gl<bf16, -1, -1, -1, -1, vec_type>;
    using globals = fwd_globals<F>;

    tile_gl q_gl{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), B, H, T, F};
    tile_gl k_gl{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), B, H, T, F};
    tile_gl v_gl{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), B, H, T, F};
    tile_gl o_gl{reinterpret_cast<bf16*>(Out.data_ptr<at::BFloat16>()), B, H, T, F};

    tile_gl w1_init_gl{reinterpret_cast<bf16*>(W1.data_ptr<at::BFloat16>()), B, H, F, F*K};
    vec_gl b1_init_gl{reinterpret_cast<bf16*>(b1.data_ptr<at::BFloat16>()), B, H, 1, F*K};
    tile_gl w2_init_gl{reinterpret_cast<bf16*>(W2.data_ptr<at::BFloat16>()), B, H, F*K, F};
    vec_gl b2_init_gl{reinterpret_cast<bf16*>(b2.data_ptr<at::BFloat16>()), B, H, 1, F};

    tile_gl w1_checkpoints_gl{reinterpret_cast<bf16*>(W1_checkpoints.data_ptr<at::BFloat16>()), B, H, num_checkpoints*F, F*K};
    vec_gl b1_checkpoints_gl{reinterpret_cast<bf16*>(b1_checkpoints.data_ptr<at::BFloat16>()), B, H, num_checkpoints, F*K};
    tile_gl w2_checkpoints_gl{reinterpret_cast<bf16*>(W2_checkpoints.data_ptr<at::BFloat16>()), B, H, num_checkpoints*F*K, F};
    vec_gl b2_checkpoints_gl{reinterpret_cast<bf16*>(b2_checkpoints.data_ptr<at::BFloat16>()), B, H, num_checkpoints, F};

    globals g{
        q_gl, 
        k_gl, 
        v_gl, 
        o_gl, 
        w1_init_gl, 
        b1_init_gl,
        w2_init_gl, 
        b2_init_gl,
        w1_checkpoints_gl, 
        b1_checkpoints_gl,
        w2_checkpoints_gl, 
        b2_checkpoints_gl,
        T,
        num_checkpoints,
        checkpoint_group_size
    };

    constexpr long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        fwd_ttt_mlp_ker<F>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(TP, B, H);
    fwd_ttt_mlp_ker<F><<<grid, NUM_WORKERS*32, mem_size>>>(g);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error launching kernel");
    cudaDeviceSynchronize();

    return Out;
}//*/

#else

#include "harness.cuh"

#endif
