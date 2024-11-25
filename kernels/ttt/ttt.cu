#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>

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
    constexpr static int stages = (4);
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
    const int remat_gs;
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

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int n_minibatch = g.seq_len / (K::tile_height);
    const int n_remat_groups = n_minibatch / g.remat_gs;

    int warpid = kittens::warpid(); // Global warp ID
    int wg_warpid = warpid % kittens::WARPGROUP_WARPS; // Warp ID within Warpgroup
    int warpgroupid = warpid / kittens::WARPGROUP_WARPS; // Warpgroup ID
    int cluster_wgid = warpgroupid + CONSUMER_WARPGROUPS * tp; // Cluster Warpgroup ID

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

    tile_type(&attn1_smem) = al.allocate<tile_type>();
    tile_type(&reduction_buffer) = al.allocate<tile_type>();

    // Reinterpretations for intermediates
    auto(*z2_smem) = reinterpret_cast<tile_type(*)>(grad_l_z1_smem);
    auto(*grad_l_z2_smem) = reinterpret_cast<tile_type(*)>(v_smem);
    auto(*x2_bar_smem) = reinterpret_cast<tile_type(*)>(z1_smem);
    auto(*attn2_smem) = reinterpret_cast<tile_type(*)>(grad_l_z1_smem);
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
            // Hidden state forward
            kittens::wait(k_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            warpgroup::mm_AB(cs_cs_fl_reg, k_smem[idx % K::stages], w1_smem[warpgroupid]);
            load(cs_row_fl_reg, b2_smem);
            warpgroup::mma_async_wait();
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);

            warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], w2_smem[warpgroupid]);
            if (cluster_wgid == 0) {
                load(cs_row_fl_reg, b2_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            }
            warpgroup::mma_async_wait();
            warpgroup::store(z2_smem[warpgroupid], cs_cs_fl_reg);

            // Warpgroup 0 will perform reduction
            // Warpgroup 1 will compute Attn1
            if (warpgroupid == 0) {
                warpgroup::add(z2_smem[0], z2_smem[0], z2_smem[1]); // Ideally we can do an atomic store into shared memory
                if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore, sizeof(reduction_buffer));
                tma::cluster::sync();
                if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, z2_smem[0], tp ^ 1, dsmem_semaphore);
                warpgroup::sub(grad_l_z2_smem[idx % K::stages], v_smem[idx % K::stages], z2_smem[0]);
                kittens::wait(dsmem_semaphore, idx % 2);
                warpgroup::sync(1);
                warpgroup::sub(grad_l_z2_smem[idx % K::stages], grad_l_z2_smem[idx % K::stages], reduction_buffer);
            } else {
                tma::cluster::arrive_aligned();
                kittens::wait(q_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
                warpgroup::mm_ABt(cs_cs_fl_reg, q_smem[idx % K::stages], k_smem[idx % K::stages]);
                warpgroup::mma_async_wait();
                make_causal(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(attn1_smem, cs_cs_fl_reg);   
            }

            // Wait on each other to complete
            if (warpgroup::laneid() == 0) arrive(reduction_done, 1);
            kittens::wait(reduction_done, idx % 2);

            // Calculate grad_l_wrt_Z1
            warpgroup::mm_ABt(cs_cs_fl_reg, grad_l_z2_smem[idx % K::stages], w2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(grad_l_z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::load(cs_cs_fl_reg, z1_smem[warpgroupid]);
            gelu_bwd(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::mul(grad_l_z1_smem[warpgroupid], grad_l_z1_smem[warpgroupid], z1_smem[warpgroupid]);

            // Compute Z1_bar
            warpgroup::mm_AB(cs_cs_fl_reg, q_smem[idx % K::stages], w1_smem[warpgroupid]);
            warpgroup::mma_AB(cs_cs_fl_reg, attn1_smem, grad_l_z1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_bar_smem[warpgroupid], cs_cs_fl_reg);

            // Update hidden states (W1)
            warpgroup::load(cs_cs_fl_reg, w1_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[idx % K::stages], grad_l_z1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(w1_smem[warpgroupid], cs_cs_fl_reg);

            // Save hidden state checkpoint (W1)
            if (wg_warpid == 0 && (idx + 1) % n_remat_groups == 0) {
                int4 curr_checkpoint = {batch_idx, head_idx, (idx + 1) / n_remat_groups, cluster_wgid};
                tma::store_async(g.w1_checkpoints, w1_smem[warpgroupid], curr_checkpoint);
            }

            // Compute Attn2
            warpgroup::mm_ABt(cs_cs_fl_reg, x2_bar_smem[warpgroupid], x2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            make_causal(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(attn2_smem[warpgroupid], cs_cs_fl_reg);

            // Compute Z2_bar
            warpgroup::mm_AB(cs_cs_fl_reg, x2_bar_smem[warpgroupid], w2_smem[warpgroupid]);
            warpgroup::mma_AB(cs_cs_fl_reg, attn2_smem[warpgroupid], grad_l_z2_smem[idx % K::stages]);
            warpgroup::mma_async_wait();
            warpgroup::store(z2_bar_smem[warpgroupid], cs_cs_fl_reg);

            // Store out Z2 Bar
            if (wg_warpid == 0) {
                tma::store_add_async(g.o, z2_bar_smem[warpgroupid], {batch_idx, head_idx, idx, 0});
            }

            // Update hidden state (W2)
            warpgroup::load(cs_cs_fl_reg, w2_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem[warpgroupid], grad_l_z2_smem[idx % K::stages]);
            warpgroup::mma_async_wait();
            warpgroup::store(w2_smem[warpgroupid], cs_cs_fl_reg);

            // Save hidden state checkpoint (W2)
            if (wg_warpid == 0 && (idx + 1) % n_remat_groups == 0) {
                int4 curr_checkpoint = {batch_idx, head_idx, cluster_wgid, (idx + 1) / n_remat_groups};
                tma::store_async(g.w2_checkpoints, w2_smem[warpgroupid], curr_checkpoint);
                tma::store_commit_group();
            }

            if (warpgroup::laneid() == 0) arrive(compute_done[idx % K::stages], 1);
        }
    }
}

#include "harness.cuh"
#include "torch_harness.cuh"
