#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>
#include <cstdio>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_BACKWARD
#endif

constexpr int TP = (4);
constexpr int CONSUMER_WARPGROUPS = (1);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

using namespace kittens; // dangerous

template<ducks::st::all T>
__device__ static inline void tp_reduce(
    T &src, 
    T &reduction_buffer, 
    kittens::semaphore& dsmem_semaphore1, 
    kittens::semaphore& dsmem_semaphore2, 
    const int idx
)
{
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int tp = cluster.block_rank();
    static_assert(TP == 4, "Reduction is only implemented for tp=4.");
    const int warpid = kittens::warpid(); // Global warp ID
    const int wg_warpid = warpid % kittens::WARPGROUP_WARPS; // Warp ID within Warpgroup

    if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore1, sizeof(reduction_buffer));
    tma::cluster::sync();
    if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, src, tp ^ 1, dsmem_semaphore1);
    kittens::wait(dsmem_semaphore1, idx % 2);

    warpgroup::add(src, src, reduction_buffer);
    warpgroup::sync(1);

    if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore2, sizeof(reduction_buffer));
    tma::cluster::sync();
    if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, src, tp ^ 3, dsmem_semaphore2);
    kittens::wait(dsmem_semaphore2, idx % 2);

    warpgroup::add(src, src, reduction_buffer);
    warpgroup::sync(1);
}

__device__ static inline void tp_reduce_arrive()
{
    static_assert(TP == 4, "Reduction is only implemented for tp=4.");
    // need two here since we need two global reads to reduce 4
    tma::cluster::arrive_aligned();
    tma::cluster::arrive_aligned();
}

template <int head_dim> struct bwd_ttt_mlp_ker_tile_dims {
    constexpr static int mini_batch_size = 64;
    constexpr static int F = head_dim;
    constexpr static int stages = (2);
};

template <int head_dim> struct bwd_globals {
    using tile_dims = bwd_ttt_mlp_ker_tile_dims<head_dim>;
    // Tiles
    using CS_F_tile_type = st_bf<tile_dims::mini_batch_size, tile_dims::F>;
    using F_F_tile_type = st_bf<tile_dims::F, tile_dims::F>;
    using CS_F_tile_acc_type = st_fl<tile_dims::mini_batch_size, tile_dims::F>;
    using F_F_tile_acc_type = st_fl<tile_dims::F, tile_dims::F>;

    // Vectors
    using CS_vec_type = sv_bf<tile_dims::mini_batch_size>;
    using F_vec_type = sv_bf<tile_dims::F>;
    using CS_vec_acc_type = sv_fl<tile_dims::mini_batch_size>;
    using F_vec_acc_type = sv_fl<tile_dims::F>;
    using std_vec_acc_type = sv_fl<16>;

    // Global memory layout
    using qkvo_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using qkvo_acc_gl = gl<float, -1, -1, -1, -1, CS_F_tile_acc_type>;

    using last_eta_gl = gl<bf16, -1, -1, -1, -1, CS_vec_type>;

    using ttt_norm_weight_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using ttt_norm_bias_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    using w1_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b1_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using w2_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b2_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    // Remat checkpoints
    using w1_checkpoints_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b1_checkpoints_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using w2_checkpoints_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b2_checkpoints_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    using std_gl = gl<float, -1, -1, -1, -1, std_vec_acc_type>;

    qkvo_gl q;
    qkvo_gl k;
    qkvo_gl v;
    qkvo_gl o;

    last_eta_gl last_eta;

    ttt_norm_weight_gl ttt_norm_weight;
    ttt_norm_bias_gl ttt_norm_bias;

    w1_gl w1;
    b1_gl b1;
    w2_gl w2;
    b2_gl b2;

    w1_checkpoints_gl w1_checkpoints;
    b1_checkpoints_gl b1_checkpoints;
    w2_checkpoints_gl w2_checkpoints;
    b2_checkpoints_gl b2_checkpoints;

    // rematted activations
    w1_gl W1_init_group;
    b1_gl b1_init_group;
    w2_gl W2_init_group;
    b2_gl b2_init_group;
    qkvo_gl x_hat_ln_group;
    std_gl std_ln_group;
    qkvo_gl X2_group;
    qkvo_gl Z1_group;
    qkvo_gl Z1_bar_group;
    qkvo_gl X2_bar_group;
    qkvo_gl grad_l_wrt_Z2_group;
    qkvo_gl grad_l_wrt_Z1_group;
    qkvo_gl x_hat_fused_group;
    qkvo_gl grad_x_hat_fused_group;
    qkvo_gl grad_output_fused_group;
    std_gl std_fused_group;

    // Upstream grads
    w1_gl grad_L_W1_last;
    b1_gl grad_L_b1_last;
    w2_gl grad_L_W2_last;
    b2_gl grad_L_b2_last;
    qkvo_gl grad_L_XQW_mini_batch;

    // Output grads
    ttt_norm_weight_gl grad_L_ttt_norm_weight;
    ttt_norm_bias_gl grad_L_ttt_norm_bias;
    w1_gl grad_L_W1_init;
    b1_gl grad_L_b1_init;
    w2_gl grad_L_W2_init;
    b2_gl grad_L_b2_init;
    last_eta_gl grad_L_last_eta;
    qkvo_gl grad_L_XQ;
    qkvo_gl grad_L_XK;
    qkvo_gl grad_L_XV;

    const int seq_len;
    const int num_checkpoints;
    const int checkpoint_group_size;
};

template <int head_dim>
__cluster_dims__(TP)
__global__ __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void bwd_ttt_mlp_ker(const __grid_constant__ bwd_globals<head_dim> g) {
    using globals = bwd_globals<head_dim>;
    using K = bwd_ttt_mlp_ker_tile_dims<head_dim>;

    using CS_F_tile_type = globals::CS_F_tile_type;
    using F_F_tile_type = globals::F_F_tile_type;
    using CS_F_tile_acc_type = globals::CS_F_tile_acc_type;
    using F_F_tile_acc_type = globals::F_F_tile_acc_type;

    // Vectors
    using CS_vec_type = globals::CS_vec_type;
    using F_vec_type = globals::F_vec_type;
    using CS_vec_acc_type = globals::CS_vec_acc_type;
    using F_vec_acc_type = globals::F_vec_acc_type;

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int tp = cluster.block_rank();

    // For input indexing
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int n_minibatch = g.seq_len / (K::mini_batch_size);
    const int n_remat_groups = g.num_checkpoints;
    const int checkpoint_group_size = g.checkpoint_group_size;

    // Block info
    const int warpid = kittens::warpid(); // Global warp ID
    const int wg_warpid = warpid % kittens::WARPGROUP_WARPS; // Warp ID within Warpgroup
    const int warpgroupid = warpid / kittens::WARPGROUP_WARPS; // Warpgroup ID
    const int tp_shard_rank = tp;
    const bool is_producer = warpgroupid == NUM_WARPGROUPS - 1;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    if (is_producer)
    {
        warpgroup::decrease_registers<24>(); // producer doesnt need many registers due to tma
    }
    else
    {
        warpgroup::increase_registers<240>(); // consumer needs all of the registers
    }
    


    // Shared memory for hidden states
    F_F_tile_acc_type(&w1_smem) = al.allocate<F_F_tile_acc_type>();
    F_vec_acc_type(&b1_smem) = al.allocate<F_vec_acc_type>();
    F_F_tile_acc_type(&w2_smem) = al.allocate<F_F_tile_acc_type>();
    F_vec_acc_type(&b2_smem) = al.allocate<F_vec_acc_type>();

    // Shared memory for inputs (staged)
    CS_F_tile_type(&q_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_F_tile_type(&k_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_F_tile_type(&v_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_vec_type(&last_eta_smem)[K::stages] = al.allocate<CS_vec_type, K::stages>();

    // Shared memory for ttt norm params
    F_vec_acc_type(&ttt_norm_weight_smem) = al.allocate<F_vec_acc_type>();
    F_vec_acc_type(&ttt_norm_bias_smem) = al.allocate<F_vec_acc_type>();

    CS_F_tile_type(&z1_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&x2_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&ln_tile_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&grad_l_z1_smem) = al.allocate<CS_F_tile_type>();
    sv_fl<16>(&ln_smem)[4] = al.allocate<sv_fl<16>, 4>();
    CS_F_tile_type(&matmul_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_acc_type(&b_acc_smem) = al.allocate<CS_F_tile_acc_type>();
    CS_F_tile_type(&cs_f_store_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&cs_f_store2_smem) = al.allocate<CS_F_tile_type>();

    // Reinterpretations for intermediates
    auto(&reduction_buffer) = matmul_smem;
    auto(&z2_smem) = grad_l_z1_smem;
    auto(&grad_l_z2_smem) = v_smem[0];


    // Create locks to make sure reads aren't premature
    __shared__ kittens::semaphore 
        w1_arrived,
        w2_arrived,
        b1_arrived,
        b2_arrived,
        q_sem_arrived[K::stages],
        k_sem_arrived[K::stages], 
        v_sem_arrived[K::stages],
        last_eta_sem_arrived[K::stages],
        ttt_norm_weight_arrived,
        ttt_norm_bias_arrived,
        compute_done[K::stages],
        dsmem_semaphore1,
        dsmem_semaphore2,
        second_dsmem_semaphore1,
        second_dsmem_semaphore2;

    if (threadIdx.x == 0) {
        init_semaphore(w1_arrived, 0, 1);
        init_semaphore(b1_arrived, 0, 1);
        init_semaphore(w2_arrived, 0, 1);
        init_semaphore(b2_arrived, 0, 1);
        init_semaphore(ttt_norm_weight_arrived, 0, 1);
        init_semaphore(ttt_norm_bias_arrived, 0, 1);
        init_semaphore(dsmem_semaphore1, 0, 1);
        init_semaphore(dsmem_semaphore2, 0, 1);
        init_semaphore(second_dsmem_semaphore1, 0, 1);
        init_semaphore(second_dsmem_semaphore2, 0, 1);
        for (int i = 0; i < K::stages; i++) {
            init_semaphore(q_sem_arrived[i], 0, 1);
            init_semaphore(k_sem_arrived[i], 0, 1);
            init_semaphore(v_sem_arrived[i], 0, 1);
            init_semaphore(last_eta_sem_arrived[i], 0, 1);
            init_semaphore(compute_done[i], CONSUMER_WARPGROUPS, 0);
        }

        // ttt norm params
        tma::expect_bytes(ttt_norm_weight_arrived, sizeof(ttt_norm_weight_smem));
        tma::expect_bytes(ttt_norm_bias_arrived, sizeof(ttt_norm_bias_smem));
        tma::load_async(ttt_norm_weight_smem, g.ttt_norm_weight, {0, head_idx, 0, 0}, ttt_norm_weight_arrived);
        tma::load_async(ttt_norm_bias_smem, g.ttt_norm_bias, {0, head_idx, 0, 0}, ttt_norm_bias_arrived);
    }
    __syncthreads();


    // Allow producer to start loading in 
    if (!is_producer)
    {
        kittens::wait(ttt_norm_weight_arrived, 0);
        kittens::wait(ttt_norm_bias_arrived, 0);
    }

    int semaphore_idx = 0;
    
    for (int checkpoint_idx = g.num_checkpoints - 1; checkpoint_idx >= 0; --checkpoint_idx)
    {
        if (is_producer && warpid == NUM_WORKERS - 4)
        {
            // TODO: Will have to hide this latency later
            int4 curr_checkpoint = {batch_idx, head_idx, checkpoint_idx, tp_shard_rank};
            const int sharded_checkpoint_offset = checkpoint_idx * TP + tp_shard_rank;
            const int4 W2_checkpoint = {batch_idx, head_idx, sharded_checkpoint_offset, 0};
            // Load hidden states from checkpoint
            tma::expect_bytes(w1_arrived, sizeof(w1_smem));
            tma::expect_bytes(b1_arrived, sizeof(b1_smem));
            tma::expect_bytes(w2_arrived, sizeof(w2_smem));
            tma::expect_bytes(b2_arrived, sizeof(b2_smem));
            tma::load_async(w1_smem, g.w1_checkpoints, curr_checkpoint, w1_arrived);
            tma::load_async(b1_smem, g.b1_checkpoints, curr_checkpoint, b1_arrived);
            tma::load_async(w2_smem, g.w2_checkpoints, W2_checkpoint, w2_arrived);
            tma::load_async(b2_smem, g.b2_checkpoints, {batch_idx, head_idx, checkpoint_idx, 0}, b2_arrived);
        }
        else if (!is_producer)
        {
            const int semaphore_phase = (g.num_checkpoints - checkpoint_idx - 1) % 2;
            kittens::wait(w1_arrived, semaphore_phase);
            kittens::wait(b1_arrived, semaphore_phase);
            kittens::wait(w2_arrived, semaphore_phase);
            kittens::wait(b2_arrived, semaphore_phase);
        }

        // backward-forward
        for (int mini_batch_in_group_idx = 0; mini_batch_in_group_idx < g.checkpoint_group_size; ++mini_batch_in_group_idx)
        {
            const int global_mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_in_group_idx;

            if (is_producer)
            {
                if (warpid == NUM_WORKERS - 4)
                {
                    int4 tile_idx = {batch_idx, head_idx, global_mini_batch_idx, 0};

                    const int curr_stage = semaphore_idx % K::stages;

                    // Wait on previous stage to finish computation
                    if (semaphore_idx / K::stages > 0)
                    {
                        const int last_idx = semaphore_idx - K::stages;
                        kittens::wait(compute_done[last_idx % K::stages], (last_idx / K::stages) % 2);
                    }
                    
                    tma::expect_bytes(k_sem_arrived[curr_stage], sizeof(k_smem[0]));
                    tma::load_async(k_smem[curr_stage], g.k, tile_idx, k_sem_arrived[curr_stage]);
                    tma::expect_bytes(v_sem_arrived[curr_stage], sizeof(v_smem[0]));
                    tma::load_async(v_smem[curr_stage], g.v, tile_idx, v_sem_arrived[curr_stage]);
                    tma::expect_bytes(q_sem_arrived[curr_stage], sizeof(q_smem[0]));
                    tma::load_async(q_smem[curr_stage], g.q, tile_idx, q_sem_arrived[curr_stage]);
                    tma::expect_bytes(last_eta_sem_arrived[curr_stage], sizeof(last_eta_smem[0]));
                    tma::load_async(last_eta_smem[curr_stage], g.last_eta, tile_idx, last_eta_sem_arrived[curr_stage]);

                    tp_reduce_arrive(); // might be slow to do this, idk
                    tp_reduce_arrive(); // Should be two reduces
                }            
            }
            else
            {
                if (wg_warpid == 0) {
                    tma::store_async(g.W1_init_group, w1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.b1_init_group, b1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.W2_init_group, w2_smem, {batch_idx, head_idx, mini_batch_in_group_idx * TP + tp_shard_rank, 0});
                    tma::store_async(g.b2_init_group, b2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_commit_group();
                }

                rt_fl<16, K::F> cs_cs_fl_reg;
                rt_fl<16, K::F> cs_cs_fl_reg2;
                typeof(cs_cs_fl_reg)::row_vec cs_row_fl_reg;
                typeof(cs_cs_fl_reg)::col_vec cs_col_fl_reg;

                const int curr_stage = (semaphore_idx) % K::stages;
                const int semaphore_phase = (semaphore_idx / K::stages) % 2;
                
                // Hidden state forward
                kittens::wait(k_sem_arrived[curr_stage], semaphore_phase);
                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, k_smem[curr_stage], matmul_smem);
                load(cs_row_fl_reg, b1_smem);
                warpgroup::mma_async_wait();
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(z1_smem, cs_cs_fl_reg);
                
                gelu(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(x2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.Z1_group, z1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.X2_group, x2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_commit_group();
                }

                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, x2_smem, matmul_smem);
                warpgroup::mma_async_wait();
                // Only add b2 to one of the sharded Z2
                // Else, post reduction it will result in adding 4*b2
                if (tp_shard_rank == 0) {
                    load(cs_row_fl_reg, b2_smem);
                    add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                }
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+4);

                // Reductions
                tp_reduce(z2_smem, reduction_buffer, dsmem_semaphore1, dsmem_semaphore2, semaphore_idx);

                // LN
                // mean
                zero(cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // mu_fused

                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // Z2 - mu_fused, first part of xhat
                warpgroup::store(ln_tile_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // var
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg);

                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // var
                add(cs_col_fl_reg, cs_col_fl_reg, 1e-6f); 
                sqrt(cs_col_fl_reg, cs_col_fl_reg); // std
                store(ln_smem[wg_warpid], cs_col_fl_reg);

                // finish x_hat
                warpgroup::load(cs_cs_fl_reg, ln_tile_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // final x_hat
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0)
                {
                    tma::store_async(g.x_hat_fused_group, z2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                }
                    

                tma::store_async(g.std_fused_group, ln_smem[wg_warpid], {batch_idx, head_idx, mini_batch_in_group_idx, wg_warpid});
                tma::store_commit_group();
                
                
                
                // compute y
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                load(cs_row_fl_reg, ttt_norm_bias_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg); // y
                
                // LN BWD
                // grad_output
                warpgroup::load(cs_cs_fl_reg2, v_smem[curr_stage]);
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, k_smem[curr_stage]);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg);



                // grad x_hat
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(cs_f_store2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.grad_output_fused_group, cs_f_store_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_async(g.grad_x_hat_fused_group, cs_f_store2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_commit_group();
                }

                warpgroup::load(cs_cs_fl_reg2, z2_smem);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg, cs_cs_fl_reg2);
                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, z2_smem);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg); // 3rd line, not negative

                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                add_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);

                mul(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                div(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));

                load(cs_col_fl_reg, ln_smem[wg_warpid]);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg); // untouched grad_l_wrt_Z2
                mul(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f); // negate to prepare for grad step
                kittens::wait(last_eta_sem_arrived[curr_stage], semaphore_phase);

                warpgroup::store(grad_l_z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul_row(grad_l_z2_smem, grad_l_z2_smem, last_eta_smem[curr_stage]);

                if (wg_warpid == 0) {
                    tma::store_async(g.grad_l_wrt_Z2_group, cs_f_store_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_commit_group();
                }

                // Calculate grad_l_wrt_Z1
                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_ABt(cs_cs_fl_reg, grad_l_z2_smem, matmul_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(grad_l_z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::load(cs_cs_fl_reg, z1_smem);
                gelu_bwd(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1); // I dont think this one is necessary
                warpgroup::mul(grad_l_z1_smem, grad_l_z1_smem, z1_smem);

                // recalc grad_l_wrt_z1 without eta for fwd comp
                zero(cs_cs_fl_reg);
                warpgroup::mm_ABt(cs_cs_fl_reg, cs_f_store_smem, matmul_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul(cs_f_store_smem, cs_f_store_smem, z1_smem); // already gelu bwd for Z1
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.grad_l_wrt_Z1_group, cs_f_store_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_commit_group();
                }
                
                // Update W2
                warpgroup::load(cs_cs_fl_reg, w2_smem);
                warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem, grad_l_z2_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(w2_smem, cs_cs_fl_reg);

                // Update b2
                warpgroup::copy(b_acc_smem, grad_l_z2_smem);
                warpgroup::col_sum(b2_smem, b_acc_smem, b2_smem);

                // Update W1
                warpgroup::load(cs_cs_fl_reg, w1_smem);
                warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[curr_stage], grad_l_z1_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(w1_smem, cs_cs_fl_reg);

                // Update b1
                warpgroup::copy(b_acc_smem, grad_l_z1_smem);
                warpgroup::col_sum(b1_smem, b_acc_smem, b1_smem);

                warpgroup::sync(warpgroupid+1);
                

                // Compute output
                zero(cs_cs_fl_reg);
                kittens::wait(q_sem_arrived[curr_stage], semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, w1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, q_smem[curr_stage], matmul_smem);
                warpgroup::mma_async_wait();
                load(cs_row_fl_reg, b1_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(z1_smem, cs_cs_fl_reg);
                gelu(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(x2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.Z1_bar_group, z1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.X2_bar_group, x2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_commit_group();
                }

                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, x2_smem, matmul_smem);
                warpgroup::mma_async_wait();
                // Only add b2 to one of the sharded Z2
                // Else, post reduction it will result in adding 4*b2
                if (tp_shard_rank == 0) {
                    load(cs_row_fl_reg, b2_smem);
                    add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                }
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // Reduction
                tp_reduce(z2_smem, reduction_buffer, second_dsmem_semaphore1, second_dsmem_semaphore2, semaphore_idx);

                // mean
                zero(cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // mu_fused

                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // Z2 - mu_fused, first part of xhat
                warpgroup::store(ln_tile_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // var
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg);

                // zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // var
                add(cs_col_fl_reg, cs_col_fl_reg, 1e-6f); 
                sqrt(cs_col_fl_reg, cs_col_fl_reg); // std
                store(ln_smem[wg_warpid], cs_col_fl_reg);

                // finish x_hat
                warpgroup::load(cs_cs_fl_reg, ln_tile_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // final x_hat
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0)
                {
                    tma::store_async(g.x_hat_ln_group, z2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                }
                tma::store_async(g.std_ln_group, ln_smem[wg_warpid], {batch_idx, head_idx, mini_batch_in_group_idx, wg_warpid});
                tma::store_commit_group();
                
                // // compute y
                // load(cs_row_fl_reg, ttt_norm_weight_smem);
                // mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                // load(cs_row_fl_reg, ttt_norm_bias_smem);
                // add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg); // y

                // // Residual
                // warpgroup::load(cs_cs_fl_reg2, q_smem[curr_stage]);
                // add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                
                // warpgroup::store(z2_smem, cs_cs_fl_reg);
                // warpgroup::sync(warpgroupid+1);
                // // Store output to global
                // if (tp_shard_rank == 0 && wg_warpid == 0) {
                //     tma::store_async(g.o, z2_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                //     tma::store_commit_group();
                // }


                if (warpgroup::laneid() == 0) arrive(compute_done[curr_stage], 1);
                
            }

            ++semaphore_idx;
        }



        // At this point, hidden states should be the last in the checkpoint group

        // backward-backward
        for (int mini_batch_in_group_idx = g.checkpoint_group_size; mini_batch_in_group_idx >= 0; --mini_batch_in_group_idx)
        {

        }
    }
}

#if TORCH_COMPILE

#include "common/pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

torch::Tensor ttt_backward(
    const torch::Tensor XQ,
    const torch::Tensor XK,
    const torch::Tensor XV,
    const torch::Tensor last_eta,
    const torch::Tensor ttt_norm_weight,
    const torch::Tensor ttt_norm_bias,
    const torch::Tensor W1,
    const torch::Tensor b1,
    const torch::Tensor W2,
    const torch::Tensor b2,
    const torch::Tensor W1_checkpoints,
    const torch::Tensor b1_checkpoints,
    const torch::Tensor W2_checkpoints,
    const torch::Tensor b2_checkpoints,
    const torch::Tensor Out,
    const torch::Tensor W1_init_group,
    const torch::Tensor b1_init_group,
    const torch::Tensor W2_init_group,
    const torch::Tensor b2_init_group,
    const torch::Tensor x_hat_ln_group,
    const torch::Tensor std_ln_group,
    const torch::Tensor X2_group,
    const torch::Tensor Z1_group,
    const torch::Tensor Z1_bar_group,
    const torch::Tensor X2_bar_group,
    const torch::Tensor grad_l_wrt_Z2_group,
    const torch::Tensor grad_l_wrt_Z1_group,
    const torch::Tensor x_hat_fused_group,
    const torch::Tensor grad_x_hat_fused_group,
    const torch::Tensor grad_output_fused_group,
    const torch::Tensor std_fused_group,
    const torch::Tensor grad_L_W1_last,
    const torch::Tensor grad_L_b1_last,
    const torch::Tensor grad_L_W2_last,
    const torch::Tensor grad_L_b2_last,
    const torch::Tensor grad_L_XQW_mini_batch,
    const torch::Tensor grad_L_ttt_norm_weight,
    const torch::Tensor grad_L_ttt_norm_bias,
    const torch::Tensor grad_L_W1_init,
    const torch::Tensor grad_L_b1_init,
    const torch::Tensor grad_L_W2_init,
    const torch::Tensor grad_L_b2_init,
    const torch::Tensor grad_L_last_eta,
    const torch::Tensor grad_L_XQ,
    const torch::Tensor grad_L_XK,
    const torch::Tensor grad_L_XV
) {
    constexpr int F = 64;
    constexpr int K = 4;
    const unsigned long B = XQ.size(0);
    const unsigned long H = XQ.size(1);
    const unsigned long T = XQ.size(2) * XQ.size(3); // seq len
    const unsigned long NC = XQ.size(2);
    const unsigned long CS = XQ.size(3);
    const unsigned long num_checkpoints = static_cast<int>(W1_checkpoints.size(2));
    const unsigned long checkpoint_group_size = NC / num_checkpoints;

    TORCH_CHECK(NC % num_checkpoints == 0, "N % R == 0");
    
    TORCH_CHECK(XQ.device().is_cuda() && XQ.is_contiguous() && XQ.dim() == 5 && XQ.size(4) == F, "Invalid dims for XQ");
    TORCH_CHECK(XK.device().is_cuda() && XK.is_contiguous() && XK.dim() == 5 && XK.size(4) == F, "Invalid dims for XK");
    TORCH_CHECK(XV.device().is_cuda() && XV.is_contiguous() && XV.dim() == 5 && XV.size(4) == F, "Invalid dims for XV");
    TORCH_CHECK(W1.device().is_cuda() && W1.is_contiguous() && W1.dim() == 4 && W1.size(0) == B && W1.size(1) == H && W1.size(2) == F && W1.size(3) == F*K, "Invalid dims for W1");
    TORCH_CHECK(W2.device().is_cuda() && W2.is_contiguous() && W2.dim() == 4 && W2.size(0) == B && W2.size(1) == H && W2.size(2) == F*K && W2.size(3) == F, "Invalid dims for W2");
    TORCH_CHECK(W1_checkpoints.device().is_cuda() && W1_checkpoints.is_contiguous() && W1_checkpoints.dim() == 5 && W1_checkpoints.size(0) == B && W1_checkpoints.size(1) == H && W1_checkpoints.size(2) == num_checkpoints && W1_checkpoints.size(3) == F && W1_checkpoints.size(4) == F*K, "Invalid dims for W1_checkpoints");
    TORCH_CHECK(W2_checkpoints.device().is_cuda() && W2_checkpoints.is_contiguous() && W2_checkpoints.dim() == 5 && W2_checkpoints.size(0) == B && W2_checkpoints.size(1) == H && W2_checkpoints.size(2) == num_checkpoints && W2_checkpoints.size(3) == F*K && W2_checkpoints.size(4) == F, "Invalid dims for W2_checkpoints");
    TORCH_CHECK(Out.device().is_cuda() && Out.is_contiguous() && Out.dim() == 5 && Out.size(4) == F, "Invalid dims for Out");

    TORCH_CHECK(ttt_norm_weight.device().is_cuda() && ttt_norm_weight.is_contiguous() && ttt_norm_weight.dim() == 4 && ttt_norm_weight.size(0) == 1 && ttt_norm_weight.size(1) == H && ttt_norm_weight.size(2) == 1 && ttt_norm_weight.size(2) == 1 && ttt_norm_weight.size(3) == F, "Invalid dims for ttt_norm_weight");

    using globals = bwd_globals<F>;

    using CS_F_tile_type = globals::CS_F_tile_type;
    using F_F_tile_type = globals::F_F_tile_type;
    using CS_F_tile_acc_type = globals::CS_F_tile_acc_type;
    using F_F_tile_acc_type = globals::F_F_tile_acc_type;

    // Vectors
    using CS_vec_type = globals::CS_vec_type;
    using F_vec_type = globals::F_vec_type;
    using CS_vec_acc_type = globals::CS_vec_acc_type;
    using F_vec_acc_type = globals::F_vec_acc_type;
    using std_vec_acc_type = globals::std_vec_acc_type;

    using CS_F_tile_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using F_F_tile_gl = gl<bf16, -1, -1, -1, -1, F_F_tile_type>;
    using CS_F_tile_acc_gl = gl<float, -1, -1, -1, -1, CS_F_tile_acc_type>;
    using F_F_tile_acc_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;

    using CS_vec_gl = gl<bf16, -1, -1, -1, -1, CS_vec_type>;
    using F_vec_gl = gl<bf16, -1, -1, -1, -1, F_vec_type>;
    using CS_vec_acc_gl = gl<float, -1, -1, -1, -1, CS_vec_acc_type>;
    using F_vec_acc_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using std_gl = gl<float, -1, -1, -1, -1, std_vec_acc_type>;

    CS_F_tile_gl q_gl{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl k_gl{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl v_gl{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl o_gl{reinterpret_cast<bf16*>(Out.data_ptr<at::BFloat16>()), B, H, T, F};

    CS_vec_gl last_eta_gl{reinterpret_cast<bf16*>(last_eta.data_ptr<at::BFloat16>()), B, H, NC, CS};

    F_vec_acc_gl ttt_norm_weight_gl{reinterpret_cast<float*>(ttt_norm_weight.data_ptr<float>()), 1, H, 1, F};
    F_vec_acc_gl ttt_norm_bias_gl{reinterpret_cast<float*>(ttt_norm_bias.data_ptr<float>()), 1, H, 1, F};

    F_F_tile_acc_gl w1_init_gl{reinterpret_cast<float*>(W1.data_ptr<float>()), B, H, F, F*K};
    F_vec_acc_gl b1_init_gl{reinterpret_cast<float*>(b1.data_ptr<float>()), B, H, 1, F*K};
    F_F_tile_acc_gl w2_init_gl{reinterpret_cast<float*>(W2.data_ptr<float>()), B, H, F*K, F};
    F_vec_acc_gl b2_init_gl{reinterpret_cast<float*>(b2.data_ptr<float>()), B, H, 1, F};

    F_F_tile_acc_gl w1_checkpoints_gl{reinterpret_cast<float*>(W1_checkpoints.data_ptr<float>()), B, H, num_checkpoints*F, F*K};
    F_vec_acc_gl b1_checkpoints_gl{reinterpret_cast<float*>(b1_checkpoints.data_ptr<float>()), B, H, num_checkpoints, F*K};
    F_F_tile_acc_gl w2_checkpoints_gl{reinterpret_cast<float*>(W2_checkpoints.data_ptr<float>()), B, H, num_checkpoints*F*K, F};
    F_vec_acc_gl b2_checkpoints_gl{reinterpret_cast<float*>(b2_checkpoints.data_ptr<float>()), B, H, num_checkpoints, F};

    // Rematted activations
    F_F_tile_acc_gl W1_init_group_gl{reinterpret_cast<float*>(W1_init_group.data_ptr<float>()), B, H, checkpoint_group_size*F, F*K};
    F_vec_acc_gl b1_init_group_gl{reinterpret_cast<float*>(b1_init_group.data_ptr<float>()), B, H, checkpoint_group_size, F*K};
    F_F_tile_acc_gl W2_init_group_gl{reinterpret_cast<float*>(W2_init_group.data_ptr<float>()), B, H, checkpoint_group_size*F*K, F};
    F_vec_acc_gl b2_init_group_gl{reinterpret_cast<float*>(b2_init_group.data_ptr<float>()), B, H, checkpoint_group_size, F};
    CS_F_tile_gl x_hat_ln_group_gl{reinterpret_cast<bf16*>(x_hat_ln_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    std_gl std_ln_group_gl{reinterpret_cast<float*>(std_ln_group.data_ptr<float>()), B, H, checkpoint_group_size, CS};
    CS_F_tile_gl X2_group_gl{reinterpret_cast<bf16*>(X2_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl Z1_group_gl{reinterpret_cast<bf16*>(Z1_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl Z1_bar_group_gl{reinterpret_cast<bf16*>(Z1_bar_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl X2_bar_group_gl{reinterpret_cast<bf16*>(X2_bar_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl grad_l_wrt_Z2_group_gl{reinterpret_cast<bf16*>(grad_l_wrt_Z2_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    CS_F_tile_gl grad_l_wrt_Z1_group_gl{reinterpret_cast<bf16*>(grad_l_wrt_Z1_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl x_hat_fused_group_gl{reinterpret_cast<bf16*>(x_hat_fused_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    CS_F_tile_gl grad_x_hat_fused_group_gl{reinterpret_cast<bf16*>(grad_x_hat_fused_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    CS_F_tile_gl grad_output_fused_group_gl{reinterpret_cast<bf16*>(grad_output_fused_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    std_gl std_fused_group_gl{reinterpret_cast<float*>(std_fused_group.data_ptr<float>()), B, H, checkpoint_group_size, CS};

    // Upstream grads
    F_F_tile_acc_gl grad_L_W1_last_gl{reinterpret_cast<float*>(grad_L_W1_last.data_ptr<float>()), B, H, F, F*K};
    F_vec_acc_gl grad_L_b1_last_gl{reinterpret_cast<float*>(grad_L_b1_last.data_ptr<float>()), B, H, 1, F*K};
    F_F_tile_acc_gl grad_L_W2_last_gl{reinterpret_cast<float*>(grad_L_W2_last.data_ptr<float>()), B, H, F*K, F};
    F_vec_acc_gl grad_L_b2_last_gl{reinterpret_cast<float*>(grad_L_b2_last.data_ptr<float>()), B, H, 1, F};
    CS_F_tile_gl grad_L_XQW_mini_batch_gl{reinterpret_cast<bf16*>(grad_L_XQW_mini_batch.data_ptr<at::BFloat16>()), B, H, T, F};

    // Output grads
    F_vec_acc_gl grad_L_ttt_norm_weight_gl{reinterpret_cast<float*>(grad_L_ttt_norm_weight.data_ptr<float>()), 1, H, 1, F};
    F_vec_acc_gl grad_L_ttt_norm_bias_gl{reinterpret_cast<float*>(grad_L_ttt_norm_bias.data_ptr<float>()), 1, H, 1, F};
    F_F_tile_acc_gl grad_L_W1_init_gl{reinterpret_cast<float*>(grad_L_W1_init.data_ptr<float>()), B, H, F, F*K};
    F_vec_acc_gl grad_L_b1_init_gl{reinterpret_cast<float*>(grad_L_b1_init.data_ptr<float>()), B, H, 1, F*K};
    F_F_tile_acc_gl grad_L_W2_init_gl{reinterpret_cast<float*>(grad_L_W2_init.data_ptr<float>()), B, H, F*K, F};
    F_vec_acc_gl grad_L_b2_init_gl{reinterpret_cast<float*>(grad_L_b2_init.data_ptr<float>()), B, H, 1, F};
    CS_vec_gl grad_L_last_eta_gl{reinterpret_cast<bf16*>(grad_L_last_eta.data_ptr<at::BFloat16>()), B, H, NC, CS};
    CS_F_tile_gl grad_L_XQ_gl{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl grad_L_XK_gl{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl grad_L_XV_gl{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), B, H, T, F};

    globals g{
        q_gl, 
        k_gl, 
        v_gl, 
        o_gl, 
        last_eta_gl,
        ttt_norm_weight_gl,
        ttt_norm_bias_gl,
        w1_init_gl, 
        b1_init_gl,
        w2_init_gl, 
        b2_init_gl,
        w1_checkpoints_gl, 
        b1_checkpoints_gl,
        w2_checkpoints_gl, 
        b2_checkpoints_gl,
        W1_init_group_gl,
        b1_init_group_gl,
        W2_init_group_gl,
        b2_init_group_gl,
        x_hat_ln_group_gl,
        std_ln_group_gl,
        X2_group_gl,
        Z1_group_gl,
        Z1_bar_group_gl,
        X2_bar_group_gl,
        grad_l_wrt_Z2_group_gl,
        grad_l_wrt_Z1_group_gl,
        x_hat_fused_group_gl,
        grad_x_hat_fused_group_gl,
        grad_output_fused_group_gl,
        std_fused_group_gl,
        grad_L_W1_last_gl,
        grad_L_b1_last_gl,
        grad_L_W2_last_gl,
        grad_L_b2_last_gl,
        grad_L_XQW_mini_batch_gl,
        grad_L_ttt_norm_weight_gl,
        grad_L_ttt_norm_bias_gl,
        grad_L_W1_init_gl,
        grad_L_b1_init_gl,
        grad_L_W2_init_gl,
        grad_L_b2_init_gl,
        grad_L_last_eta_gl,
        grad_L_XQ_gl,
        grad_L_XK_gl,
        grad_L_XV_gl,
        static_cast<int>(T),
        static_cast<int>(num_checkpoints),
        checkpoint_group_size
    };

    auto stream = at::cuda::getCurrentCUDAStream().stream(); 


    constexpr long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        bwd_ttt_mlp_ker<F>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(TP, B, H);
    bwd_ttt_mlp_ker<F><<<grid, NUM_WORKERS*32, mem_size, stream>>>(g);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Ensure the kernel execution completes
    cudaStreamSynchronize(stream);
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("CUDA Kernel Execution Error: %s\n", cudaGetErrorString(syncErr));
    }

    return Out;
}//*/

#else

#include "harness.cuh"

#endif
