#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_MLP_FORWARD_TP
#endif

#define CUDA_ASSERT(cond, tidx) if (!(cond)) { if (tidx == -1 || (threadIdx.x == tidx && tp_idx == 0)) { printf("Kernel assert failed: %s\n", #cond); } return; }

constexpr int CONSUMER_WARPGROUPS = (4);
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
    using tile_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using vec_gl = gl<bf16, -1, -1, -1, -1, vec_type>;

    tile_gl q, k, v, o;
    
    tile_gl w1, w2;
    vec_gl b1, b2;
    
    tile_gl w1_checkpoints, w2_checkpoints;
    vec_gl b1_checkpoints, b2_checkpoints;

    const int seq_len;
    const int remat_gs;
};

template <ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void warpgroup_store_add(ST &dst, const RT &src) {
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2, "Only row-major bf layout supported");
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height % warpgroup::GROUP_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height % warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width == RT::width, "Group load / store requires tile widths to match.");
    int local_warpid = warpgroup::warpid();
    using T2 = RT::dtype;
    using U = ST::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = kittens::laneid();

    #pragma unroll
    for (int i = 0; i < warp_height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            int row = (local_warpid * warp_height + i) * src.tile_size + (warp_laneid / 4);
            int col = j * src.tile_size + 2 * (warp_laneid % 4);

            U2 tmp[4];
            tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]); 
            tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);

            atomicAdd(reinterpret_cast<U2 *>(dst.idx(&dst.data[0], {row + 0, col + 0})), tmp[0]);
            atomicAdd(reinterpret_cast<U2 *>(dst.idx(&dst.data[0], {row + 8, col + 0})), tmp[1]);
            atomicAdd(reinterpret_cast<U2 *>(dst.idx(&dst.data[0], {row + 0, col + 8})), tmp[2]);
            atomicAdd(reinterpret_cast<U2 *>(dst.idx(&dst.data[0], {row + 8, col + 8})), tmp[3]);
        }
    }
}

template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void warp_store_add(SV &dst, const RV &src) {
    static_assert(dst.length == src.length);
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    int laneid = ::kittens::laneid();
    static_assert(std::is_same_v<typename RV::layout, align_l>, "Only align layout supported");
    __syncwarp();
    #pragma unroll
    for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
        int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
        int o_dim = w*4 + (laneid/4) / 2;
        int i_dim = (laneid/4) % 2;
        if(idx < src.outer_dim*16) {
            atomicAdd(&(reinterpret_cast<U2*>(dst.data)[idx/2]), base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]));
        }
    }
}

template <int head_dim>
__global__ __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_ttt_mlp_ker(const __grid_constant__ fwd_globals<head_dim> g) {
    using K = fwd_ttt_mlp_ker_tile_dims<head_dim>;
    using tile_type = st_bf<K::tile_height, K::tile_width>;
    using vec_type = sv_bf<K::tile_height>;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int n_minibatch = g.seq_len / (K::tile_height);
    const int n_remat_groups = n_minibatch / g.remat_gs;

    int warpid = kittens::warpid(); // Global warp ID
    int wg_warpid = warpgroup::warpid(); // Warp ID within Warpgroup
    int warpgroupid = warpgroup::groupid(); // Warpgroup ID

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    // Shared memory for hidden states
    tile_type(&w1_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    tile_type(&w2_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    vec_type(&b1_smem)[CONSUMER_WARPGROUPS] = al.allocate<vec_type, CONSUMER_WARPGROUPS>();
    vec_type(&b2_smem) = al.allocate<vec_type>();
    
    // Shared memory for inputs (staged)
    tile_type(&q_smem)[K::stages] = al.allocate<tile_type, K::stages>();
    tile_type(&k_smem)[K::stages] = al.allocate<tile_type, K::stages>();
    tile_type(&v_smem)[K::stages] = al.allocate<tile_type, K::stages>();
    
    // Shared memory for intermediates
    tile_type(&x2_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    tile_type(&grad_l_z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();
    tile_type(&temp_smem)[CONSUMER_WARPGROUPS] = al.allocate<tile_type, CONSUMER_WARPGROUPS>();

    // Reinterpretations for intermediates
    auto(*z1_smem) = reinterpret_cast<tile_type(*)>(temp_smem);
    auto(*grad_l_z2_smem) = reinterpret_cast<tile_type(*)>(v_smem);
    auto(*attn1_smem) = reinterpret_cast<tile_type(*)>(temp_smem);
    auto(*x2_bar_smem) = reinterpret_cast<tile_type(*)>(grad_l_z1_smem);
    auto(*attn2_smem) = reinterpret_cast<tile_type(*)>(x2_smem);
    auto(*z2_bar_smem) = reinterpret_cast<tile_type(*)>(temp_smem);

    __shared__ kittens::semaphore 
        w0_arrived,
        reduction1_done,
        reduction2_done,
        q_sem_arrived[K::stages],
        k_sem_arrived[K::stages], 
        v_sem_arrived[K::stages],
        compute_done[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(w0_arrived, 0, 1);
        init_semaphore(reduction1_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(reduction2_done, CONSUMER_WARPGROUPS, 0);
        for (int i = 0; i < K::stages; i++) {
            init_semaphore(q_sem_arrived[i], 0, 1);
            init_semaphore(k_sem_arrived[i], 0, 1);
            init_semaphore(v_sem_arrived[i], 0, 1);
            init_semaphore(compute_done[i], CONSUMER_WARPGROUPS, 0);
        }

        // Load hidden states across consumer warpgroups
        tma::expect_bytes(w0_arrived, sizeof(w1_smem)+sizeof(w2_smem)+sizeof(b1_smem)+sizeof(b2_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            tma::load_async(w1_smem[wg], g.w1, {batch_idx, head_idx, 0, wg}, w0_arrived);
            tma::load_async(w2_smem[wg], g.w2, {batch_idx, head_idx, wg, 0}, w0_arrived);
            tma::load_async(b1_smem[wg], g.b1, {batch_idx, head_idx, 0, wg}, w0_arrived);
        }
        tma::load_async(b2_smem, g.b2, {batch_idx, head_idx, 0, 0}, w0_arrived);

        // Preload 1 minibatch
        int4 tile_idx = {batch_idx, head_idx, 0, 0};
        tma::expect_bytes(k_sem_arrived[0], sizeof(tile_type));
        tma::load_async(k_smem[0], g.k, tile_idx, k_sem_arrived[0]);
        tma::expect_bytes(v_sem_arrived[0], sizeof(tile_type));
        tma::load_async(v_smem[0], g.v, tile_idx, v_sem_arrived[0]);
        tma::expect_bytes(q_sem_arrived[0], sizeof(tile_type));
        tma::load_async(q_smem[0], g.q, tile_idx, q_sem_arrived[0]);
    }
    __syncthreads();

    // First warp in last warpgroup is the consumer
    if (warpgroupid == NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<32>();

        int iters = n_minibatch - 1;
        if (warpid == NUM_WORKERS - 4) {
            for (auto idx = 0; idx < iters; idx++) {
                int4 tile_idx = {batch_idx, head_idx, idx + 1, 0};

                tma::expect_bytes(k_sem_arrived[(idx + 1) % K::stages], sizeof(tile_type));
                tma::load_async(k_smem[(idx + 1) % K::stages], g.k, tile_idx, k_sem_arrived[(idx + 1) % K::stages]);
                tma::expect_bytes(v_sem_arrived[(idx + 1) % K::stages], sizeof(tile_type));
                tma::load_async(v_smem[(idx + 1) % K::stages], g.v, tile_idx, v_sem_arrived[(idx + 1) % K::stages]);
                tma::expect_bytes(q_sem_arrived[(idx + 1) % K::stages], sizeof(tile_type));
                tma::load_async(q_smem[(idx + 1) % K::stages], g.q, tile_idx, q_sem_arrived[(idx + 1) % K::stages]);

                // Wait on previous stage to finish computation
                kittens::wait(compute_done[idx % K::stages], (idx / K::stages) % 2);
            }
        }
    } else {
        warpgroup::increase_registers<112>();

        rt_fl<16, K::tile_height> cs_cs_fl_reg;
        typeof(cs_cs_fl_reg)::row_vec cs_row_fl_reg;

        kittens::wait(w0_arrived, 0);

        for (auto idx = 0; idx < n_minibatch; idx++) {
            // Hidden state forward
            kittens::wait(k_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            // Z1 = XK @ W1 + B1
            warpgroup::mm_AB(cs_cs_fl_reg, k_smem[idx % K::stages], w1_smem[warpgroupid]);
            load(cs_row_fl_reg, b1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);

            // X2 = gelu(Z1)
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);
            // Z2 = X2 @ W2 + B2
            warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], w2_smem[warpgroupid]);
            if (warpgroupid == 0) load(cs_row_fl_reg, b2_smem);
            warpgroup::mma_async_wait();
            if (warpgroupid == 0) add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);

            // -grad_l_z2 = -V + Z2
            // We use negative gradients to use the WGMMA accumulator
            // Assumes V is passed into the kernel already negated
            // Note: -V buffer no longer represents -V after this
            kittens::wait(v_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            warpgroup_store_add(grad_l_z2_smem[idx % K::stages], cs_cs_fl_reg);
            // Wait for reduction to be complete
            if (warpgroup::laneid() == 0) arrive(reduction1_done, 1);
            kittens::wait(reduction1_done, idx % 2);

            // Update hidden state W2 += X2 @ -grad_l_z2
            warpgroup::load(cs_cs_fl_reg, w2_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem[warpgroupid], grad_l_z2_smem[idx % K::stages]);
            warpgroup::mma_async_wait();
            warpgroup::store(w2_smem[warpgroupid], cs_cs_fl_reg);
            // B2 += sum(-grad_l_z2, axis=-2)
            warpgroup::load(cs_cs_fl_reg, grad_l_z2_smem[idx % K::stages]);
            col_sum(cs_row_fl_reg, cs_cs_fl_reg);
            warp_store_add(b2_smem, cs_row_fl_reg);

            // W2 and B2 checkpoint
            if (wg_warpid == 0 && (idx + 1) % n_remat_groups == 0) {
                int4 curr_checkpoint = {batch_idx, head_idx, ((idx + 1) / n_remat_groups) * 4 + warpgroupid, 0};
                tma::store_async(g.w2_checkpoints, w2_smem[warpgroupid], curr_checkpoint);
                if (warpgroupid == 0) {
                    curr_checkpoint = {batch_idx, head_idx, ((idx + 1) / n_remat_groups), 0};
                    tma::store_async(g.b2_checkpoints, b2_smem, curr_checkpoint);
                }
                tma::store_commit_group();
            }

            // -grad_l_wrt_Z1 = -grad_l_z2 @ W2^T
            warpgroup::mm_ABt(cs_cs_fl_reg, grad_l_z2_smem[idx % K::stages], w2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(grad_l_z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::load(cs_cs_fl_reg, z1_smem[warpgroupid]);
            gelu_bwd(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg); //TODO: write shr<-shr*reg function to avoid store to temp_smem
            mul(grad_l_z1_smem[warpgroupid], grad_l_z1_smem[warpgroupid], z1_smem[warpgroupid]);

            // Attn1 = tril(XQ @ XK^T)
            kittens::wait(q_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            warpgroup::mm_ABt(cs_cs_fl_reg, q_smem[idx % K::stages], k_smem[idx % K::stages]);
            warpgroup::mma_async_wait();
            make_causal(cs_cs_fl_reg, cs_cs_fl_reg, base_types::constants<bf16>::zero());
            warpgroup::store(attn1_smem[warpgroupid], cs_cs_fl_reg);

            // W1 += XK^T @ -grad_l_wrt_z1
            warpgroup::load(cs_cs_fl_reg, w1_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[idx % K::stages], grad_l_z1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(w1_smem[warpgroupid], cs_cs_fl_reg);
            // B1 += sum(-grad_l_wrt_z1, axis=-2)
            warpgroup::load(cs_cs_fl_reg, grad_l_z1_smem[warpgroupid]);
            col_sum(cs_row_fl_reg, cs_cs_fl_reg);
            warp_store_add(b1_smem[warpgroupid], cs_row_fl_reg);

            // W1 and B1 checkpoint
            if (wg_warpid == 0 && (idx + 1) % n_remat_groups == 0) {
                int4 curr_checkpoint = {batch_idx, head_idx, (idx + 1) / n_remat_groups, warpgroupid};
                tma::store_async(g.w1_checkpoints, w1_smem[warpgroupid], curr_checkpoint);
                tma::store_async(g.b1_checkpoints, b1_smem[warpgroupid], curr_checkpoint);
                tma::store_commit_group();
            }

            // Z1_bar = XQ @ W1 - Attn1 @ grad_l_wrt_Z1 + B1 - grad_l_wrt_Z1
            warpgroup::load(cs_cs_fl_reg, grad_l_z1_smem[warpgroupid]);
            warpgroup::mma_AB(cs_cs_fl_reg, q_smem[idx % K::stages], w1_smem[warpgroupid]);
            warpgroup::mma_AB(cs_cs_fl_reg, attn1_smem[warpgroupid], grad_l_z1_smem[warpgroupid]);
            load(cs_row_fl_reg, b1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);

            // X2_bar = gelu(Z1_bar)
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_bar_smem[warpgroupid], cs_cs_fl_reg);

            // Attn2 = tril(X2_bar @ X2^T)
            warpgroup::mm_ABt(cs_cs_fl_reg, x2_bar_smem[warpgroupid], x2_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            make_causal(cs_cs_fl_reg, cs_cs_fl_reg, base_types::constants<bf16>::zero());
            warpgroup::store(attn2_smem[warpgroupid], cs_cs_fl_reg);

            // Z2_bar = X2_bar @ W2 - Attn2 @ grad_l_wrt_Z2 + B2 - grad_l_wrt_Z2
            warpgroup::load(cs_cs_fl_reg, grad_l_z2_smem[idx % K::stages]);
            warpgroup::mma_AB(cs_cs_fl_reg, x2_bar_smem[warpgroupid], w2_smem[warpgroupid]);
            warpgroup::mma_AB(cs_cs_fl_reg, attn2_smem[warpgroupid], grad_l_z2_smem[idx % K::stages]);
            if (warpgroupid == 0) load(cs_row_fl_reg, b2_smem);
            warpgroup::mma_async_wait();
            if (warpgroupid == 0) add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup_store_add(z2_bar_smem[idx % K::stages], cs_cs_fl_reg);
            // Wait for reduction to be complete
            if (warpgroup::laneid() == 0) arrive(reduction2_done, 1);
            kittens::wait(reduction2_done, idx % 2);

            // Store out Z2_bar
            if (warpid == 0) {
                tma::store_add_async(g.o, z2_bar_smem[idx % K::stages], {batch_idx, head_idx, idx, 0});
                tma::store_commit_group();
            }

            if (warpgroup::laneid() == 0) arrive(compute_done[idx % K::stages], 1);
        }
    }
}

#include "harness.impl"
