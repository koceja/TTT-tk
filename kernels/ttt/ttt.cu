#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_MLP_FORWARD_TP
#endif

#define CUDA_ASSERT(cond, tidx)                                                                                        \
        if (!(cond)) {                                                                                                 \
                if (tidx == -1 || threadIdx.x == tidx && tp_idx == 0)                                                  \
                        printf("Kernel assert failed: %s\n", #cond);                                                   \
                return;                                                                                                \
        }

constexpr int CONSUMER_WARPGROUPS = (4);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);
constexpr int SM_TP = (1);

using namespace kittens;
namespace cg = cooperative_groups;

template <int D> struct fwd_attend_ker_tile_dims {};
template <> struct fwd_attend_ker_tile_dims<64> {
        constexpr static int tile_width = (64);
        constexpr static int tile_height = (64);
        constexpr static int stages = (2);
};

template <int D> struct fwd_globals {
        using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
        using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
        using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
        using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;

        using w1_tile = st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
        using w2_tile = st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;

        using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
        using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
        using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
        using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

        using w1_gl = gl<bf16, -1, -1, -1, -1, w1_tile>;
        using w2_gl = gl<bf16, -1, -1, -1, -1, w2_tile>;

        q_gl q;
        k_gl k;
        v_gl v;
        o_gl o;
        w1_gl w1;
        w2_gl w2;

        const int N;
};

template <ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void warpgroup_store_add(ST &dst, const RT &src) {
        static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2,
                      "Only row-major bf layout supported");

        constexpr int height = ST::height;
        constexpr int warp_height = RT::height;
        static_assert(height % warpgroup::GROUP_WARPS == 0,
                      "Group load / store requires tile height to be a multiple of N_WARPS.");
        static_assert(height % warp_height == 0, "Group load / store requires tile height to be a multiple of "
                                                 "the RT height.");
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

template <int TP, ducks::st::all ST>
__device__ __forceinline__ void square_all_reduce(ST &tile, ST &tile_other, int tp) {
        if constexpr (TP == 1) {
                tma::cluster::arrive_aligned();
        } else {
                static_assert(TP == 4, "TP must be 4 for this square_all_reduce implementation");
                __shared__ semaphore dsmem_semaphore[2];

                if (warpgroup::warpid() == 0) {
                        init_semaphore(dsmem_semaphore[0], 0, 1);
                        tma::expect_bytes(dsmem_semaphore[0], sizeof(tile_other));
                        init_semaphore(dsmem_semaphore[1], 0, 1);
                        tma::expect_bytes(dsmem_semaphore[1], sizeof(tile_other));
                }
                tma::cluster::sync();

                for (int stage = 0; stage < 2; stage++) {
                        if (warpgroup::warpid() == 0) {
                                tma::cluster::store_async(tile_other, tile, tp ^ (1 << stage), dsmem_semaphore[stage]);
                                kittens::wait(dsmem_semaphore[stage], 0);
                        }
                        warpgroup::sync(1);
                        warpgroup::add(tile, tile, tile_other);
                }
        }
}

template <int D>
__global__ __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
    __cluster_dims__(SM_TP) void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
        extern __shared__ int __shm[];
        tma_swizzle_allocator al((int *)&__shm[0]);
        int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

        using K = fwd_attend_ker_tile_dims<D>;

        using q_tile = st_bf<K::tile_height, K::tile_width>;
        using k_tile = st_bf<K::tile_height, K::tile_width>;
        using v_tile = st_bf<K::tile_height, K::tile_width>;

        using z1_tile = st_bf<K::tile_height, K::tile_width>;
        using x2_tile = st_bf<K::tile_height, K::tile_width>;
        using z2_tile = st_bf<K::tile_height, K::tile_width>;
        using grad_z1_tile = st_bf<K::tile_height, K::tile_width>;
        using temp_tile = st_bf<K::tile_height, K::tile_width>;

        using w1_tile = st_bf<K::tile_height, K::tile_width>;
        using w2_tile = st_bf<K::tile_height, K::tile_width>;

        w1_tile(&w1_smem)[CONSUMER_WARPGROUPS] = al.allocate<w1_tile, CONSUMER_WARPGROUPS>();
        w2_tile(&w2_smem)[CONSUMER_WARPGROUPS] = al.allocate<w2_tile, CONSUMER_WARPGROUPS>();

        q_tile(&q_smem)[K::stages] = al.allocate<q_tile, K::stages>();
        k_tile(&k_smem)[K::stages] = al.allocate<k_tile, K::stages>();
        v_tile(&v_smem)[K::stages] = al.allocate<v_tile, K::stages>();

        z1_tile(&z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<z1_tile, CONSUMER_WARPGROUPS>();
        x2_tile(&x2_smem)[CONSUMER_WARPGROUPS] = al.allocate<x2_tile, CONSUMER_WARPGROUPS>();
        grad_z1_tile(&grad_z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<grad_z1_tile, CONSUMER_WARPGROUPS>();

        temp_tile(&temp_smem)[CONSUMER_WARPGROUPS] = al.allocate<temp_tile, CONSUMER_WARPGROUPS>();
        auto(*z1_bar_smem) = reinterpret_cast<z1_tile(*)>(q_smem);
        auto(*z2_smem) = reinterpret_cast<z2_tile(*)>(v_smem);

        int batch_idx = blockIdx.y;
        int head_idx = blockIdx.z;
        int n_minibatch = g.N / (K::tile_height);

        cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
        int tp_idx = cluster.block_rank();
        CUDA_ASSERT(tp_idx == blockIdx.x, 0);

        __shared__ kittens::semaphore w1_smem_arrived, w2_smem_arrived, first_reduction_complete,
            second_reduction_complete, q_sem_arrived[K::stages], k_sem_arrived[K::stages], v_sem_arrived[K::stages],
            compute_done[K::stages];
        if (threadIdx.x == 0) {
                init_semaphore(w1_smem_arrived, 0, 1);
                init_semaphore(w2_smem_arrived, 0, 1);
                init_semaphore(first_reduction_complete, CONSUMER_WARPGROUPS, 0);
                init_semaphore(second_reduction_complete, CONSUMER_WARPGROUPS, 0);
                for (int i = 0; i < K::stages; i++) {
                        init_semaphore(q_sem_arrived[i], 0, 1);
                        init_semaphore(k_sem_arrived[i], 0, 1);
                        init_semaphore(v_sem_arrived[i], 0, 1);
                        init_semaphore(compute_done[i], CONSUMER_WARPGROUPS, 0);
                }

                // Load hidden states across consumer warpgroups
                tma::expect_bytes(w1_smem_arrived, sizeof(w1_smem));
                tma::expect_bytes(w2_smem_arrived, sizeof(w2_smem));
                for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
                        tma::load_async(w1_smem[wg], g.w1, {batch_idx, head_idx, 0, wg}, w1_smem_arrived);
                        tma::load_async(w2_smem[wg], g.w2, {batch_idx, head_idx, wg, 0}, w2_smem_arrived);
                }

                // Preload 1 minibatch
                int4 tile_idx = {batch_idx, head_idx, 0, 0};

                tma::expect_bytes(k_sem_arrived[0], sizeof(k_tile));
                tma::load_async(k_smem[0], g.k, tile_idx, k_sem_arrived[0]);

                tma::expect_bytes(v_sem_arrived[0], sizeof(v_tile));
                tma::load_async(v_smem[0], g.v, tile_idx, v_sem_arrived[0]);

                tma::expect_bytes(q_sem_arrived[0], sizeof(q_tile));
                tma::load_async(q_smem[0], g.q, tile_idx, q_sem_arrived[0]);
        }
        __syncthreads();

        if (warpgroupid == NUM_WARPGROUPS - 1) {
                warpgroup::decrease_registers<24>();
                // tma::cluster::arrive_aligned();

                int iters;
                iters = n_minibatch - 1;

                if (warpid == NUM_WORKERS - 4) {
                        for (auto idx = 0; idx < iters; idx++) {
                                int4 tile_idx = {batch_idx, head_idx, idx + 1, 0};

                                tma::expect_bytes(k_sem_arrived[(idx + 1) % K::stages], sizeof(k_tile));
                                tma::load_async(k_smem[(idx + 1) % K::stages], g.k, tile_idx,
                                                k_sem_arrived[(idx + 1) % K::stages]);

                                tma::expect_bytes(v_sem_arrived[(idx + 1) % K::stages], sizeof(v_tile));
                                tma::load_async(v_smem[(idx + 1) % K::stages], g.v, tile_idx,
                                                v_sem_arrived[(idx + 1) % K::stages]);

                                tma::expect_bytes(q_sem_arrived[(idx + 1) % K::stages], sizeof(q_tile));
                                tma::load_async(q_smem[(idx + 1) % K::stages], g.q, tile_idx,
                                                q_sem_arrived[(idx + 1) % K::stages]);

                                // Wait on previous stage to finish computation
                                kittens::wait(compute_done[idx % K::stages], (idx / K::stages) % 2);
                                // tma::cluster::arrive_aligned();
                        }
                }
        } else {
                warpgroup::increase_registers<112>();

                rt_fl<16, K::tile_height> cs_cs_fl_reg;
                rt_fl<16, K::tile_height> cs_cs_2_fl_reg;
                rt_bf<16, K::tile_height> cs_cs_bf_reg;

                kittens::wait(w1_smem_arrived, 0);
                kittens::wait(w2_smem_arrived, 0);

                for (auto idx = 0; idx < n_minibatch; idx++) {
                        // Hidden State Forward
                        kittens::wait(k_sem_arrived[idx % K::stages], (idx / K::stages) % 2);

                        warpgroup::mm_AB(cs_cs_fl_reg, k_smem[idx % K::stages], w1_smem[warpgroupid]);
                        warpgroup::mma_async_wait();
                        warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);

                        gelu(cs_cs_fl_reg, cs_cs_fl_reg);
                        warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);

                        warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], w2_smem[warpgroupid]);
                        warpgroup::mma_async_wait();

                        // Reduction over WG / SM
                        // We use negative gradients to use the WGMMA accumulator
                        // Atomic across warpgroups and wait across warpgroups
                        kittens::wait(v_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
                        warpgroup_store_add(z2_smem[idx % K::stages],
                                            cs_cs_fl_reg); // z2_smem = v_smem

                        // Wait for reduction to be complete
                        if (warpgroup::laneid() == 0)
                                arrive(first_reduction_complete, 1);
                        kittens::wait(first_reduction_complete, idx % 2);

                        // Calculate grad_l_wrt_Z1
                        warpgroup::mm_ABt(cs_cs_fl_reg, z2_smem[idx % K::stages], w2_smem[warpgroupid]);
                        warpgroup::mma_async_wait();
                        warpgroup::load(cs_cs_2_fl_reg, z1_smem[warpgroupid]);
                        gelu_bwd(cs_cs_2_fl_reg, cs_cs_2_fl_reg);
                        mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_2_fl_reg);
                        warpgroup::store(grad_z1_smem[warpgroupid],
                                         cs_cs_fl_reg); // grad_z1_smem --> grad_l_wrt_Z1

                        // Compute Attn1
                        kittens::wait(q_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
                        warpgroup::mm_ABt(cs_cs_fl_reg, q_smem[idx % K::stages], k_smem[idx % K::stages]);
                        warpgroup::mma_async_wait();
                        make_causal(cs_cs_fl_reg, cs_cs_fl_reg, base_types::constants<float>::zero());
                        warpgroup::store(temp_smem[warpgroupid],
                                         cs_cs_fl_reg); // temp_smem --> Attn1

                        // Compute Z1_bar
                        warpgroup::mm_AB(cs_cs_fl_reg, q_smem[idx % K::stages], w1_smem[warpgroupid]);
                        warpgroup::mma_AB(cs_cs_fl_reg, temp_smem[warpgroupid], grad_z1_smem[warpgroupid]);
                        warpgroup::mma_async_wait();
                        gelu(cs_cs_fl_reg, cs_cs_fl_reg);
                        warpgroup::store(z1_bar_smem[idx % K::stages],
                                         cs_cs_fl_reg); // z1_bar_smem --> Z1_bar

                        // Compute Attn2
                        warpgroup::mm_ABt(cs_cs_fl_reg, temp_smem[warpgroupid], x2_smem[warpgroupid]);
                        warpgroup::mma_async_wait();
                        make_causal(cs_cs_fl_reg, cs_cs_fl_reg, base_types::constants<float>::zero());
                        warpgroup::store(temp_smem[warpgroupid],
                                         cs_cs_fl_reg); // temp_smem --> Attn2

                        // Compute Z2_bar
                        warpgroup::mm_AB(cs_cs_fl_reg, z1_bar_smem[idx % K::stages], w2_smem[warpgroupid]);
                        warpgroup::mma_AB(cs_cs_fl_reg, temp_smem[warpgroupid], z2_smem[idx % K::stages]);
                        warpgroup::mma_async_wait();

                        // Store out Z2 Bar
                        warpgroup_store_add(temp_smem[idx % K::stages], cs_cs_fl_reg);
                        if (warpgroup::laneid() == 0)
                                arrive(second_reduction_complete, 1);
                        kittens::wait(second_reduction_complete, idx % 2);
                        if (warpid == 0) {
                                tma::store_add_async(g.o, temp_smem[idx % K::stages], {batch_idx, head_idx, idx, 0});
                                tma::store_commit_group();
                        }

                        // Update hidden states
                        warpgroup::load(cs_cs_fl_reg, w1_smem[warpgroupid]);
                        warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[idx % K::stages], grad_z1_smem[warpgroupid]);
                        warpgroup::mma_async_wait();
                        warpgroup::store(w1_smem[warpgroupid], cs_cs_fl_reg);

                        warpgroup::load(cs_cs_fl_reg, w2_smem[warpgroupid]);
                        warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem[warpgroupid], z2_smem[idx % K::stages]);
                        warpgroup::mma_async_wait();
                        warpgroup::store(w2_smem[warpgroupid], cs_cs_fl_reg);

                        if (warpgroup::laneid() == 0)
                                arrive(compute_done[idx % K::stages], 1);
                }
        }
}

// Modified ttt_mlp_forward function
#ifdef TK_COMPILE_TTT_MLP_FORWARD_TP
#include "common/pyutils/torch_helpers.cuh"
void ttt_mlp_forward_tp(
    // const torch::Tensor ttt_norm_weight,
    // const torch::Tensor ttt_norm_bias,
    const torch::Tensor W1_init,
    // const torch::Tensor b1_init,
    const torch::Tensor W2_init,
    // const torch::Tensor b2_init,
    const torch::Tensor XQ_batch, const torch::Tensor XV_batch, const torch::Tensor XK_batch, torch::Tensor output
    // const torch::Tensor eta_batch
) {
        // Initalize data pointers
        auto *d_q = reinterpret_cast<bf16 *>(XQ_batch.data_ptr<at::BFloat16>());
        auto *d_k = reinterpret_cast<bf16 *>(XV_batch.data_ptr<at::BFloat16>());
        auto *d_v = reinterpret_cast<bf16 *>(XK_batch.data_ptr<at::BFloat16>());
        auto *d_w1 = reinterpret_cast<bf16 *>(W1_init.data_ptr<at::BFloat16>());
        auto *d_w2 = reinterpret_cast<bf16 *>(W2_init.data_ptr<at::BFloat16>());
        auto *d_o = reinterpret_cast<bf16 *>(output.data_ptr<at::BFloat16>());

        constexpr int BATCH_SIZE = 1;
        constexpr int HEADS = 1;
        constexpr int TP = 4;

        constexpr int SEQ_LEN = 64;
        constexpr int HEAD_DIM = 64;
        constexpr int EXP_DIM = 256;
        constexpr int BLOCK_SIZE = (NUM_WORKERS * 32); // Number of threads in a block

        using globals = fwd_globals<HEAD_DIM>;

        globals::q_gl qg_arg{d_q, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};
        globals::k_gl kg_arg{d_k, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};
        globals::v_gl vg_arg{d_v, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};
        globals::o_gl og_arg{d_o, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};

        globals::w1_gl w1g_arg{d_w1, BATCH_SIZE, HEADS, HEAD_DIM, EXP_DIM};
        globals::w2_gl w2g_arg{d_w2, BATCH_SIZE, HEADS, EXP_DIM, HEAD_DIM};

        globals g{qg_arg, kg_arg, vg_arg, w1g_arg, w2g_arg, og_arg, SEQ_LEN};

        // Set shared memory to use max dynamic
        unsigned long mem_size = kittens::MAX_SHARED_MEMORY; // need to launch two blocks if possible.

        cudaFuncSetAttribute(fwd_attend_ker<HEAD_DIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        dim3 grid(TP, BATCH_SIZE, HEADS);

        cudaDeviceSynchronize();
        fwd_attend_ker<HEAD_DIM><<<grid, BLOCK_SIZE, mem_size>>>(g);
        cudaDeviceSynchronize();
}
#endif

#include "harness.impl"
