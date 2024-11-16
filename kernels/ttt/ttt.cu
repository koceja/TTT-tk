#include "kittens.cuh"
#include "cooperative_groups.h"
#include <iostream>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_MLP_FORWARD_TP
#endif

#define CUDA_ASSERT(cond, tidx) if (!(cond)) { if (tidx == -1 || threadIdx.x == tidx && tp_idx == 0) printf("Kernel assert failed: %s\n", #cond); return; }

constexpr int CONSUMER_WARPGROUPS = (1); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 
constexpr int TP                  = (4);

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int tile_height = (64);
};

template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using w1_tile   =         st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using w2_tile   =         st_bf<fwd_attend_ker_tile_dims<D>::tile_height, fwd_attend_ker_tile_dims<D>::tile_width>;

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

template<int TP, ducks::st::all ST>
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

        for(int stage = 0; stage < 2; stage++) {
            if (warpgroup::warpid() == 0) {
                tma::cluster::store_async(tile_other, tile, tp ^ (1 << stage), dsmem_semaphore[stage]);
                kittens::wait(dsmem_semaphore[stage], 0);
            }
            warpgroup::sync(1);
            warpgroup::add(tile, tile, tile_other);
        }
    }
}

template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
__cluster_dims__(4)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::tile_height, K::tile_width>;
    using k_tile    =         st_bf<K::tile_height, K::tile_width>;
    using v_tile    =         st_bf<K::tile_height, K::tile_width>;

    using z1_tile   =         st_bf<K::tile_height, K::tile_width>;
    using z2_tile   =         st_bf<K::tile_height, K::tile_width>;
    using grad_z1_tile   =         st_bf<K::tile_height, K::tile_width>;
    using rd_buffer_tile   =         st_bf<K::tile_height, K::tile_width>;

    using w1_tile   =         st_bf<K::tile_height, K::tile_width>;
    using w2_tile   =         st_bf<K::tile_height, K::tile_width>;

    w1_tile    (&w1_smem)                   = al.allocate<w1_tile>();
    w2_tile    (&w2_smem)                   = al.allocate<w2_tile>();

    q_tile    (&q_smem)                    = al.allocate<q_tile>();
    k_tile    (&k_smem)                    = al.allocate<k_tile>();
    v_tile    (&v_smem)                    = al.allocate<v_tile>();

    z1_tile    (&z1_smem)                   = al.allocate<z1_tile>();
    z2_tile    (&z2_smem)                   = al.allocate<z2_tile>();
    grad_z1_tile    (&grad_z1_smem)         = al.allocate<grad_z1_tile>();
    rd_buffer_tile    (&rd_buffer_smem)      = al.allocate<rd_buffer_tile>();

    int batch_idx   = blockIdx.y;
    int head_idx    = blockIdx.z;
    int n_minibatch   = g.N / (K::tile_height);

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    int tp_idx = cluster.block_rank();
    CUDA_ASSERT(tp_idx == blockIdx.x, 0);

    __shared__ kittens::semaphore w1_smem_arrived, w2_smem_arrived, q_sem_arrived, k_sem_arrived, v_sem_arrived, reduction_done, compute_done;
    if (threadIdx.x == 0) { 
        init_semaphore(w1_smem_arrived, 0, 1);
        init_semaphore(w2_smem_arrived, 0, 1);
        init_semaphore(q_sem_arrived, 0, 1); 
        init_semaphore(k_sem_arrived, 0, 1); 
        init_semaphore(v_sem_arrived, 0, 1); 
        init_semaphore(reduction_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(compute_done, CONSUMER_WARPGROUPS, 0);

        tma::expect_bytes(w1_smem_arrived, sizeof(w1_tile));
        tma::load_async(w1_smem, g.w1, {batch_idx, head_idx, 0, tp_idx}, w1_smem_arrived);

        tma::expect_bytes(w2_smem_arrived, sizeof(w2_tile));
        tma::load_async(w2_smem, g.w2, {batch_idx, head_idx, tp_idx, 0}, w2_smem_arrived);

        int4 tile_idx = {batch_idx, head_idx, 0, 0};

        tma::expect_bytes(q_sem_arrived, sizeof(q_tile));
        tma::load_async(q_smem, g.q, tile_idx, q_sem_arrived);

        tma::expect_bytes(k_sem_arrived, sizeof(k_tile));
        tma::load_async(k_smem, g.k, tile_idx, k_sem_arrived);

        tma::expect_bytes(v_sem_arrived, sizeof(v_tile));
        tma::load_async(v_smem, g.v, tile_idx, v_sem_arrived);
    }
    __syncthreads(); 

    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();    
        tma::cluster::arrive_aligned();
        
        int iters; 
        iters = n_minibatch - 1;

        kittens::wait(reduction_done, 0);
        if (warpid == NUM_WORKERS-4) {
            for (auto idx = 0; idx < iters; idx++) {
                kittens::wait(reduction_done, idx % 2);
                kittens::wait(compute_done, idx % 2);
                
                int4 tile_idx = {batch_idx, head_idx, idx + 1, 0};

                tma::expect_bytes(q_sem_arrived, sizeof(q_tile));
                tma::load_async(q_smem, g.q, tile_idx, q_sem_arrived);

                tma::expect_bytes(k_sem_arrived, sizeof(k_tile));
                tma::load_async(k_smem, g.k, tile_idx, k_sem_arrived);

                tma::expect_bytes(v_sem_arrived, sizeof(v_tile));
                tma::load_async(v_smem, g.v, tile_idx, v_sem_arrived);

                // warpgroup::sync(NUM_WARPGROUPS-1); TODO: Is this needed?
                tma::cluster::arrive_aligned();
            }
        }
    }
    else {
        warpgroup::increase_registers<184>();

        rt_fl<16, K::tile_height> cs_cs_fl_reg;
        rt_fl<16, K::tile_height> cs_cs_2_fl_reg;
        rt_bf<16, K::tile_height> cs_cs_bf_reg;

        kittens::wait(w1_smem_arrived, 0);
        kittens::wait(w2_smem_arrived, 0);
        
        for (auto idx = 0; idx < n_minibatch; idx++) {
            // Hidden State Forward
            kittens::wait(k_sem_arrived, idx % 2);

            warpgroup::mm_AB(cs_cs_fl_reg, k_smem, w1_smem);
            warpgroup::mma_async_wait();
            warpgroup::store(z1_smem, cs_cs_fl_reg);

            warpgroup::mm_AB(cs_cs_fl_reg, z1_smem, w2_smem);
            warpgroup::mma_async_wait();
            warpgroup::store(z2_smem, cs_cs_fl_reg);

            // Reduction over SM
            square_all_reduce<TP>(z2_smem, rd_buffer_smem, tp_idx);
            if (warpgroup::laneid() == 0) arrive(reduction_done, 1);

            // Calculate (negative) grad_l_wrt_Z2 / grad_l_wrt_Z1
            // We use negative gradients to use the WGMMA accumulator
            kittens::wait(v_sem_arrived, idx % 2);
            warpgroup::sub(z2_smem, z2_smem, v_smem); // grad_l_wrt_Z2 is stored into z2_smem
            warpgroup::mm_ABt(cs_cs_fl_reg, z2_smem, w2_smem);
            warpgroup::mma_async_wait();
            warpgroup::store(grad_z1_smem, cs_cs_fl_reg);

            // Compute Attn1 and Z1_bar partial (on registers)
            kittens::wait(q_sem_arrived, idx % 2);
            warpgroup::mm_ABt(cs_cs_fl_reg, q_smem, k_smem);
            warpgroup::mm_AB(cs_cs_2_fl_reg, q_smem, w1_smem); // Z1_bar partial
            
            // Compute Z1_bar using Z1_bar partial (on registers)
            copy(cs_cs_bf_reg, cs_cs_fl_reg);
            make_causal(cs_cs_bf_reg, cs_cs_bf_reg, base_types::constants<bf16>::zero());
            warpgroup::mma_AB(cs_cs_2_fl_reg, cs_cs_bf_reg, grad_z1_smem); // Z1_bar
            warpgroup::mma_async_wait();

            // Compute Attn2 and Z2_bar partial (on registers)
            copy(cs_cs_bf_reg, cs_cs_2_fl_reg);
            warpgroup::mm_ABt(cs_cs_fl_reg, cs_cs_bf_reg, z1_smem); // Attn2
            warpgroup::mm_AB(cs_cs_2_fl_reg, cs_cs_bf_reg, w2_smem); // Z2_bar partial
            warpgroup::mma_async_wait();

            // Compute Z2_bar using Z2_bar partial (on registers)
            copy(cs_cs_bf_reg, cs_cs_fl_reg);
            make_causal(cs_cs_bf_reg, cs_cs_bf_reg, base_types::constants<bf16>::zero());
            warpgroup::mma_AB(cs_cs_2_fl_reg, cs_cs_bf_reg, z2_smem); // Z2_bar
            warpgroup::mma_async_wait();

            // Store Z2_bar into global memory
            warpgroup::store(z2_smem, cs_cs_2_fl_reg);
            if (warpgroup::warpid() == 0) {
                tma::store_add_async(g.o, z2_smem, {batch_idx, head_idx, idx, 0});
                tma::store_commit_group();
            }

            // Update hidden states
            warpgroup::load(cs_cs_fl_reg, w1_smem);
            warpgroup::mma_AtB(cs_cs_fl_reg, k_smem, grad_z1_smem);
            warpgroup::mma_async_wait();
            warpgroup::store(w1_smem, cs_cs_fl_reg);

            warpgroup::load(cs_cs_fl_reg, w2_smem);
            warpgroup::mma_AtB(cs_cs_fl_reg, z1_smem, z2_smem);
            warpgroup::mma_async_wait();
            warpgroup::store(w2_smem, cs_cs_fl_reg);

            if (warpgroup::laneid() == 0) arrive(compute_done, 1);
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
    const torch::Tensor XQ_batch,
    const torch::Tensor XV_batch,
    const torch::Tensor XK_batch,
    torch::Tensor output
    // const torch::Tensor eta_batch
)
{
    // Initalize data pointers
    auto *d_q = reinterpret_cast<bf16*>(XQ_batch.data_ptr<at::BFloat16>());
    auto *d_k = reinterpret_cast<bf16*>(XV_batch.data_ptr<at::BFloat16>());
    auto *d_v = reinterpret_cast<bf16*>(XK_batch.data_ptr<at::BFloat16>());
    auto *d_w1 = reinterpret_cast<bf16*>(W1_init.data_ptr<at::BFloat16>());
    auto *d_w2 = reinterpret_cast<bf16*>(W2_init.data_ptr<at::BFloat16>());
    auto *d_o = reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>());

    constexpr int BATCH_SIZE = 1;
    constexpr int HEADS = 1;
    constexpr int TP = 4;

    constexpr int SEQ_LEN = 64; 
    constexpr int HEAD_DIM = 64; 
    constexpr int EXP_DIM = 256;
    constexpr int BLOCK_SIZE = (NUM_WORKERS*32); // Number of threads in a block

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
    
    cudaFuncSetAttribute(
        fwd_attend_ker<HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    dim3 grid(TP, BATCH_SIZE, HEADS);

    cudaDeviceSynchronize();
    fwd_attend_ker<HEAD_DIM><<<grid, BLOCK_SIZE, mem_size>>>(g);
    cudaDeviceSynchronize();
}
#endif

#include "harness.impl"
