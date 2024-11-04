
#include <stdio.h>
#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include "kittens.cuh"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_MLP_FORWARD
#endif

const constexpr int T = 8192;

const constexpr int B = 4;
const constexpr int NH = 32;
const constexpr int CS = 16;
const constexpr int NC = T / CS;
const constexpr int F = 128;
const constexpr int F_mult = 4;




// template <typename H, typename T>
// __global__ void ttt_mlp_forward_kernel(
//     const int NH, const int n_mini_batch, const int mini_batch_size, const int HF, const int HF_prime,
//     T* __W1,
//     T* __b1,
//     T* __W2,
//     T* __b2,
//     const T* __ln_weight,
//     const T* __ln_bias,
//     // const T* __make_last_b_matrix,
//     // const T* __make_last_eta_1_matrix, const T* __make_last_eta_2_matrix,
//     const T* __XV, const T* __XK, const T* __XQ, const T* __Eta
//     // T* __Output
// )
// {
//     H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF * HF_prime);
//     H *_W2       = reinterpret_cast<H*>(__W2) + blockIdx.x * (HF_prime * HF);
//     H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (mini_batch_size * HF_prime);
//     H *_b2       = reinterpret_cast<H*>(__b2) + blockIdx.x * (mini_batch_size * HF);

//     const H *_ln_weight = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * (mini_batch_size * HF);
//     const H *_ln_bias   = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * (mini_batch_size * HF);

//     // const H *_make_last_b_matrix         = reinterpret_cast<const H*>(__make_last_b_matrix);
//     // const H *_make_last_eta_1_matrix   = reinterpret_cast<const H*>(__make_last_eta_1_matrix);
//     // const H *_make_last_eta_2_matrix   = reinterpret_cast<const H*>(__make_last_eta_2_matrix);

//     const H *_XV   = reinterpret_cast<const H*>(__XV) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);
//     const H *_XK   = reinterpret_cast<const H*>(__XK) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);
//     const H *_XQ   = reinterpret_cast<const H*>(__XQ) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);
//     const H *_Eta  = reinterpret_cast<const H*>(__Eta) + blockIdx.x * (n_mini_batch * mini_batch_size * mini_batch_size);
//     // H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);

//     // This is the CUDA shared memory
//     extern __shared__ alignment_dummy __shm[];
//     shared_allocator al((int*)&__shm[0]);

//     st_hf<1, 4, ducks::st_layout::swizzle> (&XV_smem)[2] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, 2>();
//     st_hf<1, 4, ducks::st_layout::swizzle> (&XK_smem)[2] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, 2>();
//     st_hf<1, 4, ducks::st_layout::swizzle> (&XQ_smem)[2] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, 2>();
//     st_hf<1, 1, ducks::st_layout::swizzle> (&Eta_smem)[2] = al.allocate<st_hf<1, 1, ducks::st_layout::swizzle>, 2>();

//     rt_hf<4, 16, kittens::ducks::rt_layout::col> W1_col_reg;
//     rt_hf<16, 4, kittens::ducks::rt_layout::col> W2_col_reg;
//     rt_hf<1, 16> b1_reg;
//     rt_hf<1, 4> b2_reg;

//     load(W1_col_reg, _W1, W1_col_reg.cols);
//     load(W2_col_reg, _W2, W2_col_reg.cols);

//     load(b1_reg, _b1, b1_reg.cols);
//     load(b2_reg, _b2, b2_reg.cols);

//     rt_hf<1, 4> ln_w_reg;
//     rt_hf<1, 4> ln_b_reg;
//     load(ln_w_reg, _ln_weight, ln_w_reg.cols);
//     load(ln_b_reg, _ln_bias, ln_b_reg.cols);

    // rt_hf<1, 1> make_last_b_matrix_bf;
    // rt_hf<1, 4, kittens::ducks::rt_layout::col> make_last_eta_1_matrix_col;
    // rt_hf<1, 16, kittens::ducks::rt_layout::col> make_last_eta_2_matrix_col;
    // make_last_b_matrix: broadcast last row of b_bar
    // load(make_last_b_matrix_bf, _make_last_b_matrix, make_last_b_matrix_bf.cols);
    // // make_last_eta_1_matrix_col: broadcast last col of eta_transposed for multiplying X1: [bs,HF_prime]
    // load(make_last_eta_1_matrix_col, _make_last_eta_1_matrix, make_last_eta_1_matrix_col.cols);
    // // make_last_eta_2_matrix_col: broadcast last col of eta_transposed for multiplying X2: [bs,HF]
    // load(make_last_eta_2_matrix_col, _make_last_eta_2_matrix, make_last_eta_2_matrix_col.cols);

    // // 2-stage pipeline
    // int tic = 0, toc = 1;
    // auto block = cooperative_groups::this_thread_block();
    // __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> qkve_barrier;
    // if (threadIdx.x == 0) {init(&qkve_barrier, block.size());}
    // block.sync();

    // load_async(XV_smem[tic],  _XV  , 64,  qkve_barrier);
    // load_async(XK_smem[tic],  _XK  , 64,  qkve_barrier);
    // load_async(XQ_smem[tic],  _XQ  , 64,  qkve_barrier);
    // load_async(Eta_smem[tic], _Eta , 16,  qkve_barrier);

    // int tgt_offset;

    // for (int i = 0; i < n_mini_batch; i++) {

    //     qkve_barrier.arrive_and_wait();

    //     // Z1 = XK @ W1 + b1
    //     rt_hf<1, 4> XK_reg;
    //     load(XK_reg, XK_smem[tic]);

    //     rt_hf<1, 16> Z1_reg;
    //     mma_AB(Z1_reg, XK_reg, W1_col_reg, b1_reg);

    //     // X2 = gelu(Z1)
    //     rt_hf<1, 16> X2_reg;
    //     gelu(X2_reg, Z1_reg);

    //     // Z2 = X2 @ W2 + b2
    //     rt_hf<1, 4> Z2_reg;
    //     mma_AB(Z2_reg, X2_reg, W2_col_reg, b2_reg);

    //     // l2_tgt = XV - XK
    //     rt_hf<1, 4> l2_target_reg;
    //     load(l2_target_reg, XV_smem[tic]);
    //     sub(l2_target_reg, l2_target_reg, XK_reg);

    //     rt_hf<1, 4> dl_dZ2_reg;
    //     ln_fused_l2_bwd_fp16(HF, Z2_reg, l2_target_reg, ln_w_reg, ln_b_reg, dl_dZ2_reg);

    //     tgt_offset = i + 1;
    //     if (tgt_offset < n_mini_batch) {
    //         load_async(XV_smem[toc],  _XV  + tgt_offset * X_STRIDE,   64,  qkve_barrier);
    //         load_async(XK_smem[toc],  _XK  + tgt_offset * X_STRIDE,   64,  qkve_barrier);
    //         load_async(XQ_smem[toc],  _XQ  + tgt_offset * X_STRIDE,   64,  qkve_barrier);
    //         load_async(Eta_smem[toc], _Eta + tgt_offset * Eta_STRIDE, 16,  qkve_barrier);
    //     }

    //     // eta: [bs,bs], each row corresp to eta for 1 token in mini-batch
    //     // eta_transpose: [bs,bs], each col corresp to eta for 1 token in mini-batch (the last col corresp to last token's)
    //     rt_hf<1, 1> eta_reg;
    //     rt_hf<1, 1> eta_transpose_reg;
    //     load(eta_reg, Eta_smem[tic]);
    //     transpose_sep(eta_transpose_reg, eta_reg);

    //     // eta_last_X2 = (eta_transpose @ [0...0|1].t) * X2
    //     rt_hf<1, 16> eta_last_X2_reg;
    //     zero(eta_last_X2_reg);
    //     mma_AB(eta_last_X2_reg, eta_transpose_reg, make_last_eta_2_matrix_col, eta_last_X2_reg);
    //     mul(eta_last_X2_reg, X2_reg, eta_last_X2_reg);
    //     rt_hf<1, 16, ducks::rt_layout::col> &eta_last_X2_col_reg = swap_layout_inplace(eta_last_X2_reg);

    //     // delta W2 = eta_last_X2.transpose(-1,-2) @ dl_dZ2
    //     rt_hf<16, 4> delta_W2_reg;
    //     rt_hf<1, 4, ducks::rt_layout::col> dl_dZ2_col_reg;
    //     swap_layout(dl_dZ2_col_reg, dl_dZ2_reg);
    //     zero(delta_W2_reg);
    //     mma_AtB(delta_W2_reg, eta_last_X2_col_reg, dl_dZ2_col_reg, delta_W2_reg);
    //     rt_hf<16, 4, ducks::rt_layout::col> &delta_W2_col_reg = swap_layout_inplace(delta_W2_reg);

    //     // dl_dX2 = dl_dZ2 @ W2.transpose(-1,-2)
    //     rt_hf<1, 16> dl_dZ1_reg;
    //     zero(dl_dZ1_reg);
    //     rt_hf<16, 4, kittens::ducks::rt_layout::row> W2_reg;
    //     swap_layout(W2_reg, W2_col_reg);
    //     mma_ABt(dl_dZ1_reg, dl_dZ2_reg, W2_reg, dl_dZ1_reg);

    //     // dl_dZ1 = dl_dX2 * diff_gelu(Z1)
    //     rt_hf<1, 16> &diff_gelu_Z1_reg = Z1_reg;
    //     diff_gelu(diff_gelu_Z1_reg, Z1_reg);
    //     mul(dl_dZ1_reg, dl_dZ1_reg, diff_gelu_Z1_reg);

    //     // delta b1 = (eta_chunk * Attn_b) @ dl_dZ1
    //     rt_hf<1, 16> delta_b1_reg;
    //     rt_hf<1, 1> Attn_reg;
    //     rt_hf<1, 16, ducks::rt_layout::col> &dl_dZ1_col_reg = swap_layout_inplace(dl_dZ1_reg);
    //     zero(delta_b1_reg);
    //     // mul(Attn_reg, eta_reg, cumsum_matrix_bf);
    //     make_causal(eta_reg, eta_reg, base_types::constants<half>::zero());
    //     mma_AB(delta_b1_reg, eta_reg, dl_dZ1_col_reg, delta_b1_reg);
    //     // b1_bar = b1 - delta_b1
    //     sub(b1_reg, b1_reg, delta_b1_reg);

    //     // delta b2 = (eta_chunk * Attn_b) @ dl_dZ2
    //     rt_hf<1, 4> delta_b2_reg;
    //     zero(delta_b2_reg);
    //     mma_AB(delta_b2_reg, eta_reg, dl_dZ2_col_reg, delta_b2_reg);
    //     // b2_bar = b2 - delta_b2
    //     sub(b2_reg, b2_reg, delta_b2_reg);

    //     // eta_last_X1 = (eta_transpose @ [0...0|1].t) * X1
    //     rt_hf<1, 4> eta_last_X1_reg;
    //     zero(eta_last_X1_reg);
    //     mma_AB(eta_last_X1_reg, eta_transpose_reg, make_last_eta_1_matrix_col, eta_last_X1_reg);
    //     mul(eta_last_X1_reg, XK_reg, eta_last_X1_reg);
    //     rt_hf<1, 4, ducks::rt_layout::col> &eta_last_X1_col_reg = swap_layout_inplace(eta_last_X1_reg);

    //     // delta W1 = eta_last_X1.transpose(-1,-2) @ dl_dZ1
    //     rt_hf<4, 16> delta_W1_reg;
    //     zero(delta_W1_reg);
    //     mma_AtB(delta_W1_reg, eta_last_X1_col_reg, dl_dZ1_col_reg, delta_W1_reg);
    //     rt_hf<4, 16, ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);

    //     // Attn1 = eta * Tril(XQ @ XK.t)
    //     rt_hf<1, 4> XQ_reg;
    //     load(XQ_reg, XQ_smem[tic]);
    //     zero(Attn_reg);
    //     mma_ABt(Attn_reg, XQ_reg, XK_reg, Attn_reg);
    //     make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
    //     mul(Attn_reg, eta_reg, Attn_reg);

    //     // Z1_bar = XQ @ W1 - Attn1 @ dl_dZ1 + b1_bar
    //     rt_hf<1, 16> Z1_bar_term_1_reg;
    //     mma_AB(Z1_bar_term_1_reg, XQ_reg, W1_col_reg, b1_reg);

    //     // Update W1 = W1 - delta_W1 once the old W1 is no longer needed
    //     sub(W1_col_reg, W1_col_reg, delta_W1_col_reg);

    //     // Update b1
    //     rt_hf<1, 16, kittens::ducks::rt_layout::col> b1_bar_col_reg;
    //     swap_layout(b1_bar_col_reg, b1_reg);
    //     zero(b1_reg);
    //     mma_AB(b1_reg, make_last_b_matrix_bf, b1_bar_col_reg, b1_reg);

    //     rt_hf<1, 16> Z1_bar_term_2_reg;
    //     zero(Z1_bar_term_2_reg);
    //     mma_AB(Z1_bar_term_2_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_reg);

    //     sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

    //     // X2_bar = gelu(Z1_bar)
    //     rt_hf<1, 16> &X2_bar_reg = Z1_bar_term_1_reg;
    //     gelu(X2_bar_reg, Z1_bar_term_1_reg);

    //     // Attn2 = eta * Tril(X2_bar @ X2.t)
    //     zero(Attn_reg);
    //     mma_ABt(Attn_reg, X2_bar_reg, X2_reg, Attn_reg);
    //     make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
    //     mul(Attn_reg, eta_reg, Attn_reg);

    //     // Z2_bar = X2_bar @ W2 - Attn2 @ dl_dZ2 + b2_bar
    //     rt_hf<1, 4> Z2_bar_term_1_reg;
    //     mma_AB(Z2_bar_term_1_reg, X2_bar_reg, W2_col_reg, b2_reg);

    //     // Updated W2
    //     sub(W2_col_reg, W2_col_reg, delta_W2_col_reg);

    //     // Update b2
    //     rt_hf<1, 4, kittens::ducks::rt_layout::col> b2_bar_col_reg;
    //     swap_layout(b2_bar_col_reg, b2_reg);
    //     zero(b2_reg);
    //     mma_AB(b2_reg, make_last_b_matrix_bf, b2_bar_col_reg, b2_reg);

    //     rt_hf<1, 4> Z2_bar_term_2_reg;
    //     zero(Z2_bar_term_2_reg);
    //     mma_AB(Z2_bar_term_2_reg, Attn_reg, dl_dZ2_col_reg, Z2_bar_term_2_reg);

    //     sub(Z2_bar_term_1_reg, Z2_bar_term_1_reg, Z2_bar_term_2_reg);

    //     rt_hf<1, 4> &Z2_bar_reg = Z2_bar_term_1_reg;
    //     rt_hf<1, 4> LN_out_bar_reg;
    //     LN_fwd_fp16(HF, Z2_bar_reg, ln_w_reg, ln_b_reg, LN_out_bar_reg);

    //     // Output = XQ + LN(Z2_bar)
    //     add(LN_out_bar_reg, LN_out_bar_reg, XQ_reg);

    //     // Store Output
    //     store(_Output + i * X_STRIDE, LN_out_bar_reg, LN_out_bar_reg.cols); // Make async

    //     tic ^= 1;
    //     toc ^= 1;

    // }

//     store(_W1, W1_col_reg, W1_col_reg.cols);
//     store(_W2, W2_col_reg, W2_col_reg.cols);
//     store(_b1, b1_reg, b1_reg.cols);
//     store(_b2, b2_reg, b2_reg.cols);

    


// }

using namespace kittens;

template <int B, int NH, int NC, int CS, int F>
__global__ void ttt_mlp_forward_kernel(
    float* W1_init_ptr,
    float* b1_init_ptr,
    float* W2_init_ptr,
    float* b2_init_ptr,
    float* XQ_batch_ptr,
    float* XV_batch_ptr,
    float* XK_batch_ptr,
    float* eta_batch_ptr,
    float* ttt_norm_weight_ptr,
    float* ttt_norm_bias_ptr
)
{
    constexpr auto F_prime = F * F_mult;

    

    // const gl<float, B, NH, F, F_prime> W1_init_gl{W1_init_ptr, nullptr, nullptr, nullptr, nullptr};
    // const gl<float, 16, 16, 16, 16> W2_init_gl{W2_init_ptr, nullptr, nullptr, nullptr, nullptr};

    const auto* W1_init_offset = W1_init_ptr + blockIdx.x * (F * F_prime);
    const auto* b1_init_offset = b1_init_ptr + blockIdx.x * (CS * F_prime);
    const auto* W2_init_offset = W2_init_ptr + blockIdx.x * (F_prime * F);
    const auto* b2_init_offset = b2_init_ptr + blockIdx.x * (CS * F);

    const auto* ln_weight_offset = ttt_norm_weight_ptr + blockIdx.x % NH * CS * F_prime;
    const auto* ln_bias_offset   = ttt_norm_bias_ptr + (blockIdx.x % NH) * (CS * F);

    // const H *_make_last_b_matrix         = reinterpret_cast<const H*>(__make_last_b_matrix);
    // const H *_make_last_eta_1_matrix   = reinterpret_cast<const H*>(__make_last_eta_1_matrix);
    // const H *_make_last_eta_2_matrix   = reinterpret_cast<const H*>(__make_last_eta_2_matrix);

    const auto* XV_batch_offset = XV_batch_ptr + blockIdx.x * (NC * CS * F);
    const auto* XK_batch_offset = XK_batch_ptr + blockIdx.x * (NC * CS * F);
    const auto* XQ_batch_offset = XQ_batch_ptr + blockIdx.x * (NC * CS * F);
    const auto* eta_batch_offset  = eta_batch_ptr + blockIdx.x * (NC * CS * CS);
    // H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);

    kittens::rt_fl<64, 64*4, kittens::ducks::rt_layout::col> W1_init;
    kittens::rt_fl<64*4, 64, kittens::ducks::rt_layout::col> W2_init;

    const kittens::coord idx1{W1_init.cols};
    const kittens::coord idx2{W2_init.cols};

    printf("Hello from the device! Thread ID: %d\n", threadIdx.x);


    // while (true)
    // {
    //     kittens::load(W1_init, W1_init_gl, idx1);
    //     kittens::load(W2_init, W2_init_gl, idx2);

    //     add(W1_init, W1_init, W1_init);
    //     add(W2_init, W2_init, W2_init);


    //     kittens::store(W1_init_gl, W1_init, idx1);
    //     kittens::store(W2_init_gl, W2_init, idx2);
    // }
}


struct MlpForwardInputs
{
    // Member tensors
    const torch::Tensor ttt_norm_weight;
    const torch::Tensor ttt_norm_bias;
    const torch::Tensor W1_init;
    const torch::Tensor b1_init;
    const torch::Tensor W2_init;
    const torch::Tensor b2_init;
    const torch::Tensor XQ_batch;
    const torch::Tensor XV_batch;
    const torch::Tensor XK_batch;
    const torch::Tensor eta_batch;

    // Constructor
    MlpForwardInputs(
        int B,
        int NH,
        int NC,
        int CS,
        int F
    )
    : ttt_norm_weight(torch::randn({NH, F}, torch::kCUDA)),
      ttt_norm_bias(torch::randn({NH, F}, torch::kCUDA)),
      W1_init(torch::randn({B, NH, F, F * 4}, torch::kCUDA)),
      b1_init(torch::randn({B, NH, 1, F * 4}, torch::kCUDA)),
      W2_init(torch::randn({B, NH, F * 4, F}, torch::kCUDA)),
      b2_init(torch::randn({B, NH, 1, F}, torch::kCUDA)),
      XQ_batch(torch::randn({B, NH, NC, CS, F}, torch::kCUDA)),
      XV_batch(torch::randn({B, NH, NC, CS, F}, torch::kCUDA)),
      XK_batch(torch::randn({B, NH, NC, CS, F}, torch::kCUDA)),
      eta_batch(torch::randn({B, NH, NC, CS, CS}, torch::kCUDA))
    {
        // nop
    }
    
};

template <typename... Args>
MlpForwardInputs get_mlp_forward_inputs(Args&&... args) {
    torch::manual_seed(0);
    return MlpForwardInputs(std::forward<Args>(args)...);
}

void run_benchmark()
{
    auto inputs = get_mlp_forward_inputs(B, NH, NC, CS, F);

    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;


    const auto run_kernel = [&]()
    {
        auto* W1_init_ptr = inputs.W1_init.data_ptr<float>();
        auto* b1_init_ptr = inputs.b1_init.data_ptr<float>();
        auto* W2_init_ptr = inputs.W2_init.data_ptr<float>();
        auto* b2_init_ptr = inputs.b2_init.data_ptr<float>();
        auto* XQ_batch_ptr = inputs.XQ_batch.data_ptr<float>();
        auto* XV_batch_ptr = inputs.XV_batch.data_ptr<float>();
        auto* XK_batch_ptr = inputs.XK_batch.data_ptr<float>();
        auto* eta_batch_ptr = inputs.eta_batch.data_ptr<float>();
        auto* ttt_norm_weight_ptr = inputs.ttt_norm_weight.data_ptr<float>();
        auto* ttt_norm_bias_ptr = inputs.ttt_norm_bias.data_ptr<float>();


        ttt_mlp_forward_kernel<B, NH, NC, CS, F><<<B * NH, threads>>>(
            W1_init_ptr,
            b1_init_ptr,
            W2_init_ptr,
            b2_init_ptr,
            XQ_batch_ptr,
            XV_batch_ptr,
            XK_batch_ptr,
            eta_batch_ptr,
            ttt_norm_weight_ptr,
            ttt_norm_bias_ptr
            // NH, NC, CS, F, F*4
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            std::cout << "Success" << std::endl;
        }
        else
        {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        }
    };
    

    // Warmup kernel
    for(int i = 0; i < 5; i++)
    {
        run_kernel();
    }

    // Start timing
    // cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 10;
    for (int i = 0; i < ITERS; i++)
    {
        run_kernel();
    }

    // cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    std::cout << "Kernel avg execution time: " << useconds << " microseconds.";
    cudaCheckErrors("cuda error!");
}

// Modified ttt_mlp_forward function
#ifdef TK_COMPILE_TTT_MLP_FORWARD
#include "common/pyutils/torch_helpers.cuh"
torch::Tensor ttt_mlp_forward(
    const torch::Tensor ttt_norm_weight,
    const torch::Tensor ttt_norm_bias,
    const torch::Tensor W1_init,
    const torch::Tensor b1_init,
    const torch::Tensor W2_init,
    const torch::Tensor b2_init,
    const torch::Tensor XQ_batch,
    const torch::Tensor XV_batch,
    const torch::Tensor XK_batch,
    const torch::Tensor eta_batch,
)
{
    do_nothing_kernel<<<1, 1>>>();

    return torch::ones({2, 3}, torch::TensorOptions().device(torch::kCUDA));
}
#endif



int main()
{
    

    run_benchmark();
}