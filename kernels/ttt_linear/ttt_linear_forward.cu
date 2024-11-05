
#include <stdio.h>
#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>


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
#define TK_COMPILE_TTT_LINEAR_FORWARD
#endif

using namespace kittens;


const constexpr int T = 8192;

const constexpr int B = 4;
const constexpr int NH = 32;
const constexpr int CS = 16;
const constexpr int NC = T / CS;
// const constexpr int F = 128;
const constexpr int F = 64;

const constexpr int F_mult = 4;



__device__ static inline void LN_fwd_fp16(
        const int HF,
        rt_fl<16, F> &Z1_reg,
        rt_fl<16, F> &ln_w_reg,
        rt_fl<16, F> &ln_b_reg,
        rt_fl<16, F> &LN_out_reg
){

    rt_fl<16, F>::col_vec Z1_mean_reg;
    row_sum(Z1_mean_reg, Z1_reg);
    div(Z1_mean_reg, Z1_mean_reg, float(HF));

    rt_fl<16, F> Z1_square_reg;
    sub_row(Z1_square_reg, Z1_reg, Z1_mean_reg);
    mul(Z1_square_reg, Z1_square_reg, Z1_square_reg);

    rt_fl<16, F>::col_vec Z1_std_reg;
    row_sum(Z1_std_reg, Z1_square_reg);
    div(Z1_std_reg, Z1_std_reg, float(HF));
    add(Z1_std_reg, Z1_std_reg, 1e-6f);
    sqrt(Z1_std_reg, Z1_std_reg);

    // Z1_hat = (Z - mu) / std
    rt_fl<16, F> Z1_hat;
    sub_row(Z1_hat, Z1_reg, Z1_mean_reg);
    div_row(Z1_hat, Z1_hat, Z1_std_reg);

    // LN_out = ln_w * Z1_hat + ln_b
    mul(LN_out_reg, Z1_hat, ln_w_reg);
    add(LN_out_reg, LN_out_reg, ln_b_reg);

}

__device__ static inline void ln_fused_l2_bwd_fp16(
        const int HF,
        rt_fl<16, F> &Z1_reg,
        rt_fl<16, F> &l2_target_reg,
        rt_fl<16, F> &ln_w_reg,
        rt_fl<16, F> &ln_b_reg,
        rt_fl<16, F> &dl_dZ1
){
    rt_fl<16, F>::col_vec Z1_mean_reg;
    row_sum(Z1_mean_reg, Z1_reg);
    div(Z1_mean_reg, Z1_mean_reg, float(HF));

    rt_fl<16, F> Z1_square_reg;
    sub_row(Z1_square_reg, Z1_reg, Z1_mean_reg);
    mul(Z1_square_reg, Z1_square_reg, Z1_square_reg);

    rt_fl<16, F>::col_vec Z1_std_reg;
    row_sum(Z1_std_reg, Z1_square_reg);
    div(Z1_std_reg, Z1_std_reg, float(HF));
    add(Z1_std_reg, Z1_std_reg, 1e-6f);
    sqrt(Z1_std_reg, Z1_std_reg);

    // Z1_hat = (Z - mu) / std
    rt_fl<16, F> Z1_hat;
    sub_row(Z1_hat, Z1_reg, Z1_mean_reg);
    div_row(Z1_hat, Z1_hat, Z1_std_reg);

    // LN_out = ln_w * Z1_hat + ln_b
    rt_fl<16, F> LN_out_reg;
    mul(LN_out_reg, Z1_hat, ln_w_reg);
    add(LN_out_reg, LN_out_reg, ln_b_reg);

    // dl_dLN_out = LN_out - l2_target
    // dl_dZ1_hat = dl_dLN_out * ln_weight
    rt_fl<16, F> dl_dZ1_hat;
    sub(dl_dZ1_hat, LN_out_reg, l2_target_reg);
    mul(dl_dZ1_hat, dl_dZ1_hat, ln_w_reg);

    // LN bwd
    // dl_dZ1 = (HF * dl_dZ1_hat -
    //           dl_dZ1_hat.sum(dim=-1, keepdim=True) -
    //           Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    //           ) / (std * HF)

    // HF * dl_dZ1_hat
    mul(dl_dZ1, dl_dZ1_hat, float(HF));

    // HF * dl_dZ1_hat - dl_dZ1_hat.sum(dim=-1, keepdim=True)
    rt_fl<16, F>::col_vec dl_dZ1_vec_term;
    row_sum(dl_dZ1_vec_term, dl_dZ1_hat);
    sub_row(dl_dZ1, dl_dZ1, dl_dZ1_vec_term);

    // Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    rt_fl<16, F> dl_dZ1_term_3;
    mul(dl_dZ1_term_3, dl_dZ1_hat, Z1_hat);
    row_sum(dl_dZ1_vec_term, dl_dZ1_term_3);
    mul_row(dl_dZ1_term_3, Z1_hat, dl_dZ1_vec_term);

    sub(dl_dZ1, dl_dZ1, dl_dZ1_term_3);
    mul(Z1_std_reg, Z1_std_reg, float(HF));
    div_row(dl_dZ1, dl_dZ1, Z1_std_reg);

}



// for (int i = 0; i < n_mini_batch; i++) {

//         // Prefetch a mini-batch into shared memory
//         load(XV_smem[0],  _XV  + i * X_STRIDE,  64);
//         load(XK_smem[0],  _XK  + i * X_STRIDE,  64);
//         load(XQ_smem[0],  _XQ  + i * X_STRIDE,  64);
//         load(Eta_smem[0], _Eta + i * Eta_STRIDE, 16);

//         // Z1 = XK @ W1 + b1
//         rt_fl<1, 4> XK_reg;
//         load(XK_reg, XK_smem[0]);

//         rt_fl<1, 4> Z1_reg;
//         mma_AB(Z1_reg, XK_reg, W1_reg, b1_reg);

//         rt_fl<1, 4> l2_target_reg;
//         load(l2_target_reg, XV_smem[0]);
//         // l2_tgt = XV - XK
//         sub(l2_target_reg, l2_target_reg, XK_reg);

//         rt_fl<1, 4> dl_dZ1;
//         ln_fused_l2_bwd_fp16(HF, Z1_reg, l2_target_reg, ln_w_reg, ln_b_reg, dl_dZ1);

//         // b1_bar = b1 - (eta * Attn_b) @ dl_dZ1
//         rt_fl<1, 4, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);
//         rt_fl<1, 4> delta_b1_reg;
//         zero(delta_b1_reg);
//         rt_fl<1, 1> eta_reg;
//         load(eta_reg, Eta_smem[0]);
//         rt_fl<1, 1> Attn1_reg;
//         make_causal(eta_reg, eta_reg, base_types::constants<half>::zero());
//         mma_AB(delta_b1_reg, eta_reg, dl_dZ1_col, delta_b1_reg);
//         sub(b1_reg, b1_reg, delta_b1_reg);

//         // Z2 = XQ @ W1 - (eta * Attn1) @ dl_dZ1 + b1_bar
//         rt_fl<1, 4> XQ_reg;
//         load(XQ_reg, XQ_smem[0]);

//         zero(Attn1_reg);
//         mma_ABt(Attn1_reg, XQ_reg, XK_reg, Attn1_reg);

//         make_causal(Attn1_reg, Attn1_reg, base_types::constants<half>::zero());
//         mul(Attn1_reg, eta_reg, Attn1_reg);

//         rt_fl<1, 4> Z1_bar_term_1_reg;
//         mma_AB(Z1_bar_term_1_reg, XQ_reg, W1_reg, b1_reg);

//         rt_fl<1, 4> Z1_bar_term_2_reg;
//         zero(Z1_bar_term_2_reg);
//         mma_AB(Z1_bar_term_2_reg, Attn1_reg, dl_dZ1_col, Z1_bar_term_2_reg);

//         sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

//         rt_fl<1, 4> &Z1_bar_reg = Z1_bar_term_1_reg;
//         rt_fl<1, 4> LN_out_bar_reg;
//         LN_fwd_fp16(HF, Z1_bar_reg, ln_w_reg, ln_b_reg, LN_out_bar_reg);

//         // Output = XQ + LN(Z1_bar)
//         add(LN_out_bar_reg, LN_out_bar_reg, XQ_reg);

//         store(_Output + i * mini_batch_size * HF, LN_out_bar_reg, LN_out_bar_reg.cols);

//         // delta_W1 of the last token in the mini-batch
//         // delta_W1 = (eta_mini_batch_last * XK_mini_batch).transpose(-1, -2) @ dl_dZ1
//         rt_fl<1, 4> eta_1_last_reg;
//         zero(eta_1_last_reg);
//         rt_fl<1, 1> &eta_transpose_reg = transpose_inplace(eta_reg);
//         mma_AB(eta_1_last_reg, eta_transpose_reg, make_last_eta_1_matrix_col, eta_1_last_reg);
//         mul(XK_reg, XK_reg, eta_1_last_reg);

//         rt_fl<1, 4, kittens::ducks::rt_layout::col> &XK_col_reg = swap_layout_inplace(XK_reg);
//         rt_fl<4, 4> delta_W1_reg;
//         zero(delta_W1_reg);
//         mma_AtB(delta_W1_reg, XK_col_reg, dl_dZ1_col, delta_W1_reg);

//         // W1_new = W1 - delta_W1
//         rt_fl<4, 4, kittens::ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);
//         sub(W1_reg, W1_reg, delta_W1_col_reg);

//         // delta_b1 = b1_bar[-1]
//         // b1_new = b1 - delta_b1
//         rt_fl<1, 4, kittens::ducks::rt_layout::col> b1_bar_col_reg;
//         swap_layout(b1_bar_col_reg, b1_reg);
//         zero(b1_reg);
//         mma_AB(b1_reg, make_last_b_matrix, b1_bar_col_reg, b1_reg);

//     }

//     store(_W1, W1_reg, W1_reg.cols);
//     store(_b1, b1_reg, b1_reg.cols);

// }

template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void matmul(
    D &d,
    const A &a,
    const B &b,
    const C &c
)
{
    // Need to convert float32 -> bfloat16 for mma
    // static_assert();

    rt_bf<a.rows, a.cols, ducks::rt_layout::row> a_bf;
    rt_bf<b.rows, b.cols, ducks::rt_layout::col> b_bf;

    copy(a_bf, a);
    copy(b_bf, b);

    mma_AB(d, a_bf, b_bf, c);
}

template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void matmul_trans(
    D &d,
    const A &a,
    const B &b,
    const C &c
)
{
    // Need to convert float32 -> bfloat16 for mma
    // static_assert();

    rt_bf<a.rows, a.cols, ducks::rt_layout::row> a_bf;
    rt_bf<b.rows, b.cols, ducks::rt_layout::row> b_bf;

    copy(a_bf, a);
    copy(b_bf, b);

    mma_ABt(d, a_bf, b_bf, c);
}

template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void matmul_AtB(
    D &d,
    const A &a,
    const B &b,
    const C &c
)
{
    // Need to convert float32 -> bfloat16 for mma
    // static_assert();

    rt_bf<a.rows, a.cols, ducks::rt_layout::col> a_bf;
    rt_bf<b.rows, b.cols, ducks::rt_layout::col> b_bf;

    copy(a_bf, a);
    copy(b_bf, b);

    mma_AtB(d, a_bf, b_bf, c);
}


__global__ void ttt_linear_forward_kernel(
    gl<float, 1, B*NH, F, F> W1_init_gl,
    gl<float, 1, B*NH, 1, F> b1_init_gl,
    gl<float, B*NH, NC, CS, F> XQ_batch_gl,
    gl<float, B*NH, NC, CS, F> XV_batch_gl,
    gl<float, B*NH, NC, CS, F> XK_batch_gl,
    gl<float, B*NH, NC, CS, F> eta_batch_gl,
    gl<float, 1, 1, NH, F> ttt_norm_weight_gl,
    gl<float, 1, 1, NH, F> ttt_norm_bias_gl,
    gl<float, 1, 1, CS, CS> make_last_b_matrix_gl,
    gl<float, 1, 1, CS, F> make_last_coeff_1_matrix_gl,
    gl<float, B*NH, NC, CS, F> output_gl
)
{
    // if (blockIdx.x == 10 && threadIdx.x == 1)
    // {
    //     printf("started Kernel\n");
    // }

    const int B_NH = static_cast<int>(blockIdx.x);

    kittens::rt_fl<F, F, kittens::ducks::rt_layout::col> W1_init;
    kittens::rt_fl<16, F> b1_init;

    const kittens::coord idx1{0, B_NH, 0, 0};

    // printf("Hello from the device! Thread ID: %d\n", threadIdx.x);

    kittens::load(W1_init, W1_init_gl, {B_NH, 0, 0});
    kittens::load(b1_init, b1_init_gl, idx1);

    // This is the CUDA shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_fl<16, F> (&XK_smem)[1] = al.allocate<st_fl<16, F>, 1>();
    st_fl<16, F> (&XQ_smem)[1] = al.allocate<st_fl<16, F>, 1>();
    st_fl<16, F> (&XV_smem)[1] = al.allocate<st_fl<16, F>, 1>();
    st_fl<16, 16> (&Eta_smem)[1] = al.allocate<st_fl<16, 16>, 1>();

    rt_fl<16, F> ln_w_reg;
    rt_fl<16, F> ln_b_reg;
    load(ln_w_reg, ttt_norm_weight_gl, {B_NH, 0});
    load(ln_b_reg, ttt_norm_bias_gl, {B_NH, 0});

    rt_fl<16, 16> make_last_b_matrix;
    rt_fl<16, F, kittens::ducks::rt_layout::col> make_last_eta_1_matrix_col;

    

    // make_last_b_matrix: broadcast last row of b_bar
    load(make_last_b_matrix, make_last_b_matrix_gl, {0});
    // make_last_eta_1_matrix_col: broadcast last col of eta_transposed for multiplying X1: [bs,HF]
    load(make_last_eta_1_matrix_col, make_last_coeff_1_matrix_gl, {0});

    // // make_last_b_matrix: broadcast last row of b_bar
    // load(make_last_b_matrix, _make_last_b_matrix, make_last_b_matrix.cols);
    // // make_last_eta_1_matrix_col: broadcast last col of eta_transposed for multiplying X1: [bs,HF]
    // load(make_last_eta_1_matrix_col, _make_last_eta_1_matrix, make_last_eta_1_matrix_col.cols);

    for (int i = 0; i < NC; ++i) {

        // Prefetch a mini-batch into shared memory

        // tma::load_async(

        load(XV_smem[0],  XV_batch_gl, {B_NH, i, 0, 0});
        load(XK_smem[0],  XK_batch_gl, {B_NH, i, 0, 0});
        load(XQ_smem[0],  XQ_batch_gl, {B_NH, i, 0, 0});
        load(Eta_smem[0], eta_batch_gl, {B_NH, i, 0, 0});

        // Z1 = XK @ W1 + b1
        rt_fl<16, F> XK_reg;
        load(XK_reg, XK_smem[0]);

        // // // load(XK_reg, XK_batch_gl, {blockIdx.x, i, 0, 0});



        rt_fl<16, F> Z1_reg;
        matmul(Z1_reg, XK_reg, W1_init, b1_init);

        // // mma_AB(Z1_reg, XK_reg, W1_init, b1_init);

        rt_fl<16, F> l2_target_reg;
        load(l2_target_reg, XV_smem[0]);
        // l2_tgt = XV - XK
        sub(l2_target_reg, l2_target_reg, XK_reg);

        rt_fl<16, F> dl_dZ1;
        ln_fused_l2_bwd_fp16(F, Z1_reg, l2_target_reg, ln_w_reg, ln_b_reg, dl_dZ1);

        // b1_bar = b1 - (eta * Attn_b) @ dl_dZ1
        rt_fl<16, F, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);
        rt_fl<16, F> delta_b1_init;
        zero(delta_b1_init);
        rt_fl<16, 16> eta_reg;
        load(eta_reg, Eta_smem[0]);
        rt_fl<16, 16> Attn1_reg;
        make_causal(eta_reg, eta_reg, base_types::constants<float>::zero());
        // mma_AB(delta_b1_init, eta_reg, dl_dZ1_col, delta_b1_init);
        matmul(delta_b1_init, eta_reg, dl_dZ1_col, delta_b1_init);
        sub(b1_init, b1_init, delta_b1_init);

        // Z2 = XQ @ W1 - (eta * Attn1) @ dl_dZ1 + b1_bar
        rt_fl<16, F> XQ_reg;
        load(XQ_reg, XQ_smem[0]);

        zero(Attn1_reg);
        matmul_trans(Attn1_reg, XQ_reg, XK_reg, Attn1_reg);
        // mma_ABt(Attn1_reg, XQ_reg, XK_reg, Attn1_reg);

        make_causal(Attn1_reg, Attn1_reg, base_types::constants<float>::zero());
        mul(Attn1_reg, eta_reg, Attn1_reg);

        rt_fl<16, F> Z1_bar_term_1_reg;
        matmul(Z1_bar_term_1_reg, XQ_reg, W1_init, b1_init);
        // mma_AB(Z1_bar_term_1_reg, XQ_reg, W1_init, b1_init);

        rt_fl<16, F> Z1_bar_term_2_reg;
        zero(Z1_bar_term_2_reg);
        matmul(Z1_bar_term_2_reg, Attn1_reg, dl_dZ1_col, Z1_bar_term_2_reg);
        // // mma_AB(Z1_bar_term_2_reg, Attn1_reg, dl_dZ1_col, Z1_bar_term_2_reg);

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

        rt_fl<16, F> &Z1_bar_reg = Z1_bar_term_1_reg;
        rt_fl<16, F> LN_out_bar_reg;
        LN_fwd_fp16(F, Z1_bar_reg, ln_w_reg, ln_b_reg, LN_out_bar_reg);

        // Output = XQ + LN(Z1_bar)
        add(LN_out_bar_reg, LN_out_bar_reg, XQ_reg);

        kittens::store(output_gl, LN_out_bar_reg, {B_NH, i, 0, 0});

        // store(_Output + i * mini_batch_size * HF, LN_out_bar_reg, LN_out_bar_reg.cols);

        // delta_W1 of the last token in the mini-batch
        // delta_W1 = (eta_mini_batch_last * XK_mini_batch).transpose(-1, -2) @ dl_dZ1
        rt_fl<16, F> eta_1_last_reg;
        zero(eta_1_last_reg);
        rt_fl<16, 16> &eta_transpose_reg = transpose_inplace(eta_reg);
        matmul(eta_1_last_reg, eta_transpose_reg, make_last_eta_1_matrix_col, eta_1_last_reg);
        // mma_AB(eta_1_last_reg, eta_transpose_reg, make_last_eta_1_matrix_col, eta_1_last_reg);
        mul(XK_reg, XK_reg, eta_1_last_reg);

        rt_fl<16, F, kittens::ducks::rt_layout::col> &XK_col_reg = swap_layout_inplace(XK_reg);
        rt_fl<F, F> delta_W1_init;
        zero(delta_W1_init);
        matmul_AtB(delta_W1_init, XK_col_reg, dl_dZ1_col, delta_W1_init);
        // mma_AtB(delta_W1_init, XK_col_reg, dl_dZ1_col, delta_W1_init);

        // W1_new = W1 - delta_W1
        rt_fl<F, F, kittens::ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_init);
        sub(W1_init, W1_init, delta_W1_col_reg);

        // delta_b1 = b1_bar[-1]
        // b1_new = b1 - delta_b1
        rt_fl<16, F, kittens::ducks::rt_layout::col> b1_bar_col_reg;
        swap_layout(b1_bar_col_reg, b1_init);
        zero(b1_init);
        matmul(b1_init, make_last_b_matrix, b1_bar_col_reg, b1_init);
        // mma_AB(b1_init, make_last_b_matrix, b1_bar_col_reg, b1_init);

    }

    // store(_W1, W1_init, W1_init.cols);
    // store(_b1, b1_init, b1_init.cols);

    kittens::store(W1_init_gl, W1_init, idx1);
    kittens::store(b1_init_gl, b1_init, idx1);
    // kittens::store(W2_init_gl, W2_init, {blockIdx.x, 0, 0});


    // if (blockIdx.x == 10 && threadIdx.x == 1)
    // {
    //     printf("Finished Kernel\n");
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

    auto* W1_init_ptr = inputs.W1_init.data_ptr<float>();
    auto* b1_init_ptr = inputs.b1_init.data_ptr<float>();
    auto* XQ_batch_ptr = inputs.XQ_batch.data_ptr<float>();
    auto* XV_batch_ptr = inputs.XV_batch.data_ptr<float>();
    auto* XK_batch_ptr = inputs.XK_batch.data_ptr<float>();
    auto* eta_batch_ptr = inputs.eta_batch.data_ptr<float>();
    auto* ttt_norm_weight_ptr = inputs.ttt_norm_weight.data_ptr<float>();
    auto* ttt_norm_bias_ptr = inputs.ttt_norm_bias.data_ptr<float>();

    auto output = torch::empty_like(inputs.XQ_batch);

    auto* output_ptr = output.data_ptr<float>();

    auto make_last_b_matrix = torch::zeros({CS, CS}, torch::kCUDA);
    auto make_last_coeff_1_matrix = torch::zeros({CS, F}, torch::kCUDA);

    make_last_b_matrix.index_put_({torch::indexing::Slice(), -1}, 1.0);
    make_last_coeff_1_matrix.index_put_({-1, torch::indexing::Slice()}, 1.0);

    gl<float, 1, B*NH, F, F> W1_init_gl{W1_init_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, 1, B*NH, 1, F> b1_init_gl{b1_init_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, B*NH, NC, CS, F> XQ_batch_gl{XQ_batch_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, B*NH, NC, CS, F> XV_batch_gl{XV_batch_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, B*NH, NC, CS, F> XK_batch_gl{XK_batch_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, B*NH, NC, CS, F> eta_batch_gl{eta_batch_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, 1, 1, NH, F> ttt_norm_weight_gl{ttt_norm_weight_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<float, 1, 1, NH, F> ttt_norm_bias_gl{ttt_norm_bias_ptr, nullptr, nullptr, nullptr, nullptr};

    gl<float, 1, 1, CS, CS> make_last_b_matrix_gl{make_last_b_matrix.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr};
    gl<float, 1, 1, CS, F> make_last_coeff_1_matrix_gl{make_last_coeff_1_matrix.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr};

    gl<float, B*NH, NC, CS, F> output_gl{output_ptr, nullptr, nullptr, nullptr, nullptr};


    const auto run_kernel = [&]()
    {
        

        ttt_linear_forward_kernel<<<B * NH, threads, 49152>>>(
            W1_init_gl,
            b1_init_gl,
            XQ_batch_gl,
            XV_batch_gl,
            XK_batch_gl,
            eta_batch_gl,
            ttt_norm_weight_gl,
            ttt_norm_bias_gl,
            make_last_b_matrix_gl,
            make_last_coeff_1_matrix_gl,
            output_gl

            // W1_init_ptr,
            // b1_init_ptr,
            // XQ_batch_ptr,
            // XV_batch_ptr,
            // XK_batch_ptr,
            // eta_batch_ptr,
            // ttt_norm_weight_ptr,
            // ttt_norm_bias_ptr
            // NH, NC, CS, F, F*4
        );
    };
    

    // Warmup kernel
    for(int i = 0; i < 5; i++)
    {
        run_kernel();

        // Check for launch errors
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(launchErr) << std::endl;
            // Handle the error (e.g., exit or cleanup)
            exit(1);
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            std::cout << "Success" << std::endl;
        }
        else
        {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        }
    }

    // Start timing
    // cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 1000;
    for (int i = 0; i < ITERS; i++)
    {
        run_kernel();
    }

    cudaError_t err = cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    std::cout << "Kernel avg execution time: " << useconds << " microseconds.";
    // cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        std::cout << "Success" << std::endl;
    }
    else
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}

// Modified ttt_mlp_forward function
#ifdef TK_COMPILE_TTT_LINEAR_FORWARD
#include "common/pyutils/torch_helpers.cuh"
torch::Tensor ttt_linear_forward(
    const torch::Tensor ttt_norm_weight,
    const torch::Tensor ttt_norm_bias,
    const torch::Tensor W1_init,
    const torch::Tensor b1_init,
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
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "L1 Cache Size: " << prop.l1CacheSize << " bytes" << std::endl;

    run_benchmark();
}

// #include <iostream>

// int main() {
//     cudaDeviceProp prop;
//     cudaError_t err = cudaGetDeviceProperties(&prop, 0);
//     if (err != cudaSuccess) {
//         std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
//         return -1;
//     }

//     std::cout << "Max Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
//     return 0;
// }