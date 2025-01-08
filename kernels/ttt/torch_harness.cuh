#if TORCH_COMPILE

#include <torch/torch.h>
#include <c10/util/BFloat16.h>

// extern torch::Tensor ttt_forward(
//     const torch::Tensor XQ,
//     const torch::Tensor XK,
//     const torch::Tensor XV,
//     const torch::Tensor W1,
//     const torch::Tensor b1,
//     const torch::Tensor W2,
//     const torch::Tensor b2,
//     const torch::Tensor W1_checkpoints,
//     const torch::Tensor b1_checkpoints,
//     const torch::Tensor W2_checkpoints,
//     const torch::Tensor b2_checkpoints,
//     const torch::Tensor Out
// ) {
//     constexpr int F = 64;
//     constexpr int K = 4;
//     unsigned long B = XQ.size(0);
//     unsigned long H = XQ.size(1);
//     unsigned long N = XQ.size(2);
//     unsigned long R = W1_checkpoints.size(2);
//     TORCH_CHECK(N % R == 0, "N % R == 0");
    
//     TORCH_CHECK(XQ.device().is_cuda() && XQ.is_contiguous() && XQ.dim() == 4 && XQ.size(0) == B && XQ.size(1) == H && XQ.size(2) == N && XQ.size(3) == F, "XQ");
//     TORCH_CHECK(XK.device().is_cuda() && XK.is_contiguous() && XK.dim() == 4 && XK.size(0) == B && XK.size(1) == H && XK.size(2) == N && XK.size(3) == F, "XK");
//     TORCH_CHECK(XV.device().is_cuda() && XV.is_contiguous() && XV.dim() == 4 && XV.size(0) == B && XV.size(1) == H && XV.size(2) == N && XV.size(3) == F, "XV");
//     TORCH_CHECK(W1.device().is_cuda() && W1.is_contiguous() && W1.dim() == 4 && W1.size(0) == B && W1.size(1) == H && W1.size(2) == F && W1.size(3) == F*K, "W1");
//     TORCH_CHECK(W2.device().is_cuda() && W2.is_contiguous() && W2.dim() == 4 && W2.size(0) == B && W2.size(1) == H && W2.size(2) == F*K && W2.size(3) == F, "W2");
//     TORCH_CHECK(W1_checkpoints.device().is_cuda() && W1_checkpoints.is_contiguous() && W1_checkpoints.dim() == 5 && W1_checkpoints.size(0) == B && W1_checkpoints.size(1) == H && W1_checkpoints.size(2) == R && W1_checkpoints.size(3) == F && W1_checkpoints.size(4) == F*K, "W1_checkpoints");
//     TORCH_CHECK(W2_checkpoints.device().is_cuda() && W2_checkpoints.is_contiguous() && W2_checkpoints.dim() == 5 && W2_checkpoints.size(0) == B && W2_checkpoints.size(1) == H && W2_checkpoints.size(2) == R && W2_checkpoints.size(3) == F*K && W2_checkpoints.size(4) == F, "W2_checkpoints");
//     TORCH_CHECK(Out.device().is_cuda() && Out.is_contiguous() && Out.dim() == 4 && Out.size(0) == B && Out.size(1) == H && Out.size(2) == N && Out.size(3) == F, "Out");

//     using tile_type = st_bf<fwd_ttt_mlp_ker_tile_dims<F>::tile_height, fwd_ttt_mlp_ker_tile_dims<F>::tile_width>;
//     using tile_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
//     using vec_type = sv_bf<fwd_ttt_mlp_ker_tile_dims<F>::tile_height>;
//     using vec_gl = gl<bf16, -1, -1, -1, -1, vec_type>;
//     using globals = fwd_globals<F>;

//     tile_gl q_gl{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), B, H, N, F};
//     tile_gl k_gl{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), B, H, N, F};
//     tile_gl v_gl{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), B, H, N, F};
//     tile_gl o_gl{reinterpret_cast<bf16*>(Out.data_ptr<at::BFloat16>()), B, H, N, F};

//     tile_gl w1_init_gl{reinterpret_cast<bf16*>(W1.data_ptr<at::BFloat16>()), B, H, F, F*K};
//     vec_gl b1_init_gl{reinterpret_cast<bf16*>(b1.data_ptr<at::BFloat16>()), B, H, 1, F*K};
//     tile_gl w2_init_gl{reinterpret_cast<bf16*>(W2.data_ptr<at::BFloat16>()), B, H, F*K, F};
//     vec_gl b2_init_gl{reinterpret_cast<bf16*>(b2.data_ptr<at::BFloat16>()), B, H, 1, F};

//     tile_gl w1_checkpoints_gl{reinterpret_cast<bf16*>(W1_checkpoints.data_ptr<at::BFloat16>()), B, H, R*F, F*K};
//     vec_gl b1_checkpoints_gl{reinterpret_cast<bf16*>(b1_checkpoints.data_ptr<at::BFloat16>()), B, H, R, F*K};
//     tile_gl w2_checkpoints_gl{reinterpret_cast<bf16*>(W2_checkpoints.data_ptr<at::BFloat16>()), B, H, R*F*K, F};
//     vec_gl b2_checkpoints_gl{reinterpret_cast<bf16*>(b2_checkpoints.data_ptr<at::BFloat16>()), B, H, R, F};

//     globals g{
//         q_gl, 
//         k_gl, 
//         v_gl, 
//         o_gl, 
//         w1_init_gl, 
//         b1_init_gl,
//         w2_init_gl, 
//         b2_init_gl,
//         w1_checkpoints_gl, 
//         b1_checkpoints_gl,
//         w2_checkpoints_gl, 
//         b2_checkpoints_gl,
//         static_cast<int>(N),
//         static_cast<int>(N/R),
//     };

//     constexpr long mem_size = kittens::MAX_SHARED_MEMORY;
//     cudaFuncSetAttribute(
//         fwd_ttt_mlp_ker<F>,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         mem_size
//     );
//     dim3 grid(TP, B, H);
//     fwd_ttt_mlp_ker<F><<<grid, NUM_WORKERS*32, mem_size>>>(g);

//     TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error launching kernel");

//     return Out;
// }//*/


#endif
