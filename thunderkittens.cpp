#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

/*

HOW TO REGISTER YOUR OWN, CUSTOM SET OF KERNELS:

1. Decide on the identifier which will go in config.py. For example, "attn_inference" is the identifier for the first set below.
2. Add the identifier to the dict of sources in config.py.
3. Add the identifier to the list of kernels you want compiled.
4. The macro defined here, when that kernel is compiled, will be "TK_COMPILE_{IDENTIFIER_IN_ALL_CAPS}." You need to add two chunks to this file.
4a. the extern declaration at the top.
4b. the registration of the function into the module.

m.def("attention_inference_forward", attention_inference_forward);

*/



#ifdef TK_COMPILE_TTT
extern torch::Tensor ttt_forward(
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
);
#endif

#ifdef TK_COMPILE_ATTN
extern std::vector<torch::Tensor> attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal
); 
extern std::vector<torch::Tensor> attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor l_vec, torch::Tensor og, 
    bool causal
);
#endif

#ifdef TK_COMPILE_FUSED_LAYERNORM
extern std::tuple<torch::Tensor, torch::Tensor> fused_layernorm(
    const torch::Tensor x,
    const torch::Tensor residual,
    const torch::Tensor norm_weight,
    const torch::Tensor norm_bias,
    float dropout_p
);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Kernels"; // optional module docstring

#ifdef TK_COMPILE_TTT
    m.def("ttt_forward", &ttt_forward, "TTT Forward.");
#endif

#ifdef TK_COMPILE_ATTN
    m.def("mha_forward",  torch::wrap_pybind_function(attention_forward), "Bidirectional forward MHA. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
    m.def("mha_backward", torch::wrap_pybind_function(attention_backward), "Bidirectional backward MHA. Takes Q,K,V,O,Og,Qg,Kg,Vg in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally requres norm vec l_vec, and (TODO) d_vec memory.");
#endif

#ifdef TK_COMPILE_FUSED_LAYERNORM
    m.def("fused_layernorm", fused_layernorm, "LayerNorm TK. Takes tensors (x, residual, norm_weight, norm_bias, dropout_p). x, residual, norm_weight, norm_bias are bf16. dropout_p is float. Returns (B, H, N, 128) in bf16.");
#endif

}