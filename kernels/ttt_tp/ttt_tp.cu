#include "cooperative_groups.h"
#include "kittens.cuh"
#include <torch/torch.h>
#include <math_constants.h>
#include <c10/util/BFloat16.h>

#define ASSERT(cond) if (!(cond)) { printf("Assertion failed: %s\n", #cond); return 1; }

using namespace kittens;

constexpr int NUM_WORKERS = 4; // TODO: why is one warpgroup optimal?

template<int B=1, int H=1, int N=16, int F=64, int K=4, int TP=4>
__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__cluster_dims__(TP)
__global__ void ttt_tp_forward_ker(
    // TODO: make B and H runtime dimensions by instantiating templates with -1 args
    const __grid_constant__ gl<bf16, B, H, N, F> XQ_gl,
    const __grid_constant__ gl<bf16, B, H, N, F> XK_gl,
    const __grid_constant__ gl<bf16, B, H, N, F> XV_gl,
    const __grid_constant__ gl<bf16, B, H, F, F*K> W1_gl,
    const __grid_constant__ gl<bf16, B, H, F*K, F> W2_gl,
    const __grid_constant__ gl<bf16, B, H, N, F> out_gl,
    bool *signal
) {
    int b = blockIdx.y;
    int h = blockIdx.z;
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    int tp = cluster.block_rank();
    
    rt_bf<N, F, ducks::rt_layout::row> XK;
    rt_bf<F, F*K/TP, ducks::rt_layout::col> W1;
    rt_fl<N, F*K/TP, ducks::rt_layout::row> Z1;
    rt_bf<N, F*K/TP, ducks::rt_layout::row> X2;
    rt_bf<F*K/TP, F, ducks::rt_layout::col> W2;
    rt_fl<N, F, ducks::rt_layout::row> out;
    st_fl<N, F> out_part;
    
    load(XK, XK_gl, {b, h, 0, 0});
    load(W1, W1_gl, {b, h, 0, tp*(F*K/TP)});
    zero(Z1);
    mma_AB(Z1, XK, W1, Z1);
    copy(X2, Z1); //dtype conversion
    mma_AB(out, X2, W2, out);
    
    // store(out_part, out);
    for (int i = 0; i < N*F; i+=blockDim.x)
        out_part[i + threadIdx.x] = tp;

    extern __shared__ KITTENS_DEFAULT_ALIGN float shm[N * F];
    float *dsmem = cluster.map_shared_rank(&shm[0], 0);
        for (int i = 0; i < N*F; i+=blockDim.x)
            atomicAdd(&dsmem[i + threadIdx.x], out_part[i + threadIdx.x]);
    cluster.sync();

    if (tp == 0 && threadIdx.x == 0)
        for (int i = 0; i < N*F; i++)
            printf("%.1f ", shm[i]);
    cluster.sync();

    *signal = true;
}

extern torch::Tensor ttt_tp_forward(
    const torch::Tensor XQ,
    const torch::Tensor XK,
    const torch::Tensor XV,
    const torch::Tensor W1,
    const torch::Tensor W2,
    const torch::Tensor out
) {
    constexpr int B = 1, H = 1, N = 16, F = 64, K = 4, TP = 4;

    // TODO: better macro
    TORCH_CHECK(XQ.device().is_cuda() && XQ.is_contiguous() && XQ.dim() == 4 && XQ.size(0) == B && XQ.size(1) == H && XQ.size(2) == N && XQ.size(3) == F, "XQ");
    TORCH_CHECK(XK.device().is_cuda() && XK.is_contiguous() && XK.dim() == 4 && XK.size(0) == B && XK.size(1) == H && XK.size(2) == N && XK.size(3) == F, "XK");
    TORCH_CHECK(XV.device().is_cuda() && XV.is_contiguous() && XV.dim() == 4 && XV.size(0) == B && XV.size(1) == H && XV.size(2) == N && XV.size(3) == F, "XV");
    TORCH_CHECK(W1.device().is_cuda() && W1.is_contiguous() && W1.dim() == 4 && W1.size(0) == B && W1.size(1) == H && W1.size(2) == F && W1.size(3) == F*K, "W1");
    TORCH_CHECK(W2.device().is_cuda() && W2.is_contiguous() && W2.dim() == 4 && W2.size(0) == B && W2.size(1) == H && W2.size(2) == F*K && W2.size(3) == F, "W2");
    TORCH_CHECK(out.device().is_cuda() && out.is_contiguous() && out.dim() == 4 && out.size(0) == B && out.size(1) == H && out.size(2) == N && out.size(3) == F, "out");

    bool *h_signal = (bool *)malloc(sizeof(bool)), *signal;
    cudaMalloc(&signal, sizeof(bool));
    *h_signal = false;
    cudaMemcpy(signal, h_signal, sizeof(bool), cudaMemcpyHostToDevice);

    ttt_tp_forward_ker<B, H, N, F, K, TP><<<dim3(TP, B, H), NUM_WORKERS*kittens::WARP_THREADS>>>(
        gl<bf16, B, H, N, F>{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, N, F>{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, N, F>{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, F, F*K>{reinterpret_cast<bf16*>(W1.data_ptr<at::BFloat16>()), nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, F*K, F>{reinterpret_cast<bf16*>(W2.data_ptr<at::BFloat16>()), nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, N, F>{reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()), nullptr, nullptr, nullptr, nullptr},
        signal
    );

    cudaMemcpy(h_signal, signal, sizeof(bool), cudaMemcpyDeviceToHost);
    TORCH_CHECK(*h_signal, "Kernel failed, *signal=true not set");

    return out;
}//*/

int main() {
    constexpr int B = 1, H = 1, N = 16, F = 64, K = 4, TP = 4;

    bf16 *h_XQ, *h_XK, *h_XV, *h_W1, *h_W2, *h_out;

    // Allocate host memory
    h_XQ = (bf16*)malloc(B*H*N*F*sizeof(bf16));
    h_XK = (bf16*)malloc(B*H*N*F*sizeof(bf16));
    h_XV = (bf16*)malloc(B*H*N*F*sizeof(bf16));
    h_W1 = (bf16*)malloc(B*H*F*F*K*sizeof(bf16));
    h_W2 = (bf16*)malloc(B*H*F*K*F*sizeof(bf16));
    h_out = (bf16*)malloc(B*H*N*F*sizeof(bf16));

    // Initialize host arrays
    for (int i = 0; i < B*H*N*F; i++) {
        h_XQ[i] = __int2bfloat16_rn(i);
        h_XK[i] = __int2bfloat16_rn(i);
        h_XV[i] = __int2bfloat16_rn(i);
        h_out[i] = __int2bfloat16_rn(-1);
    }
    for (int i = 0; i < B*H*F*F*K; i++) {
        h_W1[i] = __int2bfloat16_rn(i);
        h_W2[i] = __int2bfloat16_rn(i);
    }

    bf16 *XQ, *XK, *XV, *W1, *W2, *out;

    // Allocate device memory
    cudaMalloc(&XQ, B*H*N*F*sizeof(bf16));
    cudaMalloc(&XK, B*H*N*F*sizeof(bf16));
    cudaMalloc(&XV, B*H*N*F*sizeof(bf16));
    cudaMalloc(&W1, B*H*F*F*K*sizeof(bf16));
    cudaMalloc(&W2, B*H*F*K*F*sizeof(bf16));
    cudaMalloc(&out, B*H*N*F*sizeof(bf16));

    // Copy data from host to device
    cudaMemcpy(XQ, h_XQ, B*H*N*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(XK, h_XK, B*H*N*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(XV, h_XV, B*H*N*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(W1, h_W1, B*H*F*F*K*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, B*H*F*K*F*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(out, h_out, B*H*N*F*sizeof(bf16), cudaMemcpyHostToDevice);

    bool *h_signal = (bool *)malloc(sizeof(bool)), *signal;
    cudaMalloc(&signal, sizeof(bool));
    *h_signal = false;
    cudaMemcpy(signal, h_signal, sizeof(bool), cudaMemcpyHostToDevice);

    printf("Launching kernel\n");

    ttt_tp_forward_ker<B, H, N, F, K, TP><<<dim3(TP, B, H), NUM_WORKERS*kittens::WARP_THREADS>>>(
        gl<bf16, B, H, N, F>{XQ, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, N, F>{XK, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, N, F>{XV, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, F, F*K>{W1, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, F*K, F>{W2, nullptr, nullptr, nullptr, nullptr},
        gl<bf16, B, H, N, F>{out, nullptr, nullptr, nullptr, nullptr},
        signal
    );

    cudaMemcpy(h_signal, signal, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT(*h_signal);

    printf("Ran successfully\n");
}