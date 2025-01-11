#ifndef TORCH_COMPILE

#include <iostream>
#include <string>
#include <fstream>

constexpr int BATCH_SIZE = 4;
constexpr int HEADS = 32;

constexpr int SEQ_LEN = 32768; 
constexpr int REMAT_GS = 32;
constexpr int CS = 64;
constexpr int NUM_CHECKPOINTS = SEQ_LEN / CS / REMAT_GS;

constexpr int HEAD_DIM = 64; 
constexpr int EXP_DIM = 256;
constexpr int BLOCK_SIZE = (NUM_WORKERS*32); // Number of threads in a block

constexpr int ITER = 1;

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

int main(int argc, char **argv) {
    std::cout << "Entered main!" << std::endl;

    constexpr int TOTAL_ELEMENTS = BATCH_SIZE*HEADS*SEQ_LEN*HEAD_DIM;
    
    constexpr int TOTAL_HS_ELEMENTS = BATCH_SIZE*HEADS*EXP_DIM*HEAD_DIM;
    constexpr int TOTAL_B1_ELEMENTS = BATCH_SIZE*HEADS*EXP_DIM;
    constexpr int TOTAL_B2_ELEMENTS = BATCH_SIZE*HEADS*HEAD_DIM;

    constexpr int TOTAL_CHECKPOINT_ELEMENTS = BATCH_SIZE*HEADS*NUM_CHECKPOINTS*EXP_DIM*HEAD_DIM;
    constexpr int TOTAL_B1_CHECKPOINT_ELEMENTS = BATCH_SIZE*HEADS*NUM_CHECKPOINTS*EXP_DIM;
    constexpr int TOTAL_B2_CHECKPOINT_ELEMENTS = BATCH_SIZE*HEADS*NUM_CHECKPOINTS*HEAD_DIM;

    bf16 *q_bf, *k_bf, *v_bf, *o_bf, *w1_bf, *b1_bf, *w2_bf, *b2_bf, *w1_checkpoints_bf, *b1_checkpoints_bf, *w2_checkpoints_bf, *b2_checkpoints_bf;

    q_bf = (bf16*)malloc(TOTAL_ELEMENTS * sizeof(bf16));
    k_bf = (bf16*)malloc(TOTAL_ELEMENTS * sizeof(bf16));
    v_bf = (bf16*)malloc(TOTAL_ELEMENTS * sizeof(bf16));
    o_bf = (bf16*)malloc(TOTAL_ELEMENTS * sizeof(bf16));

    w1_bf = (bf16*)malloc(TOTAL_HS_ELEMENTS * sizeof(bf16));
    b1_bf = (bf16*)malloc(TOTAL_B1_ELEMENTS * sizeof(bf16));
    w2_bf = (bf16*)malloc(TOTAL_HS_ELEMENTS * sizeof(bf16));
    b2_bf = (bf16*)malloc(TOTAL_B2_ELEMENTS * sizeof(bf16));

    w1_checkpoints_bf = (bf16*)malloc(TOTAL_CHECKPOINT_ELEMENTS * sizeof(bf16));
    b1_checkpoints_bf = (bf16*)malloc(TOTAL_B1_CHECKPOINT_ELEMENTS * sizeof(bf16));
    w2_checkpoints_bf = (bf16*)malloc(TOTAL_CHECKPOINT_ELEMENTS * sizeof(bf16));
    b2_checkpoints_bf = (bf16*)malloc(TOTAL_B2_CHECKPOINT_ELEMENTS * sizeof(bf16));

    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        q_bf[i] = __int2bfloat16_rn(i % 256);
        k_bf[i] = __int2bfloat16_rn(i % 256);
        v_bf[i] = __int2bfloat16_rn(i % 256);
        o_bf[i] = __int2bfloat16_rn(0);
    }
    for (int i = 0; i < TOTAL_HS_ELEMENTS; i++) {
        w1_bf[i] = __int2bfloat16_rn(i % 256);
        w2_bf[i] = __int2bfloat16_rn(i % 256);
    }
    for (int i = 0; i < TOTAL_B1_ELEMENTS; i++) {
        b1_bf[i] = __int2bfloat16_rn(i % 256);
    }
    for (int i = 0; i < TOTAL_B2_ELEMENTS; i++) {
        b2_bf[i] = __int2bfloat16_rn(i % 256);
    }
    for (int i = 0; i < TOTAL_CHECKPOINT_ELEMENTS; i++) {
        w1_checkpoints_bf[i] = __int2bfloat16_rn(0);
        w2_checkpoints_bf[i] = __int2bfloat16_rn(0);
    }
    for (int i = 0; i < TOTAL_B1_CHECKPOINT_ELEMENTS; i++) {
        b1_checkpoints_bf[i] = __int2bfloat16_rn(0);
    }
    for (int i = 0; i < TOTAL_B2_CHECKPOINT_ELEMENTS; i++) {
        b2_checkpoints_bf[i] = __int2bfloat16_rn(0);
    }

    bf16 *d_q, *d_k, *d_v, *d_w1, *d_b1, *d_w2, *d_b2, *d_o, *d_w1_checkpoints, *d_b1_checkpoints, *d_w2_checkpoints, *d_b2_checkpoints;

    cudaMalloc(&d_q, (TOTAL_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_k, (TOTAL_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_v, (TOTAL_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_o, (TOTAL_ELEMENTS) * sizeof(bf16));

    cudaMalloc(&d_w1, (TOTAL_HS_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_b1, (TOTAL_B1_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_w2, (TOTAL_HS_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_b2, (TOTAL_B2_ELEMENTS) * sizeof(bf16));

    cudaMalloc(&d_w1_checkpoints, (TOTAL_CHECKPOINT_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_b1_checkpoints, (TOTAL_B1_CHECKPOINT_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_w2_checkpoints, (TOTAL_CHECKPOINT_ELEMENTS) * sizeof(bf16));
    cudaMalloc(&d_b2_checkpoints, (TOTAL_B2_CHECKPOINT_ELEMENTS) * sizeof(bf16));

    cudaMemcpy(d_q, q_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);

    cudaMemcpy(d_w1, w1_bf, TOTAL_HS_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1_bf, TOTAL_B1_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2_bf, TOTAL_HS_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2_bf, TOTAL_B2_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);

    cudaMemcpy(d_w1_checkpoints, w1_checkpoints_bf, TOTAL_CHECKPOINT_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1_checkpoints, b1_checkpoints_bf, TOTAL_B1_CHECKPOINT_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2_checkpoints, w2_checkpoints_bf, TOTAL_CHECKPOINT_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2_checkpoints, b2_checkpoints_bf, TOTAL_B2_CHECKPOINT_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);

    using tile_type = st_bf<fwd_ttt_mlp_ker_tile_dims<HEAD_DIM>::tile_height, fwd_ttt_mlp_ker_tile_dims<HEAD_DIM>::tile_width>;
    using vec_type = sv_bf<fwd_ttt_mlp_ker_tile_dims<HEAD_DIM>::tile_height>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using o_gl = gl<bf16, -1, -1, -1, -1, tile_type>;

    using w1_init_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b1_init_gl = gl<bf16, -1, -1, -1, -1, vec_type>;
    using w2_init_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b2_init_gl = gl<bf16, -1, -1, -1, -1, vec_type>;

    using w1_checkpoints_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b1_checkpoints_gl = gl<bf16, -1, -1, -1, -1, vec_type>;
    using w2_checkpoints_gl = gl<bf16, -1, -1, -1, -1, tile_type>;
    using b2_checkpoints_gl = gl<bf16, -1, -1, -1, -1, vec_type>;

    using globals = fwd_globals<HEAD_DIM>;

    q_gl qg_arg{d_q, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};
    k_gl kg_arg{d_k, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};
    v_gl vg_arg{d_v, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};
    o_gl og_arg{d_o, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM};

    w1_init_gl w1g_arg{d_w1, BATCH_SIZE, HEADS, HEAD_DIM, EXP_DIM};
    b1_init_gl b1g_arg{d_b1, BATCH_SIZE, HEADS, 1, EXP_DIM};
    w2_init_gl w2g_arg{d_w2, BATCH_SIZE, HEADS, EXP_DIM, HEAD_DIM};
    b2_init_gl b2g_arg{d_b2, BATCH_SIZE, HEADS, 1, HEAD_DIM};

    w1_checkpoints_gl w1_checkpoints_g_arg{d_w1_checkpoints, BATCH_SIZE, HEADS, NUM_CHECKPOINTS*HEAD_DIM, EXP_DIM};
    b1_checkpoints_gl b1_checkpoints_g_arg{d_b1_checkpoints, BATCH_SIZE, HEADS, NUM_CHECKPOINTS, EXP_DIM};
    w2_checkpoints_gl w2_checkpoints_g_arg{d_w2_checkpoints, BATCH_SIZE, HEADS, EXP_DIM, NUM_CHECKPOINTS*HEAD_DIM};
    b2_checkpoints_gl b2_checkpoints_g_arg{d_b2_checkpoints, BATCH_SIZE, HEADS, NUM_CHECKPOINTS, HEAD_DIM};

    globals g{
        qg_arg, 
        kg_arg, 
        vg_arg, 
        og_arg, 
        w1g_arg,
        b1g_arg, 
        w2g_arg, 
        b2g_arg, 
        w1_checkpoints_g_arg, 
        b1_checkpoints_g_arg, 
        w2_checkpoints_g_arg, 
        b2_checkpoints_g_arg, 
        SEQ_LEN,
        REMAT_GS
    };

    std::cout << "Allocated and set memory on GPU!" << std::endl;
    
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY; // need to launch two blocks if possible.
    
    cudaFuncSetAttribute(
        fwd_ttt_mlp_ker<HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    std::cout << "Set max dynamic memory!" << std::endl;

    dim3 grid(TP, BATCH_SIZE, HEADS);

    cudaDeviceSynchronize();
    std::cout << "Starting kernel" << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITER; i++) {
        fwd_ttt_mlp_ker<HEAD_DIM><<<grid, BLOCK_SIZE, mem_size>>>(g);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    const auto finish = std::chrono::high_resolution_clock::now();

    CudaCheckError();
    std::cout << "Finished kernel\n";
    std::cout << "Average fwd execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / ITER << " us" << std::endl;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_w1);
    cudaFree(d_w2);

    delete[] q_bf, k_bf, v_bf, o_bf, w1_bf, w2_bf;
    return 0;
}

#endif