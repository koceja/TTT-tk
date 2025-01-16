import torch
import thunderkittens
import multiprocessing
import time
from kernels.ttt.triton_comps.linear_forward import ttt_linear_scan_forward

from kernels.ttt.triton_comps.mlp_forward_split import (
    ttt_mlp_stage_1 as fwd_stage_1,
    ttt_mlp_stage_2 as fwd_stage_2,
    ttt_mlp_stage_3 as fwd_stage_3,
)

def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

def compare_outputs(output_hw, output_ref, name):
    abs_diff = torch.abs(output_hw - output_ref)
    max_diff = torch.max(torch.abs(output_hw - output_ref)).item()
    median_diff = torch.median(torch.abs(output_hw - output_ref)).item()
    
    # Avoid division by zero and calculate relative absolute error
    with torch.no_grad():
        nonzero_mask = output_ref != 0
        relative_error = torch.zeros_like(output_ref)
        relative_error[nonzero_mask] = abs_diff[nonzero_mask] / torch.abs(output_ref[nonzero_mask])
        max_relative_error = torch.max(relative_error).item()
        median_relative_error = torch.median(relative_error).item()

    print(f"{name} - Max Difference: {max_diff}, Median Difference: {median_diff}, "
          f"Max Relative Error: {max_relative_error}, Median Relative Error: {median_relative_error}")

def compute_mini_batch_shard(W1, b1, W2, b2, xq_mb, xk_mb, xv_mb, shard_size):
    """
    Sharded mini batch forward for TTT MLP.

    xq_mb: [CS, F]
    xk_mb: [CS, F]
    xv_mb: [CS, F]
    W1: [F, K]
    b1: [1, K]
    W2: [K, F]
    b2: [1, F]

    Dimension Key:
    B: Batch size
    H: Num of heads
    CS: Mini-batch size
    F: Head dimension
    K: Expansion dimension

    Excludes:
    - ILR
    - LayerNorm
    - Residual connection
    """
    # Simulate sharding weights
    F, K = W1.shape
    assert K % shard_size == 0
    W1_sharded = W1.reshape(F, shard_size, K // shard_size).permute(1, 0, 2)
    b1_sharded = b1.reshape(1, shard_size, K // shard_size).permute(1, 0, 2)
    W2_sharded = W2.reshape(shard_size, K // shard_size, F)

    # Inner model forward (parallel)
    Z1 = xk_mb @ W1_sharded + b1_sharded
    X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2_sharded

    # Reduction across shards and add bias
    Z2_reduce = Z2.sum(dim=0) + b2

    grad_l_wrt_Z2 = xv_mb - Z2_reduce

    # Gradient calculation (parallel)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_sharded.transpose(-2, -1) * gelu_bwd(Z1)

    # Dual form (parallel)
    Attn1 = torch.tril(xq_mb @ xk_mb.T) # Can be on one consumer
    b1_bar_sharded = b1_sharded - grad_l_wrt_Z1
    Z1_bar = xq_mb @ W1_sharded - Attn1 @ grad_l_wrt_Z1 + b1_bar_sharded

    X2_bar = torch.nn.functional.gelu(Z1_bar, approximate="tanh")

    Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
    Z2_bar = X2_bar @ W2_sharded - Attn2 @ grad_l_wrt_Z2 

    # Reduce across shards and add bias
    b2_bar = b2 - grad_l_wrt_Z2
    Z2_bar_reduce = Z2_bar.sum(dim=0) + b2_bar

    # Weight updates
    W1_next_sharded = W1_sharded - xk_mb.T @ grad_l_wrt_Z1
    b1_next_sharded = b1_sharded - grad_l_wrt_Z1.sum(dim=1, keepdim=True)

    W2_next_sharded = W2_sharded - X2.transpose(-2, -1) @ grad_l_wrt_Z2
    b2_next = b2 - grad_l_wrt_Z2.sum(dim=0, keepdim=True) # Not sharded, hence the special update rule

    # Reshape for return
    W1_next = W1_next_sharded.permute(1, 0, 2).reshape(F, K)
    b1_next = b1_next_sharded.permute(1, 0, 2).reshape(1, K)
    W2_next = W2_next_sharded.reshape(K, F)

    return Z2_bar_reduce, W1_next, b1_next, W2_next, b2_next

def compute_mini_batch(W1, b1, W2, b2, xq_mb, xk_mb, xv_mb):
    """
    Mini batch forward for TTT MLP.

    xq_mb: [CS, F]
    xk_mb: [CS, F]
    xv_mb: [CS, F]
    W1: [F, K]
    b1: [1, K]
    W2: [K, F]
    b2: [1, F]

    Dimension Key:
    B: Batch size
    H: Num of heads
    CS: Mini-batch size
    F: Head dimension
    K: Expansion dimension

    Excludes:
    - ILR
    - LayerNorm
    - Residual connection
    """
    # Inner model forward
    Z1 = xk_mb @ W1 + b1
    X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2

    # Gradient calculation
    grad_l_wrt_Z2 = xv_mb - Z2
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.T * gelu_bwd(Z1)

    # Dual form
    Attn1 = torch.tril(xq_mb @ xk_mb.T)
    b1_bar = b1 - grad_l_wrt_Z1
    Z1_bar = xq_mb @ W1 - Attn1 @ grad_l_wrt_Z1 + b1_bar

    X2_bar = torch.nn.functional.gelu(Z1_bar, approximate="tanh")
    
    Attn2 = torch.tril(X2_bar @ X2.T)
    b2_bar = b2 - grad_l_wrt_Z2
    Z2_bar = X2_bar @ W2 - Attn2 @ grad_l_wrt_Z2 + b2_bar

    # Weight updates
    W1_next = W1 - xk_mb.T @ grad_l_wrt_Z1
    b1_next = b1 - grad_l_wrt_Z1.sum(dim=0, keepdim=True)

    W2_next = W2 - X2_bar.T @ grad_l_wrt_Z2
    b2_next = b2 - grad_l_wrt_Z2.sum(dim=0, keepdim=True)

    return Z2_bar, W1_next, b1_next, W2_next, b2_next


def compute_mini_batch_no_dual(
    W1, 
    b1, 
    W2, 
    b2, 
    xq_mb, 
    xk_mb, 
    xv_mb, 
    eta_mb,
    ttt_norm_weight,
    ttt_norm_bias,
):
    """
    Mini batch forward for TTT MLP.

    xq_mb: [CS, F]
    xk_mb: [CS, F]
    xv_mb: [CS, F]
    W1: [F, K]
    b1: [1, K]
    W2: [K, F]
    b2: [1, F]

    Dimension Key:
    B: Batch size
    H: Num of heads
    CS: Mini-batch size
    F: Head dimension
    K: Expansion dimension
    """
    num_heads = xk_mb.shape[1]
    head_dim = xk_mb.shape[-1]

    # Inner model forward
    Z1 = xk_mb @ W1 + b1
    X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2

    reconstruction_target = xv_mb - xk_mb

    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

    # Stage 2: LnFusedL2BWD

    eps = 1e-6
    mu_fused = Z2.mean(dim=-1, keepdim=True)
    var_fused = Z2.var(dim=-1, keepdim=True, unbiased=False)

    std_fused = torch.sqrt(var_fused + eps)
    x_hat_fused = (Z2 - mu_fused) / std_fused

    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z2 = (
        (1.0 / head_dim)
        * (
            head_dim * grad_x_hat_fused
            - grad_x_hat_fused.sum(dim=-1, keepdim=True)
            - x_hat_fused * (grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True)
        )
        / std_fused
    )

    # Gradient calculation
    # grad_l_wrt_Z2 = xv_mb - Z2
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-1,-2) * gelu_bwd(Z1)

    # Weight updates
    last_eta_mini_batch = eta_mb[:, :, -1, :, None]

    W1_next = W1 - (last_eta_mini_batch * xk_mb).transpose(-1,-2) @ grad_l_wrt_Z1
    b1_next = b1 - (last_eta_mini_batch * grad_l_wrt_Z1).sum(dim=-2, keepdim=True)

    W2_next = W2 - (last_eta_mini_batch * X2).transpose(-1,-2) @ grad_l_wrt_Z2
    b2_next = b2 - (last_eta_mini_batch * grad_l_wrt_Z2).sum(dim=-2, keepdim=True)

    # Post grad forward
    Z1_bar = xq_mb @ W1_next + b1_next
    X2_bar = torch.nn.functional.gelu(Z1_bar, approximate="tanh")
    Z2_bar = X2_bar @ W2_next + b2_next

    # Ln
    mu_ln = Z2_bar.mean(dim=-1, keepdim=True)
    var_ln = Z2_bar.var(dim=-1, keepdim=True, unbiased=False)
    std_ln = torch.sqrt(var_ln + eps)
    x_hat_ln = (Z2_bar - mu_ln) / std_ln

    Z2_bar_ln = ln_weight * x_hat_ln + ln_bias

    # Residual
    XQW_mini_batch = xq_mb + Z2_bar_ln

    return XQW_mini_batch, W1_next, b1_next, W2_next, b2_next


def triton_linear_forward(B, NH, NC, mini_batch_size, head_dim, K):
    NC = NC * 4
    mini_batch_size = 16
    CS = mini_batch_size
    F = head_dim
    expansion_dim = 4 * F
    device = 'cuda'
    dtype = torch.bfloat16
    intermediate_dtype = torch.float32
    mp_dtype = torch.bfloat16
    checkpoint_group_size = NC // K
    reset_states = None

    def get_inputs(dtype):
        torch.manual_seed(0)
        # Create inputs
        xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xv = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

        eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous() * 0.02
        last_eta = eta[:, :, :, -1, :, None].repeat(1, 1, 1, 1, head_dim).contiguous()

        ttt_norm_weight = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02
        ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

        W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        b1 = torch.zeros(B, NH, 1, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        b2 = torch.zeros(B, NH, 1, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02

        return xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    
    XQ_batch, XK_batch, XV_batch, eta_batch, last_eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init = get_inputs(torch.bfloat16)

    # Cast and make inputs contiguous
    XQ_batch = XQ_batch.contiguous()
    XV_batch = XV_batch.contiguous()
    XK_batch = XK_batch.contiguous()
    eta_batch = eta_batch.to(mp_dtype).contiguous()

    W1_init = W1_init.to(torch.float32).contiguous()
    b1_init = b1_init.to(torch.float32).contiguous()
    W2_init = W2_init.to(torch.float32).contiguous()
    b2_init = b2_init.to(torch.float32).contiguous()


    # Output pointers
    W1_last = torch.empty(B, NH, F, F, device=device, dtype=torch.float32)
    b1_last = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)
    XQW_batch = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)

    # Context pointers
    W1_checkpoints = torch.empty(B, NH, K, F, F, device=device, dtype=torch.float32)
    b1_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=torch.float32)

    # Strides
    CS_F_stride = CS * F
    F_F_stride = F * F
    CS_CS_stride = CS * CS
    F_stride = F

    enable_reset_states = reset_states is not None

    grid = (B, NH)

    torch.cuda.empty_cache()

    for _ in range(10):
        ttt_linear_scan_forward[grid](
            # Scan inputs
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            reset_states.contiguous() if enable_reset_states else None,
            # Outputs
            W1_last,
            b1_last,
            XQW_batch,
            # Context pointers
            W1_checkpoints,
            b1_checkpoints,
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
            enable_reset_states,
        )
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(50):
        ttt_linear_scan_forward[grid](
            # Scan inputs
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            reset_states.contiguous() if enable_reset_states else None,
            # Outputs
            W1_last,
            b1_last,
            XQW_batch,
            # Context pointers
            W1_checkpoints,
            b1_checkpoints,
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
            enable_reset_states,
        )
        torch.cuda.synchronize()

    end = time.perf_counter()
    
    return (end - start) / 50
    

def triton_forward(B, NH, NC, mini_batch_size, head_dim, K):
    NC = NC * 4
    mini_batch_size = 16
    CS = mini_batch_size
    F = head_dim
    expansion_dim = 4 * F
    device = 'cuda'
    dtype = torch.bfloat16
    intermediate_dtype = torch.float32
    mp_dtype = torch.bfloat16
    checkpoint_group_size = NC // K

    def get_inputs(dtype):
        torch.manual_seed(0)
        # Create inputs
        xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xv = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

        eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous() * 0.02
        last_eta = eta[:, :, :, -1, :, None].repeat(1, 1, 1, 1, head_dim).contiguous()

        ttt_norm_weight = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02
        ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

        W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        b1 = torch.zeros(B, NH, 1, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        b2 = torch.zeros(B, NH, 1, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02

        return xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    
    XQ_batch, XK_batch, XV_batch, eta_batch, last_eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init = get_inputs(torch.bfloat16)

    Z2_bar_ln = torch.empty(B, NH, NC, CS, F, device=device, dtype=intermediate_dtype).contiguous()

    # Context pointers (Checkpoints are always saved in Float32)
    W1_checkpoints = torch.empty(B, NH, K, F, F * 4, device=device, dtype=torch.float32).contiguous()
    b1_checkpoints = torch.empty(B, NH, K, 1, F * 4, device=device, dtype=torch.float32).contiguous()
    W2_checkpoints = torch.empty(B, NH, K, F * 4, F, device=device, dtype=torch.float32).contiguous()
    b2_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=torch.float32).contiguous()


    reset_states = None
    # Intermediates between kernels
    CS_F4_buffer_1 = torch.empty(B, NH, CS, F * 4, device=device, dtype=intermediate_dtype).contiguous()
    CS_F_buffer_1 = torch.empty(B, NH, CS, F, device=device, dtype=intermediate_dtype).contiguous()
    CS_F4_buffer_2 = torch.empty(B, NH, CS, F * 4, device=device, dtype=intermediate_dtype).contiguous()

    # Strides
    CS_F_stride = CS * F
    F_F4_stride = F * F * 4
    CS_CS_stride = CS * CS
    F_stride = F
    F4_stride = F * 4

    grid = (B, NH)

    # Cast and make inputs contiguous
    XQ_batch = XQ_batch.contiguous()
    XV_batch = XV_batch.contiguous()
    XK_batch = XK_batch.contiguous()
    eta_batch = eta_batch.to(mp_dtype).contiguous()

    W1_init = W1_init.to(torch.float32).contiguous()
    b1_init = b1_init.to(torch.float32).contiguous()
    W2_init = W2_init.to(torch.float32).contiguous()
    b2_init = b2_init.to(torch.float32).contiguous()

    # Initial states for potential resets
    if reset_states is not None:
        W1_0 = W1_init[0].clone().contiguous()
        b1_0 = b1_init[0].clone().contiguous()
        W2_0 = W2_init[0].clone().contiguous()
        b2_0 = b2_init[0].clone().contiguous()


    torch.cuda.empty_cache()

    for _ in range(10):

        for i in range(NC):

            # Save checkpoints
            if i % checkpoint_group_size == 0:
                W1_checkpoints[:, :, i // checkpoint_group_size] = W1_init
                b1_checkpoints[:, :, i // checkpoint_group_size] = b1_init
                W2_checkpoints[:, :, i // checkpoint_group_size] = W2_init
                b2_checkpoints[:, :, i // checkpoint_group_size] = b2_init

            fwd_stage_1[grid](
                # Scan inputs
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XV_batch,
                XK_batch,
                # Outputs
                CS_F4_buffer_1,
                CS_F_buffer_1,
                CS_F4_buffer_2,
                # Strides
                CS_F_stride,
                F_F4_stride,
                F_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

            fwd_stage_2[grid](
                # Scan inputs
                W1_init,
                b1_init,
                XQ_batch,
                XK_batch,
                eta_batch,
                # Intermediates
                CS_F4_buffer_1,
                # Strides
                CS_F_stride,
                F_F4_stride,
                CS_CS_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

            fwd_stage_3[grid](
                # Scan inputs
                ttt_norm_weight,
                ttt_norm_bias,
                W2_init,
                b2_init,
                eta_batch,
                # Intermediates
                CS_F_buffer_1,
                CS_F4_buffer_2,
                CS_F4_buffer_1,
                # Outputs
                Z2_bar_ln,
                # Strides
                CS_F_stride,
                F_F4_stride,
                CS_CS_stride,
                F_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

        XQW_batch = Z2_bar_ln + XQ_batch

    print("STarting triton")


    start_time = time.perf_counter()

    for _ in range(50):

        for i in range(NC):

            # Save checkpoints
            if i % checkpoint_group_size == 0:
                W1_checkpoints[:, :, i // checkpoint_group_size] = W1_init
                b1_checkpoints[:, :, i // checkpoint_group_size] = b1_init
                W2_checkpoints[:, :, i // checkpoint_group_size] = W2_init
                b2_checkpoints[:, :, i // checkpoint_group_size] = b2_init

            fwd_stage_1[grid](
                # Scan inputs
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XV_batch,
                XK_batch,
                # Outputs
                CS_F4_buffer_1,
                CS_F_buffer_1,
                CS_F4_buffer_2,
                # Strides
                CS_F_stride,
                F_F4_stride,
                F_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

            fwd_stage_2[grid](
                # Scan inputs
                W1_init,
                b1_init,
                XQ_batch,
                XK_batch,
                eta_batch,
                # Intermediates
                CS_F4_buffer_1,
                # Strides
                CS_F_stride,
                F_F4_stride,
                CS_CS_stride,
                F4_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

            fwd_stage_3[grid](
                # Scan inputs
                ttt_norm_weight,
                ttt_norm_bias,
                W2_init,
                b2_init,
                eta_batch,
                # Intermediates
                CS_F_buffer_1,
                CS_F4_buffer_2,
                CS_F4_buffer_1,
                # Outputs
                Z2_bar_ln,
                # Strides
                CS_F_stride,
                F_F4_stride,
                CS_CS_stride,
                F_stride,
                # Constant expressions
                NH,
                NC,
                CS,
                F,
                # Index
                i,
                num_warps=8,
                num_stages=4,
                num_ctas=1,
            )

        XQW_batch = Z2_bar_ln + XQ_batch
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    print("Finish triton")
    return (end_time - start_time) / 50


# Wrapper function to kill the process if it takes too long
def kernel_with_timeout(kernel_function, timeout=10):
    process = multiprocessing.Process(target=kernel_function)
    process.start()

    # Wait for the kernel process to complete
    process.join(timeout)

    # Check if the process is still alive after the timeout
    if process.is_alive():
        print(f"Kernel timeout exceeded ({timeout} seconds). Terminating process...")
        process.terminate()
        process.join()
        raise RuntimeError("Kernel execution took too long and was forcibly terminated.")

    print("Kernel execution completed within the timeout.")



def main():
    torch.manual_seed(0)
    # Define shapes
    B = 1
    NH = 48
    K = 16
    
    seq_len = 32768
    mini_batch_size = 64
    NC = seq_len // mini_batch_size
    checkpoint_group_size = NC // K

    head_dim = 64
    expansion_dim = 256
    shard_size = 4

    dtype = torch.bfloat16
    full_dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    def get_inputs(dtype):
        torch.manual_seed(0)
        # Create inputs
        xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xv = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

        # eta = torch.arange(B*NH*NC*mini_batch_size*mini_batch_size, dtype=dtype, device=device).reshape(B, NH, NC, mini_batch_size, mini_batch_size).contiguous() * 0.02
        eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous() * 0.1
        last_eta = eta[:, :, :, -1, :].contiguous()
        # last_eta = eta[:, :, :, -1, :, None].repeat(1, 1, 1, 1, head_dim).contiguous()

        ttt_norm_weight = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous()
        ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous()

        W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
        b1 = torch.randn(B, NH, 1, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
        W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=dtype, device=device).contiguous() * 0.02
        b2 = torch.randn(B, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

        return xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    

    # xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    W1_checkpoints = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=full_dtype, device=device).contiguous()
    b1_checkpoints = torch.empty(B, NH, K, 1, expansion_dim, dtype=full_dtype, device=device).contiguous()
    W2_checkpoints = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=full_dtype, device=device).contiguous()
    b2_checkpoints = torch.empty(B, NH, K, 1, head_dim, dtype=full_dtype, device=device).contiguous()

    W1_checkpoints_ref = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=full_dtype, device=device).contiguous()
    b1_checkpoints_ref = torch.empty(B, NH, K, 1, expansion_dim, dtype=full_dtype, device=device).contiguous()
    W2_checkpoints_ref = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=full_dtype, device=device).contiguous()
    b2_checkpoints_ref = torch.empty(B, NH, K, 1, head_dim, dtype=full_dtype, device=device).contiguous()
    W1_checkpoints_ref_bf = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=dtype, device=device).contiguous()
    b1_checkpoints_ref_bf = torch.empty(B, NH, K, 1, expansion_dim, dtype=dtype, device=device).contiguous()
    W2_checkpoints_ref_bf = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=dtype, device=device).contiguous()
    b2_checkpoints_ref_bf = torch.empty(B, NH, K, 1, head_dim, dtype=dtype, device=device).contiguous()

    # Create output buffers
    output_ref = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=full_dtype, device=device).contiguous()
    output_ref_bf = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    Z2_bar_pt_shard = torch.zeros(NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    output_tk = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()


    xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2 = get_inputs(torch.float32)

    xq = xq.to(torch.bfloat16).contiguous()
    xk = xk.to(torch.bfloat16).contiguous()
    xv = xv.to(torch.bfloat16).contiguous()
    last_eta = last_eta.to(torch.bfloat16)
    eta = eta.to(torch.bfloat16)
    # ttt_norm_weight = ttt_norm_weight.to(torch.bfloat16)
    # ttt_norm_bias = ttt_norm_bias.to(torch.bfloat16)
    # W1 = W1.to(torch.bfloat16)
    # W2 = W2.to(torch.bfloat16)

    
    # W1.

    # for _ in range(10):
    #     triton_forward(B, NH, NC, mini_batch_size, head_dim, K)

    # for _ in range(10):
    

    torch.cuda.empty_cache()
    for _ in range(10):
        thunderkittens.ttt_forward(
            xq,
            xk,
            xv,
            last_eta,
            ttt_norm_weight,
            ttt_norm_bias,
            W1,
            b1,
            W2,
            b2,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            output_tk
        )

    print("Starting tk kernel")
    start = time.perf_counter()
    for _ in range(50):
        thunderkittens.ttt_forward(
            xq,
            xk,
            xv,
            last_eta,
            ttt_norm_weight,
            ttt_norm_bias,
            W1,
            b1,
            W2,
            b2,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            output_tk
        )

    end = time.perf_counter()  # End the timer

    elapsed_time = (end - start) / 50
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print("Finished tk kernel")

    torch.cuda.empty_cache()

    triton_runtime = triton_forward(B, NH, NC, mini_batch_size, head_dim, K)

    print(f"Triton M2 runtime={triton_runtime} seconds")

    torch.cuda.empty_cache()

    m1_runtime = triton_linear_forward(B, NH, NC, mini_batch_size, head_dim, K)

    print(f"M1 runtime={m1_runtime} seconds")


    # thunderkittens.ttt_forward(
    #     xq,
    #     xk,
    #     xv,
    #     last_eta,
    #     ttt_norm_weight,
    #     ttt_norm_bias,
    #     W1,
    #     b1,
    #     W2,
    #     b2,
    #     W1_checkpoints,
    #     b1_checkpoints,
    #     W2_checkpoints,
    #     b2_checkpoints,
    #     output_tk
    # )

    # _, _, _, _, _, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init = get_inputs(torch.bfloat16)


    # W1_curr, b1_curr, W2_curr, b2_curr = (W1_init, b1_init, W2_init, b2_init)

    # # Compute mini-batches for PyTorch
    # for i in range(NC):
    #     if i % checkpoint_group_size == 0:
    #         checkpoint_idx = i // checkpoint_group_size
    #         W1_checkpoints_ref_bf[:, :, checkpoint_idx] = W1_curr
    #         b1_checkpoints_ref_bf[:, :, checkpoint_idx] = b1_curr
    #         W2_checkpoints_ref_bf[:, :, checkpoint_idx] = W2_curr
    #         b2_checkpoints_ref_bf[:, :, checkpoint_idx] = b2_curr

    #     xq_mb = xq[:,:,i]
    #     xk_mb = xk[:,:,i]
    #     xv_mb = xv[:,:,i]
    #     eta_mb = eta[:, :, i]

    #     # Z2_bar_pt_shard[i], W1_other, b1_other, W2_other, b2_other = compute_mini_batch_shard(W1[0][0], b1[0][0], W2[0][0], b2[0][0], xq_mb[0][0], xk_mb[0][0], xv_mb[0][0], shard_size)
    #     (
    #         output_ref_bf[:, :, i],
    #         W1_curr,
    #         b1_curr,
    #         W2_curr,
    #         b2_curr
    #     ) = compute_mini_batch_no_dual(
    #         W1_curr, 
    #         b1_curr, 
    #         W2_curr, 
    #         b2_curr, 
    #         xq_mb, 
    #         xk_mb, 
    #         xv_mb, 
    #         eta_mb,
    #         ttt_norm_weight,
    #         ttt_norm_bias
    #     )

    # xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2 = get_inputs(torch.float32)

    # W1_curr, b1_curr, W2_curr, b2_curr = (W1, b1, W2, b2)

    # # Compute mini-batches for PyTorch
    # for i in range(NC):
    #     if i % checkpoint_group_size == 0:
    #         checkpoint_idx = i // checkpoint_group_size
    #         W1_checkpoints_ref[:, :, checkpoint_idx] = W1_curr
    #         b1_checkpoints_ref[:, :, checkpoint_idx] = b1_curr
    #         W2_checkpoints_ref[:, :, checkpoint_idx] = W2_curr
    #         b2_checkpoints_ref[:, :, checkpoint_idx] = b2_curr

    #     xq_mb = xq[:,:,i]
    #     xk_mb = xk[:,:,i]
    #     xv_mb = xv[:,:,i]
    #     eta_mb = eta[:, :, i]

    #     # Z2_bar_pt_shard[i], W1_other, b1_other, W2_other, b2_other = compute_mini_batch_shard(W1[0][0], b1[0][0], W2[0][0], b2[0][0], xq_mb[0][0], xk_mb[0][0], xv_mb[0][0], shard_size)
    #     (
    #         output_ref[:, :, i],
    #         W1_curr,
    #         b1_curr,
    #         W2_curr,
    #         b2_curr
    #     ) = compute_mini_batch_no_dual(
    #         W1_curr, 
    #         b1_curr, 
    #         W2_curr, 
    #         b2_curr, 
    #         xq_mb, 
    #         xk_mb, 
    #         xv_mb, 
    #         eta_mb,
    #         ttt_norm_weight,
    #         ttt_norm_bias
    #     )

    
 

    # # Compare outputs
    # print("Comparing Outputs")

    # # breakpoint()
    # print(output_ref)
    # print(output_tk)
    # print(output_ref_bf)
    # compare_outputs(output_ref[:,:,-1], output_ref_bf[:,:,-1].to(torch.float32), "Output baseline precision diff")
    # compare_outputs(output_tk[:,:,-1], output_ref[:,:,-1], "Output")

    # compare_outputs(W1_checkpoints[:, :, -1], W1_checkpoints_ref[:, :, -1], "W1")
    # compare_outputs(b1_checkpoints[:, :, -1], b1_checkpoints_ref[:, :, -1], "b1")
    # compare_outputs(W2_checkpoints[:, :, -1], W2_checkpoints_ref[:, :, -1], "W2")
    # compare_outputs(b2_checkpoints[:, :, -1], b2_checkpoints_ref[:, :, -1], "b2")

    # compare_outputs(W1_checkpoints_ref_bf[:, :, -1], W1_checkpoints_ref[:, :, -1], "W1 bf ref")
    # compare_outputs(b1_checkpoints_ref_bf[:, :, -1], b1_checkpoints_ref[:, :, -1], "b1 bf ref")
    # compare_outputs(W2_checkpoints_ref_bf[:, :, -1], W2_checkpoints_ref[:, :, -1], "W2 bf ref")
    # compare_outputs(b2_checkpoints_ref_bf[:, :, -1], b2_checkpoints_ref[:, :, -1], "b2 bf ref")

    # print("PT Shape: ", Z2_bar_pt.shape)
    # print("PT Shard Shape: ", Z2_bar_pt_shard.shape)
    # compare_outputs(Z2_bar_pt, Z2_bar_pt_shard, "Z2_bar: Non-Sharded vs Sharded")

    breakpoint()



if __name__ == "__main__":
    # Example usage
    main()
    # try:
    #     kernel_with_timeout(main, timeout=15)
    # except RuntimeError as e:
    #     print(e)