import torch
import thunderkittens

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


def compute_mini_batch_no_dual(W1, b1, W2, b2, xq_mb, xk_mb, xv_mb):
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
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-1,-2) * gelu_bwd(Z1)

    # Weight updates
    W1_next = W1 - xk_mb.transpose(-1,-2) @ grad_l_wrt_Z1
    b1_next = b1 - grad_l_wrt_Z1.sum(dim=-2, keepdim=True)

    W2_next = W2 - X2.transpose(-1,-2) @ grad_l_wrt_Z2
    b2_next = b2 - grad_l_wrt_Z2.sum(dim=-2, keepdim=True)

    Z1_bar = xq_mb @ W1_next + b1_next
    X2_bar = torch.nn.functional.gelu(Z1_bar, approximate="tanh")
    Z2_bar = X2_bar @ W2_next + b2_next


    return Z2_bar, W1_next, b1_next, W2_next, b2_next


# def main():
#     # Define shapes
#     seq_len = 64
#     mini_batch_size = 64
#     head_dim = 64
#     expansion_dim = 256
#     shard_size = 4

#     dtype = torch.float32
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Create inputs
#     xq = torch.randn(seq_len // mini_batch_size, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
#     xk = torch.randn(seq_len // mini_batch_size, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
#     xv = torch.randn(seq_len // mini_batch_size, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

#     W1_pt = torch.randn(head_dim, expansion_dim, dtype=dtype, device=device).contiguous()
#     b1_pt = torch.randn(1, expansion_dim, dtype=dtype, device=device).contiguous()
#     W2_pt = torch.randn(expansion_dim, head_dim, dtype=dtype, device=device).contiguous()
#     b2_pt = torch.randn(1, head_dim, dtype=dtype, device=device).contiguous()

#     # Create output buffers
#     Z2_bar_pt = torch.zeros(seq_len, head_dim, dtype=dtype, device=device).contiguous()
#     Z2_bar_pt_shard = torch.zeros(seq_len, head_dim, dtype=dtype, device=device).contiguous()

#     # Compute mini-batches for PyTorch
#     for i in range(seq_len // mini_batch_size):
#         seq_idx = i * mini_batch_size

#         xq_mb = xq[i]
#         xk_mb = xk[i]
#         xv_mb = xv[i]

#         Z2_bar_pt_shard[seq_idx:seq_idx+mini_batch_size], _, _, _, _= compute_mini_batch_shard(W1_pt, b1_pt, W2_pt, b2_pt, xq_mb, xk_mb, xv_mb, shard_size)
#         Z2_bar_pt[seq_idx:seq_idx+mini_batch_size], W1_pt, b1_pt, W2_pt, b2_pt = compute_mini_batch(W1_pt, b1_pt, W2_pt, b2_pt, xq_mb, xk_mb, xv_mb)

#     # Compare outputs
#     print("Comparing Outputs")
#     print("PT Shape: ", Z2_bar_pt.shape)
#     print("PT Shard Shape: ", Z2_bar_pt_shard.shape)
#     compare_outputs(Z2_bar_pt, Z2_bar_pt_shard, "Z2_bar: Non-Sharded vs Sharded")

def main():
    torch.manual_seed(0)
    # Define shapes
    B = 2
    NH = 2
    K = 2
    seq_len = 128
    mini_batch_size = 64
    NC = seq_len // mini_batch_size

    head_dim = 64
    expansion_dim = 256
    shard_size = 4

    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create inputs
    xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    xk_clone = xk.clone().contiguous()
    xv = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    # Create inputs
    # xq = torch.ones(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    # xk = torch.ones(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    # xk_clone = xk.clone().contiguous()
    # xv = torch.ones(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    # num_elements = B * NH * NC * mini_batch_size * head_dim
    # xq = torch.arange(num_elements, dtype=dtype, device=device).reshape(B, NH, NC, mini_batch_size, head_dim).contiguous()
    # xk = torch.arange(num_elements, dtype=dtype, device=device).reshape(B, NH, NC, mini_batch_size, head_dim).contiguous()
    # xk_clone = xk.clone().contiguous()
    # xv = torch.arange(num_elements, dtype=dtype, device=device).reshape(B, NH, NC, mini_batch_size, head_dim).contiguous()

    W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
    b1 = torch.zeros(B, NH, 1, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
    W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=dtype, device=device).contiguous() * 0.02
    b2 = torch.zeros(B, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02


    W1_checkpoints = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=dtype, device=device).contiguous()
    b1_checkpoints = torch.empty(B, NH, K, 1, expansion_dim, dtype=dtype, device=device).contiguous()
    W2_checkpoints = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=dtype, device=device).contiguous()
    b2_checkpoints = torch.empty(B, NH, K, 1, head_dim, dtype=dtype, device=device).contiguous()

    # Create output buffers
    output_ref = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    Z2_bar_pt_shard = torch.zeros(NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    output_tk = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()


    thunderkittens.ttt_forward(
        xq,
        xk_clone,
        xv,
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

    W1_curr, b1_curr, W2_curr, b2_curr = (W1, b1, W2, b2)

    # Compute mini-batches for PyTorch
    for i in range(NC):
        seq_idx = i * mini_batch_size

        xq_mb = xq[:,:,i]
        xk_mb = xk[:,:,i]
        xv_mb = xv[:,:,i]

        # Z2_bar_pt_shard[i], W1_other, b1_other, W2_other, b2_other = compute_mini_batch_shard(W1[0][0], b1[0][0], W2[0][0], b2[0][0], xq_mb[0][0], xk_mb[0][0], xv_mb[0][0], shard_size)
        output_ref[:, :, i], W1_curr, b1_curr, W2_curr, b2_curr = compute_mini_batch_no_dual(W1_curr, b1_curr, W2_curr, b2_curr, xq_mb, xk_mb, xv_mb)
        
        breakpoint()
    breakpoint()

    # # Compare outputs
    # print("Comparing Outputs")
    # print("PT Shape: ", Z2_bar_pt.shape)
    # print("PT Shard Shape: ", Z2_bar_pt_shard.shape)
    # compare_outputs(Z2_bar_pt, Z2_bar_pt_shard, "Z2_bar: Non-Sharded vs Sharded")





if __name__ == "__main__":
    main()