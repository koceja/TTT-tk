import torch

def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

def gelu_bwd_derivative(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))

    term1 = 0.79788456
    term2 = 6 * 0.79788456 * 0.044715 * x**2
    term3 = x * tanh_out * (0.79788456 + 3 * 0.79788456 * 0.044715 * x**2) ** 2

    derivative = (1 - tanh_out**2) * (term1 + term2 - term3)
    return derivative

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
    
def compute_mini_batch_shard(
    # Inputs
    W1, b1, W2, b2, xq_mb, xk_mb, xv_mb,
    # Intermediates
    Z1, grad_l_wrt_Z1, grad_l_wrt_Z2, 
    Attn1, Attn2, X2, Z1_bar, X2_bar,
    # Upstream gradients
    grad_L_W1_last, grad_L_b1_last, 
    grad_L_W2_last, grad_L_b2_last,
    grad_L_Z2_bar, shard_size
):
    """
    Sharded mini batch backward for TTT MLP.

    Excludes:
    - ILR
    - LayerNorm
    - Residual connection
    """
    # Simulate shard
    F, K = W1.shape
    mb_size = xq_mb.shape[0]
    assert K % shard_size == 0

    W1_sharded = W1.reshape(F, shard_size, K // shard_size).permute(1, 0, 2)
    b1_sharded = b1.reshape(1, shard_size, K // shard_size).permute(1, 0, 2)
    W2_sharded = W2.reshape(shard_size, K // shard_size, F)

    grad_L_W1_last_sharded = grad_L_W1_last.reshape(F, shard_size, K // shard_size).permute(1, 0, 2)
    grad_L_b1_last_sharded = grad_L_b1_last.reshape(1, shard_size, K // shard_size).permute(1, 0, 2)
    grad_L_W2_last_sharded = grad_L_W2_last.reshape(shard_size, K // shard_size, F)

    Z1_sharded = Z1.reshape(mb_size, shard_size, K // shard_size).permute(1, 0, 2)
    Z1_bar_sharded = Z1_bar.reshape(mb_size, shard_size, K // shard_size).permute(1, 0, 2)
    grad_l_wrt_Z1_sharded = grad_l_wrt_Z1.reshape(mb_size, shard_size, K // shard_size).permute(1, 0, 2)
    X2_sharded = X2.reshape(mb_size, shard_size, K // shard_size).permute(1, 0, 2)
    X2_bar_sharded = X2_bar.reshape(mb_size, shard_size, K // shard_size).permute(1, 0, 2)

    # Dual form
    grad_L_eta_Attn2 = torch.tril(grad_L_Z2_bar @ grad_l_wrt_Z2.T) # [CS, F] @ [F, CS] --> Not sharded
    grad_L_X2_bar = grad_L_Z2_bar @ W2_sharded.transpose(-2, -1) - grad_L_eta_Attn2 @ X2_sharded # [CS, F] @ [TP, F, Shard] - [CS, F] @ [TP, F, Shard]
    
    grad_L_Z1_bar = grad_L_X2_bar * gelu_bwd(Z1_bar_sharded) # [TP, CS, Shard]
    grad_L_eta_Attn1 = torch.tril(grad_L_Z1_bar @ grad_l_wrt_Z1_sharded.transpose(-2, -1)) # [TP, CS, Shard] @ [TP, Shard, CS]

    # Gradients of gradients
    grad_L_grad_l_wrt_Z1 = - (
        grad_L_Z1_bar # [TP, CS, Shard]
        + Attn1.T @ grad_L_Z1_bar # [CS, CS] @ [TP, CS, Shard]
        + xk_mb @ grad_L_W1_last_sharded # [CS, F] @ [TP, F, Shard]
        + grad_L_b1_last_sharded # [TP, 1, Shard]
    )
    grad_L_grad_l_wrt_Z2 = - (
        X2_sharded @ grad_L_W2_last_sharded # [TP, CS, Shard] @ [TP, Shard, F]
        + (grad_L_grad_l_wrt_Z1 * gelu_bwd(Z1_sharded)) @ W2_sharded # [TP, CS, Shard] @ [TP, Shard, F]
    )

    # Reduction and add additional terms
    grad_L_grad_l_wrt_Z2 = grad_L_grad_l_wrt_Z2.sum(dim=0) - (
        grad_L_Z2_bar
        + Attn2.T @ grad_L_Z2_bar
        + grad_L_b2_last
    )    
    grad_L_Z2 = grad_L_grad_l_wrt_Z2 # TODO: Verify
    grad_L_xv_mb = grad_L_Z2 # TODO: Verify

    # Inputs
    grad_L_xq_mb = (
        grad_L_Z1_bar @ W1_sharded.transpose(-2, -1) # [TP, CS, Shard] @ [TP, Shard, F]
        - grad_L_eta_Attn1 @ xk_mb # [TP, CS, CS] @ [CS, F]
    )
    grad_L_xk_mb = (
        grad_l_wrt_Z1_sharded @ grad_L_W1_last_sharded.transpose(-2, -1) # [TP, CS, Shard] @ [TP, Shard, F]
        - grad_L_eta_Attn1.transpose(-2, -1) @ xq_mb # [TP, CS, CS] @ [CS, F]
    )

    # Hidden state forward
    grad_L_X2 = (
        grad_l_wrt_Z2 @ grad_L_W2_last_sharded.transpose(-2, -1) # [CS, F] @ [TP, F, Shard]
        - grad_L_eta_Attn2.T @ X2_bar_sharded # [CS, CS] @ [TP, CS, Shard]
        + grad_L_Z2 @ W2_sharded.transpose(-2, -1) # [CS, F] @ [TP, F, Shard]
    )
    grad_L_Z1 = (
        grad_L_X2 * gelu_bwd(Z1_sharded) # [TP, CS, Shard]
        + grad_l_wrt_Z2 @ W2_sharded.transpose(-2, -1) # [CS, F] @ [TP, F, Shard]
    ) * grad_L_grad_l_wrt_Z1 * gelu_bwd_derivative(Z1_sharded)

    # Atomic Add
    grad_L_xv_mb += torch.sum(grad_L_Z1 @ W1_sharded.transpose(-2, -1), dim=0) # [CS, F]

    # Hidden state updates
    grad_L_W1_sharded = grad_L_W1_last_sharded + (
        xq_mb.T @ grad_L_Z1_bar # [CS, F] @ [TP, F, Shard]
        + xk_mb.T @ grad_L_Z1 # [CS, F] @ [TP, F, Shard]
    )
    grad_L_b1_sharded = grad_L_b1_last_sharded + (
        grad_L_Z1_bar.sum(dim=1, keepdim=True) # [TP, 1, Shard]
        + grad_L_Z1.sum(dim=1, keepdim=True) # [TP, 1, Shard]
    )
    grad_L_W2_sharded = grad_L_W2_last_sharded + (
        X2_bar_sharded.transpose(-2, -1) @ grad_L_Z2_bar # [TP, Shard, CS] @ [CS, F]
        + (grad_L_grad_l_wrt_Z1 * gelu_bwd(Z1_sharded)).transpose(-2, -1) @ grad_l_wrt_Z2 # [TP, Shard, CS] @ [CS, F]
        + X2_sharded.transpose(-2, -1) @ grad_L_Z2 # [TP, Shard, CS] @ [CS, F]
    )

    # Not sharded
    grad_L_b2 = grad_L_b2_last + (
        grad_L_Z2_bar.sum(dim=0, keepdim=True) # [1, F]
        + grad_L_Z2.sum(dim=0, keepdim=True) # [1, F]
    )

    # Reshape for return
    grad_L_W1 = grad_L_W1_sharded.permute(1, 0, 2).reshape(F, -1)
    grad_L_b1 = grad_L_b1_sharded.permute(1, 0, 2).reshape(1, -1)
    grad_L_W2 = grad_L_W2_sharded.permute(1, 0, 2).reshape(K, -1)

    grad_L_xq_mb = grad_L_xq_mb.sum(dim=0)
    grad_L_xk_mb = grad_L_xk_mb.sum(dim=0)

    return grad_L_xq_mb, grad_L_xk_mb, grad_L_xv_mb, grad_L_W1, grad_L_b1, grad_L_W2, grad_L_b2

def compute_mini_batch(
    
    # Inputs
    W1, b1, W2, b2, xq_mb, xk_mb, xv_mb,
    # Intermediates
    Z1, grad_l_wrt_Z1, grad_l_wrt_Z2, 
    Attn1, Attn2, X2, Z1_bar, X2_bar,
    # Upstream gradients
    grad_L_W1_last, grad_L_b1_last, 
    grad_L_W2_last, grad_L_b2_last,
    grad_L_Z2_bar
):
    """
    Mini batch backward for TTT MLP.

    Excludes:
    - ILR
    - LayerNorm
    - Residual connection
    """
    # Dual form
    grad_L_eta_Attn2 = torch.tril(grad_L_Z2_bar @ grad_l_wrt_Z2.T) # [CS, F] @ [F, CS]
    grad_L_X2_bar = grad_L_Z2_bar @ W2.T - grad_L_eta_Attn2 @ X2 # [CS, F] @ [F, K] - [CS, F] @ [F, K]

    grad_L_Z1_bar = grad_L_X2_bar * gelu_bwd(Z1_bar) # [CS, K]
    grad_L_eta_Attn1 = torch.tril(grad_L_Z1_bar @ grad_l_wrt_Z1.T) # [CS, K] @ [K, CS]

    # Gradients of gradients
    grad_L_grad_l_wrt_Z1 = - (
        grad_L_Z1_bar # [CS, K]
        + Attn1.T @ grad_L_Z1_bar # [CS, CS] @ [CS, K]
        + xk_mb @ grad_L_W1_last # [CS, F] @ [F, K]
        + grad_L_b1_last # [1, K]
    )
    grad_L_grad_l_wrt_Z2 = - (
        grad_L_Z2_bar # [CS, F]
        + Attn2.T @ grad_L_Z2_bar # [CS, CS] @ [CS, F]
        + X2 @ grad_L_W2_last # [CS, K] @ [K, F]
        + grad_L_b2_last # [1, F]
        + (grad_L_grad_l_wrt_Z1 * gelu_bwd(Z1)) @ W2 # [CS, K] @ [K, F]
    )

    # Inputs
    grad_L_xq_mb = (
        grad_L_Z1_bar @ W1.T # [CS, K] @ [K, F]
        - grad_L_eta_Attn1 @ xk_mb # [CS, CS] @ [CS, F]
    )
    grad_L_xk_mb = (
        grad_l_wrt_Z1 @ grad_L_W1_last.T # [CS, K] @ [F, K]
        - grad_L_eta_Attn1.T @ xq_mb # [CS, CS] @ [CS, F]
    )

    # Hidden state forward
    grad_L_Z2 = grad_L_grad_l_wrt_Z2 # TODO: Verify
    grad_L_xv_mb = grad_L_Z2 # TODO: Verify
    
    grad_L_X2 = (
        grad_l_wrt_Z2 @ grad_L_W2_last.T # [CS, F] @ [F, K]
        - grad_L_eta_Attn2.T @ X2_bar # [CS, CS] @ [CS, K]
        + grad_L_Z2 @ W2.T # [CS, F] @ [F, K]
    )
    grad_L_Z1 = (
        grad_L_X2 * gelu_bwd(Z1) # [CS, K]
        + grad_l_wrt_Z2 @ W2.T # [CS, F] @ [F, K]
    ) * grad_L_grad_l_wrt_Z1 * gelu_bwd_derivative(Z1) # [CS, K]

    grad_L_xv_mb += grad_L_Z1 @ W1.T # [CS, K] @ [K, F]

    # Hidden state updates
    grad_L_W1 = grad_L_W1_last + (
        xq_mb.T @ grad_L_Z1_bar # [F, CS] @ [CS, K]
        + xk_mb.T @ grad_L_Z1 # [F, CS] @ [CS, K]
    )
    grad_L_b1 = grad_L_b1_last + (
        grad_L_Z1_bar.sum(dim=0, keepdim=True) # [1, K]
        + grad_L_Z1.sum(dim=0, keepdim=True) # [1, K]
    )
    grad_L_W2 = grad_L_W2_last + (
        X2_bar.T @ grad_L_Z2_bar # [K, CS] @ [CS, F]
        + (grad_L_grad_l_wrt_Z1 * gelu_bwd(Z1)).T @ grad_l_wrt_Z2 # [K, CS] @ [CS, F]
        + X2.T @ grad_L_Z2 # [K, CS] @ [CS, F]
    )
    grad_L_b2 = grad_L_b2_last + (
        grad_L_Z2_bar.sum(dim=0, keepdim=True) # [1, F]
        + grad_L_Z2.sum(dim=0, keepdim=True) # [1, F]
    )

    return grad_L_xq_mb, grad_L_xk_mb, grad_L_xv_mb, grad_L_W1, grad_L_b1, grad_L_W2, grad_L_b2

def main():
    # Define shapes
    seq_len = 64
    mb_size = 64
    head_dim = 64
    exp_dim = 256
    shard_size = 4

    n_mb = seq_len // mb_size

    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create inputs
    xq = torch.randn(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()
    xk = torch.randn(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()
    xv = torch.randn(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()

    W1_pt = torch.randn(head_dim, exp_dim, dtype=dtype, device=device).contiguous()
    b1_pt = torch.randn(1, exp_dim, dtype=dtype, device=device).contiguous()
    W2_pt = torch.randn(exp_dim, head_dim, dtype=dtype, device=device).contiguous()
    b2_pt = torch.randn(1, head_dim, dtype=dtype, device=device).contiguous()

    # Create intermediates
    Z1 = torch.randn(mb_size, exp_dim, dtype=dtype, device=device).contiguous()
    Z1_bar = torch.randn(mb_size, exp_dim, dtype=dtype, device=device).contiguous()
    grad_l_wrt_Z1 = torch.randn(mb_size, exp_dim, dtype=dtype, device=device).contiguous()
    grad_l_wrt_Z2 = torch.randn(mb_size, head_dim, dtype=dtype, device=device).contiguous()
    Attn1 = torch.randn(mb_size, mb_size, dtype=dtype, device=device).contiguous()
    Attn2 = torch.randn(mb_size, mb_size, dtype=dtype, device=device).contiguous()
    X2 = torch.randn(mb_size, exp_dim, dtype=dtype, device=device).contiguous()
    X2_bar = torch.randn(mb_size, exp_dim, dtype=dtype, device=device).contiguous()

    # Create upstream gradients
    grad_L_W1_last = torch.randn(head_dim, exp_dim, dtype=dtype, device=device).contiguous()
    grad_L_b1_last = torch.randn(1, exp_dim, dtype=dtype, device=device).contiguous()
    grad_L_W2_last = torch.randn(exp_dim, head_dim, dtype=dtype, device=device).contiguous()
    grad_L_b2_last = torch.randn(1, head_dim, dtype=dtype, device=device).contiguous()
    grad_L_Z2_bar = torch.randn(seq_len, head_dim, dtype=dtype, device=device).contiguous()

    # Create output buffers
    grad_L_xq = torch.zeros(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()
    grad_L_xk = torch.zeros(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()
    grad_L_xv = torch.zeros(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()

    grad_L_xq_shard = torch.zeros(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()
    grad_L_xk_shard = torch.zeros(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()
    grad_L_xv_shard = torch.zeros(n_mb, mb_size, head_dim, dtype=dtype, device=device).contiguous()

    # Compute mini-batches for PyTorch
    for i in range(n_mb - 1, -1, -1):
        xq_mb, xk_mb, xv_mb = xq[i], xk[i], xv[i]

        grad_L_xq_shard[i], grad_L_xk_shard[i], grad_L_xv_shard[i], \
        _, _, _, _ =compute_mini_batch_shard(
            W1_pt, b1_pt, W2_pt, b2_pt, xq_mb, xk_mb, xv_mb,
            Z1, grad_l_wrt_Z1, grad_l_wrt_Z2, Attn1, Attn2, X2, Z1_bar, X2_bar,
            grad_L_W1_last, grad_L_b1_last, grad_L_W2_last, grad_L_b2_last, grad_L_Z2_bar, shard_size
        )

        grad_L_xq[i], grad_L_xk[i], grad_L_xv[i], \
        grad_L_W1_last, grad_L_b1_last, grad_L_W2_last, grad_L_b2_last = compute_mini_batch(
            W1_pt, b1_pt, W2_pt, b2_pt, xq_mb, xk_mb, xv_mb,
            Z1, grad_l_wrt_Z1, grad_l_wrt_Z2, Attn1, Attn2, X2, Z1_bar, X2_bar,
            grad_L_W1_last, grad_L_b1_last, grad_L_W2_last, grad_L_b2_last, grad_L_Z2_bar
        )

    # Compare outputs
    print("Comparing Outputs")
    compare_outputs(grad_L_xq_shard, grad_L_xq, "XQ Shard: Non-Sharded vs Sharded")
    compare_outputs(grad_L_xk_shard, grad_L_xk, "XK Shard: Non-Sharded vs Sharded")
    compare_outputs(grad_L_xv_shard, grad_L_xv, "XV Shard: Non-Sharded vs Sharded")

if __name__ == "__main__":
    main()
