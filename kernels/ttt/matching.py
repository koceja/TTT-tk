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




# def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
#     "Batch backward for LayerNorm fused with L2 loss."
#     D = x.shape[-1]

#     # Mean and variance computation
#     mu = x.mean(dim=-1, keepdim=True)
#     var = x.var(dim=-1, keepdim=True, unbiased=False)

#     # Normalization
#     std = torch.sqrt(var + eps)
#     x_hat = (x - mu) / std

#     # Scale and shift
#     y = gamma * x_hat + beta

#     grad_output = y - l2_target
#     grad_x_hat = grad_output * gamma
#     z = (
#         (1.0 / D)
#         * (
#             D * grad_x_hat
#             - grad_x_hat.sum(dim=-1, keepdim=True)
#             - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
#         )
#         / std
#     )

#     return z



# def forward(
#     ttt_norm_weight,
#     ttt_norm_bias,
#     W1_init,
#     b1_init,
#     W2_init,
#     b2_init,
#     XQ_mini_batch,
#     XV_mini_batch,
#     XK_mini_batch,
#     eta_mini_batch,
#     num_heads,
#     head_dim,
# ):
#     # Stage 1: MatMul
#     Z1 = XK_mini_batch @ W1_init + b1_init
#     X2 = F.gelu(Z1, approximate="tanh")
#     Z2 = X2 @ W2_init + b2_init
#     reconstruction_target = XV_mini_batch - XK_mini_batch

#     ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
#     ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

#     # Stage 2: LnFusedL2BWD
#     eps = 1e-6
#     mu_fused = Z2.mean(dim=-1, keepdim=True)
#     var_fused = Z2.var(dim=-1, keepdim=True, unbiased=False)

#     std_fused = torch.sqrt(var_fused + eps)
#     x_hat_fused = (Z2 - mu_fused) / std_fused

#     y = ln_weight * x_hat_fused + ln_bias
#     grad_output_fused = y - reconstruction_target
#     grad_x_hat_fused = grad_output_fused * ln_weight

#     grad_l_wrt_Z2 = (
#         (1.0 / head_dim)
#         * (
#             head_dim * grad_x_hat_fused
#             - grad_x_hat_fused.sum(dim=-1, keepdim=True)
#             - x_hat_fused * (grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True)
#         )
#         / std_fused
#     )

#     grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

#     # # Stage 3: Dual Form
#     # Attn1 = torch.tril(XQ_mini_batch @ XK_mini_batch.transpose(-2, -1))
#     # b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
#     # Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

#     # X2_bar = F.gelu(Z1_bar, approximate="tanh")

#     # Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
#     # b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
#     # Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

#     last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]

#     W1_last = W1_init - (last_eta_mini_batch * XK_mini_batch).transpose(-1, -2) @ grad_l_wrt_Z1
#     b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)

#     W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
#     b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

#     # Stage 4: LN
#     mu_ln = Z2_bar.mean(dim=-1, keepdim=True)
#     var_ln = Z2_bar.var(dim=-1, keepdim=True, unbiased=False)
#     std_ln = torch.sqrt(var_ln + eps)
#     x_hat_ln = (Z2_bar - mu_ln) / std_ln

#     Z2_bar_ln = ln_weight * x_hat_ln + ln_bias

#     XQW_mini_batch = XQ_mini_batch + Z2_bar_ln

#     return W1_last, b1_last, W2_last, b2_last, XQW_mini_batch


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


    reconstruction_target = xv_mb.to(torch.float32) - xk_mb.to(torch.float32)

    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

    # Stage 2: LnFusedL2BWD

    eps = 1e-6
    Z2 = Z2.to(torch.float32)
    mu_fused = Z2.mean(dim=-1, keepdim=True)
    var_fused = Z2.var(dim=-1, keepdim=True, unbiased=False)

    std_fused = torch.sqrt(var_fused + eps)
    x_hat_fused = (Z2 - mu_fused) / std_fused

    ln_weight = ln_weight.to(torch.float32)
    ln_bias = ln_bias.to(torch.float32)
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

    grad_l_wrt_Z2 = grad_l_wrt_Z2.to(torch.bfloat16)

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
    B = 1
    NH = 1
    K = 1
    
    seq_len = 64
    mini_batch_size = 64
    NC = seq_len // mini_batch_size
    checkpoint_group_size = NC // K

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

    eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous()
    last_eta = eta[:, :, :, -1, :, None].repeat(1, 1, 1, 1, head_dim).contiguous()

    ttt_norm_weight = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous()
    ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous()

    W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
    b1 = torch.zeros(B, NH, 1, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
    W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=dtype, device=device).contiguous() * 0.02
    b2 = torch.zeros(B, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02


    W1_checkpoints = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=dtype, device=device).contiguous()
    b1_checkpoints = torch.empty(B, NH, K, 1, expansion_dim, dtype=dtype, device=device).contiguous()
    W2_checkpoints = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=dtype, device=device).contiguous()
    b2_checkpoints = torch.empty(B, NH, K, 1, head_dim, dtype=dtype, device=device).contiguous()

    W1_checkpoints_ref = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=dtype, device=device).contiguous()
    b1_checkpoints_ref = torch.empty(B, NH, K, 1, expansion_dim, dtype=dtype, device=device).contiguous()
    W2_checkpoints_ref = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=dtype, device=device).contiguous()
    b2_checkpoints_ref = torch.empty(B, NH, K, 1, head_dim, dtype=dtype, device=device).contiguous()

    # Create output buffers
    output_ref = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    Z2_bar_pt_shard = torch.zeros(NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    output_tk = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()


    thunderkittens.ttt_forward(
        xq,
        xk_clone,
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

    W1_curr, b1_curr, W2_curr, b2_curr = (W1, b1, W2, b2)

    # Compute mini-batches for PyTorch
    for i in range(NC):
        if i % checkpoint_group_size == 0:
            checkpoint_idx = i // checkpoint_group_size
            W1_checkpoints_ref[:, :, checkpoint_idx] = W1_curr
            b1_checkpoints_ref[:, :, checkpoint_idx] = b1_curr
            W2_checkpoints_ref[:, :, checkpoint_idx] = W2_curr
            b2_checkpoints_ref[:, :, checkpoint_idx] = b2_curr

        xq_mb = xq[:,:,i]
        xk_mb = xk[:,:,i]
        xv_mb = xv[:,:,i]
        eta_mb = eta[:, :, i]

        # Z2_bar_pt_shard[i], W1_other, b1_other, W2_other, b2_other = compute_mini_batch_shard(W1[0][0], b1[0][0], W2[0][0], b2[0][0], xq_mb[0][0], xk_mb[0][0], xv_mb[0][0], shard_size)
        (
            output_ref[:, :, i],
            W1_curr,
            b1_curr,
            W2_curr,
            b2_curr
        ) = compute_mini_batch_no_dual(
            W1_curr, 
            b1_curr, 
            W2_curr, 
            b2_curr, 
            xq_mb, 
            xk_mb, 
            xv_mb, 
            eta_mb,
            ttt_norm_weight,
            ttt_norm_bias
        )
 
    breakpoint()

    # # Compare outputs
    # print("Comparing Outputs")
    # print("PT Shape: ", Z2_bar_pt.shape)
    # print("PT Shard Shape: ", Z2_bar_pt_shard.shape)
    # compare_outputs(Z2_bar_pt, Z2_bar_pt_shard, "Z2_bar: Non-Sharded vs Sharded")





if __name__ == "__main__":
    main()