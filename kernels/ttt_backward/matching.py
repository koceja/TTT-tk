import torch
# import thunderkittens
import multiprocessing
import time

from tabulate import tabulate

def compare_all_grads(gradients_to_compare):
    results = []

    for grad_hw, grad_ref, name in gradients_to_compare:
        # Calculate absolute differences and statistics
        abs_diff = torch.abs(grad_hw - grad_ref)
        max_diff = torch.max(abs_diff).item()
        median_diff = torch.median(abs_diff).item()
        
        # Calculate relative errors, avoiding division by zero
        with torch.no_grad():
            nonzero_mask = grad_ref != 0
            relative_error = torch.zeros_like(grad_ref)
            relative_error[nonzero_mask] = abs_diff[nonzero_mask] / torch.abs(grad_ref[nonzero_mask])
            
            median_relative_error = torch.median(relative_error[nonzero_mask]).item() if torch.any(nonzero_mask) else float('inf')

        # Format values to 10 digits total
        results.append((
            name,
            f"{max_diff:.10g}",
            f"{median_diff:.10g}",
            f"{median_relative_error:.10g}"
        ))

    # Prepare data for tabulation
    table_data = [["Metric", "Max Difference", "Median Difference", "Median Relative Error"]] + results

    # Print the results using tabulate
    print("\nComparison Results:")
    print(tabulate(table_data, headers="firstrow", tablefmt="pretty"))


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

    return (
        XQW_mini_batch, 
        W1_next, 
        b1_next, 
        W2_next, 
        b2_next,
        Z1,
        std_fused,
        x_hat_fused,
        grad_output_fused,
        grad_x_hat_fused,
        grad_l_wrt_Z2,
        grad_l_wrt_Z1,
        # Dual Form
        X2,
        Z1_bar,
        X2_bar,
        # LN
        std_ln,
        x_hat_ln,
    )


def get_backward_inputs(B, NH, NC, CS, F, seed=0):
    torch.manual_seed(seed)

    XQ_batch = torch.randn(B, NH, NC, CS, F, device="cuda")
    XK_batch = torch.randn(B, NH, NC, CS, F, device="cuda")
    Z1 = torch.randn(B, NH, NC, CS, F * 4, device="cuda")

    W1_init = torch.randn(B, NH, F, F * 4, device="cuda")
    b1_init = torch.randn(B, NH, 1, F * 4, device="cuda")
    W2_init = torch.randn(B, NH, F * 4, F, device="cuda")
    b2_init = torch.randn(B, NH, 1, F, device="cuda")
    
    ttt_norm_weight = torch.randn(NH, F, device="cuda")
    ttt_norm_bias = torch.randn(NH, F, device="cuda")
    
    std_fused = torch.randn(B, NH, NC, CS, 1, device="cuda")
    x_hat_fused = torch.randn(B, NH, NC, CS, F, device="cuda")
    grad_output_fused = torch.randn(B, NH, NC, CS, F, device="cuda")
    grad_x_hat_fused = torch.randn(B, NH, NC, CS, F, device="cuda")
    grad_l_wrt_Z2 = torch.randn(B, NH, NC, CS, F, device="cuda")
    grad_l_wrt_Z1 = torch.randn(B, NH, NC, CS, F * 4, device="cuda")
    
    eta_batch = torch.randn(B, NH, NC, CS, CS, device="cuda")
    Attn1 = torch.randn(B, NH, NC, CS, CS, device="cuda")
    Attn2 = torch.randn(B, NH, NC, CS, CS, device="cuda")
    X2 = torch.randn(B, NH, NC, CS, F * 4, device="cuda")
    Z1_bar = torch.randn(B, NH, NC, CS, F * 4, device="cuda")
    X2_bar = torch.randn(B, NH, NC, CS, F * 4, device="cuda")
    Z2_bar = torch.randn(B, NH, NC, CS, F * 4, device="cuda")
    
    std_ln = torch.randn(B, NH, NC, CS, 1, device="cuda")
    x_hat_ln = torch.randn(B, NH, NC, CS, F, device="cuda")

    grad_L_W1_last = torch.randn(B, NH, F, F * 4, device="cuda")
    grad_L_b1_last = torch.randn(B, NH, 1, F * 4, device="cuda")
    grad_L_W2_last = torch.randn(B, NH, F * 4, F, device="cuda")
    grad_L_b2_last = torch.randn(B, NH, 1, F, device="cuda")
    grad_L_XQW_batch = torch.randn(B, NH, NC, CS, F, device="cuda")

    return (
        XQ_batch,
        XK_batch,
        Z1,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        # LnFusedL2BWD
        ttt_norm_weight,
        ttt_norm_bias,
        std_fused,
        x_hat_fused,
        grad_output_fused,
        grad_x_hat_fused,
        grad_l_wrt_Z2,
        grad_l_wrt_Z1,
        # Dual Form
        eta_batch,
        Attn1,
        Attn2,
        X2,
        Z1_bar,
        X2_bar,
        Z2_bar,
        # LN
        std_ln,
        x_hat_ln,
        grad_L_W1_last,
        grad_L_b1_last,
        grad_L_W2_last,
        grad_L_b2_last,
        grad_L_XQW_batch,
    )



def backward(
    # MatMul
    XQ_mini_batch,
    XK_mini_batch,
    Z1,
    W1_init,
    b1_init,
    W2_init,
    b2_init,
    W2_last,
    W1_last,
    # LnFusedL2BWD
    ln_weight,
    ln_bias,
    std_fused,
    x_hat_fused,
    grad_output_fused,
    grad_x_hat_fused,
    grad_l_wrt_Z2,
    grad_l_wrt_Z1,
    # Dual Form
    eta_mini_batch,
    X2,
    Z1_bar,
    X2_bar,
    # LN
    std_ln,
    x_hat_ln,
    ttt_norm_weight,
    ttt_norm_bias,
    grad_L_W1_last,
    grad_L_b1_last,
    grad_L_W2_last,
    grad_L_b2_last,
    grad_L_XQW_mini_batch,
):

    head_dim = XQ_mini_batch.shape[-1]
    
    # Stage 4: LN
    grad_L_ln_bias_ln = grad_L_XQW_mini_batch.sum(dim=-2, keepdim=True).sum(dim=0)
    grad_L_ln_weight_ln = (grad_L_XQW_mini_batch * x_hat_ln).sum(dim=-2, keepdim=True).sum(dim=0)
    grad_L_x_hat_ln = grad_L_XQW_mini_batch * ln_weight

    grad_L_Z2_bar = (
        (1.0 / head_dim)
        * (
            head_dim * grad_L_x_hat_ln
            - grad_L_x_hat_ln.sum(dim=-1, keepdim=True)
            - x_hat_ln * (grad_L_x_hat_ln * x_hat_ln).sum(dim=-1, keepdim=True)
        )
        / std_ln
    )


    # Stage 3: Dual Form
    
    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]

    grad_L_X2_bar = grad_L_Z2_bar @ W2_last.transpose(-2, -1)

    grad_L_Z1_bar = grad_L_X2_bar * gelu_bwd(Z1_bar)

    grad_L_b2_last += grad_L_Z2_bar.sum(dim=-2, keepdim=True)
    grad_L_W2_last += X2_bar.transpose(-2, -1) @ grad_L_Z2_bar
    grad_L_b1_last += grad_L_Z1_bar.sum(dim=-2, keepdim=True)
    grad_L_W1_last += XQ_mini_batch.transpose(-2, -1) @ grad_L_Z1_bar

    grad_L_grad_l_wrt_Z1 = (
        - (last_eta_mini_batch * XK_mini_batch) @ grad_L_W1_last
        - last_eta_mini_batch * grad_L_b1_last
    )
    grad_L_grad_l_wrt_Z2 = (
        - (last_eta_mini_batch * X2) @ grad_L_W2_last
        - last_eta_mini_batch * grad_L_b2_last
        + (grad_L_grad_l_wrt_Z1 * gelu_bwd(Z1)) @ W2_init
    )

    grad_L_XQ_mini_batch = grad_L_Z1_bar @ W1_last.transpose(-2, -1)

    grad_L_XK_mini_batch = (
        - (grad_l_wrt_Z1 @ grad_L_W1_last.transpose(-2, -1)) * last_eta_mini_batch
    )

    grad_L_last_eta_in_mini_batch = (
        -((grad_l_wrt_Z2 @ grad_L_W2_last.transpose(-2, -1)) * X2).sum(dim=-1, keepdim=True)
        - (grad_L_b2_last * grad_l_wrt_Z2).sum(dim=-1, keepdim=True)
        - ((grad_l_wrt_Z1 @ grad_L_W1_last.transpose(-2, -1)) * XK_mini_batch).sum(
            dim=-1, keepdim=True
        )
        - (grad_L_b1_last * grad_l_wrt_Z1).sum(dim=-1, keepdim=True)
    )

    grad_L_eta_mini_batch = (
        torch.nn.functional.pad(grad_L_last_eta_in_mini_batch.transpose(-2, -1), (0, 0, 63, 0))
    ) # Might not need this var anymore

    # Stage 2: LnFusedL2BWD
    grad_L_W2_init = (grad_L_grad_l_wrt_Z1 * gelu_bwd(Z1)).transpose(-2, -1) @ grad_l_wrt_Z2

    grad_L_grad_x_hat_fused = (1.0 / std_fused) * (
        grad_L_grad_l_wrt_Z2 + 
        (-1.0 / head_dim) * (
            (grad_L_grad_l_wrt_Z2 ).sum(dim=-1, keepdim=True) + 
            x_hat_fused * (grad_L_grad_l_wrt_Z2 * x_hat_fused).sum(dim=-1, keepdim=True)
        )
    )

    grad_L_y = ln_weight * grad_L_grad_x_hat_fused

    grad_L_ln_weight_fused = (
        (grad_output_fused * grad_L_grad_x_hat_fused + grad_L_y * x_hat_fused).sum(dim=-2, keepdim=True).sum(dim=0)
    )
    grad_L_ln_bias_fused = grad_L_y.sum(dim=-2, keepdim=True).sum(dim=0)

    grad_L_x_hat_fused = (
        grad_L_y * ln_weight
        + (-1.0 / (head_dim * std_fused))
        * (
            grad_x_hat_fused * (grad_L_grad_l_wrt_Z2 * x_hat_fused).sum(dim=-1, keepdim=True)
            + grad_L_grad_l_wrt_Z2 * (grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True)
        )
    )

    # grad_L_std = -grad_L_x_hat_fused * ((x_hat_fused) / std_fused) - grad_L_grad_l_wrt_Z2 * (
    #     grad_l_wrt_Z2 * std_fused
    # ) / (std_fused**2)
    grad_L_std = (-grad_L_x_hat_fused * ((x_hat_fused)) - grad_L_grad_l_wrt_Z2 *
        grad_l_wrt_Z2
    ) / std_fused

    # grad_L_Z2 = (
    #     grad_L_x_hat_fused * (1.0 / std_fused)
    #     - (1.0 / head_dim) * grad_L_x_hat_fused.sum(dim=-1, keepdim=True) * (1.0 / std_fused)
    #     + (1.0 / head_dim) * (grad_L_std).sum(dim=-1, keepdim=True) * x_hat_fused
    # )

    grad_L_Z2 = (
        grad_L_x_hat_fused * (1.0 / std_fused)
        + (1.0 / head_dim) * ((grad_L_std).sum(dim=-1, keepdim=True) * x_hat_fused
        - grad_L_x_hat_fused.sum(dim=-1, keepdim=True) * (1.0 / std_fused) )
    )

    grad_L_reconstruction_target = -ln_weight * grad_L_grad_x_hat_fused

    grad_L_ttt_norm_weight = grad_L_ln_weight_ln.reshape(ttt_norm_weight.shape) + grad_L_ln_weight_fused.reshape(
        ttt_norm_weight.shape
    )
    grad_L_ttt_norm_bias = grad_L_ln_bias_ln.reshape(ttt_norm_bias.shape) + grad_L_ln_bias_fused.reshape(
        ttt_norm_bias.shape
    )

    # Stage 1: MatMul
    grad_L_X2 = (
        grad_L_Z2 @ W2_init.transpose(-2, -1)
        - (grad_l_wrt_Z2 @ grad_L_W2_last.transpose(-2, -1)) * last_eta_mini_batch
    )
        
    grad_L_Z1 = grad_L_X2 * gelu_bwd(Z1) + (
            grad_l_wrt_Z2 @ W2_init.transpose(-2, -1)
        ) * grad_L_grad_l_wrt_Z1 * gelu_bwd_derivative(Z1)

    grad_L_XQ = grad_L_XQW_mini_batch + grad_L_XQ_mini_batch
    grad_L_XK = -grad_L_reconstruction_target + grad_L_XK_mini_batch + grad_L_Z1 @ W1_init.transpose(-2, -1)
    grad_L_XV = grad_L_reconstruction_target
    
    grad_L_W2_states = grad_L_W2_last + grad_L_W2_init + X2.transpose(-2, -1) @ grad_L_Z2
    grad_L_b2_states = grad_L_b2_last + grad_L_Z2.sum(-2, keepdim=True)
    grad_L_W1_states = grad_L_W1_last + XK_mini_batch.transpose(-2, -1) @ grad_L_Z1
    grad_L_b1_states = grad_L_b1_last + grad_L_Z1.sum(-2, keepdim=True)

    grad_L_eta = grad_L_eta_mini_batch
    
    return (
        grad_L_ttt_norm_weight,
        grad_L_ttt_norm_bias,
        grad_L_W1_states,
        grad_L_b1_states,
        grad_L_W2_states,
        grad_L_b2_states,
        grad_L_XQ,
        grad_L_XV,
        grad_L_XK,
        grad_L_eta,
    )



def match_backward_pytorch():

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
    # device = 'cpu'

    def get_inputs(dtype):
        torch.manual_seed(0)
        # Create inputs
        xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xv = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

        eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous() * 0.1
        last_eta = eta[:, :, :, -1, :, None].repeat(1, 1, 1, 1, head_dim).contiguous()

        ttt_norm_weight = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02
        ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

        W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        b1 = torch.zeros(B, NH, 1, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02
        b2 = torch.zeros(B, NH, 1, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02

        return xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    
    XQ_batch, XK_batch, XV_batch, eta_batch, last_eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init = get_inputs(torch.float32)

    XQ_batch.requires_grad_(True)
    XK_batch.requires_grad_(True)
    XV_batch.requires_grad_(True)
    eta_batch.requires_grad_(True)
    last_eta.requires_grad_(True)
    ttt_norm_weight.requires_grad_(True)
    ttt_norm_bias.requires_grad_(True)
    W1_init.requires_grad_(True)
    b1_init.requires_grad_(True)
    W2_init.requires_grad_(True)
    b2_init.requires_grad_(True)

    xq_mb = XQ_batch[:,:,0,:,:]
    xk_mb = XK_batch[:,:,0,:,:]
    xv_mb = XV_batch[:,:,0,:,:]
    eta_mb = eta_batch[:,:,0,:,:]
    
    (
        XQW_mini_batch, 
        W1_last, 
        b1_last, 
        W2_last, 
        b2_last,
        Z1,
        std_fused,
        x_hat_fused,
        grad_output_fused,
        grad_x_hat_fused,
        grad_l_wrt_Z2,
        grad_l_wrt_Z1,
        X2,
        Z1_bar,
        X2_bar,
        std_ln,
        x_hat_ln
    ) = compute_mini_batch_no_dual(
        W1_init, 
        b1_init, 
        W2_init, 
        b2_init, 
        xq_mb, 
        xk_mb, 
        xv_mb, 
        eta_mb,
        ttt_norm_weight,
        ttt_norm_bias
    )

    XQW_mini_batch.requires_grad_(True)
    XQW_mini_batch.retain_grad()

    loss = (XQW_mini_batch - torch.randn_like(XQW_mini_batch)).sum()
    loss.backward()

    (
        grad_L_ttt_norm_weight,
        grad_L_ttt_norm_bias,
        grad_L_W1_states,
        grad_L_b1_states,
        grad_L_W2_states,
        grad_L_b2_states,
        grad_L_XQ,
        grad_L_XV,
        grad_L_XK,
        grad_L_eta,
    ) = backward(
        # MatMul
        xq_mb,
        xk_mb,
        Z1,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        W2_last,
        W1_last,
        # LnFusedL2BWD
        ttt_norm_weight,
        ttt_norm_bias,
        std_fused,
        x_hat_fused,
        grad_output_fused,
        grad_x_hat_fused,
        grad_l_wrt_Z2,
        grad_l_wrt_Z1,
        # Dual Form
        eta_mb,
        X2,
        Z1_bar,
        X2_bar,
        # LN
        std_ln,
        x_hat_ln,
        ttt_norm_weight,
        ttt_norm_bias,
        torch.zeros_like(W1_last),
        torch.zeros_like(b1_last),
        torch.zeros_like(W2_last),
        torch.zeros_like(b2_last),
        XQW_mini_batch.grad,
    )

    compare_all_grads([
        [grad_L_W1_states, W1_init.grad, "W1"],
        [grad_L_b1_states, b1_init.grad, "b1"],
        [grad_L_W2_states, W2_init.grad, "W2"],
        [grad_L_b2_states, b2_init.grad, "b2"],
        [grad_L_ttt_norm_weight, ttt_norm_weight.grad, "ln_weight"],
        [grad_L_ttt_norm_bias, ttt_norm_bias.grad, "ln_bias"],
        [grad_L_XQ, XQ_batch.grad[:,:,0], "XQ"],
        [grad_L_XV, XV_batch.grad[:,:,0], "XV"],
        [grad_L_XK, XK_batch.grad[:,:,0], "XK"],
        [grad_L_eta, eta_batch.grad[:,:,0], "ETA"],]
    )


def main():
    match_backward_pytorch()
    # torch.manual_seed(0)
    # # Define shapes
    # B = 1
    # NH = 48
    # K = 2
    
    # seq_len = 256
    # mini_batch_size = 64
    # CS = mini_batch_size
    # NC = seq_len // mini_batch_size
    # checkpoint_group_size = NC // K

    # head_dim = 64
    # F = head_dim
    # expansion_dim = 256
    # shard_size = 4

    # dtype = torch.bfloat16
    # full_dtype = torch.float32
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # device = 'cpu'

    # def get_inputs(dtype):
    #     torch.manual_seed(0)
    #     # Create inputs
    #     xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    #     xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    #     xv = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

    #     eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous() * 0.1
    #     last_eta = eta[:, :, :, -1, :, None].contiguous()

    #     ttt_norm_weight = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02
    #     ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

    #     W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    #     b1 = torch.randn(B, NH, 1, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    #     W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    #     b2 = torch.randn(B, NH, 1, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02

    #     return xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    
    # XQ_batch, XK_batch, XV_batch, eta_batch, last_eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init = get_inputs(torch.float32)

    # XQ_batch = XQ_batch.to(torch.bfloat16).contiguous()
    # XK_batch = XK_batch.to(torch.bfloat16).contiguous()
    # XV_batch = XV_batch.to(torch.bfloat16).contiguous()
    # last_eta = last_eta.to(torch.bfloat16).contiguous()

    # W1_checkpoints = torch.randn(B, NH, K, head_dim, expansion_dim, dtype=full_dtype, device=device).contiguous()
    # b1_checkpoints = torch.randn(B, NH, K, 1, expansion_dim, dtype=full_dtype, device=device).contiguous()
    # W2_checkpoints = torch.randn(B, NH, K, expansion_dim, head_dim, dtype=full_dtype, device=device).contiguous()
    # b2_checkpoints = torch.randn(B, NH, K, 1, head_dim, dtype=full_dtype, device=device).contiguous()
    # output_tk = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    # output_ref = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

    # grad_L_W1_last = torch.randn(B, NH, head_dim, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    # grad_L_b1_last = torch.randn(B, NH, 1, expansion_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    # grad_L_W2_last = torch.randn(B, NH, expansion_dim, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    # grad_L_b2_last = torch.randn(B, NH, 1, head_dim, dtype=torch.float32, device=device).contiguous() * 0.02
    # grad_L_XQW_mini_batch = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=torch.bfloat16, device=device).contiguous() * 0.02

    # grad_L_ttt_norm_weight = torch.empty(1, NH, 1, head_dim, dtype=full_dtype, device=device).contiguous()
    # grad_L_ttt_norm_bias = torch.empty(1, NH, 1, head_dim, dtype=full_dtype, device=device).contiguous()
    # grad_L_W1_init = torch.empty(B, NH, head_dim, expansion_dim, dtype=torch.float32, device=device).contiguous()
    # grad_L_b1_init = torch.empty(B, NH, 1, expansion_dim, dtype=torch.float32, device=device).contiguous()
    # grad_L_W2_init = torch.empty(B, NH, expansion_dim, head_dim, dtype=torch.float32, device=device).contiguous()
    # grad_L_b2_init = torch.empty(B, NH, 1, head_dim, dtype=torch.float32, device=device).contiguous()
    # grad_L_XQ = torch.empty(B, NH, NC, mini_batch_size, head_dim, dtype=torch.bfloat16, device=device).contiguous()
    # grad_L_XK = torch.empty(B, NH, NC, mini_batch_size, head_dim, dtype=torch.bfloat16, device=device).contiguous()
    # grad_L_XV = torch.empty(B, NH, NC, mini_batch_size, head_dim, dtype=torch.bfloat16, device=device).contiguous()
    # grad_L_last_eta = torch.empty_like(last_eta).contiguous()

    # # TK rematted values
    # W1_init_group = torch.empty(B, NH, checkpoint_group_size, F, F * 4, device=device, dtype=torch.float32)
    # b1_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F * 4, device=device, dtype=torch.float32)
    # W2_init_group = torch.empty(B, NH, checkpoint_group_size, F * 4, F, device=device, dtype=torch.float32)
    # b2_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F, device=device, dtype=torch.float32)

    # x_hat_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=dtype)
    # std_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=torch.float32)

    # X2_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=dtype)
    # Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=dtype)
    # Z1_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=dtype)
    # X2_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=dtype)

    # grad_l_wrt_Z2_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=dtype)
    # grad_l_wrt_Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=dtype)
    # x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=dtype)
    # grad_x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=dtype)
    # grad_output_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=dtype)
    # std_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=torch.float32)

    # # Ref rematted values
    # W1_init_group_ref = torch.empty(B, NH, checkpoint_group_size, F, F * 4, device=device, dtype=torch.float32)
    # b1_init_group_ref = torch.empty(B, NH, checkpoint_group_size, 1, F * 4, device=device, dtype=torch.float32)
    # W2_init_group_ref = torch.empty(B, NH, checkpoint_group_size, F * 4, F, device=device, dtype=torch.float32)
    # b2_init_group_ref = torch.empty(B, NH, checkpoint_group_size, 1, F, device=device, dtype=torch.float32)

    # x_hat_ln_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.float32)
    # std_ln_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=torch.float32)

    # X2_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.float32)
    # Z1_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.float32)
    # Z1_bar_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.float32)
    # X2_bar_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.float32)

    # grad_l_wrt_Z2_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.float32)
    # grad_l_wrt_Z1_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.float32)
    # x_hat_fused_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.float32)
    # grad_x_hat_fused_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.float32)
    # grad_output_fused_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.float32)
    # std_fused_group_ref = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=torch.float32)





    # thunderkittens.ttt_backward(
    #     # Inputs
    #     XQ_batch,
    #     XK_batch,
    #     XV_batch,
    #     last_eta,
    #     ttt_norm_weight,
    #     ttt_norm_bias,
    #     W1_init,
    #     b1_init,
    #     W2_init,
    #     b2_init,
    #     # Checkpoints
    #     W1_checkpoints,
    #     b1_checkpoints,
    #     W2_checkpoints,
    #     b2_checkpoints,
    #     output_tk,
    #     # Rematted Buffers
    #     W1_init_group,
    #     b1_init_group,
    #     W2_init_group,
    #     b2_init_group,
    #     x_hat_ln_group,
    #     std_ln_group,
    #     X2_group,
    #     Z1_group,
    #     Z1_bar_group,
    #     X2_bar_group,
    #     grad_l_wrt_Z2_group,
    #     grad_l_wrt_Z1_group,
    #     x_hat_fused_group,
    #     grad_x_hat_fused_group,
    #     grad_output_fused_group,
    #     std_fused_group,
    #     # Upstream grads
    #     grad_L_W1_last,
    #     grad_L_b1_last,
    #     grad_L_W2_last,
    #     grad_L_b2_last,
    #     grad_L_XQW_mini_batch,
    #     # Output grads
    #     grad_L_ttt_norm_weight,
    #     grad_L_ttt_norm_bias,
    #     grad_L_W1_init,
    #     grad_L_b1_init,
    #     grad_L_W2_init,
    #     grad_L_b2_init,
    #     grad_L_last_eta,
    #     grad_L_XQ,
    #     grad_L_XK,
    #     grad_L_XV
    # )

    # XQ_batch, XK_batch, XV_batch, eta_batch, last_eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init = get_inputs(torch.float32)

    

    # # Compute mini-batches for PyTorch
    # for checkpoint_idx in range(K-1, -1, -1):

    #     W1_curr, b1_curr, W2_curr, b2_curr = (
    #         W1_checkpoints[:, :, checkpoint_idx], 
    #         b1_checkpoints[:, :, checkpoint_idx], 
    #         W2_checkpoints[:, :, checkpoint_idx], 
    #         b2_checkpoints[:, :, checkpoint_idx]
    #     )

    #     for i in range(checkpoint_group_size):
    #         global_mini_batch_idx = checkpoint_idx * checkpoint_group_size + i
    #         xq_mb = XQ_batch[:,:,global_mini_batch_idx]
    #         xk_mb = XK_batch[:,:,global_mini_batch_idx]
    #         xv_mb = XV_batch[:,:,global_mini_batch_idx]
    #         eta_mb = eta_batch[:, :, global_mini_batch_idx]

    #         (
    #             output_ref[:, :, global_mini_batch_idx],
    #             W1_curr,
    #             b1_curr,
    #             W2_curr,
    #             b2_curr,
    #             Z1_group_ref[:,:,i],
    #             std_fused_group_ref[:,:,i],
    #             x_hat_fused_group_ref[:,:,i],
    #             grad_output_fused_group_ref[:,:,i],
    #             grad_x_hat_fused_group_ref[:,:,i],
    #             grad_l_wrt_Z2_group_ref[:,:,i],
    #             grad_l_wrt_Z1_group_ref[:,:,i],
    #             # Dual Form
    #             X2_group_ref[:,:,i],
    #             Z1_bar_group_ref[:,:,i],
    #             X2_bar_group_ref[:,:,i],
    #             # LN
    #             std_ln_group_ref[:,:,i],
    #             x_hat_ln_group_ref[:,:,i],
    #         ) = compute_mini_batch_no_dual(
    #             W1_curr, 
    #             b1_curr, 
    #             W2_curr, 
    #             b2_curr, 
    #             xq_mb, 
    #             xk_mb, 
    #             xv_mb, 
    #             eta_mb,
    #             ttt_norm_weight,
    #             ttt_norm_bias
    #         )

    #         W1_init_group_ref[:,:,i] = W1_curr
    #         b1_init_group_ref[:,:,i] = b1_curr
    #         W2_init_group_ref[:,:,i] = W2_curr
    #         b2_init_group_ref[:,:,i] = b2_curr



    # # breakpoint()

    # gradients_to_compare = [
    #     # W1_init_group
    #     (W1_init_group, W1_init_group_ref, "W1_init_group"),
    #     (b1_init_group, b1_init_group_ref, "b1_init_group"),
    #     (W2_init_group, W2_init_group_ref, "W2_init_group"),
    #     (b2_init_group, b2_init_group_ref, "b2_init_group"),

    #     # Layer Norm Groups
    #     (x_hat_ln_group, x_hat_ln_group_ref, "x_hat_ln_group"),
    #     (std_ln_group, std_ln_group_ref, "std_ln_group"),

    #     # Intermediate Computations
    #     (X2_group, X2_group_ref, "X2_group"),
    #     (Z1_group, Z1_group_ref, "Z1_group"),
    #     (Z1_bar_group, Z1_bar_group_ref, "Z1_bar_group"),
    #     (X2_bar_group, X2_bar_group_ref, "X2_bar_group"),

    #     # Gradients
    #     (grad_l_wrt_Z2_group, grad_l_wrt_Z2_group_ref, "grad_l_wrt_Z2_group"),
    #     (grad_l_wrt_Z1_group, grad_l_wrt_Z1_group_ref, "grad_l_wrt_Z1_group"),
    #     (x_hat_fused_group, x_hat_fused_group_ref, "x_hat_fused_group"),
    #     (grad_x_hat_fused_group, grad_x_hat_fused_group_ref, "grad_x_hat_fused_group"),
    #     (grad_output_fused_group, grad_output_fused_group_ref, "grad_output_fused_group"),
    #     (std_fused_group, std_fused_group_ref, "std_fused_group")
    # ]

    # # Execute the comparison
    # compare_all_grads(gradients_to_compare)




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

if __name__ == '__main__':
    is_lock_protected = False

    if not is_lock_protected:
        main()
    else:
        try:
            kernel_with_timeout(main, timeout=15)
        except RuntimeError as e:
            print(e)
