import torch
import torchtitan.models.tk.thunderkittens as tk
from einops import einsum

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

def compute_mini_batch(W1, W2, xq_mb, xk_mb, xv_mb):
    """
    Mini batch forward for TTT MLP. 
    - All operations outside of hidden state update are in bfloat16
    - Hidden state update is in float32

    xq_mb: [B, H, CS, F]
    xk_mb: [B, H, CS, F]
    xv_mb: [B, H, CS, F]
    W1: [B, H, F, K]
    W2: [B, H, K, F]

    Dimension Key:
    B: Batch size
    H: Num of heads
    CS: Mini-batch size
    F: Head dimension
    K: Expansion dimension

    Excludes:
    - Bias
    - ILR
    - LayerNorm
    - GeLU
    - Residual connection
    """
    input_dtype = xq_mb.dtype
    hidden_state_dtype = W1.dtype

    # Inner model forward
    Z1 = einsum(xk_mb, W1.to(input_dtype), 'B H CS F, B H F K -> B H CS K')
    Z2 = einsum(Z1, W2.to(input_dtype), 'B H CS K, B H K F -> B H CS F')

    # L2 loss gradient
    grad_l_wrt_Z2 = xv_mb - Z2
    grad_l_wrt_Z1 = einsum(grad_l_wrt_Z2, W2.to(input_dtype), 'B H CS F, B H K F -> B H CS K')

    # Dual form
    Attn1 = torch.tril(einsum(xq_mb, xk_mb, 'B H CS1 F, B H CS2 F -> B H CS1 CS2'))
    Z1_bar = (
        einsum(xq_mb, W1.to(input_dtype), 'B H CS F, B H F K -> B H CS K') - 
        einsum(Attn1, grad_l_wrt_Z1, 'B H CS1 CS2, B H CS2 K -> B H CS1 K')
    )

    Attn2 = torch.tril(einsum(Z1_bar, Z1, 'B H CS1 K, B H CS2 K -> B H CS1 CS2'))
    Z2_bar = (
        einsum(Z1_bar, W2.to(input_dtype), 'B H CS K, B H K F -> B H CS F') -
        einsum(Attn2, grad_l_wrt_Z2, 'B H CS1 CS2, B H CS2 F -> B H CS1 F')
    )

    # Weight update
    W1_bar = W1 - einsum(xk_mb, grad_l_wrt_Z1, 'B H CS F, B H CS K -> B H F K').to(hidden_state_dtype)
    W2_bar = W2 - einsum(Z1, grad_l_wrt_Z2, 'B H CS K, B H CS F -> B H K F').to(hidden_state_dtype)

    return Z2_bar, W1_bar, W2_bar

def main():
    # Define shapes
    batch_size = 4
    num_heads = 12
    seq_len = 32
    mini_batch_size = 16
    head_dim = 64
    expansion_dim = 4

    hidden_state_dtype = torch.float32
    input_dtype = torch.bfloat16
    output_dtype = torch.bfloat16

    # Create inputs
    xq = torch.randn(batch_size, num_heads, seq_len // mini_batch_size, mini_batch_size, head_dim, dtype=input_dtype, device='cuda').contiguous()
    xk = torch.randn(batch_size, num_heads, seq_len // mini_batch_size, mini_batch_size, head_dim, dtype=input_dtype, device='cuda').contiguous()
    xv = torch.randn(batch_size, num_heads, seq_len // mini_batch_size, mini_batch_size, head_dim, dtype=input_dtype, device='cuda').contiguous()

    W1_pt = torch.randn(batch_size, num_heads, head_dim, expansion_dim, dtype=hidden_state_dtype, device='cuda').contiguous()
    W2_pt = torch.randn(batch_size, num_heads, expansion_dim, head_dim, device='cuda').contiguous()
    W1_tk = W1_pt.clone()
    W2_tk = W2_pt.clone()

    # Create output buffers
    Z2_bar_pt = torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=output_dtype, device='cuda').contiguous()
    Z2_bar_tk = torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=output_dtype, device='cuda').contiguous()

    # Compute mini-batches for PyTorch
    for i in range(seq_len // mini_batch_size):
        seq_idx = i * mini_batch_size

        xq_mb = xq[:, :, i]
        xk_mb = xk[:, :, i]
        xv_mb = xv[:, :, i]

        Z2_bar_pt[:, :, seq_idx:seq_idx+mini_batch_size], W1_pt, W2_pt = compute_mini_batch(W1_pt, W2_pt, xq_mb, xk_mb, xv_mb)

    # Compute mini-batches for ThunderKittens (TODO)

    # Compare outputs
    print("Comparing Outputs")
    print("PT Shape: ", Z2_bar_pt.shape)
    print("TK Shape: ", Z2_bar_tk.shape)
    compare_outputs(Z2_bar_tk, Z2_bar_pt, "Z2_bar")

if __name__ == "__main__":
    main()