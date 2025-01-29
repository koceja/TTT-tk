import triton
import triton.language as tl


@triton.jit
def gelu_triton(x):
    # Constants used in the GELU approximation
    sqrt_2_over_pi = tl.constexpr(0.79788456)
    coeff = sqrt_2_over_pi * (x + 0.044715 * x * x * x)

    # Compute GELU: 0.5 * x * (1 + tanh(coeff))
    tanh_approximation = (2 / (1 + tl.exp(-2 * coeff))) - 1

    gelu_out = 0.5 * x * (1.0 + tanh_approximation)

    return gelu_out


@triton.jit
def gelu_bwd_triton(x):
    # Constants used in the GELU derivative approximation
    sqrt_2_over_pi = tl.constexpr(0.79788456)
    coeff = sqrt_2_over_pi * x * (1 + 0.044715 * x * x)

    # Compute tanh component
    tanh_out = (2 / (1 + tl.exp(-2 * coeff))) - 1

    # Compute derivative: 0.5 * (1 + tanh_out) + 0.5 * x * (1 - tanh_out^2) * (sqrt_2_over_pi + 0.1070322243 * x * x)
    derivative = 0.5 * (1.0 + tanh_out) + 0.5 * x * (1.0 - tanh_out * tanh_out) * (
        sqrt_2_over_pi + 0.1070322243 * x * x
    )

    return derivative


@triton.jit
def gelu_bwd_derivative_triton(x):
    # Constants used in the GELU derivative approximation
    sqrt_2_over_pi = tl.constexpr(0.79788456)
    coeff = sqrt_2_over_pi * x * (1 + 0.044715 * x * x)

    # Compute tanh component
    tanh_out = (2 / (1 + tl.exp(-2 * coeff))) - 1

    term1 = 6 * sqrt_2_over_pi * 0.044715 * x * x
    term2 = sqrt_2_over_pi + 3 * sqrt_2_over_pi * 0.044715 * x * x
    term2 = x * tanh_out * (term2 * term2)

    derivative = (1 - tanh_out * tanh_out) * (sqrt_2_over_pi + term1 - term2)

    return derivative


@triton.jit
def ttt_mlp_stage_1(
    # Scan inputs
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_init_ptr,
    b1_init_ptr,
    W2_init_ptr,
    b2_init_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    # Intermediate buffers
    W1_init_group_ptr,
    b1_init_group_ptr,
    W2_init_group_ptr,
    b2_init_group_ptr,
    X2_group_ptr,
    Z1_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    x_hat_fused_group_ptr,
    grad_x_hat_fused_group_ptr,
    grad_output_fused_group_ptr,
    std_fused_group_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    F_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    i,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XV_batch_ptr.type.element_ty

    CS_stride = CS

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + i * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)

    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F_F4_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F4_F_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F * 4)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_F4_offset = (
        batch * NH * checkpoint_group_size * F4_stride
        + head * checkpoint_group_size * F4_stride
        + mini_batch_idx_in_group * F4_stride
        + tl.arange(0, 1)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F_offset = (
        batch * NH * checkpoint_group_size * F_stride
        + head * checkpoint_group_size * F_stride
        + mini_batch_idx_in_group * F_stride
        + tl.arange(0, 1)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_offset = (
        batch * NH * checkpoint_group_size * CS_stride
        + head * checkpoint_group_size * CS_stride
        + mini_batch_idx_in_group * CS_stride
        + tl.arange(0, CS)[:, None]
        + tl.arange(0, 1)[None, :]
    )

    # Stage 1: MatMul
    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)
    W1_init = tl.load(W1_init_ptr + F_F4_offset).to(tl.float32)
    b1_init = tl.load(b1_init_ptr + F4_offset).to(tl.float32)
    Z1 = tl.dot(XK_mini_batch.to(mp_dtype), W1_init.to(mp_dtype), allow_tf32=False) + b1_init
    tl.store(W1_init_group_ptr + G_F_F4_offset, W1_init)
    tl.store(b1_init_group_ptr + G_F4_offset, b1_init)

    X2 = gelu_triton(Z1)
    W2_init = tl.load(W2_init_ptr + F4_F_offset).to(tl.float32)
    b2_init = tl.load(b2_init_ptr + F_offset).to(tl.float32)
    Z2 = tl.dot(X2.to(mp_dtype), W2_init.to(mp_dtype), allow_tf32=False) + b2_init
    tl.store(W2_init_group_ptr + G_F4_F_offset, W2_init)
    tl.store(b2_init_group_ptr + G_F_offset, b2_init)

    XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset).to(tl.float32)
    reconstruction_target = XV_mini_batch - XK_mini_batch

    # Stage 2: LnFusedL2BWD
    mu_fused = (tl.sum(Z2, axis=1) / F)[:, None]
    var_fused = (tl.sum((Z2 - mu_fused) * (Z2 - mu_fused), axis=1) / F)[:, None]

    std_fused = tl.sqrt(var_fused + 1e-6)
    x_hat_fused = (Z2 - mu_fused) / std_fused

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset).to(tl.float32)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset).to(tl.float32)[None, :]
    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z2 = (
        (1.0 / F)
        * (
            F * grad_x_hat_fused
            - tl.sum(grad_x_hat_fused, axis=1)[:, None]
            - x_hat_fused * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        )
        / std_fused
    )

    grad_l_wrt_Z2_W2 = tl.dot(grad_l_wrt_Z2.to(mp_dtype), tl.trans(W2_init).to(mp_dtype), allow_tf32=False)
    grad_l_wrt_Z1 = grad_l_wrt_Z2_W2 * gelu_bwd_triton(Z1)

    tl.store(std_fused_group_ptr + G_CS_offset, std_fused)
    tl.store(grad_l_wrt_Z2_group_ptr + G_CS_F_offset, grad_l_wrt_Z2)
    tl.store(grad_l_wrt_Z1_group_ptr + G_CS_F4_offset, grad_l_wrt_Z1)
    tl.store(x_hat_fused_group_ptr + G_CS_F_offset, x_hat_fused)
    tl.store(grad_x_hat_fused_group_ptr + G_CS_F_offset, grad_x_hat_fused)
    tl.store(grad_output_fused_group_ptr + G_CS_F_offset, grad_output_fused)
    tl.store(X2_group_ptr + G_CS_F4_offset, X2)
    tl.store(Z1_group_ptr + G_CS_F4_offset, Z1)


@triton.jit
def ttt_mlp_stage_2(
    # Scan inputs
    W1_init_ptr,
    b1_init_ptr,
    XQ_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    # Intermediate buffers
    Attn1_group_ptr,
    Z1_bar_group_ptr,
    X2_bar_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    i,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XK_batch_ptr.type.element_ty

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + i * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )
    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]

    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_CS_CS_offset = (
        batch * NH * checkpoint_group_size * CS_CS_stride
        + head * checkpoint_group_size * CS_CS_stride
        + mini_batch_idx_in_group * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )

    # Stage 3: Dual Form
    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

    W1_init = tl.load(W1_init_ptr + F_F4_offset).to(tl.float32)
    b1_init = tl.load(b1_init_ptr + F4_offset).to(tl.float32)
    grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_F4_offset).to(tl.float32)
    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)

    W1_last = W1_init - tl.dot(
        tl.trans(last_eta_mini_batch * XK_mini_batch).to(mp_dtype), grad_l_wrt_Z1.to(mp_dtype), allow_tf32=False
    )
    b1_last = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]
    tl.store(W1_init_ptr + F_F4_offset, W1_last)
    tl.store(b1_init_ptr + F4_offset, b1_last)

    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
    XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset).to(tl.float32)

    Attn1 = tl.where(
        mask, tl.dot(XQ_mini_batch.to(mp_dtype), tl.trans(XK_mini_batch).to(mp_dtype), allow_tf32=False), 0
    )
    b1_bar = b1_init - tl.dot(
        tl.where(mask, eta_mini_batch, 0).to(mp_dtype), grad_l_wrt_Z1.to(mp_dtype), allow_tf32=False
    )
    Z1_bar = (
        tl.dot(XQ_mini_batch.to(mp_dtype), W1_init.to(mp_dtype), allow_tf32=False)
        - tl.dot((eta_mini_batch * Attn1).to(mp_dtype), grad_l_wrt_Z1.to(mp_dtype), allow_tf32=False)
        + b1_bar
    )
    X2_bar = gelu_triton(Z1_bar)

    tl.store(Z1_bar_group_ptr + G_CS_F4_offset, Z1_bar)
    tl.store(X2_bar_group_ptr + G_CS_F4_offset, X2_bar)
    tl.store(Attn1_group_ptr + G_CS_CS_offset, Attn1)


@triton.jit
def ttt_mlp_stage_3(
    # Scan inputs
    W2_init_ptr,
    b2_init_ptr,
    eta_batch_ptr,
    # Intermediate buffers
    x_hat_ln_group_ptr,
    std_ln_group_ptr,
    Attn2_group_ptr,
    X2_group_ptr,
    X2_bar_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    i,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = eta_batch_ptr.type.element_ty

    CS_stride = CS

    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + i * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )
    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_CS_offset = (
        batch * NH * checkpoint_group_size * CS_CS_stride
        + head * checkpoint_group_size * CS_CS_stride
        + mini_batch_idx_in_group * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    G_CS_offset = (
        batch * NH * checkpoint_group_size * CS_stride
        + head * checkpoint_group_size * CS_stride
        + mini_batch_idx_in_group * CS_stride
        + tl.arange(0, CS)[:, None]
        + tl.arange(0, 1)[None, :]
    )

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

    W2_init = tl.load(W2_init_ptr + F4_F_offset).to(tl.float32)
    b2_init = tl.load(b2_init_ptr + F_offset).to(tl.float32)

    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_offset).to(tl.float32)
    X2 = tl.load(X2_group_ptr + G_CS_F4_offset).to(tl.float32)

    W2_last = W2_init - tl.dot(
        tl.trans(last_eta_mini_batch * X2).to(mp_dtype), grad_l_wrt_Z2.to(mp_dtype), allow_tf32=False
    )
    b2_last = b2_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z2, axis=0)[None, :]
    tl.store(W2_init_ptr + F4_F_offset, W2_last)
    tl.store(b2_init_ptr + F_offset, b2_last)

    X2_bar = tl.load(X2_bar_group_ptr + G_CS_F4_offset).to(tl.float32)
    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)

    Attn2 = tl.where(mask, tl.dot(X2_bar.to(mp_dtype), tl.trans(X2).to(mp_dtype), allow_tf32=False), 0)
    b2_bar = b2_init - tl.dot(
        tl.where(mask, eta_mini_batch, 0).to(mp_dtype), grad_l_wrt_Z2.to(mp_dtype), allow_tf32=False
    )
    Z2_bar = (
        tl.dot(X2_bar.to(mp_dtype), W2_init.to(mp_dtype), allow_tf32=False)
        - tl.dot((eta_mini_batch * Attn2).to(mp_dtype), grad_l_wrt_Z2.to(mp_dtype), allow_tf32=False)
        + b2_bar
    )

    # Stage 4: LN
    mu_ln = tl.sum(Z2_bar, axis=1)[:, None] / F
    var_ln = tl.sum((Z2_bar - mu_ln) * (Z2_bar - mu_ln), axis=1)[:, None] / F
    std_ln = tl.sqrt(var_ln + 1e-6)
    x_hat_ln = (Z2_bar - mu_ln) / std_ln

    tl.store(x_hat_ln_group_ptr + G_CS_F_offset, x_hat_ln)
    tl.store(std_ln_group_ptr + G_CS_offset, std_ln)
    tl.store(Attn2_group_ptr + G_CS_CS_offset, Attn2)


@triton.jit
def ttt_mlp_backward_stage_1(
    ttt_norm_weight_ptr,
    # Upstream gradients
    grad_L_XQW_mini_batch_ptr,
    grad_L_W1_last_ptr,
    # Intermediate buffers
    x_hat_ln_group_ptr,
    std_ln_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    # Other stages
    grad_L_Z2_bar_ptr,
    grad_l_wrt_Z1_Last_ptr,
    # Output buffers
    grad_L_ttt_norm_weight_ptr,
    grad_L_ttt_norm_bias_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    F_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    # NOTE: No value passed in has accurate dtype
    mp_dtype = tl.bfloat16

    CS_stride = CS

    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )

    norm_offset = head * F_stride + tl.arange(0, F)
    norm_store_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + mini_batch_idx * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_offset = (
        batch * NH * checkpoint_group_size * CS_stride
        + head * checkpoint_group_size * CS_stride
        + mini_batch_idx_in_group * CS_stride
        + tl.arange(0, CS)[:, None]
        + tl.arange(0, 1)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )

    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    F_CS_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, F)[:, None] * CS + tl.arange(0, CS)[None, :]
    )

    # Stage 4: LN
    grad_L_XQW_mini_batch = tl.load(grad_L_XQW_mini_batch_ptr + CS_F_offset).to(tl.float32)
    x_hat_ln = tl.load(x_hat_ln_group_ptr + G_CS_F_offset).to(tl.float32)
    std_ln = tl.load(std_ln_group_ptr + G_CS_offset).to(tl.float32)

    grad_L_ln_weight_ln = tl.load(grad_L_ttt_norm_weight_ptr + norm_store_offset).to(tl.float32)
    grad_L_ln_weight_ln += tl.sum(grad_L_XQW_mini_batch * x_hat_ln, axis=0)
    tl.store(grad_L_ttt_norm_weight_ptr + norm_store_offset, grad_L_ln_weight_ln)

    grad_L_ln_bias_ln = tl.load(grad_L_ttt_norm_bias_ptr + norm_store_offset).to(tl.float32)
    grad_L_ln_bias_ln += tl.sum(grad_L_XQW_mini_batch, axis=0)
    tl.store(grad_L_ttt_norm_bias_ptr + norm_store_offset, grad_L_ln_bias_ln)

    grad_L_x_hat_ln = grad_L_XQW_mini_batch * ln_weight
    grad_L_Z2_bar = (
        (1.0 / F)
        * (
            F * grad_L_x_hat_ln
            - tl.sum(grad_L_x_hat_ln, axis=1)[:, None]
            - x_hat_ln * tl.sum(grad_L_x_hat_ln * x_hat_ln, axis=1)[:, None]
        )
        / std_ln
    )

    tl.store(grad_L_Z2_bar_ptr + CS_F_intermediate_offset, grad_L_Z2_bar)

    # Section IV
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F4_offset).to(tl.float32)
    grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_F4_offset).to(tl.float32)

    grad_l_wrt_Z1_Last = tl.dot(grad_L_W1_last.to(mp_dtype), tl.trans(grad_l_wrt_Z1).to(mp_dtype), allow_tf32=False)
    tl.store(grad_l_wrt_Z1_Last_ptr + F_CS_intermediate_offset, grad_l_wrt_Z1_Last)


@triton.jit
def ttt_mlp_backward_stage_2(
    XK_batch_ptr,
    eta_batch_ptr,
    # Upstream gradients
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    # Intermediate buffers
    W2_init_group_ptr,
    Attn1_group_ptr,
    Attn2_group_ptr,
    X2_group_ptr,
    Z1_bar_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    # Other stages
    grad_L_grad_l_wrt_Z1_ptr,
    grad_L_eta_Attn2_ptr,
    grad_L_Z1_bar_ptr,
    grad_L_Z2_bar_ptr,
    # Output buffers
    grad_L_eta_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XK_batch_ptr.type.element_ty

    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + mini_batch_idx * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F4_F_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F * 4)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_CS_offset = (
        batch * NH * checkpoint_group_size * CS_CS_stride
        + head * checkpoint_group_size * CS_CS_stride
        + mini_batch_idx_in_group * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )

    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    CS_CS_intermediate_offset = (
        batch * NH * CS_CS_stride + head * CS_CS_stride + tl.arange(0, CS)[:, None] * CS + tl.arange(0, CS)[None, :]
    )

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    # Stage 3: Dual Form
    # Section I
    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_offset).to(tl.float32)
    Attn2 = tl.load(Attn2_group_ptr + G_CS_CS_offset).to(tl.float32)

    grad_L_Z2_bar = tl.load(grad_L_Z2_bar_ptr + CS_F_intermediate_offset).to(tl.float32)

    grad_L_eta_Attn2 = tl.dot(grad_L_Z2_bar.to(mp_dtype), tl.trans(grad_l_wrt_Z2).to(mp_dtype), allow_tf32=False)

    tl.store(grad_L_eta_Attn2_ptr + CS_CS_intermediate_offset, grad_L_eta_Attn2)
    grad_L_eta_mini_batch = tl.load(grad_L_eta_ptr + CS_CS_offset).to(tl.float32)
    grad_L_eta_mini_batch += -tl.where(mask, grad_L_eta_Attn2, 0) - (Attn2 * grad_L_eta_Attn2)
    tl.store(grad_L_eta_ptr + CS_CS_offset, grad_L_eta_mini_batch)

    # Section II
    W2_init = tl.load(W2_init_group_ptr + G_F4_F_offset).to(tl.float32)
    Z1_bar = tl.load(Z1_bar_group_ptr + G_CS_F4_offset).to(tl.float32)
    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
    X2 = tl.load(X2_group_ptr + G_CS_F4_offset).to(tl.float32)

    grad_L_Z1_bar = -(
        tl.dot(
            tl.where(mask, grad_L_eta_Attn2 * eta_mini_batch, 0).to(mp_dtype),
            X2.to(mp_dtype),
            allow_tf32=False,
        )
        - tl.dot(grad_L_Z2_bar.to(mp_dtype), tl.trans(W2_init).to(mp_dtype), allow_tf32=False)
    ) * gelu_bwd_triton(Z1_bar)
    tl.store(grad_L_Z1_bar_ptr + CS_F4_intermediate_offset, grad_L_Z1_bar)

    # Section III
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F4_offset).to(tl.float32)
    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + F4_offset).to(tl.float32)
    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)
    Attn1 = tl.load(Attn1_group_ptr + G_CS_CS_offset).to(tl.float32)
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)

    grad_L_grad_l_wrt_Z1 = (
        -(
            tl.dot(
                tl.trans(tl.where(mask, eta_mini_batch, 0)).to(mp_dtype),
                grad_L_Z1_bar.to(mp_dtype),
                allow_tf32=False,
            )
        )
        - (tl.dot(tl.trans(eta_mini_batch * Attn1).to(mp_dtype), grad_L_Z1_bar.to(mp_dtype), allow_tf32=False))
        - (
            tl.dot(
                (last_eta_mini_batch * XK_mini_batch).to(mp_dtype),
                grad_L_W1_last.to(mp_dtype),
                allow_tf32=False,
            )
        )
        - (last_eta_mini_batch * grad_L_b1_last)
    )
    tl.store(grad_L_grad_l_wrt_Z1_ptr + CS_F4_intermediate_offset, grad_L_grad_l_wrt_Z1)


@triton.jit
def ttt_mlp_backward_stage_3(
    XQ_batch_ptr,
    eta_batch_ptr,
    # Upstream gradients
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    # Intermediate buffers
    grad_l_wrt_Z1_group_ptr,
    # Other stages
    grad_L_XK_mini_batch_ptr,
    grad_L_Z1_bar_ptr,
    grad_l_wrt_Z1_Last_ptr,
    grad_L_b1_init_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XQ_batch_ptr.type.element_ty

    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + mini_batch_idx * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )

    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    F_CS_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, F)[:, None] * CS + tl.arange(0, CS)[None, :]
    )

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]

    # # Section IV
    grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_F4_offset).to(tl.float32)
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F4_offset).to(tl.float32)

    # Section V
    grad_L_Z1_bar = tl.load(grad_L_Z1_bar_ptr + CS_F4_intermediate_offset).to(tl.float32)
    XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset).to(tl.float32)

    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + F4_offset).to(tl.float32)
    grad_L_W1_init = grad_L_W1_last + tl.dot(
        tl.trans(XQ_mini_batch).to(mp_dtype), grad_L_Z1_bar.to(mp_dtype), allow_tf32=False
    )
    grad_L_b1_init = grad_L_b1_last + tl.sum(grad_L_Z1_bar, axis=0)

    tl.store(grad_L_W1_last_ptr + F_F4_offset, grad_L_W1_init)
    tl.store(grad_L_b1_init_ptr + F4_offset, grad_L_b1_init)

    grad_L_eta_Attn1 = tl.dot(grad_L_Z1_bar.to(mp_dtype), tl.trans(grad_l_wrt_Z1).to(mp_dtype), allow_tf32=False)

    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)

    grad_l_wrt_Z1_Last = tl.load(grad_l_wrt_Z1_Last_ptr + F_CS_intermediate_offset).to(tl.float32)
    grad_L_XK_mini_batch = -(
        tl.trans(
            tl.dot(
                tl.trans(XQ_mini_batch).to(mp_dtype),
                tl.where(mask, grad_L_eta_Attn1 * eta_mini_batch, 0).to(mp_dtype),
                allow_tf32=False,
            )
        )
    )
    grad_L_XK_mini_batch -= tl.trans(grad_l_wrt_Z1_Last) * last_eta_mini_batch

    tl.store(
        grad_L_XK_mini_batch_ptr + CS_F_intermediate_offset,
        grad_L_XK_mini_batch,
    )


@triton.jit
def ttt_mlp_backward_stage_4(
    eta_batch_ptr,
    grad_L_W2_last_ptr,
    grad_L_b2_last_ptr,
    W2_init_group_ptr,
    Attn2_group_ptr,
    X2_group_ptr,
    Z1_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    grad_L_grad_l_wrt_Z2_ptr,
    grad_L_Z1_ptr,
    grad_L_Z2_bar_ptr,
    grad_L_grad_l_wrt_Z1_ptr,
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = tl.bfloat16

    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F4_F_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F * 4)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_CS_offset = (
        batch * NH * checkpoint_group_size * CS_CS_stride
        + head * checkpoint_group_size * CS_CS_stride
        + mini_batch_idx_in_group * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )

    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]

    # Section VI
    Z1 = tl.load(Z1_group_ptr + G_CS_F4_offset).to(tl.float32)
    W2_init = tl.load(W2_init_group_ptr + G_F4_F_offset).to(tl.float32)
    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_offset).to(tl.float32)
    grad_L_grad_l_wrt_Z1 = tl.load(grad_L_grad_l_wrt_Z1_ptr + CS_F4_intermediate_offset).to(tl.float32)
    grad_L_Z1 = (
        tl.dot(grad_l_wrt_Z2.to(mp_dtype), tl.trans(W2_init).to(mp_dtype), allow_tf32=False)
        * grad_L_grad_l_wrt_Z1
        * gelu_bwd_derivative_triton(Z1)
    )  # Partial
    tl.store(grad_L_Z1_ptr + CS_F4_intermediate_offset, grad_L_Z1)

    # Section VII
    grad_L_W2_last = tl.load(grad_L_W2_last_ptr + F4_F_offset).to(tl.float32)
    grad_L_b2_last = tl.load(grad_L_b2_last_ptr + F_offset).to(tl.float32)

    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)
    grad_L_Z2_bar = tl.load(grad_L_Z2_bar_ptr + CS_F_intermediate_offset).to(tl.float32)
    Attn2 = tl.load(Attn2_group_ptr + G_CS_CS_offset).to(tl.float32)
    X2 = tl.load(X2_group_ptr + G_CS_F4_offset).to(tl.float32)

    grad_L_grad_l_wrt_Z2 = tl.dot(
        (grad_L_grad_l_wrt_Z1 * gelu_bwd_triton(Z1)).to(mp_dtype),
        W2_init.to(mp_dtype),
        allow_tf32=False,
    )

    grad_L_grad_l_wrt_Z2 += (
        -(
            tl.dot(
                tl.trans(tl.where(mask, eta_mini_batch, 0)).to(mp_dtype),
                grad_L_Z2_bar.to(mp_dtype),
                allow_tf32=False,
            )
        )
        - (tl.dot(tl.trans(eta_mini_batch * Attn2).to(mp_dtype), grad_L_Z2_bar.to(mp_dtype), allow_tf32=False))
        - (tl.dot((last_eta_mini_batch * X2).to(mp_dtype), grad_L_W2_last.to(mp_dtype), allow_tf32=False))
        - (last_eta_mini_batch * grad_L_b2_last)
    )

    tl.store(grad_L_grad_l_wrt_Z2_ptr + CS_F_intermediate_offset, grad_L_grad_l_wrt_Z2)


@triton.jit
def ttt_mlp_backward_stage_5(
    XK_batch_ptr,
    eta_batch_ptr,
    # Upstream gradients
    grad_L_b1_last_ptr,
    grad_L_W2_last_ptr,
    grad_L_b2_last_ptr,
    grad_L_XQW_mini_batch_ptr,
    # Intermediate buffers
    W1_init_group_ptr,
    Attn1_group_ptr,
    X2_group_ptr,
    Z1_group_ptr,
    X2_bar_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    # Other stages
    grad_L_Z1_bar_ptr,
    grad_L_Z2_bar_ptr,
    grad_l_wrt_Z1_Last_ptr,
    grad_L_grad_l_wrt_Z1_ptr,
    grad_L_W2_init_ptr,
    # Output buffers
    grad_L_XQ_ptr,
    grad_L_eta_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XK_batch_ptr.type.element_ty

    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + mini_batch_idx * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F_F4_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_CS_CS_offset = (
        batch * NH * checkpoint_group_size * CS_CS_stride
        + head * checkpoint_group_size * CS_CS_stride
        + mini_batch_idx_in_group * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )

    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    F_CS_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, F)[:, None] * CS + tl.arange(0, CS)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]

    Z1 = tl.load(Z1_group_ptr + G_CS_F4_offset).to(tl.float32)
    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_offset).to(tl.float32)

    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
    grad_L_Z2_bar = tl.load(grad_L_Z2_bar_ptr + CS_F_intermediate_offset).to(tl.float32)
    X2 = tl.load(X2_group_ptr + G_CS_F4_offset).to(tl.float32)

    grad_L_W2_last = tl.load(grad_L_W2_last_ptr + F4_F_offset).to(tl.float32)
    grad_L_b2_last = tl.load(grad_L_b2_last_ptr + F_offset).to(tl.float32)

    grad_L_grad_l_wrt_Z1 = tl.load(grad_L_grad_l_wrt_Z1_ptr + CS_F4_intermediate_offset).to(tl.float32)

    grad_l_wrt_Z2_Last = tl.dot(grad_L_W2_last.to(mp_dtype), tl.trans(grad_l_wrt_Z2).to(mp_dtype), allow_tf32=False)

    # Store these
    X2_bar = tl.load(X2_bar_group_ptr + G_CS_F4_offset).to(tl.float32)
    grad_L_W2_init = (
        grad_L_W2_last
        + tl.dot(tl.trans(X2_bar).to(mp_dtype), grad_L_Z2_bar.to(mp_dtype), allow_tf32=False)
        + tl.dot(
            tl.trans(grad_L_grad_l_wrt_Z1 * gelu_bwd_triton(Z1)).to(mp_dtype),
            grad_l_wrt_Z2.to(mp_dtype),
            allow_tf32=False,
        )
    )
    grad_L_b2_init = grad_L_b2_last + tl.sum(grad_L_Z2_bar, axis=0)

    tl.store(grad_L_W2_init_ptr + F4_F_offset, grad_L_W2_init)
    tl.store(grad_L_b2_last_ptr + F_offset, grad_L_b2_init)

    grad_l_wrt_Z1_Last = tl.load(grad_l_wrt_Z1_Last_ptr + F_CS_intermediate_offset).to(tl.float32)
    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + F4_offset).to(tl.float32)
    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)
    grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_F4_offset).to(tl.float32)

    grad_L_last_eta_in_mini_batch = (
        -tl.sum((tl.trans(grad_l_wrt_Z2_Last) * X2), axis=1, keep_dims=True)
        - tl.sum(
            tl.sum(grad_L_b2_last, axis=0, keep_dims=True) * grad_l_wrt_Z2,
            axis=1,
            keep_dims=True,
        )
        - tl.sum(tl.trans(grad_l_wrt_Z1_Last) * XK_mini_batch, axis=1, keep_dims=True)
        - tl.sum(
            tl.sum(grad_L_b1_last, axis=0, keep_dims=True) * grad_l_wrt_Z1,
            axis=1,
            keep_dims=True,
        )
    )

    Attn1 = tl.load(Attn1_group_ptr + G_CS_CS_offset).to(tl.float32)
    grad_L_Z1_bar = tl.load(grad_L_Z1_bar_ptr + CS_F4_intermediate_offset).to(tl.float32)

    last_mini_batch_mask = tl.arange(0, CS)[:, None] == CS - 1
    grad_L_eta_Attn1 = tl.dot(grad_L_Z1_bar.to(mp_dtype), tl.trans(grad_l_wrt_Z1).to(mp_dtype), allow_tf32=False)

    grad_L_eta_mini_batch = tl.load(grad_L_eta_ptr + CS_CS_offset).to(tl.float32)
    grad_L_eta_mini_batch += -tl.where(mask, grad_L_eta_Attn1, 0) - (Attn1 * grad_L_eta_Attn1)
    grad_L_eta_mini_batch += tl.where(last_mini_batch_mask, tl.trans(grad_L_last_eta_in_mini_batch), 0)
    tl.store(grad_L_eta_ptr + CS_CS_offset, grad_L_eta_mini_batch)

    W1_init = tl.load(W1_init_group_ptr + G_F_F4_offset).to(tl.float32)
    grad_L_XQ_mini_batch = -(
        tl.dot(
            tl.where(mask, grad_L_eta_Attn1 * eta_mini_batch, 0).to(mp_dtype),
            XK_mini_batch.to(mp_dtype),
            allow_tf32=False,
        )
    ) + tl.dot(grad_L_Z1_bar.to(mp_dtype), tl.trans(W1_init).to(mp_dtype), allow_tf32=False)

    grad_L_XQW_mini_batch = tl.load(grad_L_XQW_mini_batch_ptr + CS_F_offset).to(tl.float32)

    tl.store(grad_L_XQ_ptr + CS_F_offset, grad_L_XQ_mini_batch + grad_L_XQW_mini_batch)


@triton.jit
def ttt_mlp_backward_stage_6(
    ttt_norm_weight_ptr,
    # Upstream gradients
    grad_L_b2_last_ptr,
    # Intermediate buffers
    X2_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    x_hat_fused_group_ptr,
    grad_x_hat_fused_group_ptr,
    grad_output_fused_group_ptr,
    std_fused_group_ptr,
    # Other stages
    grad_L_grad_l_wrt_Z2_ptr,
    grad_L_XK_mini_batch_ptr,
    grad_L_Z2_ptr,
    grad_L_W2_init_ptr,
    # Output buffers
    grad_L_ttt_norm_weight_ptr,
    grad_L_ttt_norm_bias_ptr,
    grad_L_XV_ptr,
    grad_L_XK_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = tl.bfloat16

    CS_stride = CS

    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    norm_offset = head * F_stride + tl.arange(0, F)
    norm_store_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + mini_batch_idx * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_CS_offset = (
        batch * NH * checkpoint_group_size * CS_stride
        + head * checkpoint_group_size * CS_stride
        + mini_batch_idx_in_group * CS_stride
        + tl.arange(0, CS)[:, None]
        + tl.arange(0, 1)[None, :]
    )

    # From stage 1 buffer
    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )

    # Stage 2: LnFusedL2BWD
    std_fused = tl.load(std_fused_group_ptr + G_CS_offset).to(tl.float32)
    x_hat_fused = tl.load(x_hat_fused_group_ptr + G_CS_F_offset).to(tl.float32)
    grad_L_grad_l_wrt_Z2 = tl.load(grad_L_grad_l_wrt_Z2_ptr + CS_F_intermediate_offset).to(tl.float32)

    grad_L_grad_x_hat_fused = (
        (1.0 / std_fused) * grad_L_grad_l_wrt_Z2
        + (1.0 / F) * tl.sum(-grad_L_grad_l_wrt_Z2 * (1.0 / std_fused), axis=1)[:, None]
        + (1.0 / F) * x_hat_fused * tl.sum(-grad_L_grad_l_wrt_Z2 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
    )

    grad_L_y = ln_weight * grad_L_grad_x_hat_fused

    grad_output_fused = tl.load(grad_output_fused_group_ptr + G_CS_F_offset).to(tl.float32)

    grad_L_ttt_norm_weight = tl.load(grad_L_ttt_norm_weight_ptr + norm_store_offset).to(tl.float32)
    grad_L_ttt_norm_weight += tl.sum(grad_output_fused * grad_L_grad_x_hat_fused + grad_L_y * x_hat_fused, axis=0)
    tl.store(grad_L_ttt_norm_weight_ptr + norm_store_offset, grad_L_ttt_norm_weight)

    grad_L_ttt_norm_bias = tl.load(grad_L_ttt_norm_bias_ptr + norm_store_offset).to(tl.float32)
    grad_L_ttt_norm_bias += tl.sum(grad_L_y, axis=0)
    tl.store(grad_L_ttt_norm_bias_ptr + norm_store_offset, grad_L_ttt_norm_bias)

    grad_x_hat_fused = tl.load(grad_x_hat_fused_group_ptr + G_CS_F_offset).to(tl.float32)
    grad_L_x_hat_fused = (
        grad_L_y * ln_weight
        + (1.0 / F)
        * grad_x_hat_fused
        * tl.sum(-grad_L_grad_l_wrt_Z2 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
        + (1.0 / F)
        * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        * (-grad_L_grad_l_wrt_Z2 * (1.0 / std_fused))
    )

    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_offset).to(tl.float32)

    grad_L_std = -grad_L_x_hat_fused * (x_hat_fused / std_fused) - (
        grad_L_grad_l_wrt_Z2 * (grad_l_wrt_Z2 * std_fused) / (std_fused * std_fused)
    )

    grad_L_Z2 = (
        grad_L_x_hat_fused * (1.0 / std_fused)
        - (1.0 / F) * tl.sum(grad_L_x_hat_fused, axis=1)[:, None] * (1.0 / std_fused)
        + (1.0 / F) * tl.sum(grad_L_std, axis=1)[:, None] * x_hat_fused
    )

    tl.store(grad_L_Z2_ptr + CS_F_intermediate_offset, grad_L_Z2)

    X2 = tl.load(X2_group_ptr + G_CS_F4_offset).to(tl.float32)

    grad_L_W2_last = tl.load(grad_L_W2_init_ptr + F4_F_offset).to(tl.float32)
    grad_L_b2_last = tl.load(grad_L_b2_last_ptr + F_offset).to(tl.float32)

    grad_L_W2_last += tl.dot(tl.trans(X2).to(mp_dtype), grad_L_Z2.to(mp_dtype), allow_tf32=False)
    grad_L_b2_last += tl.sum(grad_L_Z2, axis=0)

    tl.store(grad_L_W2_init_ptr + F4_F_offset, grad_L_W2_last)
    tl.store(grad_L_b2_last_ptr + F_offset, grad_L_b2_last)

    grad_L_reconstruction_target = -ln_weight * grad_L_grad_x_hat_fused
    tl.store(grad_L_XV_ptr + CS_F_offset, grad_L_reconstruction_target)

    # TODO: Use grad_L_XK instead to save allocation, only need to accumulate
    grad_L_XK_mini_batch = tl.load(grad_L_XK_mini_batch_ptr + CS_F_intermediate_offset).to(tl.float32)
    grad_L_XK_mini_batch -= grad_L_reconstruction_target

    tl.store(grad_L_XK_ptr + CS_F_offset, grad_L_XK_mini_batch)


@triton.jit
def ttt_mlp_backward_stage_7(
    eta_batch_ptr,
    # Upstream gradients
    grad_L_W2_last_ptr,
    # Intermediate buffers
    W2_init_group_ptr,
    Z1_group_ptr,
    X2_bar_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    # Other stages
    grad_L_eta_Attn2_ptr,
    grad_L_Z1_ptr,
    grad_L_Z2_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = tl.bfloat16

    F4_F_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, tl.constexpr(F * 4))[:, None] * F
        + tl.arange(0, F)[None, :]
    )

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + tl.arange(0, CS)[:, None] * CS
        + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = (
        batch * NH * NC * CS_CS_stride
        + head * NC * CS_CS_stride
        + mini_batch_idx * CS_CS_stride
        + (CS - 1) * CS
        + tl.arange(0, CS)[:, None]
    )

    # Create offsets for intermediate values from buffer
    G_CS_F_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride
        + head * checkpoint_group_size * CS_F_stride
        + mini_batch_idx_in_group * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )
    G_CS_F4_offset = (
        batch * NH * checkpoint_group_size * CS_F_stride * 4
        + head * checkpoint_group_size * CS_F_stride * 4
        + mini_batch_idx_in_group * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    G_F4_F_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F * 4)[:, None] * F
        + tl.arange(0, F)[None, :]
    )

    # From stage 1 buffer
    CS_F_intermediate_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )
    CS_CS_intermediate_offset = (
        batch * NH * CS_CS_stride + head * CS_CS_stride + tl.arange(0, CS)[:, None] * CS + tl.arange(0, CS)[None, :]
    )

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]

    # Stage 1: MatMul
    X2_bar = tl.load(X2_bar_group_ptr + G_CS_F4_offset).to(tl.float32)
    grad_L_W2_last = tl.load(grad_L_W2_last_ptr + F4_F_offset).to(tl.float32)
    eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
    last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)
    grad_L_eta_Attn2 = tl.load(grad_L_eta_Attn2_ptr + CS_CS_intermediate_offset).to(tl.float32)

    W2_init = tl.load(W2_init_group_ptr + G_F4_F_offset).to(tl.float32)

    grad_l_wrt_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_offset).to(tl.float32)
    grad_L_Z2 = tl.load(grad_L_Z2_ptr + CS_F_intermediate_offset).to(tl.float32)

    grad_L_X2 = (
        -tl.trans(tl.dot(grad_L_W2_last.to(mp_dtype), tl.trans(grad_l_wrt_Z2).to(mp_dtype), allow_tf32=False))
        * last_eta_mini_batch
        - tl.trans(
            tl.dot(
                tl.trans(X2_bar).to(mp_dtype),
                tl.where(mask, grad_L_eta_Attn2 * eta_mini_batch, 0).to(mp_dtype),
                allow_tf32=False,
            )
        )
        + tl.dot(grad_L_Z2.to(mp_dtype), tl.trans(W2_init).to(mp_dtype), allow_tf32=False)
    )

    Z1 = tl.load(Z1_group_ptr + G_CS_F4_offset).to(tl.float32)
    grad_L_Z1 = tl.load(grad_L_Z1_ptr + CS_F4_intermediate_offset).to(tl.float32)

    grad_L_Z1 = grad_L_X2 * gelu_bwd_triton(Z1) + grad_L_Z1

    tl.store(grad_L_Z1_ptr + CS_F4_intermediate_offset, grad_L_Z1)


@triton.jit
def ttt_mlp_backward_stage_8(
    XK_batch_ptr,
    # Upstream gradients
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    # Intermediate buffers
    W1_init_group_ptr,
    # Other stages
    grad_L_Z1_ptr,
    grad_L_b1_init_ptr,
    # Output buffers
    grad_L_XK_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F4_stride: tl.constexpr,
    F4_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    checkpoint_idx,
    mini_batch_idx_in_group,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XK_batch_ptr.type.element_ty

    F_F4_offset = (
        batch * NH * F_F4_stride
        + head * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, tl.constexpr(F * 4))[None, :]
    )
    F4_offset = batch * NH * F4_stride + head * F4_stride + tl.arange(0, tl.constexpr(F * 4))[None, :]

    # Overall index of mini-batch in input
    mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

    CS_F_offset = (
        batch * NH * NC * CS_F_stride
        + head * NC * CS_F_stride
        + mini_batch_idx * CS_F_stride
        + tl.arange(0, CS)[:, None] * F
        + tl.arange(0, F)[None, :]
    )

    # Create offsets for intermediate values from buffer
    G_F_F4_offset = (
        batch * NH * checkpoint_group_size * F_F4_stride
        + head * checkpoint_group_size * F_F4_stride
        + mini_batch_idx_in_group * F_F4_stride
        + tl.arange(0, F)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )

    # From stage 1 buffer
    CS_F4_intermediate_offset = (
        batch * NH * CS_F_stride * 4
        + head * CS_F_stride * 4
        + tl.arange(0, CS)[:, None] * F * 4
        + tl.arange(0, F * 4)[None, :]
    )

    grad_L_Z1 = tl.load(grad_L_Z1_ptr + CS_F4_intermediate_offset)

    W1_init = tl.load(W1_init_group_ptr + G_F_F4_offset).to(
        tl.float32
    )  # TODO: an also maybe load this in as transposed version
    grad_L_XK = tl.load(grad_L_XK_ptr + CS_F_offset).to(tl.float32)
    grad_L_XK += tl.dot(grad_L_Z1.to(mp_dtype), tl.trans(W1_init).to(mp_dtype), allow_tf32=False)
    tl.store(grad_L_XK_ptr + CS_F_offset, grad_L_XK)

    XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)

    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F4_offset).to(tl.float32)
    grad_L_b1_last = tl.load(grad_L_b1_init_ptr + F4_offset).to(tl.float32)

    grad_L_W1_last += tl.dot(tl.trans(XK_mini_batch).to(mp_dtype), grad_L_Z1.to(mp_dtype), allow_tf32=False)
    grad_L_b1_last += tl.sum(grad_L_Z1, axis=0)

    tl.store(grad_L_W1_last_ptr + F_F4_offset, grad_L_W1_last)
    tl.store(grad_L_b1_last_ptr + F4_offset, grad_L_b1_last)
