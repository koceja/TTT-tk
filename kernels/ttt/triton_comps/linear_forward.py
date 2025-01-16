import torch
import triton
import numpy as np
import triton.language as tl


@triton.jit
def ttt_linear_scan_forward(
    # Scan inputs
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_init_ptr,
    b1_init_ptr,
    XQ_batch_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    reset_states_ptr,
    # Ouputs
    W1_last_ptr,
    b1_last_ptr,
    XQW_batch_ptr,
    # Context pointers
    W1_checkpoints_ptr,
    b1_checkpoints_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    K: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
    enable_state_reset: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XQ_batch_ptr.type.element_ty

    K_F_F_stride = K * F * F
    K_F_stride = K * F

    F_F_offset = batch * NH * F_F_stride + head * F_F_stride + tl.arange(0, F)[:, None] * F + tl.arange(0, F)[None, :]
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)

    W1_init = tl.load(W1_init_ptr + F_F_offset).to(tl.float32)
    b1_init = tl.load(b1_init_ptr + F_offset).to(tl.float32)

    W1_0 = tl.load(W1_init_ptr + F_F_offset).to(tl.float32)
    b1_0 = tl.load(b1_init_ptr + F_offset).to(tl.float32)

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset).to(tl.float32)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset).to(tl.float32)[None, :]

    for i in range(NC):
        if i % checkpoint_group_size == 0:
            curr_k = i // checkpoint_group_size
            W1_checkpoint_offset = (
                batch * NH * K_F_F_stride
                + head * K_F_F_stride
                + curr_k * F_F_stride
                + tl.arange(0, F)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            b1_checkpoint_offset = (
                batch * NH * K_F_stride
                + head * K_F_stride
                + curr_k * F_stride
                + tl.arange(0, 1)[:, None] * F
                + tl.arange(0, F)[None, :]
            )

            tl.store(W1_checkpoints_ptr + W1_checkpoint_offset, W1_init)
            tl.store(b1_checkpoints_ptr + b1_checkpoint_offset, b1_init)

        # Compute mini-batch offsets
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

        XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset).to(tl.float32)
        XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)
        XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset).to(tl.float32)
        eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset).to(tl.float32)
        last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)

        if enable_state_reset == 1:
            reset_states_offset = batch * NC + i
            reset_states = tl.load(reset_states_ptr + reset_states_offset)

        # Stage 1: MatMul
        Z1 = tl.dot(XK_mini_batch.to(mp_dtype), W1_init.to(mp_dtype)) + b1_init
        reconstruction_target = XV_mini_batch - XK_mini_batch

        # Stage 2: LnFusedL2BWD
        mu_fused = (tl.sum(Z1, axis=1) / F)[:, None]
        var_fused = (tl.sum((Z1 - mu_fused) * (Z1 - mu_fused), axis=1) / F)[:, None]

        std_fused = tl.sqrt(var_fused + 1e-6)
        x_hat_fused = (Z1 - mu_fused) / std_fused

        y = ln_weight * x_hat_fused + ln_bias
        grad_output_fused = y - reconstruction_target
        grad_x_hat_fused = grad_output_fused * ln_weight

        grad_l_wrt_Z1 = (
            (1.0 / F)
            * (
                F * grad_x_hat_fused
                - tl.sum(grad_x_hat_fused, axis=1)[:, None]
                - x_hat_fused * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
            )
            / std_fused
        )

        # Stage 3: Dual Form
        mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
        Attn1 = tl.where(
            mask,
            tl.dot(XQ_mini_batch.to(mp_dtype), tl.trans(XK_mini_batch).to(mp_dtype)),
            0,
        )
        b1_bar = b1_init - tl.dot(tl.where(mask, eta_mini_batch, 0).to(mp_dtype), grad_l_wrt_Z1.to(mp_dtype))
        Z1_bar = (
            tl.dot(XQ_mini_batch.to(mp_dtype), W1_init.to(mp_dtype))
            - tl.dot((eta_mini_batch * Attn1).to(mp_dtype), grad_l_wrt_Z1.to(mp_dtype))
            + b1_bar
        )

        # NOTE: Accumulation in FP32
        if enable_state_reset == 1:
            W1_init = tl.where(
                reset_states == 1,
                W1_0,
                W1_init
                - tl.dot(
                    tl.trans(last_eta_mini_batch * XK_mini_batch).to(mp_dtype),
                    grad_l_wrt_Z1.to(mp_dtype),
                ),
            )
            b1_init = tl.where(
                reset_states == 1,
                b1_0,
                b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :],
            )
        else:
            W1_init = W1_init - tl.dot(
                tl.trans(last_eta_mini_batch * XK_mini_batch).to(mp_dtype),
                grad_l_wrt_Z1.to(mp_dtype),
            )
            b1_init = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]

        # Stage 4: LN
        mu_ln = tl.sum(Z1_bar, axis=1)[:, None] / F
        var_ln = tl.sum((Z1_bar - mu_ln) * (Z1_bar - mu_ln), axis=1)[:, None] / F
        std_ln = tl.sqrt(var_ln + 1e-6)
        x_hat_ln = (Z1_bar - mu_ln) / std_ln

        Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

        XQW_mini_batch = XQ_mini_batch + Z1_bar_ln
        tl.store(XQW_batch_ptr + CS_F_offset, XQW_mini_batch)

    tl.store(W1_last_ptr + F_F_offset, W1_init)
    tl.store(b1_last_ptr + F_offset, b1_init)
