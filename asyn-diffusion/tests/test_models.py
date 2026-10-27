# =============================================================================
# COSMIO: A Benchmark for Cross-Survey Modality Imputation
#
# Copyright (c) 2026 Futurewei Technologies, Inc.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =============================================================================

from __future__ import annotations

import torch

from asyndiff.models import GaussianDiffusion, UNet


def test_unet_forward_shape() -> None:
    model = UNet(
        in_channels=9,
        out_channels=9,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_dim=64,
    )
    x = torch.randn(2, 9, 64, 64)
    t = torch.randint(0, 20, (2,))
    y = model(x, t)
    assert y.shape == x.shape

    t_mod = torch.randint(0, 20, (2, 3))
    y_mod = model(x, t_mod)
    assert y_mod.shape == x.shape


def test_unet_multihead_selects_requested_head() -> None:
    model = UNet(
        in_channels=9,
        out_channels=9,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_dim=64,
        num_heads=3,
    )
    x = torch.randn(2, 9, 64, 64)
    t = torch.randint(0, 20, (2,))
    y = model(x, t, head_index=torch.tensor([0, 2]))
    assert y.shape == x.shape


def test_diffusion_sampling_shapes() -> None:
    model = UNet(
        in_channels=9,
        out_channels=9,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_dim=64,
    )
    diffusion = GaussianDiffusion(model=model, timesteps=20)
    x0 = torch.randn(2, 9, 64, 64)
    t = torch.randint(0, 20, (2,))
    q = diffusion.q_sample(x0, t)
    assert q.shape == x0.shape

    mask = torch.zeros_like(x0)
    mask[:, :6] = 1.0
    out = diffusion.sample_ddim_with_mask(
        x0,
        mask,
        device=torch.device("cpu"),
        fix_mode="noised",
        steps=diffusion.build_linear_steps(20, 8),
        eta=1.0,
    )
    assert out.shape == x0.shape

    time_mask = torch.zeros((2, 9), dtype=torch.float32)
    # Masked modalities: euclid(5:6) + roman(6:9) share asynchronous step; lsst remains at 0.
    time_mask[:, 5:9] = 1.0
    model_time_mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    out_async = diffusion.sample_ddim_with_mask(
        x0,
        mask,
        device=torch.device("cpu"),
        fix_mode="noised",
        steps=diffusion.build_linear_steps(20, 8),
        eta=1.0,
        time_mask=time_mask,
        model_time_mask=model_time_mask,
    )
    assert out_async.shape == x0.shape


def test_diffusion_loss_scalar() -> None:
    model = UNet(
        in_channels=9,
        out_channels=9,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_dim=64,
    )
    diffusion = GaussianDiffusion(model=model, timesteps=20)
    x0 = torch.randn(2, 9, 64, 64)
    loss = diffusion.loss(x0)
    assert loss.ndim == 0

    channel_t = torch.zeros((2, 9), dtype=torch.long)
    channel_t[:, 6:9] = 7
    model_t = torch.tensor([[0, 0, 7], [0, 0, 7]], dtype=torch.long)
    loss_mask = torch.zeros_like(x0)
    loss_mask[:, 6:9] = 1.0
    async_loss = diffusion.loss(x0, t=channel_t, model_t=model_t, loss_mask=loss_mask)
    assert async_loss.ndim == 0


def test_diffusion_loss_updates_only_selected_head() -> None:
    model = UNet(
        in_channels=9,
        out_channels=9,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_dim=64,
        num_heads=3,
    )
    diffusion = GaussianDiffusion(model=model, timesteps=20)
    x0 = torch.randn(2, 9, 64, 64)
    loss_mask = torch.ones_like(x0)
    head_index = torch.tensor([1, 1], dtype=torch.long)
    loss = diffusion.loss(x0, loss_mask=loss_mask, head_index=head_index)
    loss.backward()

    inactive_grad_0 = model.final_convs[0].weight.grad.abs().sum().item()
    active_grad = model.final_convs[1].weight.grad.abs().sum().item()
    inactive_grad_2 = model.final_convs[2].weight.grad.abs().sum().item()
    assert active_grad > 0.0
    assert inactive_grad_0 == 0.0
    assert inactive_grad_2 == 0.0

