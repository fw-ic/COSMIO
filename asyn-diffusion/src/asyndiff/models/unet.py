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

"""Time-conditioned UNet denoiser.

A standard residual UNet with sinusoidal timestep embeddings, in the form
introduced with the diffusion models of Ho et al. (2020). The multi-head output
selected by ``head_index`` is specific to COSMIO's per-modality heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels):
    groups = 8
    if channels < groups:
        groups = 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        num_modalities=3,
        num_heads=1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_modalities = int(num_modalities)
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.multi_time_proj = nn.Sequential(
            nn.Linear(time_dim * self.num_modalities, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(in_ch if j == 0 else ch, ch, time_dim))
            self.down_blocks.append(blocks)
            in_ch = ch
            if i != len(channels) - 1:
                self.downsamples.append(Downsample(in_ch))

        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_block = in_ch + ch if j == 0 else ch
                blocks.append(ResBlock(in_block, ch, time_dim))
            self.up_blocks.append(blocks)
            in_ch = ch
            if i != len(channels) - 1:
                self.upsamples.append(Upsample(in_ch))

        self.final_norm = _group_norm(in_ch)
        self.final_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1) for _ in range(self.num_heads)]
        )

    def _prepare_head_index(self, head_index, batch, device):
        if head_index is None:
            if self.num_heads == 1:
                return None
            raise ValueError("head_index is required when UNet.num_heads > 1.")

        out = torch.as_tensor(head_index, device=device, dtype=torch.long)
        if out.dim() == 0:
            out = out.expand(batch)
        elif out.dim() == 1:
            if out.shape[0] == 1 and batch > 1:
                out = out.expand(batch)
            elif out.shape[0] != batch:
                raise ValueError(f"Head index batch mismatch: {out.shape[0]} vs {batch}")
        else:
            raise ValueError(f"Unsupported head_index shape: {tuple(out.shape)}")

        if torch.any(out < 0) or torch.any(out >= self.num_heads):
            raise ValueError(f"head_index values must be in [0, {self.num_heads - 1}]")
        return out

    def forward(self, x, t, head_index=None):
        if t.dim() == 1:
            t_emb = self.time_embed(t)
        elif t.dim() == 2:
            if t.shape[1] != self.num_modalities:
                raise ValueError(
                    f"Expected modality timestep shape (B,{self.num_modalities}), got {tuple(t.shape)}."
                )
            flat = t.reshape(-1)
            per_modality = self.time_embed(flat).view(t.shape[0], self.num_modalities, -1)
            t_emb = self.multi_time_proj(per_modality.reshape(t.shape[0], -1))
        else:
            raise ValueError(f"Unsupported timestep rank for UNet.forward: {tuple(t.shape)}")

        h = self.init_conv(x)

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        h = F.silu(self.final_norm(h))
        if self.num_heads == 1:
            return self.final_convs[0](h)

        prepared_head_index = self._prepare_head_index(head_index, int(x.shape[0]), x.device)
        all_outputs = torch.stack([head(h) for head in self.final_convs], dim=1)
        gather_index = prepared_head_index.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            self.out_channels,
            all_outputs.shape[-2],
            all_outputs.shape[-1],
        )
        return all_outputs.gather(1, gather_index).squeeze(1)

