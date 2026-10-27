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

"""Gaussian diffusion with asynchronous per-modality masking.

The forward process, the epsilon-parameterised reverse step and the loss follow
Ho et al., "Denoising Diffusion Probabilistic Models" (2020). The deterministic
and stochastic accelerated samplers follow Song et al., "Denoising Diffusion
Implicit Models" (2021). Neither paper's source is reproduced here; the
asynchronous masking over LSST, Euclid and Roman is specific to COSMIO.
"""

import torch
import torch.nn.functional as F
from asyndiff.ops.random import randn_per_sample, randn_like_per_sample
from asyndiff.ops.validation import require_finite_batch


def make_beta_schedule(schedule, timesteps, beta_start, beta_end):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion(torch.nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule="linear",
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        objective="eps",
        clip_x0=True,
    ):
        super().__init__()
        self.model = model
        self.timesteps = int(timesteps)
        self.objective = str(objective).lower()
        self.clip_x0 = bool(clip_x0)

        betas = make_beta_schedule(beta_schedule, self.timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1.0e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _as_diffusion_t(self, t, x_shape):
        batch = int(x_shape[0])
        channels = int(x_shape[1])
        out = torch.as_tensor(t, device=self.betas.device, dtype=torch.long)
        if out.dim() == 0:
            out = out.expand(batch)

        if out.dim() == 1:
            if out.shape[0] != batch:
                raise ValueError(f"Timestep batch mismatch: {out.shape[0]} vs {batch}")
        elif out.dim() == 2:
            if out.shape != (batch, channels):
                raise ValueError(
                    f"Channel-wise timestep shape must be {(batch, channels)}, got {tuple(out.shape)}"
                )
        else:
            raise ValueError(f"Unsupported timestep shape: {tuple(out.shape)}")

        if torch.any(out < 0) or torch.any(out >= self.timesteps):
            raise ValueError(f"Timestep values must be in [0, {self.timesteps - 1}]")
        return out

    def _as_model_t(self, model_t, t, x_shape):
        batch = int(x_shape[0])
        if model_t is None:
            if t.dim() == 1:
                return t
            raise ValueError("model_t is required when using channel-wise diffusion timesteps.")

        out = torch.as_tensor(model_t, device=self.betas.device, dtype=torch.long)
        if out.dim() == 0:
            out = out.expand(batch)
        if out.dim() == 1 and out.shape[0] != batch:
            raise ValueError(f"Model timestep batch mismatch: {out.shape[0]} vs {batch}")
        if out.dim() == 2 and out.shape[0] != batch:
            raise ValueError(f"Model timestep batch mismatch: {out.shape[0]} vs {batch}")
        if out.dim() not in {1, 2}:
            raise ValueError(f"Unsupported model timestep shape: {tuple(out.shape)}")
        if torch.any(out < 0) or torch.any(out >= self.timesteps):
            raise ValueError(f"Model timestep values must be in [0, {self.timesteps - 1}]")
        return out

    @staticmethod
    def _broadcast_t_nonzero(t, x_shape):
        batch = int(x_shape[0])
        if t.dim() == 1:
            return (t > 0).view(batch, *([1] * (len(x_shape) - 1)))
        return (t > 0).view(batch, t.shape[1], *([1] * (len(x_shape) - 2)))

    @staticmethod
    def _broadcast_t_zero(t, x_shape):
        batch = int(x_shape[0])
        if t.dim() == 1:
            return (t == 0).view(batch, *([1] * (len(x_shape) - 1)))
        return (t == 0).view(batch, t.shape[1], *([1] * (len(x_shape) - 2)))

    @staticmethod
    def _prepare_time_mask(time_mask, x0):
        if time_mask is None:
            return None
        batch, channels = int(x0.shape[0]), int(x0.shape[1])
        out = torch.as_tensor(time_mask, device=x0.device, dtype=x0.dtype)

        if out.dim() == 0:
            out = out.view(1, 1).expand(batch, channels)
        elif out.dim() == 1:
            if out.shape[0] == channels:
                out = out.view(1, channels).expand(batch, channels)
            elif out.shape[0] == batch:
                out = out.view(batch, 1).expand(batch, channels)
            else:
                raise ValueError(f"Unsupported 1D time_mask shape: {tuple(out.shape)}")
        elif out.dim() == 2:
            if out.shape[0] == 1 and batch > 1:
                out = out.expand(batch, out.shape[1])
            if out.shape[1] == 1 and channels > 1:
                out = out.expand(out.shape[0], channels)
            if out.shape != (batch, channels):
                raise ValueError(
                    f"time_mask must resolve to {(batch, channels)}, got {tuple(out.shape)}"
                )
        elif out.dim() == 4:
            if out.shape[0] == 1 and batch > 1:
                out = out.expand(batch, -1, -1, -1)
            if out.shape[:2] != (batch, channels):
                raise ValueError(
                    f"time_mask must match x0 in batch/channels {(batch, channels)}, got {tuple(out.shape)}"
                )
            out = out[:, :, 0, 0]
        else:
            raise ValueError(f"Unsupported time_mask shape: {tuple(out.shape)}")

        return (out > 0).to(dtype=x0.dtype)

    @staticmethod
    def _prepare_model_time_mask(model_time_mask, batch, device):
        if model_time_mask is None:
            return None
        out = torch.as_tensor(model_time_mask, device=device, dtype=torch.float32)
        if out.dim() == 0:
            out = out.view(1, 1).expand(batch, 1)
        elif out.dim() == 1:
            out = out.view(1, -1).expand(batch, -1)
        elif out.dim() == 2:
            if out.shape[0] == 1 and batch > 1:
                out = out.expand(batch, out.shape[1])
            if out.shape[0] != batch:
                raise ValueError(f"Model time mask batch mismatch: {out.shape[0]} vs {batch}")
        else:
            raise ValueError(f"Unsupported model_time_mask shape: {tuple(out.shape)}")
        return (out > 0).to(dtype=torch.float32)

    def _prepare_head_index(self, head_index, batch, device):
        if head_index is None:
            if getattr(self.model, "num_heads", 1) == 1:
                return None
            raise ValueError("head_index is required when diffusion model has multiple heads.")

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

        num_heads = int(getattr(self.model, "num_heads", 1))
        if torch.any(out < 0) or torch.any(out >= num_heads):
            raise ValueError(f"head_index values must be in [0, {num_heads - 1}]")
        return out

    def _extract(self, arr, t, x_shape):
        if t.dim() == 1:
            out = arr.gather(0, t)
            return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

        if t.dim() == 2:
            batch = t.shape[0]
            out = arr.gather(0, t.reshape(-1)).view(batch, t.shape[1])
            return out.view(batch, t.shape[1], *([1] * (len(x_shape) - 2)))

        raise ValueError(f"Unsupported timestep shape for extract: {tuple(t.shape)}")

    def q_sample(self, x0, t, noise=None, generators=None):
        t = self._as_diffusion_t(t, x0.shape)
        if noise is None:
            noise = randn_like_per_sample(x0, generators)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        sampled = sqrt_alpha * x0 + sqrt_one_minus * noise
        # Keep exact clean values at t=0, which is required for visible modalities in async masking.
        zero_mask = self._broadcast_t_zero(t, x0.shape)
        return torch.where(zero_mask, x0, sampled)

    @staticmethod
    def _cat_condition(x, cond):
        if cond is None:
            return x
        if cond.shape[0] != x.shape[0]:
            raise ValueError(f"cond batch mismatch: {cond.shape[0]} vs {x.shape[0]}")
        if cond.shape[2:] != x.shape[2:]:
            raise ValueError(f"cond spatial mismatch: {tuple(cond.shape[2:])} vs {tuple(x.shape[2:])}")
        return torch.cat([x, cond], dim=1)

    def _predict_x0_from_eps(self, x_t, t, eps):
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def p_mean_variance(self, x_t, t, cond=None, model_t=None, head_index=None):
        t = self._as_diffusion_t(t, x_t.shape)
        model_timesteps = self._as_model_t(model_t, t, x_t.shape)
        prepared_head_index = self._prepare_head_index(head_index, int(x_t.shape[0]), x_t.device)

        model_in = self._cat_condition(x_t, cond)
        model_out = self.model(model_in, model_timesteps, head_index=prepared_head_index)
        if self.objective == "eps":
            pred_eps = model_out
            pred_x0 = self._predict_x0_from_eps(x_t, t, pred_eps)
        elif self.objective == "x0":
            pred_x0 = model_out
            pred_eps = (x_t - self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * pred_x0) / self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        require_finite_batch(model_out, list(range(x_t.shape[0])), "diffusion model output (batch index)")
        require_finite_batch(pred_x0, list(range(x_t.shape[0])), "diffusion pre-clipping x0 (batch index)")
        if self.clip_x0:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean, var, pred_x0, pred_eps

    @torch.no_grad()
    def p_sample(self, x_t, t, cond=None, model_t=None, head_index=None, generators=None):
        t = self._as_diffusion_t(t, x_t.shape)
        mean, var, pred_x0, _ = self.p_mean_variance(
            x_t,
            t,
            cond=cond,
            model_t=model_t,
            head_index=head_index,
        )
        if torch.all(t == 0):
            return pred_x0
        noise = randn_like_per_sample(x_t, generators)
        nonzero = self._broadcast_t_nonzero(t, x_t.shape).to(dtype=x_t.dtype)
        return mean + torch.sqrt(var) * noise * nonzero

    @torch.no_grad()
    def ddim_step(self, x_t, t, t_next, cond=None, eta=0.0, model_t=None, head_index=None, generators=None):
        """DDIM update from timestep t -> t_next (t_next <= t)."""

        t = self._as_diffusion_t(t, x_t.shape)
        t_next = self._as_diffusion_t(t_next, x_t.shape)
        if t_next.shape != t.shape:
            raise ValueError(f"t_next shape must match t shape: {tuple(t_next.shape)} vs {tuple(t.shape)}")
        if torch.any(t_next > t):
            raise ValueError("t_next must be <= t (for all entries)")

        _, _, pred_x0, pred_eps = self.p_mean_variance(
            x_t,
            t,
            cond=cond,
            model_t=model_t,
            head_index=head_index,
        )

        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_next = self._extract(self.alphas_cumprod, t_next, x_t.shape)

        if float(eta) < 0.0:
            raise ValueError("eta must be >= 0")
        eta = float(eta)
        sigma = eta * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)) * torch.sqrt(
            1.0 - alpha_bar_t / alpha_bar_next
        )

        mean = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(
            (1.0 - alpha_bar_next) - sigma**2
        ) * pred_eps

        if torch.all(t_next == 0):
            return pred_x0, pred_x0

        noise = randn_like_per_sample(x_t, generators)
        noise_mask = self._broadcast_t_nonzero(t_next, x_t.shape).to(dtype=x_t.dtype)
        x_next = mean + sigma * noise * noise_mask
        return x_next, pred_x0

    @torch.no_grad()
    def sample(self, shape, device, cond=None, head_index=None, generators=None):
        x = randn_per_sample(shape, device=device, generators=generators)
        if cond is not None:
            cond = cond.to(device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond=cond, head_index=head_index, generators=generators)
        return x

    @staticmethod
    def build_linear_steps(num_timesteps: int, num_steps: int | None):
        """Unique, strictly-decreasing steps from (T-1) -> 0."""

        if num_steps is None:
            return None
        num_steps = int(num_steps)
        if num_steps <= 0:
            return None
        num_timesteps = int(num_timesteps)
        num_steps = min(num_steps, num_timesteps)
        if num_steps == 1:
            return torch.tensor([0], dtype=torch.long)

        raw = torch.linspace(num_timesteps - 1, 0, num_steps).round().long()
        raw[0] = num_timesteps - 1
        raw[-1] = 0

        steps = []
        last = None
        for s in raw.tolist():
            s = int(s)
            if last is None or s < last:
                steps.append(s)
                last = s
        if steps[-1] != 0:
            steps.append(0)
        return torch.tensor(steps, dtype=torch.long)

    @torch.no_grad()
    def sample_ddim(self, shape, device, *, cond=None, steps=None, eta=0.0, head_index=None, generators=None):
        """Accelerated sampling using DDIM over a subset of steps."""

        x = randn_per_sample(shape, device=device, generators=generators)
        if cond is not None:
            cond = cond.to(device)

        if steps is None:
            steps = self.build_linear_steps(self.timesteps, 200)
        steps = torch.as_tensor(steps, device=device, dtype=torch.long)
        if steps.numel() == 0:
            return x

        for idx in range(int(steps.numel()) - 1):
            t = int(steps[idx].item())
            t_next = int(steps[idx + 1].item())
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_next_batch = torch.full((shape[0],), t_next, device=device, dtype=torch.long)
            x, _ = self.ddim_step(x, t_batch, t_next_batch, cond=cond, eta=eta, head_index=head_index, generators=generators)
        if int(steps[-1].item()) != 0:
            t_last = int(steps[-1].item())
            t_batch = torch.full((shape[0],), t_last, device=device, dtype=torch.long)
            t0 = torch.zeros((shape[0],), device=device, dtype=torch.long)
            x, _ = self.ddim_step(x, t_batch, t0, cond=cond, eta=eta, head_index=head_index, generators=generators)
        return x

    @torch.no_grad()
    def sample_with_mask(
        self,
        x0,
        mask,
        *,
        device=None,
        fix_mode="noised",
        time_mask=None,
        model_time_mask=None,
        head_index=None,
        generators=None,
    ):
        """Sample while clamping a known subset of channels/pixels."""

        if device is None:
            device = x0.device
        device = torch.device(device)

        x0 = x0.to(device)
        if x0.dim() != 4:
            raise ValueError(f"x0 must be 4D (B,C,H,W), got shape={tuple(x0.shape)}")

        if fix_mode not in {"noised", "clean"}:
            raise ValueError("fix_mode must be one of: {'noised', 'clean'}")

        mask = mask.to(device=device, dtype=x0.dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and x0.shape[0] > 1:
            mask = mask.expand(x0.shape[0], -1, -1, -1)
        if mask.shape != x0.shape:
            raise ValueError(
                f"mask must match x0 shape (or broadcast on batch), got mask={tuple(mask.shape)} x0={tuple(x0.shape)}"
            )

        B = int(x0.shape[0])
        x = randn_like_per_sample(x0, generators)
        time_mask = self._prepare_time_mask(time_mask, x0)
        model_time_mask = self._prepare_model_time_mask(model_time_mask, B, device)

        has_mask = bool(mask.sum().item() > 0)
        if has_mask:
            if fix_mode == "noised":
                if time_mask is None:
                    t_init = torch.full((B,), self.timesteps - 1, device=device, dtype=torch.long)
                else:
                    t_init = (time_mask * float(self.timesteps - 1)).long()
                x_known = self.q_sample(x0, t_init, generators=generators)
            else:
                x_known = x0
            x = x * (1.0 - mask) + x_known * mask

        for step in reversed(range(self.timesteps)):
            if time_mask is None:
                t = torch.full((B,), step, device=device, dtype=torch.long)
            else:
                t = (time_mask * float(step)).long()
            if model_time_mask is None:
                model_t = None
            else:
                model_t = (model_time_mask * float(step)).long()

            x = self.p_sample(x, t, cond=None, model_t=model_t, head_index=head_index, generators=generators)

            if not has_mask:
                continue

            if step == 0 or fix_mode == "clean":
                x = x * (1.0 - mask) + x0 * mask
            else:
                if time_mask is None:
                    t_next = torch.full((B,), step - 1, device=device, dtype=torch.long)
                else:
                    t_next = (time_mask * float(step - 1)).long()
                x_known = self.q_sample(x0, t_next, generators=generators)
                x = x * (1.0 - mask) + x_known * mask

        return x

    @torch.no_grad()
    def sample_ddim_with_mask(
        self,
        x0,
        mask,
        *,
        device=None,
        fix_mode="noised",
        steps=None,
        eta=0.0,
        time_mask=None,
        model_time_mask=None,
        head_index=None,
        generators=None,
    ):
        """DDIM sampling while clamping a known subset of channels/pixels."""

        if device is None:
            device = x0.device
        device = torch.device(device)
        x0 = x0.to(device)
        if x0.dim() != 4:
            raise ValueError(f"x0 must be 4D (B,C,H,W), got shape={tuple(x0.shape)}")

        if fix_mode not in {"noised", "clean"}:
            raise ValueError("fix_mode must be one of: {'noised', 'clean'}")

        mask = mask.to(device=device, dtype=x0.dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and x0.shape[0] > 1:
            mask = mask.expand(x0.shape[0], -1, -1, -1)
        if mask.shape != x0.shape:
            raise ValueError(
                f"mask must match x0 shape (or broadcast on batch), got mask={tuple(mask.shape)} x0={tuple(x0.shape)}"
            )

        B = int(x0.shape[0])
        x = randn_like_per_sample(x0, generators)
        time_mask = self._prepare_time_mask(time_mask, x0)
        model_time_mask = self._prepare_model_time_mask(model_time_mask, B, device)

        if steps is None:
            steps = self.build_linear_steps(self.timesteps, 200)
        steps = torch.as_tensor(steps, device=device, dtype=torch.long)
        if steps.numel() == 0:
            return x

        has_mask = bool(mask.sum().item() > 0)
        if has_mask:
            t_init = int(steps[0].item())
            if fix_mode == "noised" and t_init > 0:
                if time_mask is None:
                    t_batch = torch.full((B,), t_init, device=device, dtype=torch.long)
                else:
                    t_batch = (time_mask * float(t_init)).long()
                x_known = self.q_sample(x0, t_batch, generators=generators)
            else:
                x_known = x0
            x = x * (1.0 - mask) + x_known * mask

        for idx in range(int(steps.numel()) - 1):
            t = int(steps[idx].item())
            t_next = int(steps[idx + 1].item())

            if time_mask is None:
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                t_next_batch = torch.full((B,), t_next, device=device, dtype=torch.long)
            else:
                t_batch = (time_mask * float(t)).long()
                t_next_batch = (time_mask * float(t_next)).long()

            if model_time_mask is None:
                model_t_batch = None
            else:
                model_t_batch = (model_time_mask * float(t)).long()

            x, _ = self.ddim_step(
                x,
                t_batch,
                t_next_batch,
                cond=None,
                eta=eta,
                model_t=model_t_batch,
                head_index=head_index,
                generators=generators,
            )

            if not has_mask:
                continue

            if t_next == 0 or fix_mode == "clean":
                x = x * (1.0 - mask) + x0 * mask
            else:
                x_known = self.q_sample(x0, t_next_batch, generators=generators)
                x = x * (1.0 - mask) + x_known * mask

        return x

    def loss(self, x0, cond=None, *, t=None, model_t=None, loss_mask=None, head_index=None):
        batch = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch,), device=x0.device, dtype=torch.long)
        else:
            t = self._as_diffusion_t(t, x0.shape)

        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        model_timesteps = self._as_model_t(model_t, t, x0.shape)
        prepared_head_index = self._prepare_head_index(head_index, int(batch), x0.device)
        pred = self.model(
            self._cat_condition(x_t, cond),
            model_timesteps,
            head_index=prepared_head_index,
        )

        if self.objective == "eps":
            target = noise
        elif self.objective == "x0":
            target = x0
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        if loss_mask is None:
            return F.mse_loss(pred, target)

        mask = loss_mask.to(device=x0.device, dtype=pred.dtype)
        if mask.dim() == 2:
            mask = mask[:, :, None, None]
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and pred.shape[0] > 1:
            mask = mask.expand(pred.shape[0], -1, -1, -1)
        if mask.shape != pred.shape:
            raise ValueError(f"loss_mask must match prediction shape, got {tuple(mask.shape)} vs {tuple(pred.shape)}")

        denom = mask.sum()
        if float(denom.item()) <= 0.0:
            raise ValueError("loss_mask must include at least one element.")
        sq = (pred - target) ** 2
        return (sq * mask).sum() / denom

    @torch.no_grad()
    def reconstruct(self, x0, t_start, cond=None, head_index=None, generators=None):
        t_start = int(t_start)
        if t_start <= 0:
            return x0
        t = torch.full((x0.shape[0],), t_start, device=x0.device, dtype=torch.long)
        x = self.q_sample(x0, t, generators=generators)
        if cond is not None:
            cond = cond.to(x0.device)
        for step in reversed(range(t_start + 1)):
            t = torch.full((x0.shape[0],), step, device=x0.device, dtype=torch.long)
            x = self.p_sample(x, t, cond=cond, head_index=head_index, generators=generators)
        return x

