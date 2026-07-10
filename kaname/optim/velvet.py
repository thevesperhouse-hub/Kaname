"""Velvet Optimizer (v2 port): AdamW + PGM + LVS.

PGM (Perplexity-Guided Momentum): adapts beta1 from a normalized loss ratio.
LVS (Loss-Velocity Scaling): scales LR from dual log-loss EMAs (current vs anchor),
with plateau-triggered cyclical bursts.

Ported from v1, pure-PyTorch, with one correctness fix: the bias correction for the
first moment now uses the *effective* beta1 actually applied to `m`, instead of the
base beta1 (v1 mixed the two, biasing the update whenever PGM moved beta1).

NOTE ON EVALUATION: to claim a win over AdamW, run both at the *same* base LR and LR
schedule and multiple seeds. LVS changes the effective LR trajectory, so an uncontrolled
comparison is not evidence. `entropy_adaptive=False, perplexity_guided=False` gives you
a clean AdamW baseline from this same class.
"""

import math
import torch
from torch.optim import Optimizer


class VelvetOptimizer(Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, entropy_adaptive=True, perplexity_guided=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        entropy_adaptive=entropy_adaptive, perplexity_guided=perplexity_guided)
        super().__init__(params, defaults)

        self._ema_current = None
        self._ema_anchor = None
        self._current_window = 100
        self._anchor_window_min = 200
        self._anchor_window_max = 500
        self._ema_current_alpha = 2.0 / 101
        self._ema_anchor_alpha = 2.0 / 201
        self._lvs_phase_steps = 20000
        self._entropy_scale = 1.0
        self._lvs_momentum_up = 0.995
        self._lvs_momentum_down = 0.9
        self._plateau_counter = 0
        self._plateau_threshold = 0.005
        self._plateau_patience = 200
        self._burst_duration = 50
        self._burst_multiplier = 1.3
        self._burst_step = -1
        self._perplexity_scale = 1.0
        self._global_step = 0

    def set_training_steps(self, max_steps: int):
        self._lvs_phase_steps = max(max_steps, 1)
        self._current_window = max(50, int(max_steps * 0.02))
        self._anchor_window_min = max(self._current_window * 3, int(max_steps * 0.10))
        self._anchor_window_max = max(self._anchor_window_min, int(max_steps * 0.25))
        self._ema_current_alpha = 2.0 / (self._current_window + 1)
        self._ema_anchor_alpha = 2.0 / (self._anchor_window_min + 1)

    def set_loss_metrics(self, loss_val: float, vocab_size: int):
        if not math.isfinite(loss_val):
            return
        max_entropy = math.log(max(vocab_size, 2))
        ratio = min(loss_val, max_entropy) / max_entropy

        if ratio > 0.6:
            pgm = 1.0
        elif ratio > 0.2:
            pgm = 0.8 + 0.5 * (ratio - 0.2)
        else:
            pgm = 0.8 + 2.0 * (0.2 - ratio)
        target = max(0.7, min(1.3, pgm))
        delta = max(-0.02, min(0.02, target - self._perplexity_scale))
        self._perplexity_scale += delta

        log_loss = math.log(max(loss_val, 1e-8))
        if self._ema_current is None:
            self._ema_current = log_loss
            self._ema_anchor = log_loss
            return

        progress = min(1.0, self._global_step / max(self._lvs_phase_steps, 1))
        anchor_window = self._anchor_window_min + (self._anchor_window_max - self._anchor_window_min) * progress
        self._ema_anchor_alpha = 2.0 / (anchor_window + 1)

        self._ema_current += self._ema_current_alpha * (log_loss - self._ema_current)
        self._ema_anchor += self._ema_anchor_alpha * (log_loss - self._ema_anchor)

        gap = max(-1.0, min(1.0, self._ema_current - self._ema_anchor))
        phase = progress
        max_boost = 1.1 - 0.05 * phase
        max_damp = 0.9 + 0.05 * phase
        if gap < -0.005:
            raw = max_boost
        elif gap > 0.005:
            raw = 1.0 - (1.0 - max_damp) * min(1.0, gap / 0.05)
        else:
            raw = 1.0

        mom = self._lvs_momentum_up if raw >= self._entropy_scale else self._lvs_momentum_down
        self._entropy_scale = mom * self._entropy_scale + (1.0 - mom) * raw

        if abs(gap) < self._plateau_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = max(0, self._plateau_counter - 5)

        if self._burst_step >= 0:
            bp = (self._global_step - self._burst_step) / self._burst_duration
            if bp < 1.0:
                spike = 0.5 * (1.0 - math.cos(2.0 * math.pi * bp))
                self._entropy_scale *= 1.0 + (self._burst_multiplier - 1.0) * spike
            else:
                self._burst_step = -1
                self._plateau_counter = 0
        elif self._plateau_counter >= self._plateau_patience and self._global_step > 500:
            self._burst_step = self._global_step
            self._plateau_counter = 0
        burst_max = max_boost * self._burst_multiplier if self._burst_step >= 0 else max_boost
        self._entropy_scale = max(max_damp, min(burst_max, self._entropy_scale))

    @property
    def effective_lr(self) -> float:
        lr = self.param_groups[0]["lr"]
        if self.defaults["entropy_adaptive"]:
            lr *= self._entropy_scale
        return lr

    @property
    def lr_scale(self) -> float:
        return self._entropy_scale

    @property
    def effective_beta1(self) -> float:
        b1 = self.defaults["betas"][0]
        if self.defaults["perplexity_guided"]:
            b1 = max(0.5, min(0.999, b1 * self._perplexity_scale))
        return b1

    @property
    def perplexity_scale(self) -> float:
        return self._perplexity_scale

    @property
    def is_bursting(self) -> bool:
        return self._burst_step >= 0

    @property
    def global_step(self) -> int:
        return self._global_step

    # keys of the LVS/PGM controller state that must survive a checkpoint resume
    _VELVET_KEYS = (
        "_ema_current", "_ema_anchor", "_current_window", "_anchor_window_min",
        "_anchor_window_max", "_ema_current_alpha", "_ema_anchor_alpha",
        "_lvs_phase_steps", "_entropy_scale", "_plateau_counter", "_burst_step",
        "_perplexity_scale", "_global_step",
    )

    def state_dict(self):
        sd = super().state_dict()
        sd["velvet"] = {k: getattr(self, k) for k in self._VELVET_KEYS}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        velvet = sd.pop("velvet", {})
        super().load_state_dict(sd)
        for k, v in velvet.items():
            setattr(self, k, v)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps, base_lr, wd = group["eps"], group["lr"], group["weight_decay"]
            eff_beta1 = (max(0.5, min(0.999, beta1 * self._perplexity_scale))
                         if group["perplexity_guided"] else beta1)
            eff_lr = base_lr * self._entropy_scale if group["entropy_adaptive"] else base_lr

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Velvet does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]

                # bias correction uses the beta actually applied (correctness fix vs v1)
                bc1 = 1.0 - eff_beta1 ** t
                bc2 = 1.0 - beta2 ** t

                p_f32 = p.data.float()
                g = p.grad.float()
                p_f32.mul_(1.0 - base_lr * wd)                     # decoupled weight decay
                state["m"].mul_(eff_beta1).add_(g, alpha=1.0 - eff_beta1)
                state["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                update = (state["m"] / bc1) / ((state["v"] / bc2).sqrt() + eps)
                p_f32.add_(update, alpha=-eff_lr)
                p.data.copy_(p_f32.to(p.dtype))

        return loss
