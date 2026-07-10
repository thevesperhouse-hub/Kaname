"""Adaptive Computation Routing (ACR), v2.

In v1 the router gated *layers*, which broke causality and the gradient. Here it
does something causal and useful: it decides how many memory slots each segment is
compressed into. A boring segment -> 1 slot (SKIM); a dense one -> up to k_max (FOCUS).

The route weights are turned into a differentiable per-slot keep-gate, so the whole
thing trains end-to-end from the LM loss plus a compute penalty that rewards
compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class SegmentScanner(nn.Module):
    """Pools a segment into a summary and predicts routing + write priority."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = RMSNorm(d_model)
        hidden = d_model // 2
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 4, bias=False),   # 3 route logits + 1 write-priority
        )

    def forward(self, seg: torch.Tensor):
        # seg: (B, W, D) -> summary via single learned query
        B, W, D = seg.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(self.query).expand(B, -1).view(B, 1, H, hd).transpose(1, 2)
        k = self.k_proj(seg).view(B, W, H, hd).transpose(1, 2)
        v = self.v_proj(seg).view(B, W, H, hd).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        summary = self.out_proj((attn @ v).transpose(1, 2).reshape(B, D))
        out = self.head(self.norm(summary))
        return out[:, :3], torch.sigmoid(out[:, 3])   # route_logits (B,3), write_priority (B,)


class AdaptiveRouter(nn.Module):
    """Route logits -> route weights (Gumbel straight-through in training)."""

    def __init__(self, cfg):
        super().__init__()
        self.tau_start = cfg.router_tau_start
        self.tau_end = cfg.router_tau_end
        self.tau_steps = cfg.router_tau_steps
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @property
    def tau(self) -> float:
        p = min(self.step.item() / max(self.tau_steps, 1), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * p

    def forward(self, route_logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            tau = self.tau
            g = -torch.empty_like(route_logits).exponential_().log()
            y_soft = F.softmax((route_logits + g) / tau, dim=-1)
            idx = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(route_logits).scatter_(-1, idx, 1.0)
            self.step += 1
            return y_hard - y_soft.detach() + y_soft   # straight-through
        return F.softmax(route_logits / self.tau_end, dim=-1)


def slot_gates(route_weights: torch.Tensor, cfg) -> torch.Tensor:
    """Turn route weights into a smooth per-slot keep-gate of shape (B, k_max).

    expected_k = weighted slot count; gate_j ~ sigmoid(beta * (expected_k - (j+0.5))).
    Differentiable in route_weights, so the router learns from the LM + compute loss.
    """
    device = route_weights.device
    counts = torch.tensor(
        [cfg.k_skim, cfg.k_process, cfg.k_max], device=device, dtype=route_weights.dtype
    )
    expected_k = (route_weights * counts).sum(-1, keepdim=True)          # (B,1)
    j = torch.arange(cfg.k_max, device=device, dtype=route_weights.dtype) + 0.5
    return torch.sigmoid(cfg.gate_beta * (expected_k - j[None, :]))       # (B, k_max)


def acr_losses(route_weights: torch.Tensor, route_logits: torch.Tensor,
               gates: torch.Tensor, cfg) -> dict:
    """ACR auxiliary losses aggregated over all segments in the batch."""
    device = route_weights.device
    avg = route_weights.mean(dim=0)                                       # (3,)
    target = torch.tensor(cfg.target_route_dist, device=device, dtype=avg.dtype)
    balance = F.kl_div(avg.clamp(min=1e-8).log(), target.clamp(min=1e-8), reduction="sum")

    probs = F.softmax(route_logits, dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()

    # compute cost = mean fraction of memory slots actually kept -> pressure to compress
    compute = (gates.sum(-1) / cfg.k_max).mean()

    return {"balance": balance, "entropy": entropy, "compute": compute}
