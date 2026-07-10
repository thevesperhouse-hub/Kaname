"""Segment compressor: turns a processed segment (W tokens) into k_max memory slots.

Learned query vectors cross-attend to the segment's hidden states, forcing salient
information through a bottleneck. The router's per-slot gate decides how many of the
k_max slots survive (effective compression ratio). This module is small and shared —
it is *not* a second transformer tower.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class _CrossAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.ffn_norm = RMSNorm(d_model)
        hidden = int(ffn_mult * d_model * 2 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False), nn.SiLU(),
            nn.Linear(hidden, d_model, bias=False), nn.Dropout(dropout),
        )

    def forward(self, q_in, kv_in):
        B, K, D = q_in.shape
        W = kv_in.shape[1]
        H, hd = self.n_heads, self.head_dim
        q = self.q_norm(self.q_proj(self.norm_q(q_in)).view(B, K, H, hd)).transpose(1, 2)
        kv = self.norm_kv(kv_in)
        k = self.k_norm(self.k_proj(kv).view(B, W, H, hd)).transpose(1, 2)
        v = self.v_proj(kv).view(B, W, H, hd).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, K, D)
        x = q_in + self.out_proj(out)
        return x + self.ffn(self.ffn_norm(x))


class SegmentCompressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k_max = cfg.k_max
        self.queries = nn.Parameter(torch.randn(cfg.k_max, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList([
            _CrossAttn(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_compress_layers)
        ])
        self.out_norm = RMSNorm(cfg.d_model)

    def forward(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        # seg_hidden: (B, W, D) -> slots (B, k_max, D)
        B = seg_hidden.shape[0]
        c = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            c = layer(c, seg_hidden)
        return self.out_norm(c)
