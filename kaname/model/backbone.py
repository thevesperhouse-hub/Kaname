"""The single shared decoder tower.

There is exactly one stack of blocks. It is reused for every segment; the only
thing that carries information across segments is the compressed `memory`. This is
what keeps the parameter budget honest — no separate "local" and "global" towers.
"""

import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLU, MemoryAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MemoryAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_mult, dropout)

    def forward(self, x, cos, sin, memory=None, mem_gate=None):
        # Memory is normed with the same attn_norm so keys live in one space.
        m = self.attn_norm(memory) if (memory is not None and memory.shape[1] > 0) else None
        x = x + self.attn(self.attn_norm(x), cos, sin, memory=m, mem_gate=mem_gate)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, memory=None, mem_gate=None):
        for layer in self.layers:
            x = layer(x, cos, sin, memory=memory, mem_gate=mem_gate)
        return self.norm(x)
