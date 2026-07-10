"""Core building blocks: RMSNorm, RoPE, SwiGLU, and memory-augmented causal attention.

The attention here is the crux of v2: a query block of length T attends to
[compressed_memory (M slots) ++ its own causal window (T tokens)]. Memory columns
are always visible (they encode strictly-past segments) and carry a soft per-slot
gate; the local window is causal. RoPE is applied only to the local q/k, with
positions restarting each segment — memory slots are content-addressable, not
positional, so unbounded context never stretches the rotary range.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryCache:
    """Precomputed cos/sin for RoPE positions 0..max_pos-1."""

    def __init__(self, head_dim: int, max_pos: int, theta: float, device, dtype):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_pos, device=device).float()
        freqs = torch.outer(t, inv_freq)              # (max_pos, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)       # (max_pos, head_dim)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def get(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd); cos/sin: (T, hd)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_mult: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(ffn_mult * d_model * 2 / 3)
        hidden = (hidden + 63) // 64 * 64          # round to a friendly multiple
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MemoryAttention(nn.Module):
    """Causal self-attention over a length-T window, augmented with M memory slots."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-norm: stabilizes attention logits, cheap, helps small-width models.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.drop = nn.Dropout(dropout)

    def _shape(self, t: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,L,hd)

    def forward(self, x, cos, sin, memory=None, mem_gate=None):
        """
        x:        (B, T, D)  current window
        cos/sin:  (T, hd)    rope for local positions
        memory:   (B, M, D)  compressed past (already normed by caller) or None
        mem_gate: (B, M)     soft keep-gate per memory slot in [0,1] or None
        """
        B, T, D = x.shape
        q = self.q_norm(self._shape(self.q_proj(x), B, T))
        k = self.k_norm(self._shape(self.k_proj(x), B, T))
        v = self._shape(self.v_proj(x), B, T)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        p = self.drop.p if self.training else 0.0

        if memory is not None and memory.shape[1] > 0:
            M = memory.shape[1]
            mk = self.k_norm(self._shape(self.k_proj(memory), B, M))   # no rope on memory
            mv = self._shape(self.v_proj(memory), B, M)
            k = torch.cat([mk, k], dim=2)                             # (B,H,M+T,hd)
            v = torch.cat([mv, v], dim=2)

            # additive float mask (B,1,T,M+T): memory cols carry the soft gate log-bias
            # (fully visible), local cols are causal. SDPA consumes this and runs a fused
            # (cuDNN / mem-efficient) kernel instead of a hand-rolled softmax.
            mask = torch.zeros(B, 1, T, M + T, dtype=q.dtype, device=x.device)
            mask[..., M:] = torch.triu(
                torch.full((T, T), float("-inf"), dtype=q.dtype, device=x.device), diagonal=1)
            if mem_gate is not None:
                mask[..., :M] += torch.log(mem_gate.clamp(min=1e-6))[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
