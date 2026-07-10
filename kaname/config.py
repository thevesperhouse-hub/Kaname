"""Configuration for Kaname.

One dataclass drives model + training. Keep it flat and explicit; a committee (and
future-you) should be able to read the whole model shape in one screen.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class KanameConfig:
    # ---- vocab / width ----
    vocab_size: int = 100_000
    d_model: int = 768
    n_layers: int = 12            # single shared decoder tower (no separate local/global towers)
    n_heads: int = 12
    ffn_mult: float = 4.0         # SwiGLU hidden = ffn_mult * d_model (before the 2/3 SwiGLU factor)
    dropout: float = 0.0
    rope_theta: float = 10_000.0

    # ---- compressive memory / ACR ----
    segment_size: int = 512       # W: tokens processed per recurrent step; RoPE restarts each segment
    k_max: int = 16               # max memory slots produced per segment (FOCUS route)
    k_process: int = 4            # PROCESS route effective slots
    k_skim: int = 1               # SKIM route effective slots
    n_compress_layers: int = 2    # depth of the segment compressor (cross-attn)
    max_memory_slots: int = 512   # hard cap on the compressed KV memory (StreamingLLM-style eviction)
    gate_beta: float = 4.0        # sharpness of the soft slot-count gate

    # ---- ACR routing ----
    router_tau_start: float = 1.0
    router_tau_end: float = 0.1
    router_tau_steps: int = 50_000
    target_route_dist: tuple = (0.6, 0.3, 0.1)   # SKIM / PROCESS / FOCUS

    # ---- ACR aux-loss weights ----
    lambda_compute: float = 0.01   # push toward compression (fewer effective slots)
    lambda_balance: float = 0.01   # keep routes from collapsing
    lambda_entropy: float = 0.001  # mild entropy regularization on route logits
    compress_warmup_steps: int = 2000  # ramp compute pressure 0->1 so the model first
                                       # learns to USE memory before it's told to compress

    # ---- sequence ----
    max_seq_len: int = 8192

    # ---- perf ----
    ce_chunk: int = 2048   # chunk the vocab-100k cross-entropy over tokens (0 = full);
                           # caps the giant logits tensor -> big VRAM saving

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_segments_max(self) -> int:
        return self.max_seq_len // self.segment_size

    @classmethod
    def from_yaml(cls, path: str) -> "KanameConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if "target_route_dist" in raw:
            raw["target_route_dist"] = tuple(raw["target_route_dist"])
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
