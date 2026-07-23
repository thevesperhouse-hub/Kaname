"""Kaname top-level model.

Segment-recurrent compressive decoder:

    for each segment i:
        h_i = backbone(embed(seg_i), memory=compressed_past)   # causal, attends past memory
        slots_i, gate_i = compress(h_i)   with the ACR router picking the ratio
        memory <- append(memory, slots_i)                      # bounded, evicts oldest

The memory *is* the KV cache. Causality is structural: segment i only ever sees
compressed slots from segments < i, plus its own causal window. Every position after
the first segment benefits from the compressed past — no masked-out no-op like v1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import RMSNorm, RotaryCache
from .backbone import Backbone
from .compressor import SegmentCompressor
from .router import SegmentScanner, AdaptiveRouter, slot_gates, acr_losses


class Kaname(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W = cfg.segment_size

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.backbone = Backbone(cfg)
        self.scanner = SegmentScanner(cfg.d_model, cfg.n_heads)
        self.router = AdaptiveRouter(cfg)
        self.compressor = SegmentCompressor(cfg)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight       # tie

        self._rope = None
        self.grad_checkpoint = False
        self._init_weights()

    def set_grad_checkpoint(self, on: bool = True):
        self.grad_checkpoint = on

    def _init_weights(self):
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2 and "embed" not in name:
                if "out_proj" in name or "w_down" in name:
                    nn.init.xavier_normal_(p, gain=scale)
                else:
                    nn.init.xavier_normal_(p, gain=0.5)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def _rope_cache(self, device, dtype):
        if self._rope is None or self._rope.cos.device != device or self._rope.cos.dtype != dtype:
            self._rope = RotaryCache(self.cfg.head_dim, self.W, self.cfg.rope_theta, device, dtype)
        return self._rope

    def forward(self, input_ids: torch.Tensor, detach_memory: bool = False,
                return_hidden: bool = False, use_memory: bool = True) -> dict:
        """use_memory=False withholds the compressed past from attention (each segment
        only sees its own local window) — the ablation that isolates the memory's value."""
        B, L = input_ids.shape
        W = self.W
        device = input_ids.device

        pad = (W - L % W) % W
        if pad:
            input_ids = F.pad(input_ids, (0, pad), value=0)
        n_seg = input_ids.shape[1] // W

        x_all = self.embed_drop(self.token_embed(input_ids))         # (B, L_pad, D)
        cos, sin = self._rope_cache(device, x_all.dtype).get(W)

        memory = None          # (B, M, D)
        mem_gate = None        # (B, M)
        outs = []
        route_w_list, route_l_list, gate_list, wp_list = [], [], [], []

        for i in range(n_seg):
            seg = x_all[:, i * W:(i + 1) * W, :]
            mem_in, gate_in = (memory, mem_gate) if use_memory else (None, None)
            if self.grad_checkpoint and self.training:
                h = checkpoint(self.backbone, seg, cos, sin, mem_in, gate_in,
                               use_reentrant=False)
            else:
                h = self.backbone(seg, cos, sin, memory=mem_in, mem_gate=gate_in)
            outs.append(h)

            route_logits, wp = self.scanner(h)
            route_w = self.router(route_logits)
            gate = slot_gates(route_w, self.cfg)                     # (B, k_max)
            slots = self.compressor(h)                               # (B, k_max, D)

            route_w_list.append(route_w)
            route_l_list.append(route_logits)
            gate_list.append(gate)
            wp_list.append(wp)

            new_mem = slots if memory is None else torch.cat([memory, slots], dim=1)
            new_gate = gate if mem_gate is None else torch.cat([mem_gate, gate], dim=1)
            cap = self.cfg.max_memory_slots
            if new_mem.shape[1] > cap:                               # StreamingLLM eviction
                new_mem = new_mem[:, -cap:, :]
                new_gate = new_gate[:, -cap:]
            memory = new_mem.detach() if detach_memory else new_mem
            mem_gate = new_gate.detach() if detach_memory else new_gate

        h_full = torch.cat(outs, dim=1)[:, :L, :]

        route_w = torch.cat(route_w_list, dim=0)                     # (B*n_seg, 3)
        route_l = torch.cat(route_l_list, dim=0)
        gates = torch.cat(gate_list, dim=0)                          # (B*n_seg, k_max)
        aux = acr_losses(route_w, route_l, gates, self.cfg)

        eff_slots = gates.sum(-1).mean()                            # avg slots kept / segment
        stats = {
            "route_dist": route_w.mean(0).detach(),                # (3,)
            "eff_slots": eff_slots.detach(),
            "mem_slots": float(memory.shape[1]) if memory is not None else 0.0,
            "compression_ratio": (W / eff_slots.clamp(min=1e-3)).detach(),
            "router_tau": self.router.tau,
            # per-segment routing, for analysis: (B, n_seg, 3) and (B, n_seg)
            "seg_routes": torch.stack(route_w_list, dim=1).detach(),
            "seg_write_priority": torch.stack(wp_list, dim=1).detach(),
        }
        result = {"aux": aux, "stats": stats}
        if return_hidden:
            result["hidden"] = h_full        # loss path fuses lm_head into chunked CE
        else:
            result["logits"] = self.lm_head(h_full)
        return result

    def _cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Chunked cross-entropy over the tied 100k head: never materializes the full
        (tokens x vocab) logits tensor, only one chunk at a time."""
        D = hidden.size(-1)
        hf = hidden.reshape(-1, D)
        tf = targets.reshape(-1)
        Wt = self.lm_head.weight
        chunk = self.cfg.ce_chunk
        if not chunk or chunk >= hf.size(0):
            return F.cross_entropy(F.linear(hf, Wt), tf, ignore_index=-100)
        total = hf.new_zeros(())
        count = tf.new_zeros(())
        for i in range(0, hf.size(0), chunk):
            lo = F.linear(hf[i:i + chunk], Wt)
            total = total + F.cross_entropy(lo, tf[i:i + chunk], ignore_index=-100, reduction="sum")
            count = count + (tf[i:i + chunk] != -100).sum()
        return total / count.clamp(min=1)

    def lm_loss(self, input_ids: torch.Tensor, targets: torch.Tensor,
                detach_memory: bool = False, compress_scale: float = 1.0) -> dict:
        """`compress_scale` in [0,1] ramps the compression pressure (see trainer warmup):
        early on it's ~0 so the router first learns to USE memory, then it rises so the
        model is rewarded for compressing what it safely can."""
        out = self.forward(input_ids, detach_memory=detach_memory, return_hidden=True)
        ce = self._cross_entropy(out.pop("hidden"), targets)
        aux = out["aux"]
        c = self.cfg
        total = (ce
                 + compress_scale * c.lambda_compute * aux["compute"]
                 + c.lambda_balance * aux["balance"]
                 - c.lambda_entropy * aux["entropy"])
        out.update({"loss": total, "ce": ce.detach(), "compress_scale": compress_scale})
        return out

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embed.weight.numel()
        return n

    def param_breakdown(self) -> dict:
        def c(m): return sum(p.numel() for p in m.parameters())
        return {
            "token_embed": self.token_embed.weight.numel(),
            "backbone": c(self.backbone),
            "scanner": c(self.scanner),
            "router": c(self.router),
            "compressor": c(self.compressor),
            "total": sum(p.numel() for p in self.parameters()),
        }
