"""Smoke test: plumbing + causality + a few real optimizer steps, all on CPU.

The causality check is the one that matters: changing an input token at position p
must not alter any logits at positions < p. v1 lost this guarantee inside the
expansion/compression path; v2 is causal by construction, and this test proves it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kaname.config import KanameConfig
from kaname.model import Kaname


def tiny_cfg():
    return KanameConfig(
        vocab_size=512, d_model=64, n_layers=2, n_heads=4, ffn_mult=4.0,
        segment_size=8, k_max=4, k_process=2, k_skim=1, n_compress_layers=2,
        max_memory_slots=32, max_seq_len=32,
    )


def test_forward_backward():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = Kaname(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model.lm_loss(x, y)
    out["loss"].backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    assert torch.isfinite(out["loss"]), "loss not finite"
    print(f"[forward/backward] loss={out['loss'].item():.4f} ce={out['ce'].item():.4f} "
          f"grads flowing to {n_grad}/{n_total} tensors")
    print(f"[stats] {  {k: (round(v.item(),3) if torch.is_tensor(v) else v) for k,v in out['stats'].items() if k!='route_dist'} }")
    return model, cfg


@torch.no_grad()
def test_causality(model, cfg):
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    l1 = model(x)["logits"]
    p = cfg.max_seq_len - 1
    x2 = x.clone()
    # change the input token at position p to something different
    x2[0, p] = (x2[0, p] + 1) % cfg.vocab_size
    l2 = model(x2)["logits"]
    diff_before = (l1[:, :p] - l2[:, :p]).abs().max().item()
    diff_at = (l1[:, p:] - l2[:, p:]).abs().max().item()
    print(f"[causality] max|dlogits| positions < p : {diff_before:.2e}  (must be ~0)")
    print(f"[causality] max|dlogits| positions >= p: {diff_at:.2e}  (should be > 0)")
    assert diff_before < 1e-4, "CAUSALITY VIOLATION: past logits changed when a future token changed"
    assert diff_at > 1e-6, "changing a token had no effect at all — something is disconnected"
    print("[causality] PASS  OK")


def test_velvet_steps(model, cfg):
    from kaname.optim import VelvetOptimizer
    model.train()
    opt = VelvetOptimizer(model.parameters(), lr=1e-3)
    opt.set_training_steps(50)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    losses = []
    for i in range(20):
        opt.zero_grad()
        out = model.lm_loss(x, y)
        out["loss"].backward()
        opt.set_loss_metrics(out["ce"].item(), cfg.vocab_size)
        opt.step()
        losses.append(out["ce"].item())
    print(f"[velvet] ce {losses[0]:.4f} -> {losses[-1]:.4f} over 20 steps "
          f"(overfit tiny batch), lr_scale={opt.lr_scale:.3f}, β1={opt.effective_beta1:.3f}")
    assert losses[-1] < losses[0], "model failed to overfit a fixed tiny batch"
    print("[velvet] PASS  OK")


def main():
    print("=" * 64)
    model, cfg = test_forward_backward()
    bd = model.param_breakdown()
    print(f"[params] {  {k: f'{v/1e3:.1f}K' for k,v in bd.items()} }")
    print("-" * 64)
    test_causality(model, cfg)
    print("-" * 64)
    test_velvet_steps(model, cfg)
    print("=" * 64)
    print("ALL SMOKE TESTS PASSED  OK")


if __name__ == "__main__":
    main()
