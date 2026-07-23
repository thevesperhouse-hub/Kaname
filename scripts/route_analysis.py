"""Is the ACR router content-adaptive, or does it route noise?

The router sends each segment to SKIM / PROCESS / FOCUS (increasing compute + memory
slots). If it is intelligent, it should send information-dense / hard segments to FOCUS
and boilerplate to SKIM. We test that directly:

  - route per segment  = argmax of the router (from the normal forward)
  - difficulty per seg = local next-token CE with memory OFF (intrinsic hardness)

If mean difficulty rises SKIM -> PROCESS -> FOCUS (positive correlation), the routing
is content-adaptive. Flat / no correlation => the router isn't doing real work yet.

  python scripts/route_analysis.py --ckpt outputs/kaname_1.3b_001/ckpt_step200000.pt \
      --eval-len 2048 --docs 200
"""

import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from kaname.config import KanameConfig
from kaname.model import Kaname
from kaname.train import load_tokenizer

FINEWEB = "HuggingFaceFW/fineweb-edu-score-2"
DEFAULT_TOK = "AkiraXan/velvet-tok-100k-unigram"
ROUTES = ["SKIM", "PROCESS", "FOCUS"]


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = KanameConfig(**ck["cfg"])
    m = Kaname(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, cfg, ck.get("step", "?")


def _ac(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") \
        else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--eval-len", type=int, default=2048)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--examples", type=int, default=3, help="sample segments to print per route")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model, cfg, step = load_model(args.ckpt, args.device)
    W = cfg.segment_size
    n_seg = args.eval_len // W
    print(f"loaded step {step} · eval_len {args.eval_len} ({n_seg} segments)\n")

    from datasets import load_dataset
    ds = load_dataset(FINEWEB, split="train", streaming=True).shuffle(seed=11, buffer_size=2000)

    diff = {r: [] for r in range(3)}          # intrinsic difficulty per route
    wprio = {r: [] for r in range(3)}         # write-priority per route
    examples = {r: [] for r in range(3)}
    done = 0
    for ex in ds:
        text = ex.get("text")
        if not text:
            continue
        ids = tok.encode(text)
        if len(ids) < args.eval_len + 1:
            continue
        ids = ids[: args.eval_len + 1]
        t = torch.tensor(ids, device=args.device)
        x, y = t[:-1][None], t[1:][None]

        with _ac(args.device):
            out = model(x)                                        # normal forward (routing)
            logits_off = model(x, use_memory=False)["logits"][0].float()
        routes = out["seg_routes"][0].argmax(-1)                 # (n_seg,)
        wp = out["seg_write_priority"][0]                        # (n_seg,)
        ce = F.cross_entropy(logits_off, y[0], reduction="none")  # (L,) local difficulty
        seg_ce = ce.reshape(n_seg, W).mean(-1)                    # (n_seg,)

        for s in range(n_seg):
            r = int(routes[s])
            diff[r].append(float(seg_ce[s])); wprio[r].append(float(wp[s]))
            if len(examples[r]) < args.examples:
                snip = tok.decode(ids[s * W:(s + 1) * W][:40]).replace("\n", " ")
                examples[r].append((float(seg_ce[s]), snip[:90]))
        done += 1
        if done >= args.docs:
            break

    total = sum(len(diff[r]) for r in range(3))
    print(f"segments analyzed: {total}  ({done} docs)\n")
    print(f"{'route':>8} {'share':>7} {'mean difficulty':>16} {'mean write_prio':>16}")
    means = {}
    for r in range(3):
        n = len(diff[r])
        if n == 0:
            print(f"{ROUTES[r]:>8} {0.0:>6.1%} {'--':>16} {'--':>16}"); means[r] = None; continue
        md = float(np.mean(diff[r])); mw = float(np.mean(wprio[r])); means[r] = md
        print(f"{ROUTES[r]:>8} {n/total:>6.1%} {md:>16.4f} {mw:>16.4f}")

    # correlation between route compute-level (0,1,2) and segment difficulty
    lvl = np.concatenate([[r] * len(diff[r]) for r in range(3)])
    dv = np.concatenate([diff[r] for r in range(3)])
    corr = float(np.corrcoef(lvl, dv)[0, 1]) if dv.std() > 0 else 0.0
    print(f"\ncorrelation(route level, segment difficulty) = {corr:+.3f}")

    ordered = [means[r] for r in range(3) if means[r] is not None]
    monotone = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
    if corr > 0.1 and monotone:
        print("VERDICT: routing is CONTENT-ADAPTIVE — harder segments go to FOCUS (YES)")
    elif corr > 0.05:
        print("VERDICT: weak but present content-adaptivity")
    else:
        print("VERDICT: routing shows NO clear link to content difficulty (router not doing real work yet)")

    for r in range(3):
        if examples[r]:
            print(f"\n[{ROUTES[r]}] example segments (difficulty, text):")
            for d, s in examples[r]:
                print(f"   {d:.2f}  \"{s}\"")


if __name__ == "__main__":
    main()
