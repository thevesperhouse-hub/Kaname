"""Ablation: does the compressed memory actually help? (the Kaname thesis)

For held-out long documents, compute next-token CE with the compressed memory ON vs
OFF (each segment isolated to its local window), broken down by segment position.
If the memory carries useful cross-segment information, the ON curve should sit below
OFF, with the gap GROWING for later segments (more compressed past to draw on).

Also reports bits-per-byte (BPB) — tokenizer-independent, the number to quote.

  python scripts/ablate.py --ckpt outputs/kaname_1.3b_001/ckpt_step200000.pt \
      --eval-len 4096 --docs 100
"""

import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from kaname.config import KanameConfig
from kaname.model import Kaname
from kaname.train import load_tokenizer

FINEWEB = "HuggingFaceFW/fineweb-edu-score-2"
DEFAULT_TOK = "AkiraXan/velvet-tok-100k-unigram"
LN2 = math.log(2.0)


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = KanameConfig(**ck["cfg"])
    m = Kaname(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, cfg, ck.get("step", "?")


@torch.no_grad()
def per_pos_ce(model, x, y, device, use_memory):
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") \
        else torch.autocast("cpu", enabled=False)
    with ac:
        logits = model(x, use_memory=use_memory)["logits"][0].float()   # (L, V)
    return F.cross_entropy(logits, y[0], reduction="none")               # (L,) nats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--eval-len", type=int, default=4096)
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model, cfg, step = load_model(args.ckpt, args.device)
    W = cfg.segment_size
    n_seg = args.eval_len // W
    print(f"loaded step {step} · {model.num_params()/1e6:.0f}M non-embed · "
          f"eval_len {args.eval_len} ({n_seg} segments of {W})\n")

    from datasets import load_dataset
    ds = load_dataset(FINEWEB, split="train", streaming=True).shuffle(seed=7, buffer_size=2000)

    seg_on = torch.zeros(n_seg); seg_off = torch.zeros(n_seg); seg_cnt = torch.zeros(n_seg)
    nats_on = nats_off = n_tok = n_bytes = 0.0
    done = 0
    for ex in ds:
        text = ex.get("text")
        if not text:
            continue
        ids = tok.encode(text)[: args.eval_len + 1]
        if len(ids) < 2 * W + 1:            # need >=2 segments to be meaningful
            continue
        t = torch.tensor(ids, device=args.device)
        x, y = t[:-1][None], t[1:][None]
        L = x.shape[1]

        ce_on = per_pos_ce(model, x, y, args.device, True)
        ce_off = per_pos_ce(model, x, y, args.device, False)

        pos = torch.arange(L, device=args.device) // W
        for s in range(min(n_seg, int(pos.max()) + 1)):
            m = pos == s
            seg_on[s] += ce_on[m].sum().item(); seg_off[s] += ce_off[m].sum().item()
            seg_cnt[s] += m.sum().item()

        nats_on += ce_on.sum().item(); nats_off += ce_off.sum().item(); n_tok += L
        n_bytes += len(tok.decode(y[0].tolist()).encode("utf-8"))
        done += 1
        if done >= args.docs:
            break

    ce_on_all, ce_off_all = nats_on / n_tok, nats_off / n_tok
    bpb_on = nats_on / (LN2 * n_bytes)
    bpb_off = nats_off / (LN2 * n_bytes)

    print(f"{'segment':>8} {'CE on':>9} {'CE off':>9} {'gap(off-on)':>12}")
    for s in range(n_seg):
        if seg_cnt[s] == 0:
            continue
        on, off = seg_on[s] / seg_cnt[s], seg_off[s] / seg_cnt[s]
        print(f"{s:>8} {on:>9.4f} {off:>9.4f} {off-on:>12.4f}")

    print(f"\noverall  CE on {ce_on_all:.4f} | off {ce_off_all:.4f} | "
          f"ppl on {math.exp(ce_on_all):.2f}")
    print(f"BPB      on {bpb_on:.4f} | off {bpb_off:.4f}   (tokenizer-independent)")
    rel = 100 * (ce_off_all - ce_on_all) / ce_off_all
    print(f"\nMemory reduces overall CE by {rel:.1f}%.")
    late = n_seg - 1
    if seg_cnt[late] > 0 and seg_cnt[0] > 0:
        g0 = (seg_off[0] - seg_on[0]) / seg_cnt[0]
        gl = (seg_off[late] - seg_on[late]) / seg_cnt[late]
        print(f"Gap grows from {g0:.4f} (segment 0) to {gl:.4f} (segment {late}) "
              f"-> {'memory helps MORE with more context (YES)' if gl > g0 else 'gap does not grow'}")


if __name__ == "__main__":
    main()
