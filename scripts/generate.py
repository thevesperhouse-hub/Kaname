"""Generate from / evaluate a trained Kaname checkpoint.

  # sample text from a prompt
  python scripts/generate.py --ckpt outputs/kaname_1.3b_001/ckpt_step100000.pt \
      --prompt "The mitochondria is" --max-new 200

  # interactive REPL
  python scripts/generate.py --ckpt .../ckpt_step100000.pt

  # perplexity on held-out FineWeb-Edu samples
  python scripts/generate.py --ckpt .../ckpt_step100000.pt --eval --eval-docs 50

Generation re-runs the full forward each step (O(n^2) but fine for testing a few
hundred tokens); the model is causal so the last real position's logits predict next.
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


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = KanameConfig(**ck["cfg"])
    model = Kaname(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("step", "?")


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") \
        else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=200, temperature=0.8, top_k=50):
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        with _autocast(device):
            logits = model(x)["logits"][:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, nxt], dim=1)
        if nxt.item() == tok.eos_id:
            break
    return tok.decode(x[0].tolist())


@torch.no_grad()
def perplexity(model, tok, device, n_docs=50, seq_len=2048):
    from kaname.train import StreamingTextDataset
    ds = StreamingTextDataset(FINEWEB, tok, seq_len, shuffle_buffer=0, seed=1234)
    it = iter(ds)
    tot, n = 0.0, 0
    for _ in range(n_docs):
        x, y = next(it)
        x, y = x[None].to(device), y[None].to(device)
        with _autocast(device):
            out = model.lm_loss(x, y)
        tot += out["ce"].item(); n += 1
    ce = tot / max(n, 1)
    return ce, math.exp(min(ce, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-docs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model, cfg, step = load_model(args.ckpt, args.device)
    print(f"loaded {args.ckpt} (step {step}) · {model.num_params()/1e6:.0f}M non-embed · "
          f"tokenizer {tok.name}")

    if args.eval:
        ce, ppl = perplexity(model, tok, args.device, args.eval_docs, cfg.max_seq_len)
        print(f"\nheld-out ({args.eval_docs} docs):  ce {ce:.4f}   ppl {ppl:.2f}")

    if args.prompt is not None:
        print("\n" + "=" * 60)
        print(generate(model, tok, args.prompt, args.device,
                       args.max_new, args.temperature, args.top_k))
        print("=" * 60)
    elif not args.eval:
        print("\ninteractive mode — type a prompt (empty line or Ctrl-C to quit)\n")
        while True:
            try:
                p = input("prompt> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not p.strip():
                break
            print(generate(model, tok, p, args.device,
                           args.max_new, args.temperature, args.top_k) + "\n")


if __name__ == "__main__":
    main()
