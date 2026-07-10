"""Entry point: train Kaname.

Real run (streaming FineWeb-Edu-score-2, v1 tokenizer):
  python scripts/train.py --config configs/base.yaml \
      --hf-path HuggingFaceFW/fineweb-edu-score-2 \
      --tokenizer AkiraXan/velvet-tok-100k-unigram \
      --max-steps 2000 --batch-size 4 --grad-accum 4

Quick plumbing (random tokens, no download):
  python scripts/train.py --config configs/base.yaml --max-steps 20 --no-tui

AdamW baseline (same class, controlled comparison):
  python scripts/train.py ... --no-velvet
"""

import sys, os, argparse, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kaname.config import KanameConfig
from kaname.model import Kaname
from kaname.train import (
    Trainer, BinTokenDataset, RandomTokenDataset, StreamingTextDataset, load_tokenizer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--hf-path", default=None, help="HF dataset id to stream (e.g. HuggingFaceFW/fineweb-edu-score-2)")
    ap.add_argument("--hf-name", default=None, help="HF dataset config name (None = default/all)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", default=None, help="local dir or HF repo id")
    ap.add_argument("--data", default=None, help="pre-tokenized uint32 .bin (alternative to --hf-path)")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--no-velvet", action="store_true")
    ap.add_argument("--no-tui", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run-name", default="kaname")
    # long-run infra
    ap.add_argument("--ckpt-dir", default="outputs")
    ap.add_argument("--save-every", type=int, default=0, help="0 = no checkpoints")
    ap.add_argument("--keep-last-n", type=int, default=3)
    ap.add_argument("--resume", default=None, help="path to ckpt_stepN.pt")
    ap.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing (fit bigger models)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    args = ap.parse_args()

    cfg = KanameConfig.from_yaml(args.config)
    if args.seq_len:
        cfg.max_seq_len = args.seq_len
    seq_len = cfg.max_seq_len

    # ---- data ----
    tok = None
    if args.hf_path:
        assert args.tokenizer, "--tokenizer is required with --hf-path"
        tok = load_tokenizer(args.tokenizer)
        print(f"tokenizer: {tok.name}  vocab={tok.vocab_size}  eos={tok.eos_id}")
        if tok.vocab_size != cfg.vocab_size:
            print(f"  [!] config vocab_size {cfg.vocab_size} != tokenizer {tok.vocab_size}; using tokenizer's")
            cfg.vocab_size = tok.vocab_size
        # Windows DataLoader workers can't pickle the tokenizer closure -> force 0 there.
        nw = args.num_workers if args.num_workers is not None else (0 if platform.system() == "Windows" else 2)
        train_ds = StreamingTextDataset(
            args.hf_path, tok, seq_len, hf_name=args.hf_name, split=args.split,
            text_field=args.text_field, shuffle_buffer=args.shuffle_buffer,
        )
    elif args.data:
        train_ds = BinTokenDataset(args.data, seq_len)
        nw = args.num_workers or 0
    else:
        print("No --hf-path/--data: using RandomTokenDataset (plumbing only).")
        train_ds = RandomTokenDataset(cfg.vocab_size, seq_len, n_samples=4096)
        nw = 0

    # ---- model ----
    model = Kaname(cfg)
    print(f"params: total={model.param_breakdown()['total']/1e6:.1f}M  non-embed={model.num_params()/1e6:.1f}M")

    trainer = Trainer(
        model, cfg, train_ds, lr=args.lr, min_lr=args.min_lr,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_steps=args.max_steps, warmup_steps=args.warmup, device=args.device,
        use_velvet=not args.no_velvet, run_name=args.run_name, tui=not args.no_tui,
        num_workers=nw, ckpt_dir=args.ckpt_dir, save_every=args.save_every,
        keep_last_n=args.keep_last_n, resume=args.resume,
        grad_checkpoint=args.grad_checkpoint, compile=args.compile,
    )
    trainer.train()


if __name__ == "__main__":
    main()
