"""Kaname Studio — interactive launcher (SOMA-style).

Pick model size (presets or custom dims), steps, batch, sequence length, and dataset
from a terminal menu; it shows the resulting parameter count, then hands off to the
live training dashboard.

    .venv\\Scripts\\python scripts\\studio.py
"""

import sys, os, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# cut CUDA fragmentation OOMs (must be set before torch inits the allocator)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from kaname.config import KanameConfig
from kaname.model import Kaname
from kaname.monitor.ui import C, banner, panel, gpu_stats
from kaname.train import Trainer, RandomTokenDataset, StreamingTextDataset, load_tokenizer

console = Console()

PRESETS = {                       # name: (d_model, n_layers, n_heads)
    "base":  (768, 12, 12),
    "large": (1024, 16, 16),
    "1.3b":  (2048, 20, 16),
}
DEFAULT_TOKENIZER = "AkiraXan/velvet-tok-100k-unigram"
FINEWEB = "HuggingFaceFW/fineweb-edu-score-2"


def meta_params(cfg):
    with torch.device("meta"):
        m = Kaname(cfg)
    bd = m.param_breakdown()
    return bd["total"], bd["total"] - bd["token_embed"]


def header():
    console.print(banner("KANAME"))
    g = gpu_stats()
    if g:
        line = (f"[bold {C.teal}]{g['name']}[/]  [{C.dim}]·[/]  "
                f"VRAM [white]{g['vram_used']:.1f}/{g['vram_total']:.1f} Go[/]  "
                f"[{C.dim}]·[/]  [white]{g['temp']:.0f}°C[/]")
    else:
        dev = "CPU (no CUDA)" if not torch.cuda.is_available() else torch.cuda.get_device_name(0)
        line = f"[bold {C.teal}]{dev}[/]"
    console.print(panel(line, "compressive LM · training studio", title_color=C.mint))


def choose_model():
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style=f"bold {C.mint}"); tbl.add_column(style=C.dim)
    for k, (d, L, h) in PRESETS.items():
        cfg = KanameConfig(d_model=d, n_layers=L, n_heads=h)
        tot, ne = meta_params(cfg)
        tbl.add_row(k, f"d={d} L={L} h={h}  →  {tot/1e6:.0f}M total / {ne/1e6:.0f}M non-embed")
    tbl.add_row("custom", "choose d_model / n_layers / n_heads yourself")
    console.print(panel(tbl, "model size", title_color=C.teal))

    choice = Prompt.ask(f"[{C.mint}]taille[/]", choices=list(PRESETS) + ["custom"], default="base")
    if choice == "custom":
        d = IntPrompt.ask(f"[{C.mint}]d_model[/]", default=768)
        L = IntPrompt.ask(f"[{C.mint}]n_layers[/]", default=12)
        h = IntPrompt.ask(f"[{C.mint}]n_heads[/]", default=max(1, d // 64))
        while d % h != 0:
            console.print(f"[{C.danger}]d_model doit être divisible par n_heads[/]")
            h = IntPrompt.ask(f"[{C.mint}]n_heads[/]", default=max(1, d // 64))
    else:
        d, L, h = PRESETS[choice]
    return choice, d, L, h


def run():
    header()
    if Prompt.ask(f"[{C.mint}]menu[/]", choices=["train", "quit"], default="train") == "quit":
        return

    preset, d, L, h = choose_model()
    seq = IntPrompt.ask(f"[{C.mint}]seq_len[/]", default=2048)
    cfg = KanameConfig(d_model=d, n_layers=L, n_heads=h, max_seq_len=seq)
    tot, ne = meta_params(cfg)
    console.print(panel(
        f"[white]{tot/1e6:.0f}M[/] params total  ·  [white]{ne/1e6:.0f}M[/] non-embedding  "
        f"[{C.dim}]({d}×{L}, {h} heads, seq {seq})[/]", "model", title_color=C.warm))

    batch = IntPrompt.ask(f"[{C.mint}]batch_size[/]", default=4)
    accum = IntPrompt.ask(f"[{C.mint}]grad_accum[/]", default=4)
    steps = IntPrompt.ask(f"[{C.mint}]max_steps[/]", default=2000)
    warmup = IntPrompt.ask(f"[{C.mint}]warmup_steps[/]", default=max(1, steps // 20))
    ds_kind = Prompt.ask(f"[{C.mint}]dataset[/]", choices=["fineweb", "random"], default="fineweb")
    use_velvet = Confirm.ask(f"[{C.mint}]Velvet optimizer ?[/]", default=True)
    run_name = Prompt.ask(f"[{C.mint}]run name[/]", default=f"kaname_{preset}")
    save_every = IntPrompt.ask(f"[{C.mint}]checkpoint every (0=off)[/]", default=0)
    # perf levers (see README): compile fuses kernels; grad-ckpt trades compute for VRAM.
    do_compile = Confirm.ask(f"[{C.mint}]torch.compile ? (faster, slow 1st steps)[/]", default=tot > 3e8)
    grad_ckpt = Confirm.ask(f"[{C.mint}]grad checkpointing ? (only if OOM)[/]", default=tot > 8e8)

    # ---- build dataset ----
    if ds_kind == "fineweb":
        tok = load_tokenizer(DEFAULT_TOKENIZER)
        cfg.vocab_size = tok.vocab_size
        console.print(f"[{C.dim}]tokenizer {tok.name} · vocab {tok.vocab_size} · eos {tok.eos_id}[/]")
        train_ds = StreamingTextDataset(FINEWEB, tok, seq, shuffle_buffer=10_000)
        nw = 0 if platform.system() == "Windows" else 4
    else:
        train_ds = RandomTokenDataset(cfg.vocab_size, seq, n_samples=8192)
        nw = 0

    eff_tokens = batch * accum * seq
    console.print(panel(
        f"[{C.mint}]{run_name}[/]  ·  {tot/1e6:.0f}M  ·  {ds_kind}  ·  "
        f"{steps} steps × [white]{eff_tokens:,}[/] tok = [white]{steps*eff_tokens/1e9:.2f}B[/] tokens  ·  "
        f"velvet={'on' if use_velvet else 'off (AdamW)'}", "launch", title_color=C.mint))
    if not Confirm.ask(f"[{C.mint}]go ?[/]", default=True):
        console.print(f"[{C.dim}]annulé.[/]")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Kaname(cfg)
    Trainer(
        model, cfg, train_ds, batch_size=batch, grad_accum=accum, max_steps=steps,
        warmup_steps=warmup, device=device, use_velvet=use_velvet, run_name=run_name,
        num_workers=nw, save_every=save_every, ckpt_dir=f"outputs/{run_name}",
        grad_checkpoint=grad_ckpt, compile=do_compile,
    ).train()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        console.print(f"\n[{C.dim}]bye.[/]")
