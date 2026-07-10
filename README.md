# Kaname

Single-tower **compressive decoder** with an **adaptive KV memory**. Unbounded context
at bounded memory: the past is compressed into a small, learned set of memory slots that
the decoder re-reads at every step. The compression ratio is chosen per segment by an
Adaptive Computation Routing (ACR) module.

## Why v2 (vs v1)

v1 used two separate transformer towers (local + global) plus a compressor and memory.
The concept path was masked out of the loss for the first segment and heavily attenuated
elsewhere, and it was detached from the local encoder — so the "reasoning" machinery
barely touched the training signal, while still costing parameters. v2 fixes the root
causes:

| Concern | v1 | v2 |
|---|---|---|
| Parameters | two towers + compressor + memory | **one shared tower**; compressor is a small module |
| Causality | expansion mask made the concept path a no-op on segment 0 | **causal by construction**; verified: `max\|Δlogits\|` for past positions = `0.0` |
| Gradient to compressor | `local_out.detach()` | full BPTT through the segment-recurrent memory chain |
| KV cache | awkward, bolted on | the compressed memory **is** the cache (bounded, StreamingLLM eviction) |
| ACR | gated *layers* (broke gradient) | routes the **compression ratio** (SKIM/PROCESS/FOCUS) |

## Architecture

```
for each segment i (W tokens):
    h_i    = backbone(embed(seg_i), memory=compressed_past)   # causal self-attn + attend past memory
    route  = scanner(h_i) -> router                            # SKIM / PROCESS / FOCUS (Gumbel-ST)
    slots  = compressor(h_i)            (B, k_max, D)          # learned cross-attention bottleneck
    gate   = slot_gates(route)          (B, k_max) in [0,1]    # differentiable effective slot count
    memory = append(memory, slots, gate)                       # bounded; evict oldest
logits = lm_head(cat(h_i))
```

- RoPE positions **restart each segment**; cross-segment order is carried by memory, so
  context length never stretches the rotary range.
- Loss = CE + `λ_compute·compute` + `λ_balance·balance` − `λ_entropy·entropy`.

## Layout

```
kaname/
  config.py            KanameConfig (drives model + training)
  model/
    layers.py          RMSNorm, RoPE, SwiGLU, MemoryAttention (memory-augmented causal attn)
    backbone.py        the single shared decoder tower
    compressor.py      SegmentCompressor (W tokens -> k_max slots)
    router.py          ACR: SegmentScanner + AdaptiveRouter + slot_gates + aux losses
    kaname.py          top-level segment-recurrent forward + lm_loss
  optim/velvet.py      Velvet optimizer (AdamW + PGM + LVS); bias-correction fix vs v1
  train/               trainer, datasets (bin/random)
  monitor/dashboard.py live rich TUI (loss, Velvet state, ACR/memory)
scripts/
  smoke_test.py        plumbing + causality + overfit checks (CPU)
  train.py             entry point
configs/base.yaml      ~150M non-embedding, vocab 100k
```

## Run

```bash
# venv already at .venv (torch cu128 for Blackwell / RTX 50xx)
.venv/Scripts/python scripts/studio.py             # interactive launcher: pick size/steps/dataset
.venv/Scripts/python scripts/smoke_test.py         # CPU sanity + causality proof
.venv/Scripts/python scripts/preview_dashboard.py  # live TUI demo (synthetic data)

# real run: stream FineWeb-Edu-score-2, tokenize on the fly with the v1 tokenizer
.venv/Scripts/python scripts/train.py --config configs/base.yaml \
    --hf-path HuggingFaceFW/fineweb-edu-score-2 \
    --tokenizer AkiraXan/velvet-tok-100k-unigram \
    --max-steps 2000 --batch-size 4 --grad-accum 4

.venv/Scripts/python scripts/train.py ... --no-velvet   # clean AdamW baseline
```

Data: the full FineWeb-Edu-score-2 subsample is ~1.3T tokens, so it is **streamed**
(tokenized on the fly, packed into seq_len windows) rather than pre-tokenized — same code
path scales to an H100 box. For small slices you can instead pre-tokenize to a flat
`uint32` `.bin` and pass `--data` (uint32 because vocab=100k exceeds uint16). Note: the
v1 tokenizer is French-trained; on English data it is less token-efficient (fine for now).

The ACR router would otherwise collapse to SKIM before learning to use memory, so
`compress_warmup_steps` ramps the compression penalty 0→1 (default 2000 steps).

## Deploy on Lambda (H100)

`configs/1_3b.yaml` — d_model 2048 × 20 layers, ~1.31B total (~1.11B non-embedding).

```bash
bash scripts/setup_lambda.sh     # venv + torch cu128 + pinned requirements.txt
bash scripts/run_lambda.sh       # nohup background run + logs/, resumable checkpoints
TUI=1 bash scripts/run_lambda.sh # foreground live dashboard (run inside tmux)
RESUME=outputs/kaname_1p3b_fineweb_edu/ckpt_step12000.pt bash scripts/run_lambda.sh
```

Long-run infra baked into the trainer: checkpoint save/resume (`--save-every`,
`--resume`; the Velvet LVS/PGM controller state and router step are persisted too),
activation checkpointing (`--grad-checkpoint`, needed to fit 1.3B with a decent
batch), and `--compile`. Single H100-80GB is enough for 1.3B; multi-GPU DDP is a
follow-up (the streaming dataset is already rank/worker shard-aware).

## Performance

Attention uses `F.scaled_dot_product_attention` (FlashAttention / cuDNN on a Linux
CUDA build; a fused fallback elsewhere) — not a hand-rolled softmax. Two more levers:

- **`--compile`** (torch.compile): big win via kernel fusion. First steps are slow
  (compilation); because `seq_len` is fixed the memory only takes a few shapes, so it
  compiles a handful of graphs once, then runs fast. Recommended on.
- **`--grad-checkpoint`**: recomputes each segment's forward in backward (~+33% compute)
  to save activation memory. With SDPA freeing memory, prefer it **off**; enable only if
  you OOM. On Lambda: `COMPILE=1 GC=0 bash scripts/run_lambda.sh` (drop batch if OOM).

Baseline first run was ~14% MFU (hand-rolled attention, no compile, grad-checkpoint on);
these close most of that gap.

## Evaluating Velvet honestly

`--no-velvet` gives an AdamW baseline from the *same* optimizer class. To claim a win,
run both at identical base LR + schedule + multiple seeds and compare held-out loss.
LVS changes the effective LR trajectory, so an uncontrolled comparison is not evidence.

## Status

Working end-to-end: forward/backward, causality verified (Δ=0 on past), Velvet overfits
a tiny batch, streaming FineWeb-Edu-score-2 trains (CE 11.0→6.6 in 40 steps on an RTX
5080, ~15k tok/s, 175M params bf16), checkpoint save/resume + grad checkpointing verified,
1.3B config ready for the H100 run.

Next: an ablation harness — the money-plot (val loss with/without memory, and vs
compression ratio / context length) — to prove the compressed memory actually helps.
