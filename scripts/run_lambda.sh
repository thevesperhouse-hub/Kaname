#!/usr/bin/env bash
# Launch the 1.3B run on Lambda. Streams FineWeb-Edu-score-2, checkpoints, resumable.
#
#   bash scripts/run_lambda.sh                 # background + logs (nohup)
#   TUI=1 bash scripts/run_lambda.sh           # foreground live dashboard (use inside tmux)
#   RESUME=outputs/kaname_1p3b/ckpt_step12000.pt bash scripts/run_lambda.sh
set -euo pipefail
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# export HF_TOKEN=hf_xxx          # optional: higher HF rate limits / faster streaming

CONFIG=${CONFIG:-configs/1_3b.yaml}
RUN=${RUN:-kaname_1p3b_fineweb_edu}
STEPS=${STEPS:-100000}
BATCH=${BATCH:-8}          # per H100-80GB, seq 2048, bf16 + grad checkpointing
ACCUM=${ACCUM:-8}          # -> effective 131k tokens/step
SEQ=${SEQ:-2048}
WORKERS=${WORKERS:-8}      # on-the-fly tokenization throughput
RESUME=${RESUME:-}

mkdir -p logs "outputs/$RUN"
RESUME_ARG=""; [ -n "$RESUME" ] && RESUME_ARG="--resume $RESUME"
TUI_ARG="--no-tui"; [ "${TUI:-0}" = "1" ] && TUI_ARG=""
# perf: COMPILE=1 fuses kernels (big win, slow first steps). GC=1 enables gradient
# checkpointing (trades ~33% compute for VRAM) — only if you OOM without it.
COMPILE_ARG=""; [ "${COMPILE:-1}" = "1" ] && COMPILE_ARG="--compile"
GC_ARG="";      [ "${GC:-0}" = "1" ]      && GC_ARG="--grad-checkpoint"

CMD=(python scripts/train.py
  --config "$CONFIG"
  --hf-path HuggingFaceFW/fineweb-edu-score-2
  --tokenizer AkiraXan/velvet-tok-100k-unigram
  --seq-len "$SEQ" --batch-size "$BATCH" --grad-accum "$ACCUM"
  --max-steps "$STEPS" --warmup 2000 --lr 3e-4 --min-lr 3e-5
  --num-workers "$WORKERS" --shuffle-buffer 10000
  $COMPILE_ARG $GC_ARG
  --ckpt-dir "outputs/$RUN" --save-every 1000 --keep-last-n 5
  --run-name "$RUN" $RESUME_ARG $TUI_ARG)

if [ "${TUI:-0}" = "1" ]; then
  exec "${CMD[@]}"
else
  nohup "${CMD[@]}" > "logs/$RUN.log" 2>&1 &
  echo "launched PID $! -> tail -f logs/$RUN.log"
fi
