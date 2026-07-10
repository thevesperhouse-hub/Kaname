"""Training loop for Kaname.

Wires the model + Velvet optimizer + live dashboard together. Handles grad
accumulation, bf16 autocast, cosine LR schedule with warmup, grad clipping, feeds
the loss into Velvet's LVS/PGM controllers, and — for long Lambda runs — checkpoint
save/resume, gradient checkpointing, and torch.compile.
"""

import os
import glob
import math
import time
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..model import Kaname
from ..optim import VelvetOptimizer
from ..monitor import TrainingDashboard


def cosine_lr(step, base_lr, min_lr, warmup, total):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


class Trainer:
    def __init__(self, model: Kaname, cfg, train_ds, *, lr=3e-4, min_lr=3e-5,
                 weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0, batch_size=8,
                 grad_accum=4, max_steps=1000, warmup_steps=100, device="cuda",
                 amp_dtype="bf16", use_velvet=True, run_name="kaname", tui=True,
                 log_every=1, num_workers=0, ckpt_dir="outputs", save_every=0,
                 keep_last_n=3, resume=None, grad_checkpoint=False, compile=False):
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True          # fixed shapes -> pick fast kernels
            torch.set_float32_matmul_precision("high")     # TF32 for any fp32 matmuls
        self.model = model.to(device)
        if grad_checkpoint:
            self.model.set_grad_checkpoint(True)
        self.cfg = cfg
        self.device = device
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.max_steps = max_steps
        self.base_lr, self.min_lr, self.warmup = lr, min_lr, warmup_steps
        self.log_every = log_every
        self.run_name = run_name
        self.tui = tui
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.start_step = 0

        is_iter = isinstance(train_ds, IterableDataset)
        self.loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=not is_iter, drop_last=True,
            num_workers=num_workers, persistent_workers=num_workers > 0,
            pin_memory=device.startswith("cuda"),
        )
        self.opt = VelvetOptimizer(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,
            entropy_adaptive=use_velvet, perplexity_guided=use_velvet,
        )
        self.opt.set_training_steps(max_steps)

        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.use_amp = device.startswith("cuda")

        if resume:
            self._load_ckpt(resume)
        # compile AFTER loading weights (compiled module state_dict keys differ)
        if compile:
            self.model = torch.compile(self.model)

    # ---- checkpointing ----
    def _save_ckpt(self, step, dash=None):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        raw = getattr(self.model, "_orig_mod", self.model)   # unwrap torch.compile
        path = os.path.join(self.ckpt_dir, f"ckpt_step{step}.pt")
        torch.save({
            "step": step, "model": raw.state_dict(), "optim": self.opt.state_dict(),
            "cfg": self.cfg.to_dict(),
        }, path)
        if dash:
            dash.log(f"checkpoint saved: {os.path.basename(path)}", step)
        if self.keep_last_n > 0:
            cks = sorted(glob.glob(os.path.join(self.ckpt_dir, "ckpt_step*.pt")),
                         key=lambda p: int(p.split("step")[-1].split(".")[0]))
            for old in cks[:-self.keep_last_n]:
                try: os.remove(old)
                except OSError: pass

    def _load_ckpt(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        raw = getattr(self.model, "_orig_mod", self.model)
        raw.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optim"])
        self.start_step = ck["step"]
        print(f"resumed from {path} at step {self.start_step}")

    # ---- data ----
    def _batches(self):
        while True:
            for xb, yb in self.loader:
                yield xb, yb

    # ---- train ----
    def train(self):
        self.model.train()
        gen = self._batches()
        vocab = self.cfg.vocab_size
        tokens_per_step = self.batch_size * self.grad_accum * self.cfg.max_seq_len

        # tui=True auto-detects a TTY (live dashboard) vs a pipe/nohup (plain logs);
        # tui=False forces plain logs regardless.
        enabled = None if self.tui else False
        with TrainingDashboard(self.max_steps, self.run_name, enabled=enabled) as dash:
            if self.start_step:
                dash.log(f"resumed at step {self.start_step}", self.start_step)
                dash._progress.update(dash._task, completed=self.start_step)
            try:
                for step in range(self.start_step + 1, self.max_steps + 1):
                    t0 = time.time()
                    lr = cosine_lr(step, self.base_lr, self.min_lr, self.warmup, self.max_steps)
                    for g in self.opt.param_groups:
                        g["lr"] = lr

                    cscale = min(1.0, step / max(self.cfg.compress_warmup_steps, 1))
                    self.opt.zero_grad(set_to_none=True)
                    loss_acc, ce_acc, last = 0.0, 0.0, None
                    for _ in range(self.grad_accum):
                        xb, yb = next(gen)
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        if self.use_amp:
                            with torch.autocast("cuda", dtype=self.amp_dtype):
                                out = self.model.lm_loss(xb, yb, compress_scale=cscale)
                        else:
                            out = self.model.lm_loss(xb, yb, compress_scale=cscale)
                        (out["loss"] / self.grad_accum).backward()
                        loss_acc += out["loss"].item() / self.grad_accum
                        ce_acc += out["ce"].item() / self.grad_accum
                        last = out

                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.set_loss_metrics(ce_acc, vocab)
                    self.opt.step()

                    dt = time.time() - t0
                    st = last["stats"]
                    if step % self.log_every == 0:
                        dash.update({
                            "step": step, "loss": loss_acc, "ce": ce_acc,
                            "ppl": math.exp(min(ce_acc, 20)),
                            "grad_norm": gnorm.item(), "eff_lr": self.opt.effective_lr,
                            "lr_scale": self.opt.lr_scale, "beta1": self.opt.effective_beta1,
                            "bursting": self.opt.is_bursting,
                            "route_dist": st["route_dist"].tolist(),
                            "eff_slots": st["eff_slots"].item(),
                            "compression_ratio": st["compression_ratio"].item(),
                            "mem_slots": st["mem_slots"],
                            "tok_s": tokens_per_step / max(dt, 1e-6),
                        })
                    if self.save_every and step % self.save_every == 0:
                        self._save_ckpt(step, dash)
            except KeyboardInterrupt:
                dash.log("interrupted — saving", step)
                self._save_ckpt(step, dash)
                raise
        if self.save_every:
            self._save_ckpt(self.max_steps)
        return loss_acc
