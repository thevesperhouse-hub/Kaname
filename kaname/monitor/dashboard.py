"""Live training dashboard (rich), SOMA-style.

Real-time terminal app: gradient logo, run status, progress, a braille loss plot,
the ACR/memory compression state, the Velvet optimizer internals, live GPU telemetry,
and an event journal. Degrades to plain per-step logging when stdout is not a TTY.
"""

import time
from collections import deque

from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.console import Console, Group
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from .ui import C, banner, gradient, bar, panel, loss_plot, gpu_stats


def _kv(rows, key_style=C.dim, val_style=f"bold {C.val}"):
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(style=key_style)
    t.add_column(justify="right", style=val_style)
    for k, v in rows:
        t.add_row(k, v)
    return t


class TrainingDashboard:
    def __init__(self, total_steps: int, run_name: str = "kaname", enabled: bool = None):
        self.console = Console()
        self.total_steps = total_steps
        self.run_name = run_name
        self.enabled = self.console.is_terminal if enabled is None else enabled
        self.loss_hist = deque(maxlen=400)
        self.ratio_hist = deque(maxlen=400)
        self.journal = deque(maxlen=6)
        self.start = time.time()
        self._live = None
        self._gpu = None
        self._gpu_t = 0.0
        self._progress = Progress(
            TextColumn(f"[{C.mint}]{{task.description}}"),
            BarColumn(complete_style=C.mint, finished_style=C.teal, pulse_style=C.deep),
            TextColumn(f"[{C.dim}]{{task.completed}}/{{task.total}}"),
            TextColumn("·"), TimeElapsedColumn(), TextColumn("·"), TimeRemainingColumn(),
        )
        self._task = self._progress.add_task("training", total=total_steps)
        self.log("run initialized")

    def log(self, msg: str, step: int = None):
        self.journal.append((step, msg))

    # -- lifecycle --
    def __enter__(self):
        if self.enabled:
            self._live = Live(self._render({}), console=self.console,
                              refresh_per_second=8, screen=True)
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.__exit__(*exc)

    # -- update --
    def update(self, m: dict):
        self.loss_hist.append(m.get("ce", m.get("loss", 0.0)))
        self.ratio_hist.append(m.get("compression_ratio", 1.0))
        self._progress.update(self._task, completed=m.get("step", 0))
        now = time.time()
        if now - self._gpu_t > 1.0:
            self._gpu = gpu_stats()
            self._gpu_t = now
        if self._live is not None:
            self._live.update(self._render(m))
        else:
            self.console.print(
                f"step {m.get('step',0):>7} | loss {m.get('loss',0):.4f} "
                f"ce {m.get('ce',0):.4f} | lr {m.get('eff_lr',0):.2e} "
                f"scale {m.get('lr_scale',1):.3f} | ratio {m.get('compression_ratio',1):.1f}x "
                f"slots {m.get('eff_slots',0):.1f} mem {m.get('mem_slots',0):.0f} "
                f"| {m.get('tok_s',0):.0f} tok/s"
            )

    # -- rendering --
    def _render(self, m: dict):
        layout = Layout()
        layout.split_column(
            Layout(banner("KANAME"), name="logo", size=7),
            Layout(self._status(m), name="status", size=3),
            Layout(self._progress, name="prog", size=1),
            Layout(name="main", size=13),
            Layout(name="lower", size=10),
            Layout(self._journal_panel(), name="journal", size=6),
        )
        layout["main"].split_row(
            Layout(self._loss_panel(m), ratio=3),
            Layout(self._acr_panel(m), ratio=2),
        )
        layout["lower"].split_row(
            Layout(self._train_panel(m)),
            Layout(self._velvet_panel(m)),
            Layout(self._gpu_panel()),
        )
        return layout

    def _status(self, m):
        g = self._gpu or {}
        left = Text.assemble(
            (f" {self.run_name} ", f"bold {C.mint}"), ("·", C.dim),
            (f" step {m.get('step',0):,}/{self.total_steps:,} ", C.val), ("·", C.dim),
            (f" {m.get('tok_s',0):,.0f} tok/s ", C.warm),
        )
        right = Text.assemble(
            (f" {g.get('name','GPU')} ", f"bold {C.teal}"), ("·", C.dim),
            (f" VRAM {g.get('vram_used',0):.1f}/{g.get('vram_total',0):.1f} Go ", C.val),
            ("·", C.dim), (f" {g.get('temp',0):.0f}°C ", C.val),
        )
        grid = Table.grid(expand=True)
        grid.add_column(justify="left"); grid.add_column(justify="right")
        grid.add_row(left, right)
        return panel(grid, "status")

    def _loss_panel(self, m):
        title = f"loss · ce {m.get('ce',0):.4f} · ppl {m.get('ppl',0):.1f}"
        return panel(loss_plot(self.loss_hist, w_cells=44, h_cells=8), title,
                     color=C.deep, title_color=C.mint)

    def _acr_panel(self, m):
        rd = m.get("route_dist", [0, 0, 0])
        t = Table.grid(padding=(0, 1), expand=True)
        t.add_column(style=C.dim); t.add_column(); t.add_column(justify="right", style=f"bold {C.val}")
        for name, val in zip(("SKIM", "PROCESS", "FOCUS"), rd):
            t.add_row(name, bar(val, 14, fill=C.teal), f"{val:.2f}")
        t.add_row("", "", "")
        t.add_row("eff slots", bar(m.get("eff_slots", 0) / 16, 14, fill=C.mint), f"{m.get('eff_slots',0):.2f}")
        t.add_row("compress", "", Text(f"{m.get('compression_ratio',1):.1f}×", style=f"bold {C.warm}"))
        t.add_row("mem slots", "", f"{m.get('mem_slots',0):.0f}")
        return panel(t, "ACR · memory", color=C.deep, title_color=C.mint)

    def _train_panel(self, m):
        return panel(_kv([
            ("loss", f"{m.get('loss',0):.4f}"),
            ("ce", f"{m.get('ce',0):.4f}"),
            ("ppl", f"{m.get('ppl',0):.2f}"),
            ("grad_norm", f"{m.get('grad_norm',0):.3f}"),
        ]), "training")

    def _velvet_panel(self, m):
        state = Text("● BURST", style=f"bold {C.danger}") if m.get("bursting") \
            else Text("○ steady", style=C.dim)
        return panel(_kv([
            ("eff LR", f"{m.get('eff_lr',0):.2e}"),
            ("LR scale", f"{m.get('lr_scale',1):.3f}"),
            ("PGM β1", f"{m.get('beta1',0.9):.3f}"),
            ("state", state),
        ]), "velvet optim", color=C.deep, title_color=C.teal)

    def _gpu_panel(self):
        g = self._gpu
        if not g:
            return panel(Text("no GPU telemetry", style=C.dim), "gpu")
        t = Table.grid(padding=(0, 1), expand=True)
        t.add_column(style=C.dim); t.add_column(); t.add_column(justify="right", style=f"bold {C.val}")
        t.add_row("util", bar(g["util"] / 100, 12, fill=C.mint), f"{g['util']:.0f}%")
        t.add_row("vram", bar(g["vram_used"] / g["vram_total"], 12, fill=C.teal),
                  f"{g['vram_used']:.1f}/{g['vram_total']:.0f}G")
        t.add_row("temp", "", Text(f"{g['temp']:.0f}°C · {g['power']:.0f}W", style=C.val))
        return panel(t, "gpu", color=C.deep, title_color=C.teal)

    def _journal_panel(self):
        g = Group(*[
            Text.assemble(
                (f" {step:>7} " if step is not None else "         ", C.dim),
                (f"› {msg}", C.mint if i == len(self.journal) - 1 else C.dim),
            ) for i, (step, msg) in enumerate(self.journal)
        ]) if self.journal else Text(" …", style=C.dim)
        return panel(g, "journal", color=C.faint, title_color=C.dim)
