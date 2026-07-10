"""UI toolkit for the dashboard: teal theme, gradient banner, braille plots, GPU stats.

Aesthetic target: a cohesive mint/teal palette on near-black, rounded panels with the
title set into the top border, a pixel-art logo, and a real braille scatter for the loss.
"""

import subprocess
import pyfiglet
from rich.text import Text
from rich.panel import Panel
from rich import box


# ---- palette -------------------------------------------------------------
class C:
    mint = "#5eead4"      # primary accent
    teal = "#2dd4bf"
    deep = "#0d9488"      # borders
    light = "#99f6e4"     # gradient start
    dim = "#6b7280"       # secondary text
    faint = "#374151"     # empty bar track
    warm = "#fbbf24"      # time / highlights
    danger = "#f87171"
    val = "white"


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _lerp(a, b, t):
    ca, cb = _hex(a), _hex(b)
    return "#%02x%02x%02x" % tuple(round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))


def gradient(text: str, start=C.light, end=C.deep) -> Text:
    """Horizontal color gradient across a single line."""
    t = Text()
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        t.append(ch, style=_lerp(start, end, i / n))
    return t


def banner(word: str = "KANAME") -> Text:
    art = pyfiglet.figlet_format(word, font="ansi_regular").rstrip("\n").split("\n")
    width = max(len(l) for l in art)
    out = Text(justify="center")
    for r, line in enumerate(art):
        line = line.ljust(width)
        # vertical gradient (light at top -> deep at bottom), like the SOMA logo
        col_top = _lerp(C.light, C.mint, r / max(len(art) - 1, 1))
        col_bot = _lerp(C.mint, C.deep, r / max(len(art) - 1, 1))
        for c, ch in enumerate(line):
            out.append(ch, style=_lerp(col_top, col_bot, c / width))
        out.append("\n")
    return out


def bar(frac: float, width: int = 18, fill=C.mint, track=C.faint) -> Text:
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    t = Text()
    t.append("█" * n, style=fill)
    t.append("░" * (width - n), style=track)
    return t


def panel(body, title: str, color=C.deep, title_color=C.mint):
    return Panel(body, title=Text(title, style=f"bold {title_color}"),
                 title_align="center", border_style=color, box=box.ROUNDED,
                 padding=(0, 1))


# ---- braille scatter plot ------------------------------------------------
_BR = 0x2800
_DOTS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]  # [row%4][col%2]


class Braille:
    def __init__(self, w_cells: int, h_cells: int):
        self.wc, self.hc = w_cells, h_cells
        self.pw, self.ph = w_cells * 2, h_cells * 4
        self.g = [[0] * w_cells for _ in range(h_cells)]

    def set(self, x: int, y: int):
        if 0 <= x < self.pw and 0 <= y < self.ph:
            self.g[y // 4][x // 2] |= _DOTS[y % 4][x % 2]

    def lines(self):
        return ["".join(chr(_BR + c) for c in row) for row in self.g]


def loss_plot(values, w_cells=34, h_cells=8, color=C.mint):
    """Braille scatter of a value history, with min/max labels down the left edge."""
    body = Text()
    if len(values) < 2:
        return Text("… collecting …", style=C.dim)
    v = list(values)
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    cv = Braille(w_cells, h_cells)
    n = len(v)
    for i, val in enumerate(v):
        x = round(i / (n - 1) * (cv.pw - 1))
        y = round((1 - (val - lo) / rng) * (cv.ph - 1))
        cv.set(x, y)
    lines = cv.lines()
    for r, line in enumerate(lines):
        if r == 0:
            label = f"{hi:6.3f} "
        elif r == len(lines) - 1:
            label = f"{lo:6.3f} "
        else:
            label = "       "
        body.append(label, style=C.dim)
        body.append(line + "\n", style=color)
    return body


# ---- GPU telemetry -------------------------------------------------------
def gpu_stats():
    """Query nvidia-smi; returns dict or None. Cheap enough to call ~1x/sec."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().split("\n")[0]
        name, temp, power, used, total, util = [s.strip() for s in out.split(",")]
        return {"name": name, "temp": float(temp), "power": float(power),
                "vram_used": float(used) / 1024, "vram_total": float(total) / 1024,
                "util": float(util)}
    except Exception:
        return None
