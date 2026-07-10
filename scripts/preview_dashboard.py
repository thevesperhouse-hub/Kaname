"""Live demo of the training dashboard with synthetic data (no dataset needed).

Run it in a real terminal to see the full colored, animated TUI:
    .venv\\Scripts\\python scripts\\preview_dashboard.py
"""

import sys, os, time, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaname.monitor.dashboard import TrainingDashboard

TOTAL = 2000


def main():
    dash = TrainingDashboard(total_steps=TOTAL, run_name="demo_fineweb_fr")
    with dash:
        for step in range(1, TOTAL + 1):
            i = step
            ce = 3.4 + 4.6 * math.exp(-i / 300) + random.uniform(-0.03, 0.03)
            ratio = 6 + 6 * (1 - math.exp(-i / 250)) + 3 * math.sin(i / 30) + random.uniform(-0.6, 0.6)
            slots = max(1.0, 16 / max(ratio, 1e-3) * 8)
            bursting = 900 < step < 980
            if step == 900:
                dash.log("plateau detected → velvet burst", step)
            if step % 500 == 0:
                dash.log(f"checkpoint saved: ckpt_{step}.pt", step)
            dash.update({
                "step": step, "loss": ce + 0.05, "ce": ce, "ppl": math.exp(min(ce, 20)),
                "grad_norm": 0.4 + random.uniform(-0.1, 0.1),
                "tok_s": 40000 + random.uniform(-3000, 3000),
                "eff_lr": 2.5e-4 * (1.3 if bursting else 1.0),
                "lr_scale": 1.3 if bursting else 1.0 + random.uniform(-0.02, 0.02),
                "beta1": 0.89, "bursting": bursting,
                "route_dist": [0.6 - 0.001 * (i % 100), 0.3, 0.1 + 0.001 * (i % 100)],
                "eff_slots": slots, "compression_ratio": ratio, "mem_slots": min(512, step),
            })
            time.sleep(0.02)


if __name__ == "__main__":
    main()
