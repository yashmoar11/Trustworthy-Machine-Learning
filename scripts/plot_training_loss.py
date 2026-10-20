"""README step 10: plot the training loss of all five runs in one figure, results/training_loss.png.

Reads trainer_state.json from each finished model folder ($HOME/models/<name>, where train.sh copies it; set MODELS to
look somewhere else). Run from scripts/ with the sft env:  $HOME/envs/sft/bin/python plot_training_loss.py
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("MODELS", Path.home() / "models"))
RUNS = ["random_baseline", "filtered_random", "difficulty_weighted", "topic_balanced", "hardest"]


def ema(values, alpha=0.05):
    """Exponential moving average, to make the noisy logged loss (one value every 5 steps) easier to read."""
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


fig, ax = plt.subplots(figsize=(8, 4.5))
for run in RUNS:
    state = json.loads((MODELS / run / "trainer_state.json").read_text())
    logs = [x for x in state["log_history"] if "loss" in x]
    ax.plot([x["step"] for x in logs], ema([x["loss"] for x in logs]), label=run, linewidth=1.2)
for step in (1875, 3750):  # epoch boundaries: 15,000 rows / 8 per step = 1,875 steps per epoch
    ax.axvline(step, color="grey", linestyle=":", linewidth=0.8)
ax.set(xlabel="optimizer step", ylabel="training loss (smoothed)",
       title="Training loss, Qwen2.5-3B-Instruct on each 15K subset (3 epochs)")
ax.legend()
fig.savefig(REPO / "results" / "training_loss.png", dpi=150, bbox_inches="tight")
print(f"saved -> {REPO / 'results' / 'training_loss.png'}")
