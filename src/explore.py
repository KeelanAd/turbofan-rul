"""Block 1: plot every sensor against cycle for a few engines.

Run with:  python src/explore.py
Saves figures/sensors_vs_cycle.png and prints how much each column varies.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from data import load_cmapss

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# A handful of engines is enough to see the trend without clutter
ENGINES = [1, 2, 3, 4]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, one per engine

train = load_cmapss("train_FD001.txt")
signal_columns = [c for c in train.columns if c not in ("unit", "cycle")]

# One small panel per column (3 op settings + 21 sensors = 24 panels)
fig, axes = plt.subplots(6, 4, figsize=(16, 18))
for ax, column in zip(axes.flat, signal_columns):
    for engine, colour in zip(ENGINES, COLOURS):
        one_engine = train[train["unit"] == engine]
        ax.plot(one_engine["cycle"], one_engine[column], color=colour, linewidth=1.2,
                label=f"Engine {engine}")
    ax.set_title(column, fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# One shared legend, and a shared x-axis label
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.suptitle("FD001: every sensor against cycle, first 4 engines", fontsize=14, y=0.995)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982),
           ncol=len(ENGINES), frameon=False, fontsize=11)
fig.supxlabel("Cycle", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.965))

FIGURES_DIR.mkdir(exist_ok=True)
fig.savefig(FIGURES_DIR / "sensors_vs_cycle.png", dpi=110)
print("Saved", FIGURES_DIR / "sensors_vs_cycle.png")

# Numbers to back up the plots: how much does each column vary across ALL engines?
print("\nStandard deviation and number of distinct values per column:")
summary = train[signal_columns].agg(["std", "nunique"]).T.sort_values("std")
print(summary.to_string())
