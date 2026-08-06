"""Generate manuscript figures from results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

INK = "#12304d"
MID = "#5b8db8"
PALE = "#b9cfe2"
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})


def strip_axes(ax, keep_left: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def figure1(data: pd.DataFrame, summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rng = np.random.default_rng(0)

    with_fake = data[data["fake_evidence"].notna()]
    series = [
        ("True key\n(n=63)", data["true_evidence"].to_numpy(), INK),
        ("Corrupted key\n(n=48)", with_fake["fake_evidence"].to_numpy(), MID),
        ("Permuted null\n(n=63)", data["true_null_evidence"].to_numpy(), PALE),
    ]

    for i, (label, values, colour) in enumerate(series):
        jitter = rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(i + jitter, values, s=17, color=colour, alpha=0.85,
                   edgecolor="white", linewidth=0.4, zorder=3)
        median = float(np.median(values))
        ax.plot([i - 0.28, i + 0.28], [median, median],
                color=TEXT, linewidth=1.6, zorder=4, solid_capstyle="butt")
        ax.annotate(f"median {median:.0f}", (i + 0.32, median),
                    fontsize=7, color=TEXT, va="center")

    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([label for label, _, _ in series])
    ax.set_ylabel("Columns explained")
    ax.set_yscale("symlog", linthresh=1, linscale=0.6)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylim(-0.35, 70)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    strip_axes(ax)

    ax.annotate(
        f"Every corrupted key rejected ({summary['sensitivity']:.0%}).\n"
        "A permuted key explained zero columns\nin all 63 datasets.",
        xy=(1.98, 12), fontsize=7, color=MUTED, ha="center", linespacing=1.5,
    )
    save(fig, "figure1_null_separation")


def figure2(data: pd.DataFrame, summary: dict) -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    rng = np.random.default_rng(1)
    with_signal = data[data["true_evidence"] > 0]

    order = [("clinical", "Clinical /\nbiomedical"), ("other", "Economic /\nother")]
    for i, (domain, label) in enumerate(order):
        values = with_signal.loc[with_signal["domain"] == domain,
                                 "true_evidence_frac"].to_numpy()
        colour = INK if domain == "clinical" else MID
        left.scatter(i + rng.uniform(-0.14, 0.14, len(values)), values,
                     s=19, color=colour, alpha=0.85,
                     edgecolor="white", linewidth=0.4, zorder=3)
        stats = summary["by_domain_given_signal"][domain]
        left.annotate(f"{stats['specificity']:.0%} accepted\nn={stats['n']}",
                      (i, 1.04), fontsize=7, color=TEXT, ha="center", va="bottom")

    left.axhline(0.40, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    left.annotate("acceptance threshold", (-0.44, 0.425), fontsize=7,
                  color=MUTED, ha="left")
    left.set_xticks(range(len(order)))
    left.set_xticklabels([label for _, label in order])
    left.set_ylabel("Share of columns explained")
    left.set_ylim(-0.04, 1.16)
    left.set_xlim(-0.5, 1.5)
    left.yaxis.grid(True, color=GRID, linewidth=0.5)
    left.set_axisbelow(True)
    left.set_title("Datasets where evidence exists", loc="left", pad=14)
    strip_axes(left)

    counts = [summary["n_with_signal"], summary["n_without_signal"]]
    labels = ["Evidence\navailable", "No persistent\nattribute"]
    bars = right.bar(labels, counts, color=[INK, PALE], width=0.55, zorder=3)
    for bar, count in zip(bars, counts):
        right.annotate(f"{count}", (bar.get_x() + bar.get_width() / 2, count + 1.2),
                       ha="center", fontsize=8, color=TEXT, fontweight="bold")
    right.set_ylabel("Datasets")
    right.set_ylim(0, max(counts) * 1.2)
    right.yaxis.grid(True, color=GRID, linewidth=0.5)
    right.set_axisbelow(True)
    right.set_title("The method requires signal to exist", loc="left", pad=14)
    right.annotate(
        "No method can validate a panel whose\ncolumns all vary legitimately.",
        (1, -0.30), xycoords="axes fraction", fontsize=7, color=MUTED,
        ha="right", va="top", linespacing=1.5,
    )
    strip_axes(right)

    save(fig, "figure2_domain_dependence")


def figure3(findings: dict) -> None:
    concealment = findings["concealment"]
    broken = concealment["under_positional_key"]
    recovered = concealment["under_recovered_key"]
    model_mae = concealment["published_model_mae"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    labels = ["Positional key\n(as used)", "Recovered key"]
    values = [broken["r2"], recovered["r2"]]
    bars = left.bar(labels, values, color=[PALE, INK], width=0.5, zorder=3)
    for bar, value in zip(bars, values):
        left.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value + 0.03),
                      ha="center", fontsize=8, color=TEXT, fontweight="bold")
    left.set_ylabel("Carry-forward baseline $R^2$")
    left.set_ylim(0, 1.12)
    left.yaxis.grid(True, color=GRID, linewidth=0.5)
    left.set_axisbelow(True)
    left.set_title("The same baseline, two entity keys", loc="left", pad=10)
    for x, caption in ((0, "appears useless"), (1, "nearly unbeatable")):
        left.annotate(caption, (x, values[x] + 0.09), ha="center",
                      fontsize=7, color=MUTED)
    strip_axes(left)

    mae_labels = ["Carry-forward\n(recovered key)", "Earlier model"]
    mae_values = [recovered["mae"], model_mae]
    bars = right.bar(mae_labels, mae_values, color=[INK, MID], width=0.5, zorder=3)
    for bar, value in zip(bars, mae_values):
        right.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value + 0.003),
                       ha="center", fontsize=8, color=TEXT, fontweight="bold")
    right.set_ylabel("Mean absolute error")
    right.set_ylim(0, max(mae_values) * 1.34)
    right.yaxis.grid(True, color=GRID, linewidth=0.5)
    right.set_axisbelow(True)
    right.set_title("Doing nothing beats the model", loc="left", pad=10)
    right.annotate(
        f"{concealment['published_model_worse_by']:.1f}x worse",
        (1, model_mae * 1.20), ha="center", fontsize=8, color=MUTED,
    )
    strip_axes(right)

    save(fig, "figure3_concealment")


def main() -> None:
    evaluation = json.loads((ROOT / "results" / "evaluation.json").read_text())
    findings = json.loads((ROOT / "results" / "findings.json").read_text())
    data = pd.DataFrame(evaluation["datasets"])

    print("writing figures:")
    figure1(data, evaluation["summary"])
    figure2(data, evaluation["summary"])
    figure3(findings)


if __name__ == "__main__":
    main()
