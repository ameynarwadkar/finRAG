"""Generate README charts using matplotlib. No AI — pure code."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ─── Shared style ───────────────────────────────────────────────────────────
BG_COLOR = "#0d1117"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"
ACCENT_COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#f778ba", "#d2a8ff", "#79c0ff"]

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "font.family": "sans-serif",
    "font.size": 12,
})


def generate_architecture_diagram():
    """Generate a horizontal pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("EUR-Lex\nScraper", "#58a6ff"),
        ("BM25 + Dense\nIndexing", "#3fb950"),
        ("Query\nRefinement", "#f0883e"),
        ("Hybrid Search\n+ RRF", "#f778ba"),
        ("Reranking\nCE / FlashRank", "#d2a8ff"),
        ("LLM\nGeneration", "#79c0ff"),
    ]

    box_w, box_h = 1.8, 2.0
    gap = 0.35
    start_x = 0.4
    y_center = 2.0

    for i, (label, color) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=color + "22",  # translucent fill
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2, y_center, label,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=color,
        )
        # Arrow between boxes
        if i < len(stages) - 1:
            arrow_x = x + box_w + 0.02
            ax.annotate(
                "", xy=(arrow_x + gap - 0.04, y_center),
                xytext=(arrow_x, y_center),
                arrowprops=dict(
                    arrowstyle="-|>", color=TEXT_COLOR,
                    lw=1.8, mutation_scale=18,
                ),
            )

    fig.suptitle("RAG Pipeline Architecture", fontsize=16, fontweight="bold", y=0.97, color=TEXT_COLOR)
    fig.savefig(os.path.join(ASSETS_DIR, "architecture.png"), dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("[OK] architecture.png")


def generate_evaluation_chart():
    """Generate grouped bar chart of retrieval benchmark results."""
    methods = ["BM25", "Dense", "Hybrid"]
    metrics = ["Precision@5", "Recall@5", "MRR"]
    
    # lookup, conceptual averaged
    data = {
        "BM25":   [0.585, 0.585, 0.47],
        "Dense":  [1.00,  1.00,  0.815],
        "Hybrid": [1.00,  1.00,  0.70],
    }

    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (metric, color) in enumerate(zip(metrics, ACCENT_COLORS[:3])):
        values = [data[m][i] for m in methods]
        bars = ax.bar(x + i * width, values, width, label=metric, color=color, edgecolor=color, alpha=0.85, zorder=3)
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color,
            )

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Retrieval Benchmarks — Averaged Across Query Types (k=5)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=GRID_COLOR)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(ASSETS_DIR, "evaluation.png"), dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("[OK] evaluation.png")


def generate_dataset_chart():
    """Generate a horizontal bar chart of dataset distribution."""
    regulations = ["PSD2", "MiFID II", "GDPR", "DORA"]
    counts = [117, 102, 99, 64]
    colors = ["#58a6ff", "#3fb950", "#f0883e", "#d2a8ff"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(regulations, counts, color=colors, edgecolor=[c for c in colors], alpha=0.85, height=0.55, zorder=3)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f"{count} articles", va="center", fontsize=11, fontweight="bold", color=TEXT_COLOR,
        )

    ax.set_xlabel("Number of Parsed Chunks", fontsize=13)
    ax.set_title("Dataset: 382 EU Regulation Chunks", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(counts) + 30)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12)

    fig.savefig(os.path.join(ASSETS_DIR, "dataset.png"), dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("[OK] dataset.png")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_evaluation_chart()
    generate_dataset_chart()
    print(f"\nAll charts saved to: {ASSETS_DIR}")
