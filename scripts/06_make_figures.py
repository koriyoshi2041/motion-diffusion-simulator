"""
Generate the three "hero figures" for the report / defense slides:

    Figure A — system overview (text -> 4 baselines -> 22 joints -> 4 viz)
    Figure B — three-divergence convergence  +  FID-vs-Diversity Pareto
    Figure C — Chinese-prompt 9-grid (delegated to 04_demo_grid.py)

Usage:
    python scripts/06_make_figures.py \
        --metrics runs/.../eval/metrics.json \
        --train_log runs/.../train_log.csv \
        --out runs/.../figures/
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Figure A — system overview (schematic, no real data needed)
# ---------------------------------------------------------------------------
def figure_a(out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.set_axis_off()

    def block(x, y, w, h, label, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, color=color, alpha=0.85, ec="k"))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10)

    block(0.2, 2.5, 1.6, 1.0, "中文 prompt\n（自然语言）", "#dbe9ff")
    block(2.1, 2.5, 1.4, 1.0, "DeepSeek\n改写", "#cfe1c0")
    block(3.7, 2.5, 1.4, 1.0, "英文 prompt", "#dbe9ff")

    ys = [4.5, 3.2, 1.9, 0.6]
    names = ["HY-Motion 1.0", "HY-Motion Lite", "MoMask", "MDM"]
    for y, n in zip(ys, names):
        block(5.5, y, 1.7, 0.9, n, "#ffe6b3")

    block(7.7, 2.5, 1.6, 1.0, "(T,22,3)\nxyz keypoints", "#f5d0d0")
    block(9.7, 4.0, 1.8, 0.8, "3D 火柴人 mp4", "#e6cfe6")
    block(9.7, 3.0, 1.8, 0.8, "关节强度热力图", "#e6cfe6")
    block(9.7, 2.0, 1.8, 0.8, "俯视轨迹图",     "#e6cfe6")
    block(9.7, 1.0, 1.8, 0.8, "运动能量曲线",   "#e6cfe6")

    for y in ys:
        ax.annotate("", xy=(5.5, y + 0.45), xytext=(5.1, 3.0),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.annotate("", xy=(7.7, 3.0), xytext=(7.2, y + 0.45),
                    arrowprops=dict(arrowstyle="->", color="gray"))
    for y in (4.4, 3.4, 2.4, 1.4):
        ax.annotate("", xy=(9.7, y), xytext=(9.3, 3.0),
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_title("Figure A · System Overview", fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B — three-divergence convergence + FID-vs-Diversity Pareto
# ---------------------------------------------------------------------------
def figure_b(metrics_json: Path, train_log_csv: Path | None, out_path: Path):
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Left: training divergence curves (placeholder if no train log)
    ax = axes[0]
    if train_log_csv is not None and train_log_csv.exists():
        steps, kl, js, fid = [], [], [], []
        with train_log_csv.open() as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                kl.append(float(row.get("KL(real||gen)", "nan")))
                js.append(float(row.get("JS", "nan")))
                fid.append(float(row.get("FID", "nan")))
        ax.plot(steps, kl, label="KL")
        ax.plot(steps, js, label="JS")
        ax.plot(steps, fid, label="FID", linestyle="--")
    else:
        ax.text(0.5, 0.5, "(train log absent — fill after fine-tune run)",
                transform=ax.transAxes, ha="center", va="center", color="gray")
    ax.set_xlabel("training step")
    ax.set_ylabel("divergence value")
    ax.set_title("Three-divergence convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: FID vs Diversity Pareto across baselines
    ax = axes[1]
    for name, m in metrics.items():
        ax.scatter(m.get("Diversity_gen", 0), m.get("FID", 0), s=80, label=name)
        ax.annotate(name, (m.get("Diversity_gen", 0), m.get("FID", 0)),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Diversity (higher = more varied)")
    ax.set_ylabel("FID (lower = closer to real)")
    ax.set_title("FID-vs-Diversity Pareto")
    ax.grid(alpha=0.3)

    fig.suptitle("Figure B · Convergence and Pareto front", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics",   type=Path, required=True)
    ap.add_argument("--train_log", type=Path, default=None)
    ap.add_argument("--out",       type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    figure_a(args.out / "figure_A_system_overview.png")
    figure_b(args.metrics, args.train_log, args.out / "figure_B_convergence_pareto.png")
    print(f"OK -> {args.out}")
    print("    (figure C 由 04_demo_grid.py 生成)")


if __name__ == "__main__":
    main()
