"""
Build the 9-prompt demo grid for the report's hero figure.

For each (baseline, prompt) cell, render the 4 visualisations and pick a
representative still frame. Compose into a single PNG/HTML grid.

Usage:
    python scripts/04_demo_grid.py \
        --runs runs/.../hy_motion runs/.../mdm runs/.../momask \
        --prompts_zh prompts/chinese_demo.txt \
        --out runs/.../demo_grid/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline   import load_run                       # noqa: E402
from visualizer import render_all                     # noqa: E402


def _stick_still(joints: np.ndarray, ax, title: str = ""):
    """Render a single mid-sequence stick figure into a matplotlib axis."""
    from visualizer import KINEMATIC_CHAIN
    t = joints.shape[0] // 2
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    for chain in KINEMATIC_CHAIN:
        xs = joints[t, chain, 0]
        ys = joints[t, chain, 1]
        ax.plot(xs, ys, lw=2)
    ax.set_aspect("equal", adjustable="datalim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--prompts_zh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full_videos", action="store_true",
                    help="Also render full mp4 + heatmaps + topdown + energy "
                         "for every (baseline, prompt) cell.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = args.prompts_zh.read_text(encoding="utf-8").splitlines()

    runs = {r.name: load_run(r) for r in args.runs}
    n_b, n_p = len(runs), len(prompts)

    fig, axes = plt.subplots(n_b, n_p, figsize=(2.0 * n_p, 2.4 * n_b), dpi=160)
    if n_b == 1: axes = np.array([axes])
    if n_p == 1: axes = axes.reshape(-1, 1)

    for r, (name, run) in enumerate(runs.items()):
        for c, prompt in enumerate(prompts):
            i = min(c, run["joints"].shape[0] - 1)
            j = run["joints"][i, :int(run["lengths"][i])]
            _stick_still(j, axes[r, c], title=f"{name}\n{prompt[:18]}")
            if args.full_videos:
                tag = f"{name}_{c:02d}"
                render_all(j, args.out / "videos", tag=tag)

    fig.suptitle("中文 Prompt × Baseline 对比矩阵", fontsize=14)
    fig.tight_layout()
    grid_path = args.out / "demo_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    print(f"OK -> {grid_path}")


if __name__ == "__main__":
    main()
