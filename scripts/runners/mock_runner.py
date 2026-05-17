"""
Mock baseline runner — generates plausible (T, 22, 3) random-walk skeletons
**without any model weights or GPU**.

Use this to:
    - Smoke-test the full pipeline before cluster has finished setting up
    - Demo the front-end end-to-end on a laptop
    - Verify that 02_run_baselines.py + 03_evaluate.py + 04_demo_grid.py
      all wire up correctly

Outputs follow the same contract as the real runners.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np


def fake_motion(T: int, seed: int) -> np.ndarray:
    """SMPL-22 layout, slowly drifting."""
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.15, size=(22, 3))
    base[:, 1] += np.linspace(0, 1.6, 22)               # vertical layout
    drift = np.cumsum(rng.normal(scale=0.015, size=(T, 22, 3)), axis=0)
    return (base[None] + drift).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_file",  required=True, type=Path)
    ap.add_argument("--out_dir",      required=True, type=Path)
    ap.add_argument("--duration",     type=int, default=120)
    ap.add_argument("--variant",      default="mock")
    ap.add_argument("--motion_length", type=int, default=120)
    args, _unknown = ap.parse_known_args()

    prompts = args.prompt_file.read_text(encoding="utf-8").splitlines()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    motions, lengths, t0 = [], [], time.time()
    for i, _ in enumerate(prompts):
        T = max(60, args.duration + np.random.randint(-10, 10))
        motions.append(fake_motion(T, seed=i))
        lengths.append(T)

    T_max = max(lengths)
    pad = np.zeros((len(motions), T_max, 22, 3), dtype=np.float32)
    for i, m in enumerate(motions):
        pad[i, :m.shape[0]] = m

    np.save(args.out_dir / "joints.npy",  pad)
    np.save(args.out_dir / "lengths.npy", np.asarray(lengths, dtype=np.int32))
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "baseline":      "mock",
        "n_prompts":     len(prompts),
        "total_seconds": time.time() - t0,
        "shape":         list(pad.shape),
        "warning":       "synthetic random-walk skeletons; for plumbing tests only",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK -> {args.out_dir}")


if __name__ == "__main__":
    main()
