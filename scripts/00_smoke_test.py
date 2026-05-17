"""
Standalone smoke test: verifies metrics + visualizer work end-to-end on
synthetic data, with **no model weights, no GPU, no internet** required.

Run anywhere (laptop fine):
    python scripts/00_smoke_test.py

Should produce:
    /tmp/motion_smoke/
        dummy_stickfigure.mp4
        dummy_jointmap.png
        dummy_topdown.png
        dummy_energy.png
        metrics_report.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics    import divergence_report                           # noqa: E402
from visualizer import render_all                                  # noqa: E402
from data_utils import recover_joints_from_263                     # noqa: E402


def fake_joints(T: int = 90, seed: int = 0) -> np.ndarray:
    """Synthesize a (T, 22, 3) random-walk skeleton."""
    rng = np.random.default_rng(seed)
    seed_skel = rng.normal(scale=0.2, size=(22, 3))
    seed_skel[:, 1] += np.linspace(0, 1.6, 22)              # vertical layout
    motion = np.cumsum(rng.normal(scale=0.02, size=(T, 22, 3)), axis=0)
    return seed_skel[None] + motion


def fake_humanml3d_features(N: int = 16, T: int = 80, seed: int = 0) -> np.ndarray:
    """Synth HumanML3D-style (N, T, 263) for the recover_joints test."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.05, size=(N, T, 263)).astype(np.float32)


def main():
    out = Path("/tmp/motion_smoke")
    out.mkdir(parents=True, exist_ok=True)

    # 1) visualizer
    print("[1/3] visualizer test ...")
    j = fake_joints(T=90)
    artefacts = render_all(j, out, tag="dummy", fps=20)
    print("       ->", json.dumps(artefacts, indent=2, ensure_ascii=False))

    # 2) feature recovery
    print("[2/3] data_utils.recover_joints_from_263 test ...")
    feats = fake_humanml3d_features(N=4)[0]
    recovered = recover_joints_from_263(feats)
    assert recovered.shape == (feats.shape[0], 22, 3), recovered.shape
    print(f"       263-dim ({feats.shape}) -> joints {recovered.shape}  OK")

    # 3) divergence metrics — divergences only (FID/R-Prec live in eval_t2m)
    print("[3/3] divergence metrics (KL/JS/W2) ...")
    rng = np.random.default_rng(0)
    real = rng.standard_normal((512, 263)).astype(np.float32)
    gen  = rng.standard_normal((512, 263)).astype(np.float32) + 0.2
    report = divergence_report(real, gen, target_dim=16)
    (out / "metrics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("       ", json.dumps(report, indent=2))

    print(f"\n✅ smoke test passed -> {out}")


if __name__ == "__main__":
    main()
