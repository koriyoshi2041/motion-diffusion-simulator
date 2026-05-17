"""
─────────────────────────────────────────────────────────────────────────
pipeline.py —— baseline 统一调度（driver）
─────────────────────────────────────────────────────────────────────────

我们要支持多个 motion-generation baseline（每家上游 API 完全不同）。
本模块负责把它们抽象到统一接口，下游 (metrics / visualizer / demo) 只
认这个接口，不关心 baseline 内部细节。

各家 baseline 的真实接口（May 2026 阅读上游源码后确认）：

| baseline       | 入口                                    | 真实输出                                 |
|----------------|----------------------------------------|-----------------------------------------|
| HY-Motion 1.0  | hymotion.pipeline.MotionFlowMatching   | dict["keypoints3d"]: (B, L, J, 3)       |
| MoMask         | gen_t2m.py                             | joints/<id>.npy: (T, 22, 3)             |
| MDM            | python -m sample.generate              | results.npy: dict[motion]=(N,22,3,T)    |
| HY-Motion Lite | same as HY-Motion, --variant lite      | same                                    |

所有 runner 把上述各自不同的输出**翻译**成同一个磁盘契约：

    <out_dir>/
        joints.npy          (N, T_max, 22, 3) float32，pad 短样本
        lengths.npy         (N,) int32，真实帧数
        prompts.txt         原始（英文）prompt
        manifest.json       {baseline, ckpt_path, args, timing, shape}

下游评估 / 渲染 / 9 宫格 grid 全部只读这个契约。
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class BaselineConfig:
    name: str
    runner: str                          # path to scripts/runners/<x>_runner.py
    python_bin: str = "python"
    extra_args: tuple = ()
    description: str = ""


PRESETS: dict[str, BaselineConfig] = {
    "hy_motion": BaselineConfig(
        name="HY-Motion-1.0",
        runner="scripts/runners/hy_motion_runner.py",
        extra_args=("--variant", "full"),
        description="Tencent HY-Motion 1.0 (1B), DiT + flow matching, current SOTA.",
    ),
    "hy_motion_lite": BaselineConfig(
        name="HY-Motion-1.0-Lite",
        runner="scripts/runners/hy_motion_runner.py",
        extra_args=("--variant", "lite"),
        description="HY-Motion lite (0.46B) — fast ablations.",
    ),
    "momask": BaselineConfig(
        name="MoMask",
        runner="scripts/runners/momask_runner.py",
        description="Masked generation + residual VQ (CVPR 2024).",
    ),
    "mdm": BaselineConfig(
        name="MDM",
        runner="scripts/runners/mdm_runner.py",
        description="Classifier-free DDPM on 263-dim HumanML3D (ICLR 2023).",
    ),
    "mock": BaselineConfig(
        name="Mock-Random-Walk",
        runner="scripts/runners/mock_runner.py",
        description="No model — synthetic skeleton for plumbing tests / laptop demo.",
    ),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    out_dir: Path
    joints_path: Path
    lengths_path: Path
    manifest_path: Path
    duration_sec: float


def run_baseline(cfg: BaselineConfig, prompts: Sequence[str], out_dir: Path) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = out_dir / "prompts.txt"
    prompt_file.write_text("\n".join(prompts), encoding="utf-8")

    cmd = [
        cfg.python_bin, cfg.runner,
        "--prompt_file", str(prompt_file),
        "--out_dir",     str(out_dir),
        *cfg.extra_args,
    ]
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0

    return RunResult(
        out_dir       = out_dir,
        joints_path   = out_dir / "joints.npy",
        lengths_path  = out_dir / "lengths.npy",
        manifest_path = out_dir / "manifest.json",
        duration_sec  = dt,
    )


# ---------------------------------------------------------------------------
# Contract loader
# ---------------------------------------------------------------------------

def load_run(out_dir: Path) -> dict:
    joints  = np.load(out_dir / "joints.npy")            # (N, T_max, 22, 3)
    lengths = np.load(out_dir / "lengths.npy")           # (N,)
    if joints.dtype == object or lengths.dtype == object:
        raise ValueError("object-dtype arrays not allowed")
    if joints.ndim != 4 or joints.shape[-2:] != (22, 3):
        raise ValueError(f"joints shape mismatch: {joints.shape}")

    prompts = (out_dir / "prompts.txt").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"joints": joints, "lengths": lengths, "prompts": prompts, "manifest": manifest}


def write_contract(out_dir: Path, joints: np.ndarray, lengths: np.ndarray, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "joints.npy",  joints.astype(np.float32))
    np.save(out_dir / "lengths.npy", lengths.astype(np.int32))
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=PRESETS.keys(), required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    cfg = PRESETS[args.baseline]
    prompts = Path(args.prompts).read_text(encoding="utf-8").strip().splitlines()
    res = run_baseline(cfg, prompts, Path(args.out_dir))
    print(f"OK  baseline={cfg.name}  time={res.duration_sec:.1f}s  -> {res.out_dir}")
