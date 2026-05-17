"""
MoMask runner.

MoMask's official `gen_t2m.py` writes (T, 22, 3) joint xyz to
<eval_dir>/joints/<idx>.npy. We just shell out, then collate the
per-prompt files into the project-wide contract.

Layout assumed (after running scripts/01_setup_data.sh):
    baselines/MoMask/
      gen_t2m.py
      checkpoints/...           # per upstream README
"""

from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np


def _run_cli(repo: Path, prompt_file: Path, work_dir: Path, gpu: int, motion_length: int):
    cmd = [
        "python", "gen_t2m.py",
        "--gpu_id",        str(gpu),
        "--ext",           work_dir.name,
        "--text_path",     str(prompt_file.resolve()),
        "--motion_length", str(motion_length),
        "--repeat_times",  "1",
    ]
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=repo)


def _collate(generation_dir: Path, n_prompts: int) -> tuple[np.ndarray, np.ndarray]:
    """Read joints/<idx>.npy files and stack."""
    joint_dir = generation_dir / "joints"
    arrs, lengths = [], []
    for i in range(n_prompts):
        p = joint_dir / f"{i:04d}.npy"
        if not p.exists():
            # MoMask sometimes uses different naming, try fallback
            candidates = sorted(joint_dir.glob("*.npy"))
            if i >= len(candidates):
                raise FileNotFoundError(f"missing prompt {i} under {joint_dir}")
            p = candidates[i]
        a = np.load(p)
        if a.dtype == object:
            raise ValueError("object-dtype not allowed")
        if a.ndim != 3 or a.shape[-2:] != (22, 3):
            raise ValueError(f"unexpected shape {a.shape} from MoMask")
        arrs.append(a.astype(np.float32))
        lengths.append(a.shape[0])
    T_max = max(lengths)
    padded = np.zeros((n_prompts, T_max, 22, 3), dtype=np.float32)
    for i, a in enumerate(arrs):
        padded[i, :a.shape[0]] = a
    return padded, np.asarray(lengths, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_file",   required=True, type=Path)
    ap.add_argument("--out_dir",       required=True, type=Path)
    ap.add_argument("--gpu",           type=int, default=0)
    ap.add_argument("--motion_length", type=int, default=120)
    ap.add_argument("--project_root",  type=Path, default=Path.cwd())
    args = ap.parse_args()

    repo = args.project_root / "baselines" / "MoMask"
    if not repo.exists():
        raise FileNotFoundError(f"MoMask repo not found at {repo}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt_file.read_text(encoding="utf-8").strip().splitlines()

    t0 = time.time()
    _run_cli(repo, args.prompt_file, args.out_dir, args.gpu, args.motion_length)
    dt = time.time() - t0

    # MoMask writes its outputs under repo/generation/<ext>/...
    gen_root = repo / "generation" / args.out_dir.name
    if not gen_root.exists():
        gen_root = repo / "checkpoints" / "t2m" / "Comp_v6_KLD005" / "generation" / args.out_dir.name
    if not gen_root.exists():
        # last-resort search
        cands = list(repo.rglob(f"generation/{args.out_dir.name}"))
        if not cands:
            raise FileNotFoundError("could not locate MoMask generation dir")
        gen_root = cands[0]

    joints, lengths = _collate(gen_root, len(prompts))
    np.save(args.out_dir / "joints.npy",  joints)
    np.save(args.out_dir / "lengths.npy", lengths)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "baseline":      "MoMask",
        "n_prompts":     len(prompts),
        "motion_length": args.motion_length,
        "total_seconds": dt,
        "shape":         list(joints.shape),
        "raw_outputs":   str(gen_root),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK  -> {args.out_dir}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
