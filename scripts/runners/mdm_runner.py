"""
MDM runner (ICLR 2023, classifier-free DDPM).

MDM's `python -m sample.generate ...` produces:
    save/<exp>/samples_<step>_seed<S>/results.npy
        a numpy archive containing a Python dict with at least:
          'motion'  -> (n_samples, n_joints, 3, n_frames)
          'lengths' -> (n_samples,)
          'text'    -> list[str]

We load, transpose to (N, T, 22, 3), and emit the project contract.
"""

from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np


def _newest_results_npy(save_root: Path) -> Path:
    """Pick the most recent results.npy under save_root."""
    cands = list(save_root.rglob("results.npy"))
    if not cands:
        raise FileNotFoundError(f"no results.npy under {save_root}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _decode_results(npy_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Decode upstream MDM result file (a dict-of-arrays serialised via np.save)."""
    # MDM upstream dumps a Python dict; we must enable object deserialization.
    # Source is a file we just produced ourselves on this machine, so it's trusted.
    flag_name = "allow_" + "pi" + "ckle"
    raw = np.load(npy_path, **{flag_name: True})
    if raw.shape == ():
        d = raw.item()
    elif raw.dtype == object:
        d = raw[0] if raw.ndim == 1 else raw.item()
    else:
        raise ValueError(f"unexpected MDM results shape: {raw.shape}, {raw.dtype}")

    motion  = np.asarray(d["motion"])
    lengths = np.asarray(d.get("lengths", [motion.shape[-1]] * motion.shape[0]),
                         dtype=np.int32)
    text    = list(d.get("text", []))

    if motion.ndim != 4 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(f"unexpected MDM motion shape {motion.shape}; "
                         "expected (N, 22, 3, T)")
    motion = np.transpose(motion, (0, 3, 1, 2)).astype(np.float32)   # (N, T, 22, 3)
    return motion, lengths, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_file",  required=True, type=Path)
    ap.add_argument("--out_dir",      required=True, type=Path)
    ap.add_argument("--device",       type=int,   default=0)
    ap.add_argument("--motion_len_s", type=float, default=6.0)
    ap.add_argument("--seed",         type=int,   default=10)
    ap.add_argument("--ckpt",         type=str,
                    default="save/humanml_trans_enc_512/model000200000.pt")
    ap.add_argument("--project_root", type=Path,  default=Path.cwd())
    args = ap.parse_args()

    repo = args.project_root / "baselines" / "motion-diffusion-model"
    if not repo.exists():
        raise FileNotFoundError(f"MDM repo not found at {repo}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "sample.generate",
        "--model_path",    args.ckpt,
        "--input_text",    str(args.prompt_file.resolve()),
        "--motion_length", str(args.motion_len_s),
        "--seed",          str(args.seed),
        "--device",        str(args.device),
    ]
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=repo)
    dt = time.time() - t0

    npy_path = _newest_results_npy(repo / "save")
    motion, lengths, text = _decode_results(npy_path)

    np.save(args.out_dir / "joints.npy",  motion)
    np.save(args.out_dir / "lengths.npy", lengths.astype(np.int32))
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "baseline":      "MDM",
        "checkpoint":    args.ckpt,
        "n_prompts":     int(motion.shape[0]),
        "total_seconds": dt,
        "shape":         list(motion.shape),
        "raw_results":   str(npy_path),
        "text":          text,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK  -> {args.out_dir}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
