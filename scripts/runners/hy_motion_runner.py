"""
HY-Motion 1.0 runner — verified against the upstream code (May 2026).

Real entry path (read from /tmp/HY-Motion-1.0/hymotion/utils/t2m_runtime.py):

    from hymotion.utils.t2m_runtime import T2MRuntime
    runtime = T2MRuntime(config_path="ckpts/tencent/HY-Motion-1.0-Lite",
                         ckpt_name="latest.ckpt",
                         device_ids=[0],
                         disable_prompt_engineering=True)
    runtime.load()
    pi = runtime._acquire_pipeline()
    pipeline = runtime.pipelines[pi]
    model_output = pipeline.generate(
        text="a person throws a left hook punch",
        seed_input=[0],
        duration_slider=4.0,
        cfg_scale=5.0,
        use_special_game_feat=False,
    )
    # model_output["keypoints3d"]: (B, L, J, 3) — what we want

We bypass the upstream `T2MRuntime.generate_motion` because it tries to
render FBX/HTML which needs the FBX SDK + heavy gradio HTML pipeline.

Output (project contract):
    joints.npy   (N, T_max, 22, 3) float32, padded
    lengths.npy  (N,) int32
    manifest.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _add_hy_motion_to_syspath(project_root: Path) -> Path:
    repo = project_root / "baselines" / "HY-Motion-1.0-repo"
    if not repo.exists():
        raise FileNotFoundError(f"HY-Motion repo not found at {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    return repo


def _build_runtime(repo: Path, variant: str, device: int):
    from hymotion.utils.t2m_runtime import T2MRuntime
    sub = "HY-Motion-1.0-Lite" if variant == "lite" else "HY-Motion-1.0"
    ckpt_dir = repo / "ckpts" / "tencent" / sub
    config_yml = ckpt_dir / "config.yml"
    if not (ckpt_dir / "latest.ckpt").exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir}/latest.ckpt")
    if not config_yml.exists():
        raise FileNotFoundError(f"Missing config: {config_yml}")
    runtime = T2MRuntime(
        config_path=str(config_yml),
        # T2MRuntime passes ckpt_name straight to load_in_demo() which does
        # `os.path.exists(ckpt_name)`. The relative default fails silently
        # (the model loads with random weights). Pass an absolute path.
        ckpt_name=str(ckpt_dir / "latest.ckpt"),
        device_ids=[device],
        disable_prompt_engineering=True,
    )
    return runtime


def _gen_one(runtime, prompt: str, duration: float, cfg_scale: float, seed: int) -> np.ndarray:
    pi = runtime._acquire_pipeline()
    try:
        pipeline = runtime.pipelines[pi]
        pipeline.train(False)                      # set inference mode
        out = pipeline.generate(
            text=prompt,
            seed_input=[seed],
            duration_slider=duration,
            cfg_scale=cfg_scale,
            use_special_game_feat=False,
        )
    finally:
        runtime._release_pipeline(pi)

    k3d = out["keypoints3d"]
    if hasattr(k3d, "cpu"):
        k3d = k3d.cpu().numpy()
    if k3d.shape[-2] == 52:
        k3d = k3d[..., :22, :]
    elif k3d.shape[-2] != 22:
        raise ValueError(f"unexpected joint count {k3d.shape}")
    return k3d[0].astype(np.float32)         # (L, 22, 3) — first batch elem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_file",  required=True, type=Path)
    ap.add_argument("--out_dir",      required=True, type=Path)
    ap.add_argument("--variant",      default="lite", choices=("full", "lite"))
    ap.add_argument("--duration",     type=float, default=4.0)
    ap.add_argument("--cfg_scale",    type=float, default=5.0)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--device",       type=int,   default=0)
    ap.add_argument("--project_root", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent)
    args = ap.parse_args()

    repo = _add_hy_motion_to_syspath(args.project_root)
    print(f"[hy_motion] loading variant={args.variant} ...")
    t0 = time.time()
    runtime = _build_runtime(repo, args.variant, args.device)
    print(f"[hy_motion] runtime loaded in {time.time()-t0:.1f}s")

    out_dir = (args.project_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt_file.read_text(encoding="utf-8").strip().splitlines()

    all_joints, all_lens, timings = [], [], []
    for i, prompt in enumerate(prompts):
        t1 = time.time()
        try:
            j = _gen_one(runtime, prompt, args.duration, args.cfg_scale, args.seed)
            all_joints.append(j)
            all_lens.append(j.shape[0])
            dt = time.time() - t1
            timings.append(dt)
            print(f"  [{i+1}/{len(prompts)}]  {dt:.1f}s  shape={j.shape}  '{prompt[:50]}'")
        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}]  FAILED: {e}")
            all_joints.append(np.zeros((1, 22, 3), dtype=np.float32))
            all_lens.append(1)

    T_max = max(all_lens)
    padded = np.zeros((len(all_joints), T_max, 22, 3), dtype=np.float32)
    for i, j in enumerate(all_joints):
        padded[i, :j.shape[0]] = j

    np.save(out_dir / "joints.npy",  padded)
    np.save(out_dir / "lengths.npy", np.asarray(all_lens, dtype=np.int32))
    (out_dir / "manifest.json").write_text(json.dumps({
        "baseline":     "HY-Motion-1.0" if args.variant == "full" else "HY-Motion-1.0-Lite",
        "variant":      args.variant,
        "checkpoint":   str(repo / "ckpts" / "tencent" /
                            ("HY-Motion-1.0" if args.variant == "full" else "HY-Motion-1.0-Lite")),
        "n_prompts":    len(prompts),
        "duration":     args.duration,
        "cfg_scale":    args.cfg_scale,
        "seed":         args.seed,
        "total_seconds": float(sum(timings)),
        "shape":        list(padded.shape),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK  -> {out_dir}")


if __name__ == "__main__":
    main()
