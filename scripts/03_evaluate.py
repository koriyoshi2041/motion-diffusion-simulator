"""
Evaluate baseline runs vs a reference set, two-track:

    Track A (canonical, what every paper reports):
        FID / Diversity / MultiModality  via the official MDM evaluator
        R-Precision @1/2/3              via the official MDM evaluator
        Implementation in src/eval_t2m.py — we DO NOT reinvent these.

    Track B (assignment-novel):
        D_KL(p‖q), D_KL(q‖p), D_JS(p,q), W_2(p,q)
        Computed in motion-encoder embedding space (not raw 263-dim,
        since the encoder space is what the paper FIDs live in).
        Implementation in src/metrics.py — robust by construction:
            * sample-size guard (>=256)
            * PCA-reduce to <=16D before density estimation
            * kNN estimator with split-sample JS

Inputs (the contract from src/pipeline.py + 263 features):

    runs/<exp>/<baseline>/
        joints.npy          (N, T_max, 22, 3)
        lengths.npy         (N,)
        features_263.npy    (N, T_max, 263)    [optional — required for Track A]
        prompts.txt
        manifest.json

Reference is either:
    - a saved-run dir produced by running mock or any baseline on
      HumanML3D test prompts, or
    - a HumanML3D test split path containing 263-dim features
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics  import divergence_report                       # noqa: E402
from pipeline import load_run                                # noqa: E402


# ---------------------------------------------------------------------------
# Embedding source: prefer T2M evaluator, fallback to handcrafted fingerprint
# ---------------------------------------------------------------------------

def get_embeddings(joints, lengths, features_263=None, project_root=None, device="cuda"):
    """Return (N, D) motion embeddings.

    Priority:
        1. If features_263 is given AND project_root has the MDM repo
           with downloaded text_mot_match weights, use the official encoder.
        2. Otherwise, fall back to a handcrafted fingerprint over (T,22,3).
           This still gives a usable but non-canonical metric — useful for
           plumbing tests / mock runs / when GPU is unavailable.
    """
    if features_263 is not None and project_root is not None:
        weights = (project_root / "baselines" / "motion-diffusion-model"
                   / "save" / "t2m" / "text_mot_match")
        if weights.exists():
            try:
                from eval_t2m import load_evaluator, motion_embeddings
                ev = load_evaluator(project_root=project_root, device=device)
                return motion_embeddings(ev, features_263, lengths, device=device), "t2m"
            except Exception as e:
                warnings.warn(f"T2M evaluator unavailable ({e}); using fallback fingerprint")
    return _fingerprint_set(joints, lengths), "fingerprint"


def _fingerprint(joints: np.ndarray, length: int) -> np.ndarray:
    j = joints[:length]
    if j.shape[0] < 3:
        j = np.tile(j, (3, 1, 1))[:3]
    vel = np.diff(j, axis=0)
    acc = np.diff(vel, axis=0)
    feats = []
    feats += [j.mean(axis=0).flatten()]
    feats += [j.std(axis=0).flatten()]
    feats += [vel.mean(axis=0).flatten()]
    feats += [np.abs(vel).mean(axis=0).flatten()]
    feats += [acc.std(axis=0).flatten()]
    bones = j[:, 1:] - j[:, :1]
    bone_lens = np.linalg.norm(bones, axis=-1)
    feats += [bone_lens.mean(axis=0)]
    feats += [bone_lens.std(axis=0)]
    energy = (vel ** 2).sum(axis=(1, 2))
    spec = np.abs(np.fft.rfft(energy - energy.mean()))
    bins = np.array_split(spec, 3)
    feats += [np.array([b.mean() for b in bins])]
    return np.concatenate(feats).astype(np.float32)


def _fingerprint_set(joints: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.stack([_fingerprint(joints[i], int(lengths[i])) for i in range(joints.shape[0])])


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------

def load_reference(ref_path: Path):
    """Reference is a saved run dir; we read joints + lengths + (optional) 263."""
    if ref_path.is_dir():
        run = load_run(ref_path)
        feats_263 = ref_path / "features_263.npy"
        feats_263 = np.load(feats_263) if feats_263.exists() else None
        return run["joints"], run["lengths"], feats_263
    arr = np.load(ref_path)
    if arr.ndim == 4 and arr.shape[-2:] == (22, 3):
        lengths = np.full(arr.shape[0], arr.shape[1], dtype=np.int32)
        return arr, lengths, None
    raise ValueError(f"unsupported reference at {ref_path} ({arr.shape})")


# ---------------------------------------------------------------------------
# Track A: standard metrics through T2M evaluator
# ---------------------------------------------------------------------------

def standard_metrics(real_emb: np.ndarray, gen_emb: np.ndarray) -> dict:
    """FID + Diversity (Track A canonical metrics)."""
    out = {}
    if real_emb.shape[0] >= 32 and gen_emb.shape[0] >= 32:
        from eval_t2m import fid_t2m, diversity_t2m
        out["FID"]            = fid_t2m(real_emb, gen_emb)
        out["Diversity_real"] = diversity_t2m(real_emb)
        out["Diversity_gen"]  = diversity_t2m(gen_emb)
    else:
        warnings.warn("n < 32 — skipping FID/Diversity")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs",      nargs="+", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--n_subset",  type=int, default=4096)
    ap.add_argument("--target_dim", type=int, default=16,
                    help="PCA target for divergence estimation.")
    ap.add_argument("--project_root", type=Path, default=Path.cwd())
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[ref] loading {args.reference}")
    ref_joints, ref_lengths, ref_263 = load_reference(args.reference)
    if ref_joints.shape[0] > args.n_subset:
        idx = np.random.default_rng(0).choice(ref_joints.shape[0], args.n_subset, replace=False)
        ref_joints  = ref_joints[idx]
        ref_lengths = ref_lengths[idx]
        if ref_263 is not None:
            ref_263 = ref_263[idx]
    print(f"[ref]   shape={ref_joints.shape}, n={ref_joints.shape[0]}")

    ref_emb, ref_src = get_embeddings(ref_joints, ref_lengths, ref_263,
                                      args.project_root, args.device)
    print(f"[ref]   embedding source = {ref_src}, dim = {ref_emb.shape[1]}")

    table = {}
    for run_dir in args.runs:
        print(f"\n=== {run_dir.name} ===")
        run = load_run(run_dir)
        gen_263_path = run_dir / "features_263.npy"
        gen_263 = np.load(gen_263_path) if gen_263_path.exists() else None
        gen_emb, gen_src = get_embeddings(run["joints"], run["lengths"], gen_263,
                                          args.project_root, args.device)

        report = standard_metrics(ref_emb, gen_emb)
        try:
            report.update(divergence_report(ref_emb, gen_emb, target_dim=args.target_dim))
        except ValueError as e:
            warnings.warn(f"divergence skipped: {e}")
            report["divergence_error"] = str(e)
        report["_meta"] = {
            "embedding_source": gen_src,
            "ref_n":  int(ref_emb.shape[0]),
            "gen_n":  int(gen_emb.shape[0]),
            "emb_dim": int(gen_emb.shape[1]),
        }
        table[run_dir.name] = report
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    out_path = args.out / "metrics.json"
    out_path.write_text(json.dumps(table, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nOK -> {out_path}")


if __name__ == "__main__":
    main()
