"""
─────────────────────────────────────────────────────────────────────────
eval_t2m.py —— T2M 官方 evaluator 的 thin wrapper
─────────────────────────────────────────────────────────────────────────

MDM / MoMask / HY-Motion 以及 CVPR 2022 以来几乎每篇 text-to-motion 论文
都用同一个评测器：HumanML3D 配套的 `EvaluatorMDMWrapper` —— 由 motion
encoder + text encoder 两个预训练网络构成，跑在 263 维 HumanML3D 特征上。

我们**不重复实现** FID / R-Precision / Diversity / MultiModality —— 直接
委托给 MDM 上游：
    baselines/motion-diffusion-model/save/t2m/text_mot_match/model/finest.tar

为什么这样做？
    任何自己写的 FID 都没办法和论文数字直接比较，因为对应的 motion encoder
    不一样。要拿 paper-comparable 数字，必须用同一个 evaluator。

调用方式：
    1. 先把生成的 (T, 22, 3) joints 转回 263-dim HumanML3D 特征
       (MDM/MoMask 直接出 263；HY-Motion 输出 xyz 后需要 forward kinematics)
    2. evaluator.get_motion_embeddings(motions_263, lengths) → (N, D) 特征
    3. fid_t2m / r_precision_t2m / diversity_t2m / multimodality_t2m
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Pull EvaluatorMDMWrapper into our process
# ---------------------------------------------------------------------------

@contextmanager
def _mdm_on_path(project_root: Path):
    """Temporarily prepend the MDM repo so that `data_loaders.humanml.*` resolves."""
    repo = project_root / "baselines" / "motion-diffusion-model"
    if not repo.exists():
        raise FileNotFoundError(f"MDM repo missing: {repo}; "
                                "run scripts/01_setup_data.sh first.")
    sys.path.insert(0, str(repo))
    try:
        yield repo
    finally:
        sys.path.pop(0)


def load_evaluator(project_root: Path = Path.cwd(), device: str = "cuda"):
    """Return an EvaluatorMDMWrapper bound to HumanML3D weights."""
    with _mdm_on_path(project_root):
        from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
        return EvaluatorMDMWrapper("humanml", device)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def motion_embeddings(
    evaluator,
    motions_263: np.ndarray,            # (N, T_max, 263)
    lengths:     np.ndarray,            # (N,)
    batch_size:  int = 32,
    device:      str = "cuda",
) -> np.ndarray:
    """Run the evaluator's motion encoder; return (N, D) feature matrix."""
    motions = torch.as_tensor(motions_263, dtype=torch.float32, device=device)
    lens    = torch.as_tensor(lengths,     dtype=torch.long,    device=device)
    feats = []
    for i in range(0, motions.shape[0], batch_size):
        m = motions[i:i + batch_size]
        l = lens[i:i + batch_size]
        with torch.no_grad():
            emb = evaluator.get_motion_embeddings(motions=m, m_lens=l)
        feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Standard (FID / R-Precision / Diversity / MultiModality) — all delegated
# ---------------------------------------------------------------------------

def fid_t2m(real_emb: np.ndarray, gen_emb: np.ndarray) -> float:
    """Frechet distance in T2M evaluator embedding space — the standard FID."""
    from scipy import linalg
    mu_p, mu_q = real_emb.mean(0), gen_emb.mean(0)
    sig_p = np.cov(real_emb, rowvar=False)
    sig_q = np.cov(gen_emb,  rowvar=False)
    diff = mu_p - mu_q
    covmean, _ = linalg.sqrtm(sig_p @ sig_q, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        eps = np.eye(sig_p.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sig_p + eps) @ (sig_q + eps))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    return float(diff @ diff + np.trace(sig_p) + np.trace(sig_q) - 2.0 * np.trace(covmean))


def diversity_t2m(emb: np.ndarray, n_samples: int = 300, rng=None) -> float:
    """Standard MDM/MoMask diversity protocol: avg L2 between random pairs."""
    rng = rng or np.random.default_rng(0)
    n = emb.shape[0]
    n_samples = min(n_samples, n // 2)
    a = rng.choice(n, n_samples, replace=False)
    b = rng.choice(n, n_samples, replace=False)
    return float(np.linalg.norm(emb[a] - emb[b], axis=1).mean())


def multimodality_t2m(emb_per_prompt: Sequence[np.ndarray], n_pairs: int = 10, rng=None) -> float:
    rng = rng or np.random.default_rng(0)
    scores = []
    for E in emb_per_prompt:
        if E.shape[0] < 2:
            continue
        a = rng.choice(E.shape[0], n_pairs, replace=True)
        b = rng.choice(E.shape[0], n_pairs, replace=True)
        scores.append(np.linalg.norm(E[a] - E[b], axis=1).mean())
    return float(np.mean(scores)) if scores else 0.0


def r_precision_t2m(text_emb: np.ndarray, motion_emb: np.ndarray, top_k=(1, 2, 3),
                    pool_size: int = 32, rng=None) -> dict:
    """HumanML3D protocol: pool of 32 motions per text, compute top-k retrieval."""
    rng = rng or np.random.default_rng(0)
    n = text_emb.shape[0]
    text_emb   = text_emb   / (np.linalg.norm(text_emb,   axis=1, keepdims=True) + 1e-12)
    motion_emb = motion_emb / (np.linalg.norm(motion_emb, axis=1, keepdims=True) + 1e-12)

    correct = {k: 0 for k in top_k}
    n_groups = n // pool_size
    for g in range(n_groups):
        idx = np.arange(g * pool_size, (g + 1) * pool_size)
        sim = text_emb[idx] @ motion_emb[idx].T
        ranks = (-sim).argsort(axis=1)
        for k in top_k:
            correct[k] += (ranks[:, :k] == np.arange(pool_size)[:, None]).any(axis=1).sum()
    total = n_groups * pool_size
    return {f"R@{k}": float(correct[k] / total) for k in top_k} if total else {f"R@{k}": 0.0 for k in top_k}
