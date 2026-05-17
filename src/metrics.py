"""
─────────────────────────────────────────────────────────────────────────
metrics.py —— 信息论散度模块（作业第 8 条的核心）
─────────────────────────────────────────────────────────────────────────

本模块负责所有「两个分布之间的距离」类指标：
    D_KL(p‖q)、D_JS(p,q)、W_2(p,q)

为什么不复用上游 evaluator 的 FID？
    HumanML3D 论文 / MDM / MoMask / HY-Motion 全都只报告
    FID + R-Precision + Diversity + MultiModality；
    没有人正经计算 KL/JS 散度——而作业第 8 条点名要 KL。
    这是项目自带的 novelty：在 motion-feature space 上做严格散度评估。

`src/eval_t2m.py` 负责剩下的 FID/R-Prec/Div/MM，调用 MDM 上游官方权重。

设计教训（从早期 smoke test 踩坑总结）：

    1. 高维 KL 估计是 ill-posed 的：当 n < 256 时，估计值会出负值或剧烈震荡。
       我们硬性拒绝（raise ValueError），告诉调用者去拿更多样本。
    2. 进入密度估计之前一律 PCA 降到 ≤16 维。Wasserstein-2 不需要降维
      （它在任何维度都 well-defined）。
    3. kNN-KL（Wang 2009 估计器）适合 8 < D ≤ 16；KDE 适合 D ≤ 8。
    4. JS 的 split-sample 修法：用一半样本估混合分布 m=(p+q)/2，
       用另一半样本算 KL(p‖m)、KL(q‖m)，避免自我 NN 偏差。
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import warnings

import numpy as np
from scipy import linalg
from scipy.special import digamma
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Sequence


# ---------------------------------------------------------------------------
# Sample-size pre-flight
# ---------------------------------------------------------------------------

MIN_SAMPLES_FOR_HIGH_DIM_KL = 256


def _check_sample_size(samples_p: np.ndarray, samples_q: np.ndarray,
                       what: str = "KL") -> None:
    n_p, n_q = samples_p.shape[0], samples_q.shape[0]
    if min(n_p, n_q) < MIN_SAMPLES_FOR_HIGH_DIM_KL and samples_p.shape[1] > 8:
        raise ValueError(
            f"{what} estimation in {samples_p.shape[1]}-D needs >= "
            f"{MIN_SAMPLES_FOR_HIGH_DIM_KL} samples per side; got "
            f"({n_p}, {n_q}). Either supply more samples (HumanML3D test "
            f"set has 4382), or reduce_dim_first=True before calling."
        )


def _maybe_reduce(samples_p: np.ndarray, samples_q: np.ndarray,
                  target_dim: int = 16) -> tuple[np.ndarray, np.ndarray, int]:
    """Joint PCA fit on concat(p, q); same projector applied to both.

    Float64 is forced — float32 SVD on extreme-value features triggers
    overflow warnings inside sklearn's randomised path.
    """
    D = samples_p.shape[1]
    if D <= target_dim:
        return samples_p.astype(np.float64), samples_q.astype(np.float64), D
    p64 = samples_p.astype(np.float64)
    q64 = samples_q.astype(np.float64)
    pca = PCA(n_components=target_dim, svd_solver="full")
    pca.fit(np.concatenate([p64, q64], axis=0))
    return pca.transform(p64), pca.transform(q64), target_dim


# ---------------------------------------------------------------------------
# Three divergences (assignment core)
# ---------------------------------------------------------------------------
#
# We use TWO estimators for KL/JS:
#
#   * KDE + Monte-Carlo            -- exact for small (<~10) D, smooth
#   * k-NN  (Wang et al., 2009)    -- consistent up to D ~ 100
#                                     based on Kozachenko-Leonenko entropy
#
# The kNN one is what motion-feature-space (D ~ 100s) actually needs.
# `kl_divergence` automatically picks kNN when D > 8.
# ---------------------------------------------------------------------------


def kl_knn(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    k: int = 3,
) -> float:
    """High-dim KL estimator — Wang, Kulkarni, Verdú (IEEE TIT 2009),
    with the robustness patches the original paper warned about:

        D(P||Q) ≈ (d/n) Σ log[ν_k(i)/ρ_k(i)] + log[m/(n-1)]

    Patches we apply:
        (a) When samples_p exactly equals samples_q we return 0.
        (b) When some samples_p[i] also lies in samples_q (NN dist = 0),
            we shift to the (k+1)-th neighbour so we never look at self.
        (c) We clamp the result at 0 from below (KL is non-negative;
            small estimator noise can dip to ≈ −0.05 even at n=1024).
    """
    if samples_p.shape == samples_q.shape and np.array_equal(samples_p, samples_q):
        return 0.0

    n, d = samples_p.shape
    m    = samples_q.shape[0]

    nn_p = NearestNeighbors(n_neighbors=k + 1).fit(samples_p)
    nn_q = NearestNeighbors(n_neighbors=k + 1).fit(samples_q)

    rho, _ = nn_p.kneighbors(samples_p)        # (n, k+1)
    nu,  _ = nn_q.kneighbors(samples_p)        # (n, k+1)

    rho_k = np.maximum(rho[:, k],     1e-12)   # always skip self in P
    has_self_in_q = nu[:, 0] < 1e-12
    nu_k = np.where(has_self_in_q, nu[:, k], nu[:, k - 1])
    nu_k = np.maximum(nu_k, 1e-12)

    est = (d / n) * np.sum(np.log(nu_k / rho_k)) + np.log(m / (n - 1))
    return float(max(est, 0.0))


def js_knn(samples_p: np.ndarray, samples_q: np.ndarray, k: int = 3,
           rng: np.random.Generator | None = None) -> float:
    """Symmetric JS — kNN version with split-sample mid-distribution.

    When n < 2(k+2) we fall back to (||μ_p − μ_q||² / (σ_p² + σ_q²)),
    a Bhattacharyya-style proxy that stays sensible at small sample size.
    """
    rng = rng or np.random.default_rng(0)
    n = min(len(samples_p), len(samples_q))
    half = n // 2
    if half < k + 2:
        # small-sample fallback (Gaussian proxy)
        mu_p, mu_q = samples_p.mean(0), samples_q.mean(0)
        var_p, var_q = samples_p.var(0).sum(), samples_q.var(0).sum()
        return float(np.linalg.norm(mu_p - mu_q) ** 2 / (var_p + var_q + 1e-12))
    idx_p = rng.permutation(len(samples_p))
    idx_q = rng.permutation(len(samples_q))
    p_eval, p_mix = samples_p[idx_p[:half]], samples_p[idx_p[half:2 * half]]
    q_eval, q_mix = samples_q[idx_q[:half]], samples_q[idx_q[half:2 * half]]
    mix = np.concatenate([p_mix, q_mix], axis=0)
    js = 0.5 * kl_knn(p_eval, mix, k=k) + 0.5 * kl_knn(q_eval, mix, k=k)
    # Theoretical bounds: 0 <= JS <= log(2). Estimator noise can drift
    # slightly past either side; clip to the proper interval.
    return float(np.clip(js, 0.0, float(np.log(2))))


def kl_divergence_kde(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_mc: int = 10000,
    bw: str | float = "scott",
    eps: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> float:
    """KDE + Monte-Carlo KL — only sound for low-dim (D ≤ 8)."""
    rng = rng or np.random.default_rng(0)
    p = gaussian_kde(samples_p.T, bw_method=bw)
    q = gaussian_kde(samples_q.T, bw_method=bw)
    idx = rng.choice(samples_p.shape[0], size=n_mc, replace=True)
    x = samples_p[idx].T
    log_p = np.log(np.maximum(p(x), eps))
    log_q = np.log(np.maximum(q(x), eps))
    return float(np.mean(log_p - log_q))


def js_divergence_kde(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_mc: int = 10000,
    bw: str | float = "scott",
    eps: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> float:
    """KDE Jensen-Shannon — low-dim only."""
    rng = rng or np.random.default_rng(0)
    p = gaussian_kde(samples_p.T, bw_method=bw)
    q = gaussian_kde(samples_q.T, bw_method=bw)

    def _kl(samples, dens_a, dens_b):
        idx = rng.choice(samples.shape[0], size=n_mc, replace=True)
        x = samples[idx].T
        log_a = np.log(np.maximum(dens_a(x), eps))
        log_m = np.log(np.maximum(0.5 * (dens_a(x) + dens_b(x)), eps))
        return float(np.mean(log_a - log_m))

    return 0.5 * _kl(samples_p, p, q) + 0.5 * _kl(samples_q, q, p)


def kl_divergence(samples_p: np.ndarray, samples_q: np.ndarray,
                  reduce_dim_first: bool = True, target_dim: int = 16) -> float:
    """Auto-select estimator + sample-size guard + optional PCA reduction.

    Pipeline:
        1. Refuse if high-D and tiny samples (raise ValueError).
        2. PCA-reduce to <=16D when D is large.
        3. KDE if reduced D <= 8, else kNN.
    """
    if reduce_dim_first:
        samples_p, samples_q, D = _maybe_reduce(samples_p, samples_q, target_dim)
    else:
        D = samples_p.shape[1]
    _check_sample_size(samples_p, samples_q, what="KL")
    return kl_divergence_kde(samples_p, samples_q) if D <= 8 else kl_knn(samples_p, samples_q)


def js_divergence(samples_p: np.ndarray, samples_q: np.ndarray,
                  reduce_dim_first: bool = True, target_dim: int = 16) -> float:
    if reduce_dim_first:
        samples_p, samples_q, D = _maybe_reduce(samples_p, samples_q, target_dim)
    else:
        D = samples_p.shape[1]
    _check_sample_size(samples_p, samples_q, what="JS")
    return js_divergence_kde(samples_p, samples_q) if D <= 8 else js_knn(samples_p, samples_q)


def wasserstein2_sinkhorn(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    reg: float = 0.05,
    n_iter: int = 200,
) -> float:
    """W_2 distance via Sinkhorn (entropic regularization)."""
    try:
        import ot
    except ImportError as e:
        raise ImportError("Run `pip install POT` for Wasserstein support.") from e

    a = np.full(samples_p.shape[0], 1.0 / samples_p.shape[0])
    b = np.full(samples_q.shape[0], 1.0 / samples_q.shape[0])
    M = ot.dist(samples_p, samples_q, metric="sqeuclidean")
    M /= M.max() + 1e-12
    cost = ot.sinkhorn2(a, b, M, reg=reg, numItermax=n_iter)
    return float(np.sqrt(cost))


# ---------------------------------------------------------------------------
# Divergences-only report
# ---------------------------------------------------------------------------

def divergence_report(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    target_dim: int = 16,
) -> dict:
    """Compute the three assignment-mandated divergences, in PCA-reduced space."""
    p, q, D = _maybe_reduce(real_features, gen_features, target_dim)
    _check_sample_size(p, q, what="divergence")
    return {
        "KL(real||gen)": kl_divergence(p, q, reduce_dim_first=False),
        "KL(gen||real)": kl_divergence(q, p, reduce_dim_first=False),
        "JS":            js_divergence(p, q, reduce_dim_first=False),
        "W2":            wasserstein2_sinkhorn(real_features, gen_features),
        "_meta":         {"reduced_dim": int(D), "n_real": int(real_features.shape[0]),
                          "n_gen": int(gen_features.shape[0])},
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    real = rng.standard_normal((512, 64))
    gen  = rng.standard_normal((512, 64)) + 0.1
    print(divergence_report(real, gen))
