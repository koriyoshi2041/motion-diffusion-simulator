"""
─────────────────────────────────────────────────────────────────────────
mock_server.py —— 本地无 GPU 版后端
─────────────────────────────────────────────────────────────────────────

复刻 server_hy.py 的同一套 HTTP 接口，但 generate 不调真模型，而是把
runs/ 下事先离线跑好的 (T, 22, 3) 数据按 prompt 的「分类直觉」返回。
这样无 GPU 笔记本也能看到完整 demo 流程 —— prompt 进、骨架出、4 种 viz 都能切。

接口完全等同 server_hy.py：
    GET  /api/info          模型元信息
    POST /api/generate      返回 {joints, fps, kinematic_chain, ...}
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
]

# ─── 加载所有缓存的 9 prompt 真实输出（如有），按关键词搜索匹配 ─────────
CACHED_PROMPTS = []   # list[(prompt_text_lower, joints_array)]

def _load_cached():
    """读取 runs/ 下离线产生的 (T, 22, 3)，建立 prompt → joints 索引。"""
    sources = [
        (PROJECT_ROOT / "runs" / "hy_motion_9prompts", "prompts.txt", "joints.npy"),
        (PROJECT_ROOT / "runs" / "mdm_9prompts",       "prompts.txt", "joints.npy"),
    ]
    for d, pf, jf in sources:
        if not (d / jf).exists():
            continue
        try:
            joints = np.load(d / jf)                          # (N, T, 22, 3)
            prompts = (d / pf).read_text(encoding="utf-8").splitlines() \
                if (d / pf).exists() else [f"sample_{i}" for i in range(joints.shape[0])]
            for i, p in enumerate(prompts[:joints.shape[0]]):
                CACHED_PROMPTS.append((p.lower(), joints[i]))
        except Exception as e:
            print(f"[mock] skipping {d}: {e}", file=sys.stderr)
    print(f"[mock] loaded {len(CACHED_PROMPTS)} cached prompt → motion samples", file=sys.stderr)


def _best_match(prompt: str, duration: float) -> np.ndarray:
    """启发式：词袋匹配选最像的缓存样本；都不像就用第一个或合成 idle。"""
    p_low = prompt.lower()
    p_words = set(re.findall(r"\w+", p_low))
    best_score, best_j = -1, None
    for cached_text, j in CACHED_PROMPTS:
        cached_words = set(re.findall(r"\w+", cached_text))
        score = len(p_words & cached_words)
        if score > best_score:
            best_score, best_j = score, j
    if best_j is None:
        return _synthetic_idle(int(duration * 30))
    target_T = int(duration * 30)
    if best_j.shape[0] >= target_T:
        return best_j[:target_T]
    reps = -(-target_T // best_j.shape[0])
    return np.tile(best_j, (reps, 1, 1))[:target_T]


def _synthetic_idle(T: int) -> np.ndarray:
    """毫无 cache 时的兜底：站立 + 呼吸。"""
    base = np.array([
        [+0.00, +0.00, 0.0], [+0.10, -0.08, 0.0], [-0.10, -0.08, 0.0],
        [+0.00, +0.12, 0.0], [+0.12, -0.50, 0.0], [-0.12, -0.50, 0.0],
        [+0.00, +0.24, 0.0], [+0.12, -0.92, 0.0], [-0.12, -0.92, 0.0],
        [+0.00, +0.36, 0.0], [+0.14, -0.98, 0.08], [-0.14, -0.98, 0.08],
        [+0.00, +0.50, 0.0], [+0.08, +0.42, 0.0], [-0.08, +0.42, 0.0],
        [+0.00, +0.62, 0.0], [+0.18, +0.40, 0.0], [-0.18, +0.40, 0.0],
        [+0.40, +0.40, 0.0], [-0.40, +0.40, 0.0], [+0.62, +0.40, 0.0],
        [-0.62, +0.40, 0.0],
    ], dtype=np.float32)
    out = np.zeros((max(T, 30), 22, 3), dtype=np.float32)
    for t in range(out.shape[0]):
        out[t] = base.copy()
        out[t, :, 1] += 0.012 * np.sin(2 * np.pi * t / 30)
    return out


# ─── FastAPI ────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:    str   = Field(..., min_length=1, max_length=512)
    duration:  float = Field(4.0, gt=0.5, le=12.0)
    cfg_scale: float = Field(5.0, gt=0.0, le=15.0)
    seed:      int   = Field(0, ge=0)


app = FastAPI(title="motion-diffusion-simulator · MOCK backend")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_load_cached()


@app.get("/")
async def root():
    return {"ok": True, "service": "mock-backend",
            "cached_samples": len(CACHED_PROMPTS),
            "note": "Replays cached real model outputs by word-overlap match. No GPU."}


@app.get("/api/info")
async def info():
    return {
        "model":           "MOCK (replay cached HY-Motion / MDM outputs)",
        "fps":             30,
        "max_duration_s":  12.0,
        "joint_count":     22,
        "joint_layout":    "SMPL-22 (HumanML3D order)",
        "kinematic_chain": KINEMATIC_CHAIN,
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    t0 = time.time()
    j = _best_match(req.prompt, req.duration)
    # tiny artificial latency so the UI shows the "diffusing..." stage
    time.sleep(0.2)
    return {
        "joints":           j.tolist(),
        "fps":              30,
        "kinematic_chain":  KINEMATIC_CHAIN,
        "shape":            list(j.shape),
        "inference_seconds": float(time.time() - t0),
        "request":          req.model_dump(),
        "note":             "served by mock_server.py — replaying cached output",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8888"))
    uvicorn.run(app, host="0.0.0.0", port=port)
