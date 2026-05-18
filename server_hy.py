"""
─────────────────────────────────────────────────────────────────────────
server_hy.py —— 持久化 FastAPI 后端（前端实际连这一个）
─────────────────────────────────────────────────────────────────────────

为什么需要持久化（不能用 subprocess）？
    HY-Motion 启动时要加载 22GB GPU 资源（Qwen3-8B 16G + CLIP 1.6G +
    HY-Motion 1.8G + 中间缓存），加载耗时 ~47 秒。如果每次 /api/generate
    都启子进程，等于每个请求都付 47s 启动开销 — 不可用。

做法：FastAPI lifespan 一次性加载，所有请求复用同一个 T2MRuntime 实例。
启动后每个 prompt 只需 ~2.6 秒纯 diffusion 推理。

并发：用 asyncio.Lock 串行化推理 —— GPU 一次只能跑一个 batch，无需多线程。

集群部署：
    ssh qiyuan
    tmux new -d -s hyserver '
      cd /nfsdata/wxu/motion_diffusion_simulator &&
      source /nfsdata/wxu/miniconda3/etc/profile.d/conda.sh &&
      conda activate cpm &&
      CUDA_VISIBLE_DEVICES=5 HY_VARIANT=lite HY_DEVICE=0 \
        uvicorn server_hy:app --host 0.0.0.0 --port 8888
    '

前端调用：
    POST /api/generate
        {"prompt": "a person waves", "duration": 4.0, "cfg_scale": 5.0, "seed": 0}
    → {"joints": [[[x,y,z], ...22], ...T], "fps": 30, "kinematic_chain": [...], ...}
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
HY_REPO = PROJECT_ROOT / "baselines" / "HY-Motion-1.0-repo"

# joint connectivity for SMPL-22 (HumanML3D order; first 22 of HY-Motion-52 too)
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512)
    duration: float = Field(4.0, gt=0.5, le=12.0)
    cfg_scale: float = Field(5.0, gt=0.0, le=15.0)
    seed: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Runtime singleton
# ---------------------------------------------------------------------------

class _HYRuntime:
    def __init__(self):
        self.runtime = None
        self.pipeline = None
        self.lock = asyncio.Lock()

    def load(self, variant: str = "lite", device: int = 0):
        if self.runtime is not None:
            return
        sys.path.insert(0, str(HY_REPO))
        os.chdir(HY_REPO)
        from hymotion.utils.t2m_runtime import T2MRuntime
        sub = "HY-Motion-1.0-Lite" if variant == "lite" else "HY-Motion-1.0"
        ckpt_dir = HY_REPO / "ckpts" / "tencent" / sub
        config_yml = ckpt_dir / "config.yml"
        ckpt_path = ckpt_dir / "latest.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        print(f"[server] loading HY-Motion-{variant} ...")
        t0 = time.time()
        self.runtime = T2MRuntime(
            config_path=str(config_yml),
            ckpt_name=str(ckpt_path),
            device_ids=[device],
            disable_prompt_engineering=True,
        )
        self.pipeline = self.runtime.pipelines[0]
        self.pipeline.train(False)
        print(f"[server] runtime ready in {time.time()-t0:.1f}s")

    async def generate(self, prompt: str, duration: float, cfg_scale: float, seed: int):
        async with self.lock:
            t0 = time.time()
            out = self.pipeline.generate(
                text=prompt, seed_input=[seed],
                duration_slider=duration, cfg_scale=cfg_scale,
                use_special_game_feat=False,
            )
            k3d = out["keypoints3d"]
            if hasattr(k3d, "cpu"):
                k3d = k3d.cpu().numpy()
            j = k3d[0, ..., :22, :].astype(np.float32)        # (L, 22, 3) root-relative

            # HY-Motion body_model.forward 只把 trans 加到 vertices，没加到 keypoints3d
            # 所以 root 永远停在 (0, -0.09, 0)，跳/走/蹲都看不见 —— 手动把 transl 加回去
            transl = out.get("transl") if isinstance(out, dict) else None
            if transl is not None:
                if hasattr(transl, "cpu"):
                    transl = transl.cpu().numpy()
                t = np.asarray(transl, dtype=np.float32)
                if t.ndim == 3:        # (B, L, 3) -> (L, 3)
                    t = t[0]
                if t.shape[0] == j.shape[0] and t.shape[-1] == 3:
                    j = j + t[:, None, :]     # 每帧每关节加上 root 位移
            return j, time.time() - t0


_state = _HYRuntime()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    variant = os.environ.get("HY_VARIANT", "lite")
    device = int(os.environ.get("HY_DEVICE", "0"))
    _state.load(variant=variant, device=device)
    yield


app = FastAPI(title="Motion-Diffusion-Simulator · HY-Motion Server",
              lifespan=_lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"ok": True, "service": "hy-motion-server", "loaded": _state.runtime is not None}


@app.get("/api/info")
async def info():
    return {
        "model":           "HY-Motion-1.0-" + os.environ.get("HY_VARIANT", "lite"),
        "fps":             30,
        "max_duration_s":  12.0,
        "joint_count":     22,
        "joint_layout":    "SMPL-22 (HumanML3D order)",
        "kinematic_chain": KINEMATIC_CHAIN,
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    if _state.pipeline is None:
        raise HTTPException(503, "model still loading")
    j, dt = await _state.generate(req.prompt, req.duration, req.cfg_scale, req.seed)
    return {
        "joints":          j.tolist(),                  # (T, 22, 3)
        "fps":             30,
        "kinematic_chain": KINEMATIC_CHAIN,
        "shape":           list(j.shape),
        "inference_seconds": float(dt),
        "request":         req.model_dump(),
    }


@app.post("/api/generate_sse")
async def generate_sse(req: GenerateRequest):
    """Streaming variant — useful when frontend wants progress events."""
    async def stream():
        yield _sse("stage", {"name": "queue"})
        if _state.pipeline is None:
            yield _sse("error", {"msg": "model still loading"})
            return
        yield _sse("stage", {"name": "diffuse"})
        j, dt = await _state.generate(req.prompt, req.duration, req.cfg_scale, req.seed)
        yield _sse("done", {
            "joints": j.tolist(), "fps": 30,
            "kinematic_chain": KINEMATIC_CHAIN,
            "inference_seconds": float(dt),
        })
    return StreamingResponse(stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
