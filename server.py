"""
FastAPI backend for the Motion-Diffusion-Simulator demo UI.

Endpoints (all return JSON unless noted):
    GET  /api/baselines         -> list of available baselines
    POST /api/generate          -> Server-Sent Events stream for one job
    GET  /api/result/{job_id}   -> URLs of generated assets + metrics
    POST /api/compare           -> run several baselines on one prompt
    GET  /static/...            -> serve generated mp4/png files

Run on cluster:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# project imports ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline   import PRESETS, run_baseline, load_run            # noqa: E402
from translator import PromptTranslator, TranslatorConfig          # noqa: E402
from visualizer import render_all                                  # noqa: E402
from metrics    import full_report                                 # noqa: E402
from data_utils import recover_joints_from_263                     # noqa: F401, E402

JOBS_DIR = Path("static/jobs"); JOBS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Motion Diffusion Simulator")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt_zh: str
    baseline:  str = "hy_motion"
    duration:  int = 120
    cfg_scale: float = 5.0
    seed:      int = 42
    translate: bool = True


class CompareRequest(BaseModel):
    prompt_zh: str
    baselines: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/baselines")
def list_baselines() -> dict[str, Any]:
    return {
        name: {
            "name":        cfg.name,
            "runner":      cfg.runner,
            "description": cfg.description,
        }
        for name, cfg in PRESETS.items()
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.baseline not in PRESETS:
        raise HTTPException(400, f"unknown baseline: {req.baseline}")

    job_id  = uuid.uuid4().hex[:10]
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PRESETS[req.baseline]

    def stream():
        yield _sse("stage", {"name": "start", "job_id": job_id})

        # 1) translate
        if req.translate:
            yield _sse("stage", {"name": "translate", "msg": "rewriting Chinese -> English"})
            try:
                t = PromptTranslator(TranslatorConfig())
                en = t.translate(req.prompt_zh)
            except Exception as e:
                en = req.prompt_zh
                yield _sse("warn", {"msg": f"translator failed: {e}; using raw input"})
        else:
            en = req.prompt_zh
        yield _sse("stage", {"name": "translate", "msg": en})

        # 2) inference
        yield _sse("stage", {"name": "diffuse", "msg": f"running {cfg.name}"})
        t0 = time.time()
        try:
            run_baseline(cfg, [en], out_dir)
        except Exception as e:
            yield _sse("error", {"msg": f"inference failed: {e}"})
            return
        yield _sse("stage", {"name": "diffuse", "msg": f"done in {time.time()-t0:.1f}s"})

        # 3) visualise
        yield _sse("stage", {"name": "render", "msg": "rendering 4 views"})
        run = load_run(out_dir)
        joints_one = run["joints"][0, :int(run["lengths"][0])]
        artefacts = render_all(joints_one, out_dir, tag="result", fps=20)

        # 4) (optional) self-similarity sanity metrics
        try:
            feats = np.array([joints_one.std(0).flatten()])
            metrics = full_report(feats, feats)
        except Exception:
            metrics = {}

        manifest_extra = {
            "prompt_zh": req.prompt_zh,
            "prompt_en": en,
            "artefacts": {k: f"/static/jobs/{job_id}/{Path(v).name}" for k, v in artefacts.items()},
            "metrics":   metrics,
        }
        (out_dir / "ui_manifest.json").write_text(
            json.dumps(manifest_extra, indent=2, ensure_ascii=False), encoding="utf-8")
        yield _sse("done", {"job_id": job_id, **manifest_extra})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    out = JOBS_DIR / job_id
    if not out.exists():
        raise HTTPException(404, "job not found")
    manifest = out / "ui_manifest.json"
    if not manifest.exists():
        raise HTTPException(409, "job still running or failed")
    return json.loads(manifest.read_text(encoding="utf-8"))


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Synchronously run several baselines on the same prompt."""
    out_id  = uuid.uuid4().hex[:10]
    base    = JOBS_DIR / f"compare-{out_id}"
    base.mkdir(parents=True, exist_ok=True)

    en_prompt = PromptTranslator(TranslatorConfig()).translate(req.prompt_zh)
    (base / "prompts.txt").write_text(en_prompt, encoding="utf-8")

    results = {}
    for b in req.baselines:
        if b not in PRESETS:
            results[b] = {"ok": False, "error": "unknown baseline"}
            continue
        sub = base / b
        try:
            run_baseline(PRESETS[b], [en_prompt], sub)
            run = load_run(sub)
            j = run["joints"][0, :int(run["lengths"][0])]
            artefacts = render_all(j, sub, tag=b, fps=20)
            results[b] = {
                "ok": True,
                "artefacts": {k: f"/static/jobs/compare-{out_id}/{b}/{Path(v).name}"
                              for k, v in artefacts.items()},
            }
        except Exception as e:
            results[b] = {"ok": False, "error": str(e)}

    return {"job_id": f"compare-{out_id}", "prompt_en": en_prompt, "results": results}


@app.get("/")
def root():
    return {"ok": True, "service": "motion-diffusion-simulator", "version": "0.1.0"}
