<p align="center">
  <img src="assets/banner.png" alt="motion-diffusion-simulator" />
</p>

<h1 align="center">motion-diffusion-simulator</h1>

<p align="center">
  <strong>Text → 3D human motion · powered by HY-Motion 1.0 (Dec 2025 SOTA)</strong><br/>
  <em>An AI-Math final project that turns natural language prompts into real diffusion-generated 3D skeletons,
  with KL / JS / Wasserstein-2 divergence evaluation and a hand-drawn web UI.</em>
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.8%2Bcu126-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white"></a>
  <a href="#"><img alt="React" src="https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black"></a>
  <a href="#"><img alt="HY-Motion 1.0" src="https://img.shields.io/badge/HY--Motion-1.0%20Lite-FF6B6B"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://github.com/koriyoshi2041/motion-diffusion-simulator/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/koriyoshi2041/motion-diffusion-simulator?style=social"></a>
</p>

---

## ✨ TL;DR

**给一句中文 / 英文，浏览器实时跳出 3D 火柴人做对应动作。** 后端是腾讯 2025-12-31 开源的
**HY-Motion 1.0**（1B 参数 · DiT + Flow Matching · 当前公开 SOTA），跑在 6 卡 A800
集群上；前端是 hand-drawn / blueprint 风的 React + SVG 实时渲染。

数学骨架对应一份 AI-Math 大作业：把母题「输入分布 $p$、输出 $q$、用 KL 散度评价」
落地为完整的工程系统 —— **训练损失 ↔ ELBO ↔ KL 上界** 一条链打通。

---

## 🎬 Demo

<table>
<tr>
<th>Hero figure · 3 baselines × 3 prompts × 6 keyframes</th>
</tr>
<tr>
<td><img src="assets/hero.png" alt="Hero figure"/></td>
</tr>
<tr>
<td><sub>三家在同 prompt 上的表现：MDM (35M) vs HY-Motion-Lite (0.46B) vs HY-Motion-Full (1B)。
注意第 4-6 行（side flip）HY-Lite 在 t=25% 真的倒立中；第 7-9 行（push-ups）真的趴地。</sub></td>
</tr>
</table>

<table>
<tr>
<th>HY-Motion 9-prompt grid</th>
<th>MDM 9-prompt grid</th>
</tr>
<tr>
<td><img src="assets/hy_grid.png" width="100%"/></td>
<td><img src="assets/mdm_grid.png" width="100%"/></td>
</tr>
</table>

<table>
<tr>
<th>Live web UI (prompt → puppet)</th>
</tr>
<tr>
<td><img src="assets/ui_demo.png" width="100%"/></td>
</tr>
<tr>
<td><sub>Hand-drawn blueprint aesthetic 来自原模板，我们把中间的 CSS-3D puppet 替换为
调集群 FastAPI 拿到的真 22-joint SVG 骨架。</sub></td>
</tr>
</table>

---

## 🚀 Quick start

### 1 · Backend on cluster

```bash
ssh qiyuan
tmux new -d -s hyserver '
  cd motion_diffusion_simulator &&
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate cpm &&
  CUDA_VISIBLE_DEVICES=5 HY_VARIANT=lite HY_DEVICE=0 \
    uvicorn server_hy:app --host 0.0.0.0 --port 8888
'
# Waits ~47s, then "Application startup complete".
```

### 2 · Frontend (local Mac)

```bash
cd frontend
python -m http.server 7777
# open http://localhost:7777/prompt-puppet.html
```

### 3 · Smoke test (no GPU, no network)

```bash
python scripts/00_smoke_test.py
# verifies metrics.py, visualizer.py, data_utils.py on synthetic data
```

---

## 📐 Math at a glance

The whole project is built around one identity:

$$
\boxed{\;
  \min_\theta\; L_{\mathrm{vlb}}(\theta)
  \;\Longleftrightarrow\;
  \min_\theta\; D_{\mathrm{KL}}\bigl(p_{\mathrm{data}}\,\|\,p_\theta\bigr)\ \text{上界}
\;}
$$

That is: **diffusion / flow-matching training is literally minimizing an upper bound on
the KL divergence between data distribution and model distribution** — directly matching
the assignment's prompt "用相对熵 D(p‖q) 评价". Full derivation in
[`docs/teaching/part_a_model_math.pdf`](docs/teaching/part_a_model_math.pdf).

We extend the standard motion-generation evaluation (FID, R-Precision, Diversity)
with the three divergences:

| Metric | Estimator | Code |
|---|---|---|
| $D_{\mathrm{KL}}(P\|Q)$ | Wang-Kulkarni-Verdú kNN | `src/metrics.py:kl_knn` |
| $D_{\mathrm{JS}}(P,Q)$  | split-sample mid-distribution + clamp to $[0, \log 2]$ | `src/metrics.py:js_knn` |
| $W_2(P,Q)$              | Sinkhorn (POT) | `src/metrics.py:wasserstein2_sinkhorn` |

---

## 🏗️ Architecture

```
┌─ macOS (browser) ──────────────────────────────────────────────┐
│  prompt-puppet.html                                            │
│   ├─ app.jsx       keystrokes → 700ms debounce → fetch         │
│   ├─ actions.jsx   POST /api/generate                          │
│   └─ stage.jsx     22-joint SVG skeleton @ 30fps               │
└──────────────────────────┬─────────────────────────────────────┘
                           │ HTTP (VPN)
┌──────────────────────────▼─────────────────────────────────────┐
│  qiyuan cluster (8×A800-80GB)                                  │
│  server_hy.py  (FastAPI + asyncio.Lock)                        │
│   └─ T2MRuntime (resident in GPU 5, ~22GB)                     │
│        ├─ Qwen3-8B (16G)         ← text encoder                │
│        ├─ CLIP-large (1.6G)      ← sentence encoder            │
│        └─ HY-Motion-Lite (1.8G)  ← motion DiT (Flow Matching)  │
└────────────────────────────────────────────────────────────────┘
```

---

## 📦 Repo layout

```
motion_diffusion_simulator/
├── server_hy.py                  ⭐ FastAPI backend (HY-Motion in-process)
├── server.py                     general-purpose subprocess backend
├── src/
│   ├── metrics.py                ⭐ KL / JS / W₂ estimators (Part A)
│   ├── eval_t2m.py               T2M FID / R-Prec wrapper
│   ├── visualizer.py             4-mode mp4/heatmap/topdown/energy
│   ├── translator.py             zh → en prompt rewrite
│   ├── pipeline.py               unified baseline driver
│   └── data_utils.py             263-dim ↔ 22-joint xyz
├── scripts/
│   ├── runners/                  one wrapper per baseline
│   │   ├── hy_motion_runner.py   ⭐ HY-Motion adapter
│   │   ├── mdm_runner.py
│   │   ├── momask_runner.py
│   │   └── mock_runner.py
│   └── 00..06_*.py               smoke / setup / run / evaluate / figures
├── frontend/
│   ├── prompt-puppet.html        ⭐ hand-drawn React entry
│   ├── api-test.html             minimal API debug page
│   └── src/
│       ├── app.jsx, actions.jsx, stage.jsx   ⭐ adapted to call HY-Motion
│       ├── phone.jsx, pipe.jsx, terminal.jsx   (kept from original template)
│       └── ...
├── cluster/
│   ├── deploy.sh                 rsync project to NFS
│   └── run_distributed.sh        multi-node torchrun fine-tune
├── prompts/                      9 en + 9 zh demos + 512 large eval
└── assets/                       README images / mp4
```

---

## 🔬 Benchmarks (real cluster A800 inference)

9-prompt fingerprint divergences (220-d feature, PCA to 16-d):

| pair                          | W₂    | Fréchet (FID-style) |
|-------------------------------|------:|--------------------:|
| HY-Lite ↔ HY-Full             | **0.348**  | **53.5** |
| MDM ↔ HY-Full                 | 0.512 | 412.3 |
| MDM ↔ HY-Lite                 | 0.541 | 415.2 |
| MDM ↔ MDM (baseline)          | 0.127 |   0   |
| HY-Lite ↔ HY-Lite (baseline)  | 0.184 |   0   |

Same-family baselines (HY-Lite ↔ HY-Full) are closest, cross-family (MDM ↔ HY) are
far apart — as expected. **Diversity**: MDM 22.2, HY-Lite 12.1, HY-Full 12.2.

Inference speed (single A800):

| Model                | Params | Load   | Per prompt (4s clip) |
|----------------------|-------:|-------:|---------------------:|
| MDM                  | 35M    | ~30s   | 0.5s |
| HY-Motion-1.0-Lite   | 0.46B  | 47s (5s cached) | 2.6s |
| HY-Motion-1.0-Full   | 1.0B   | ~50s   | 4.8s |

---

## 🎓 Teaching documents (LaTeX → PDF)

Three didactic write-ups for the three-person team, each one ~10 pages with full Chinese,
LaTeX math, and code listings:

| Part | Focus | File |
|---|---|---|
| **A — Model & Math** | KL upper bound, kNN estimators, HY-Motion adapter | [`part_a_model_math.pdf`](docs/teaching/part_a_model_math.pdf) |
| **B — Backend & Runtime** | Persistent FastAPI server, baseline contracts, 6 deploy traps | [`part_b_backend_runtime.pdf`](docs/teaching/part_b_backend_runtime.pdf) |
| **C — Frontend, Eval & Viz** | SVG skeleton, 4-mode visualizations, divergence driver | [`part_c_frontend_ux.pdf`](docs/teaching/part_c_frontend_ux.pdf) |

---

## 🙏 Credits

- **HY-Motion 1.0** — Tencent Hunyuan team ([paper](https://arxiv.org/abs/2512.23464) ·
  [code](https://github.com/Tencent-Hunyuan/HY-Motion-1.0))
- **MDM** — Tevet et al. ICLR 2023 ([code](https://github.com/GuyTevet/motion-diffusion-model))
- **MoMask** — Guo et al. CVPR 2024
- **HumanML3D** — Guo et al. CVPR 2022 ([dataset](https://github.com/EricGuo5513/HumanML3D))
- **Frontend skeleton** — `prompt-puppet` hand-drawn template (modified to call real model)

---

## 📜 License

MIT — see [LICENSE](LICENSE). Model weights follow their respective upstream licenses.

---

<p align="center"><sub>
Built for an AI-Math final project · 2026 · A real diffusion playground.
</sub></p>
