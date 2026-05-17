<p align="center">
  <img src="assets/banner.png" alt="motion-diffusion-simulator" />
</p>

<h1 align="center">motion-diffusion-simulator</h1>

<p align="center">
  <strong>文本驱动 3D 人体动作扩散仿真器</strong><br/>
  <em>A finals project for "AI Mathematical Foundations" — turning the assignment prompt
  "input distribution p, output q, evaluate with KL divergence" into a working text-to-motion system.</em>
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white"></a>
  <a href="#"><img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://github.com/koriyoshi2041/motion-diffusion-simulator/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/koriyoshi2041/motion-diffusion-simulator?style=social"></a>
</p>

---

## 一句话

输入一句自然语言（"a person throws a left hook punch"），返回一段符合该描述的 3D 人体动作
（22 关节 × N 帧）。基础模型用腾讯 HY-Motion 1.0 与 ICLR 2023 的 MDM 做对照。
**项目把作业母题的 KL 散度评价做成了完整的工程系统。**

---

## 📐 这门课的考查点 → 项目对应

| 作业要求 | 项目里如何实现 |
|---|---|
| **(1) 仿真器生成 AI 事件** | 文本 → (T, 22, 3) 动作序列的随机采样 |
| **(4) 随机模拟基础** | 扩散过程 = SDE 离散化采样；蒙特卡洛 + Langevin |
| **(5) 随机变量 / 分布模拟能力** | 模型学习并采样 $p_\theta(x \mid \text{text})$ |
| **(6) 用模拟数据对分布做仿真** | 在 HumanML3D 真实分布上训练，生成 q ≈ p |
| **(7) 输入分布 p → 输出符合 p 的数据** | MDM / HY-Motion 都在做这件事 |
| **(8) 用 $D(p\Vert q) = \sum_i p_i \log(p_i/q_i)$ 评价** | `src/metrics.py` 实现 kNN-KL / split-sample-JS / Sinkhorn-$W_2$，512-prompt 实测 |
| **(9) ML 学映射 fit P(X)** | 扩散损失 = ELBO = $\min D_\mathrm{KL}(p_\mathrm{data}\Vert p_\theta)$ 的上界 |
| **(10) 自然语言 → 示意图（与数据对应）** | 浏览器 SVG 实时火柴人 + 4 种 viz（骨架 / 热图 / 俯视 / 能量）|

---

## 🎬 Demo

<table>
<tr><th>三家模型对比 · 3 prompts × 6 keyframes</th></tr>
<tr><td><img src="assets/hero.png" alt="Hero figure"/></td></tr>
<tr><td><sub>MDM (35M) vs HY-Motion-Lite (0.46B) vs HY-Motion-Full (1B) 在同 prompt 上的对比。
注意第 4-6 行（side flip）HY-Lite 在 t=25% 真的倒立中；第 7-9 行（push-ups）真的趴地。</sub></td></tr>
</table>

<table>
<tr><th>HY-Motion 9-prompt</th><th>MDM 9-prompt</th></tr>
<tr><td><img src="assets/hy_grid.png" width="100%"/></td><td><img src="assets/mdm_grid.png" width="100%"/></td></tr>
</table>

<table>
<tr><th>交互界面（设计图，hand-drawn / 蓝图风）</th></tr>
<tr><td><img src="assets/ui_demo.png" width="100%"/></td></tr>
<tr><td><sub>左边手机输入 prompt → 中间神经管道 → 右上 22-joint 真骨架（可在 <code>skeleton / heat-map / top-down / energy</code> 四种 viz 间切换）→ 右下伪终端打日志。
浏览器到 <code>localhost:7777/prompt-puppet.html</code> 即可。</sub></td></tr>
</table>

---

## 🚀 一键本地跑（不需要 GPU）

```bash
git clone https://github.com/koriyoshi2041/motion-diffusion-simulator
cd motion-diffusion-simulator
./run-local.sh
```

脚本会：
1. 装 Python 依赖（fastapi / uvicorn / numpy / pydantic）
2. 启 **mock backend** 在 `:8888`（回放仓库自带的真模型缓存输出，零 GPU）
3. 启前端在 `:7777`
4. 自动打开浏览器

到浏览器后，敲一句话（"a person performs a side flip"），700ms 后火柴人开始跳。
在舞台右上角的 4 个 tab 切换不同 viz。

> 想跑真 HY-Motion 而不是 mock？看下面「真模型部署」一节。

---

## 📐 数学一眼

整个项目的数学心脏：

$$
\boxed{\;\min_\theta \, L_\text{simple}(\theta) \;\Longleftrightarrow\; \min_\theta \, D_\mathrm{KL}\!\left(p_\text{data}\,\big\|\,p_\theta\right) \text{ 的一个上界}\;}
$$

* $L_\text{simple} = \mathbb{E}_{t, x_0, \varepsilon}\,\Vert\varepsilon - \varepsilon_\theta(x_t, t, c)\Vert^2$ 是扩散 / Flow Matching 的训练损失
* ELBO 推导让它正好等于 KL 散度上界（推导见 `docs/teaching/`）

我们扩展了 motion-generation 的标准评测（FID / R-Precision / Diversity），自己加上三件套散度：

| 指标 | 估计器 | 代码 |
|---|---|---|
| $D_\mathrm{KL}(P\Vert Q)$ | Wang-Kulkarni-Verdú kNN | `src/metrics.py:kl_knn` |
| $D_\mathrm{JS}(P, Q)$ | split-sample mid-distribution + clamp $[0, \log 2]$ | `src/metrics.py:js_knn` |
| $W_2(P, Q)$ | Sinkhorn (POT) | `src/metrics.py:wasserstein2_sinkhorn` |

**512-prompt 实测**（MDM 与 HY-Lite，220-d fingerprint → PCA 16-d）：

| metric | value |
|---|---:|
| $D_\mathrm{KL}(\text{MDM}\Vert\text{HY})$ | 20.43 |
| $D_\mathrm{KL}(\text{HY}\Vert\text{MDM})$ | 26.66 |
| $D_\mathrm{JS}(\text{MDM}, \text{HY})$ | **0.6931 = log 2** |
| $W_2(\text{MDM}, \text{HY})$ | 0.3129 |
| split-half self KL: MDM | 10.38 |
| split-half self KL: HY  |  7.02 |

读数：JS 顶到理论上界 $\log 2$ → 两个生成模型分布几乎不重叠（独立训练，符合预期）。
KL ≈ 20-27 远大于同家族 split-half (7-10) → 跨家族差异远超同家族方差。

---

## 🏗️ 架构

```
┌─ 浏览器 ──────────────────────────────────────────────┐
│  prompt-puppet.html                                  │
│   ├─ app.jsx       键盘 → 700ms debounce → fetch     │
│   ├─ actions.jsx   POST /api/generate                │
│   └─ stage.jsx     22-joint SVG，4 种 viz 切换       │
└──────────────────────────┬───────────────────────────┘
                           │ HTTP
┌──────────────────────────▼───────────────────────────┐
│  后端二选一                                           │
│   • mock_server.py   本地无 GPU，回放缓存输出         │
│   • server_hy.py     生产，HY-Motion-Lite 常驻 GPU   │
└──────────────────────────────────────────────────────┘
```

---

## 📦 代码结构

```
motion-diffusion-simulator/
├── server_hy.py             生产后端（HY-Motion 常驻进程）
├── mock_server.py           本地 mock 后端（回放缓存）
├── run-local.sh             一键启动
├── src/
│   ├── metrics.py           ★ KL / JS / W₂ 散度估计器
│   ├── eval_t2m.py          T2M evaluator 包装（FID / R-Prec）
│   ├── visualizer.py        离线 4 种 viz 渲染（mp4 / png）
│   ├── translator.py        中文 → 英文 prompt 改写
│   ├── pipeline.py          多 baseline 调度抽象
│   └── data_utils.py        263-d ↔ 22-joint xyz 互转
├── scripts/
│   ├── runners/             每个 baseline 一个 wrapper
│   └── 00..06_*.py          smoke / setup / run / evaluate / figures
├── frontend/
│   ├── prompt-puppet.html   入口
│   ├── api-test.html        极简调试页
│   └── src/                 React + SVG 实现
├── prompts/                 9 zh + 9 en demos + 512 large eval
└── assets/                  README 图
```

---

## 🖥️ 真模型部署（可选）

mock backend 不需要 GPU，足够看完整 UI。但要真用 HY-Motion 推理，需要 ~22GB 显存
（Qwen3-8B 16G + CLIP-L 1.6G + HY-Motion-Lite 1.8G + activation）：

```bash
# 在有 80G 显存 GPU 的机器上
conda activate <你的 torch+cu126 环境>
CUDA_VISIBLE_DEVICES=0 HY_VARIANT=lite \
  uvicorn server_hy:app --host 0.0.0.0 --port 8888

# 然后让浏览器打集群 IP
http://localhost:7777/prompt-puppet.html?api=http://YOUR-GPU-HOST:8888/api/generate
```

权重下载方法和依赖修复见 `scripts/01_setup_data.sh`。

---

## 📊 推理速度

| 模型 | 参数 | 加载 | 单 prompt (4 秒动作) |
|---|---:|---:|---:|
| MDM | 35M | ~30 s | 0.5 s |
| HY-Motion-1.0-Lite | 0.46B | 47 s (5 s cached) | 2.6 s |
| HY-Motion-1.0-Full | 1.0B | ~50 s | 4.8 s |

---

## 🎓 学习路径（自底向上）

如果你只想理解项目背后的数学（不关心工程），按这个顺序读 `docs/teaching/`：

1. **Part 1：随机变量、概率分布、蒙特卡洛、HumanML3D 动作表示**
2. **Part 2：熵、KL、JS、Wasserstein —— 三件套散度从直觉到 kNN 估计器**
3. **Part 3：扩散过程、Flow Matching、ELBO → KL 上界、HY-Motion 内部**

每份 PDF 都自底向上，不假设读者会扩散模型；每份末尾一节才讲对应的工程实现。

---

## 🙏 Credits

- **HY-Motion 1.0** — Tencent Hunyuan ([paper](https://arxiv.org/abs/2512.23464) · [code](https://github.com/Tencent-Hunyuan/HY-Motion-1.0))
- **MDM** — Tevet et al. ICLR 2023 ([code](https://github.com/GuyTevet/motion-diffusion-model))
- **MoMask** — Guo et al. CVPR 2024
- **HumanML3D** — Guo et al. CVPR 2022 ([dataset](https://github.com/EricGuo5513/HumanML3D))

## 📜 License

MIT — see [LICENSE](LICENSE). Pretrained model weights follow their respective upstream licenses.

---

<p align="center"><sub>
A real diffusion playground · "input p, output q, evaluate with KL divergence" — done.
</sub></p>
