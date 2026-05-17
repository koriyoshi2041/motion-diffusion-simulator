# Frontend — prompt → puppet（手绘蓝图风格 demo）

> 这是一个浏览器端的演示界面，调用后端 FastAPI（HY-Motion-1.0-Lite）把
> 自然语言 prompt 实时转成 22-joint 3D 骨架动画。
>
> 设计来自原模板 `3dpeoplediffusion.zip`，我们只把中间的 CSS-3D puppet
> 替换成「调真模型 + SVG 真骨架渲染」，其他手绘元素（手机 / 管道 / 终端 /
> 注释 / 角标）全部保留。

## 怎么跑

### 1. 启后端（集群上）

```bash
ssh qiyuan
tmux new -d -s hyserver '
  cd /nfsdata/wxu/motion_diffusion_simulator &&
  source /nfsdata/wxu/miniconda3/etc/profile.d/conda.sh &&
  conda activate cpm &&
  CUDA_VISIBLE_DEVICES=5 HY_VARIANT=lite HY_DEVICE=0 \
    uvicorn server_hy:app --host 0.0.0.0 --port 8888
'
# 等约 50s 看到 "Application startup complete"
```

### 2. 启前端（本地 macOS）

```bash
cd motion_diffusion_simulator/frontend
python -m http.server 7777
# 浏览器: http://localhost:7777/prompt-puppet.html
```

### 3. 测试

在键盘上敲："a person throws a left hook punch" → 700 ms 后浏览器自动 fetch
`http://172.16.1.48:8888/api/generate` → 收到 (120, 22, 3) 骨架 → 火柴人开始播放出拳动画。

## 文件结构

| 文件 | 作用 |
|------|------|
| `prompt-puppet.html`         | 入口，加载 React UMD + 所有 jsx |
| `tweaks-panel.jsx`           | 右下角配置面板组件库（来自原模板） |
| `src/app.jsx`                | ⭐ 主组件，sendNow → async fetch |
| `src/actions.jsx`            | ⭐ fetchAction(prompt) + SMPL-22 chain |
| `src/stage.jsx`              | ⭐ 22-joint SVG 骨架渲染 + 舞台外壳 |
| `src/phone.jsx`              | 左边手机输入设备（未改） |
| `src/pipe.jsx`               | 中间神经管道动画（未改） |
| `src/terminal.jsx`           | 右下终端日志（未改） |

## 改 endpoint

默认 endpoint 是 `http://172.16.1.48:8888/api/generate`。要换：

```html
<!-- 在 prompt-puppet.html <head> 里加 -->
<script>window.HY_API = "http://your-host:port/api/generate";</script>
```

## 离线 fallback

后端连不上时（fetch 失败 / 超时），自动退回 `ACTIONS.idle` 站立呼吸姿态，
浏览器不会黑屏。可在 Tweaks 面板里看到 "[err]" 红色日志。

## Tweaks 面板（右下角）

| 项 | 默认 | 用途 |
|----|-----|------|
| puppet scale     | 1.6  | 火柴人大小 |
| theme            | dusk | 舞台背景 |
| pipe style       | neural | 管道风格 |
| auto-transmit    | true | 输入停下后 700 ms 自动发送 |
| terminal speed   | 1.0  | 终端日志逐行间隔 |
| annotations      | true | 是否显示蓝图注释 |
| **duration (s)** | 4.0  | ⭐ 让模型生成多长 |
| **cfg scale**    | 5.0  | ⭐ classifier-free guidance 强度 |

`duration` 和 `cfg_scale` 直接传给 HY-Motion，影响生成质量：
- duration 长 → 动作展开更完整；HY-Motion 训练上限 12s
- cfg 高（7-10）→ 严格跟 prompt；cfg 低（1-3）→ 多样性高但偏离 prompt

## 用 Three.js 替代 SVG？

我们最终选 SVG 而不是 Three.js，因为：
1. 22 关节 + 5 条骨架链非常少，SVG 在浏览器 60fps 没压力
2. Three.js + babel UMD 加载慢，体积大
3. 手绘风格美学下 SVG 已经足够

如果将来要做更复杂的 3D（如带身体表皮 / 多人），可以把 `stage.jsx` 的
`SkeletonSvg` 换成 Three.js `LineSegments`。数据契约不变。
