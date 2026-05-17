// ────────────────────────────────────────────────────────────────────────
// actions.jsx —— prompt → 真实动作骨架（22 关节 × N 帧）
//
// 替换原 CSS-3D 关键字驱动：现在每次 sendNow 都异步打集群 FastAPI
// (`/api/generate`)，拿回 HY-Motion-1.0-Lite 模型 diffusion 推理出的
// 真 3D 关节坐标 (T, 22, 3)。
//
// 为了保留原 demo 的「瞬时反馈」手感，我们：
//   1. 立刻把当前 action 切到 "loading idle"（轻摇摆）；
//   2. 后台 fetch 拿到 frames 后，再切到 { name, frames, fps, kinematic_chain }；
//   3. fetch 失败/超时 → 退回 fallback 站姿动作，前端始终能动。
//
// 后端约定（server_hy.py）：
//   POST /api/generate  {prompt, duration, cfg_scale, seed}
//        → {joints: number[T][22][3], fps, kinematic_chain, shape, inference_seconds}
// ────────────────────────────────────────────────────────────────────────


// ──────────────────────────────────────────────────────────────────────
// 标准 SMPL-22 骨架连接（与 HumanML3D / HY-Motion 前 22 关节顺序一致）
// 0  Pelvis    1  L_Hip     2  R_Hip    3  Spine1   4  L_Knee
// 5  R_Knee    6  Spine2    7  L_Ankle  8  R_Ankle  9  Spine3
// 10 L_Foot    11 R_Foot    12 Neck     13 L_Collar 14 R_Collar
// 15 Head      16 L_Shoulder 17 R_Shoulder
// 18 L_Elbow   19 R_Elbow   20 L_Wrist   21 R_Wrist
// ──────────────────────────────────────────────────────────────────────
const SMPL22_KINEMATIC_CHAIN = [
  [0, 2, 5, 8, 11],      // 右腿
  [0, 1, 4, 7, 10],      // 左腿
  [0, 3, 6, 9, 12, 15],  // 脊柱+头
  [9, 14, 17, 19, 21],   // 右臂
  [9, 13, 16, 18, 20],   // 左臂
];


// ──────────────────────────────────────────────────────────────────────
// 默认 server endpoint —— 直接打集群（aTrust VPN 走通后浏览器可直连）
//   优先级：URL ?api=... > window.HY_API > 集群 172.16.1.48:8889 (Full)
//   :8889 = HY-Motion-1.0 Full (1B)；:8888 = HY-Motion-1.0-Lite (0.46B)
// ──────────────────────────────────────────────────────────────────────
const CLUSTER_ENDPOINT = "http://172.16.1.48:8889/api/generate";

function _resolveEndpoint() {
  try {
    const urlApi = new URLSearchParams(window.location.search).get("api");
    if (urlApi) return urlApi;
  } catch (e) {}
  if (window.HY_API) return window.HY_API;
  return CLUSTER_ENDPOINT;
}
const DEFAULT_ENDPOINT = _resolveEndpoint();


// ──────────────────────────────────────────────────────────────────────
// fallback 帧：服务挂时也至少有一个站立 idle，前端不会黑屏
// 22 个关节按 SMPL-22 标准位置，单位米
// ──────────────────────────────────────────────────────────────────────
function makeFallbackIdle(T = 60) {
  const base = [
    [+0.00, +0.00, 0.0], // 0 pelvis
    [+0.10, -0.08, 0.0], // 1 L_Hip
    [-0.10, -0.08, 0.0], // 2 R_Hip
    [+0.00, +0.12, 0.0], // 3 spine1
    [+0.12, -0.50, 0.0], // 4 L_Knee
    [-0.12, -0.50, 0.0], // 5 R_Knee
    [+0.00, +0.24, 0.0], // 6 spine2
    [+0.12, -0.92, 0.0], // 7 L_Ankle
    [-0.12, -0.92, 0.0], // 8 R_Ankle
    [+0.00, +0.36, 0.0], // 9 spine3
    [+0.14, -0.98, 0.08],// 10 L_Foot
    [-0.14, -0.98, 0.08],// 11 R_Foot
    [+0.00, +0.50, 0.0], // 12 neck
    [+0.08, +0.42, 0.0], // 13 L_Collar
    [-0.08, +0.42, 0.0], // 14 R_Collar
    [+0.00, +0.62, 0.0], // 15 head
    [+0.18, +0.40, 0.0], // 16 L_Shoulder
    [-0.18, +0.40, 0.0], // 17 R_Shoulder
    [+0.40, +0.40, 0.0], // 18 L_Elbow
    [-0.40, +0.40, 0.0], // 19 R_Elbow
    [+0.62, +0.40, 0.0], // 20 L_Wrist
    [-0.62, +0.40, 0.0], // 21 R_Wrist
  ];
  const frames = [];
  for (let i = 0; i < T; i++) {
    const t = i / (T - 1);
    const dy = Math.sin(t * Math.PI * 2) * 0.012;     // 呼吸式上下浮动
    const f = base.map((p) => [p[0], p[1] + dy, p[2]]);
    frames.push(f);
  }
  return {
    name: "idle",
    keywords: [],
    duration: 2000,
    log: ["server=offline?", "pose=idle (fallback)", "loop=breathing"],
    frames,
    fps: 30,
    kinematic_chain: SMPL22_KINEMATIC_CHAIN,
  };
}


// ──────────────────────────────────────────────────────────────────────
// 真模型路径：fetch FastAPI 并把 (T, 22, 3) 包成 action 对象
// ──────────────────────────────────────────────────────────────────────
async function fetchAction(prompt, opts = {}) {
  const body = {
    prompt: (prompt || "").trim() || "a person stands idle",
    duration: opts.duration ?? 4.0,
    cfg_scale: opts.cfg_scale ?? 5.0,
    seed: opts.seed ?? 0,
  };
  const endpoint = opts.endpoint || DEFAULT_ENDPOINT;
  const ctl = new AbortController();
  const timeout = setTimeout(() => ctl.abort(), opts.timeout_ms ?? 60_000);
  try {
    const r = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctl.signal,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return {
      name: "diffuse",
      keywords: [],
      duration: (d.joints.length / d.fps) * 1000,
      log: [
        `endpoint=${endpoint.replace(/^https?:\/\//, "")}`,
        `model=HY-Motion-1.0-Full`,
        `frames=${d.joints.length}`,
        `fps=${d.fps}`,
        `infer=${(d.inference_seconds || 0).toFixed(2)}s`,
      ],
      frames: d.joints,                                 // (T, 22, 3)
      fps: d.fps || 30,
      kinematic_chain: d.kinematic_chain || SMPL22_KINEMATIC_CHAIN,
      inference_seconds: d.inference_seconds,
      prompt: body.prompt,
    };
  } finally {
    clearTimeout(timeout);
  }
}


// ──────────────────────────────────────────────────────────────────────
// 暴露给 app.jsx / stage.jsx
// ──────────────────────────────────────────────────────────────────────
window.SMPL22_KINEMATIC_CHAIN = SMPL22_KINEMATIC_CHAIN;
window.fetchAction = fetchAction;
window.ACTIONS = { idle: makeFallbackIdle() };
window.classifyPrompt = (_text) => window.ACTIONS.idle;   // 立即返回 idle，真 action 走 fetchAction
