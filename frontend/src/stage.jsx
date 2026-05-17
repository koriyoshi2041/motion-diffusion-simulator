// ────────────────────────────────────────────────────────────────────────
// stage.jsx —— 3D 舞台 + 4 种 viz 切换 (stickfigure / jointmap / topdown / energy)
//
// 输入：props.action = {
//   name, frames: number[T][22][3],
//   fps, kinematic_chain, duration (ms),
//   inference_seconds, prompt,
// }
//
// 四种 viz：
//   • stickfigure: 3D 骨架按 fps 实时播放 (raf 推进 frameIdx)
//   • jointmap:    22 joint × T 帧的速度模长热图（一次性算）
//   • topdown:     root 关节 (x, z) 俯视轨迹 + 起止点
//   • energy:      每帧总动能曲线
// 用户通过舞台顶部 tab 切换；状态保存在 stageMode。
// ────────────────────────────────────────────────────────────────────────


const KINEMATIC = window.SMPL22_KINEMATIC_CHAIN || [
  [0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
];

// 不同骨段的颜色（与原 demo 蓝/橙/紫色调对齐）
const CHAIN_COLORS = ["#c0392b", "#d97706", "#1d4d8a", "#6da3d6", "#9b59b6"];


// ──────────────────────────────────────────────────────────────────────
// 透视投影：3D (x, y, z) → 2D SVG 坐标
//   把 model 的 y 轴方向（向上）映射到 SVG 的负 y（屏幕坐标 y 向下）。
//   z 用来做轻微缩放（远小近大），但不做真透视裁剪（动作在 ~1m 球内）。
// ──────────────────────────────────────────────────────────────────────
function projectFrame(frame, opts) {
  const { cx, cy, scale, zScale } = opts;
  const out = new Array(frame.length);
  for (let i = 0; i < frame.length; i++) {
    const [x, y, z] = frame[i];
    const k = 1 + z * zScale;                 // 简单深度缩放
    out[i] = [cx + x * scale * k, cy - y * scale * k, z];
  }
  return out;
}


// ──────────────────────────────────────────────────────────────────────
// 骨架 SVG：给定一帧投影后的 22 个 2D 点 + chains，画 5 条线 + 22 个圆
// ──────────────────────────────────────────────────────────────────────
function SkeletonSvg({ joints2d, width, height }) {
  // 5 条骨架链
  const polylines = KINEMATIC.map((chain, i) => {
    const pts = chain.map((j) => joints2d[j]).map((p) => `${p[0]},${p[1]}`).join(" ");
    return (
      <polyline
        key={i}
        points={pts}
        fill="none"
        stroke={CHAIN_COLORS[i % CHAIN_COLORS.length]}
        strokeWidth={3.4}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.95}
      />
    );
  });

  // 22 个关节圆点
  const dots = joints2d.map((p, i) => (
    <circle
      key={i}
      cx={p[0]}
      cy={p[1]}
      r={3}
      fill="#fff8d8"
      stroke="#1d1a14"
      strokeWidth={1.2}
    />
  ));

  // 头部高亮（关节 15）
  const head = joints2d[15];

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    >
      {/* 地面投影（在 root 关节 0 下方画椭圆阴影） */}
      <ellipse
        cx={joints2d[0][0]}
        cy={joints2d[0][1] + 220}
        rx={68}
        ry={10}
        fill="rgba(29,26,20,.45)"
        filter="blur(2px)"
      />
      {polylines}
      {dots}
      {/* 头部 LED 装饰 */}
      {head && (
        <>
          <circle cx={head[0]} cy={head[1] - 16} r={4} fill="#d97706"
                  style={{ filter: "drop-shadow(0 0 6px #ffb547)" }}>
            <animate attributeName="opacity" values="0.4;1;0.4" dur="1.6s" repeatCount="indefinite"/>
          </circle>
          <line x1={head[0]} y1={head[1] - 6} x2={head[0]} y2={head[1] - 14}
                stroke="#1d1a14" strokeWidth={2}/>
        </>
      )}
    </svg>
  );
}


// ──────────────────────────────────────────────────────────────────────
// 关节运动强度热图 (jointmap): 22 joint × T-1 帧
// ──────────────────────────────────────────────────────────────────────
function JointHeatmap({ frames, width, height }) {
  if (!frames || frames.length < 2) return null;
  const T = frames.length - 1, J = 22;
  // 速度模长 vel[t][j] = |frame[t+1] - frame[t]|_xyz
  const vel = new Array(T);
  let maxV = 1e-6;
  for (let t = 0; t < T; t++) {
    vel[t] = new Array(J);
    for (let j = 0; j < J; j++) {
      const dx = frames[t+1][j][0] - frames[t][j][0];
      const dy = frames[t+1][j][1] - frames[t][j][1];
      const dz = frames[t+1][j][2] - frames[t][j][2];
      const v = Math.sqrt(dx*dx + dy*dy + dz*dz);
      vel[t][j] = v;
      if (v > maxV) maxV = v;
    }
  }
  const padL = 36, padR = 8, padT = 10, padB = 24;
  const W = width - padL - padR, H = height - padT - padB;
  const cellW = W / T, cellH = H / J;
  const cells = [];
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < J; j++) {
      const v = vel[t][j] / maxV;
      // viridis-ish: dark blue -> green -> yellow
      const r = Math.round(255 * Math.min(1, Math.max(0, v*2 - 0.5)));
      const g = Math.round(255 * Math.min(1, v*1.5));
      const b = Math.round(255 * Math.max(0, 0.4 - v*0.4));
      cells.push(<rect key={`${t}-${j}`}
                      x={padL + t*cellW} y={padT + j*cellH}
                      width={cellW + 0.5} height={cellH + 0.5}
                      fill={`rgb(${r},${g},${b})`}/>);
    }
  }
  // axis labels
  const yTicks = [];
  for (let j = 0; j <= 21; j += 7) {
    yTicks.push(<text key={`y${j}`} x={padL - 6} y={padT + j*cellH + 4}
                      fill="#9aa5b1" fontSize="9" textAnchor="end" fontFamily="JetBrains Mono">{j}</text>);
  }
  return (
    <svg width={width} height={height} style={{position: "absolute", inset: 0}}>
      {cells}
      {yTicks}
      <text x={padL/2} y={height/2} fill="#6da3d6" fontSize="10"
            transform={`rotate(-90 ${padL/2} ${height/2})`} textAnchor="middle">joint id</text>
      <text x={padL + W/2} y={height - 6} fill="#6da3d6" fontSize="10" textAnchor="middle">frame</text>
    </svg>
  );
}


// ──────────────────────────────────────────────────────────────────────
// 俯视轨迹 (topdown): root 关节 (x, z) 路径
// ──────────────────────────────────────────────────────────────────────
function TopdownTrajectory({ frames, width, height, currentIdx }) {
  if (!frames || frames.length < 2) return null;
  const T = frames.length;
  // 取 root (joint 0) 的 x, z
  const pts = frames.map(f => [f[0][0], f[0][2]]);
  const xs = pts.map(p => p[0]), zs = pts.map(p => p[1]);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let zmin = Math.min(...zs), zmax = Math.max(...zs);
  if (xmax - xmin < 0.5) { const c = (xmin+xmax)/2; xmin = c - 0.25; xmax = c + 0.25; }
  if (zmax - zmin < 0.5) { const c = (zmin+zmax)/2; zmin = c - 0.25; zmax = c + 0.25; }
  const pad = 30;
  const scale = Math.min((width - 2*pad) / (xmax - xmin), (height - 2*pad) / (zmax - zmin));
  const cx = width/2, cy = height/2;
  const project = ([x, z]) => [cx + (x - (xmin+xmax)/2) * scale,
                               cy + (z - (zmin+zmax)/2) * scale];
  const path2d = pts.map(p => project(p)).map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const start = project(pts[0]), end = project(pts[T-1]);
  const curr = currentIdx != null ? project(pts[Math.min(currentIdx, T-1)]) : null;
  return (
    <svg width={width} height={height} style={{position: "absolute", inset: 0}}>
      {/* grid */}
      <defs>
        <pattern id="topdown-grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(43,108,176,.18)" strokeWidth="0.5"/>
        </pattern>
      </defs>
      <rect width={width} height={height} fill="url(#topdown-grid)"/>
      {/* trajectory */}
      <polyline points={path2d} fill="none" stroke="#6da3d6" strokeWidth="2"/>
      {/* start (green) and end (red) markers */}
      <circle cx={start[0]} cy={start[1]} r="6" fill="#16a34a" stroke="#000" strokeWidth="1.5"/>
      <text x={start[0]+10} y={start[1]+4} fill="#16a34a" fontSize="11" fontFamily="Kalam, cursive">start</text>
      <circle cx={end[0]} cy={end[1]} r="6" fill="#c0392b" stroke="#000" strokeWidth="1.5"/>
      <text x={end[0]+10} y={end[1]+4} fill="#c0392b" fontSize="11" fontFamily="Kalam, cursive">end</text>
      {/* current frame marker (animated) */}
      {curr && <circle cx={curr[0]} cy={curr[1]} r="4" fill="#fff8d8" stroke="#000" strokeWidth="1.2"/>}
      <text x={pad} y={pad - 6} fill="#9aa5b1" fontSize="10" fontFamily="JetBrains Mono">
        top-down · root (x, z) · {T} frames
      </text>
    </svg>
  );
}


// ──────────────────────────────────────────────────────────────────────
// 能量曲线 (energy): 每帧总动能
// ──────────────────────────────────────────────────────────────────────
function EnergyCurve({ frames, width, height, currentIdx }) {
  if (!frames || frames.length < 2) return null;
  const T = frames.length - 1;
  const energy = new Array(T);
  let maxE = 1e-6;
  for (let t = 0; t < T; t++) {
    let e = 0;
    for (let j = 0; j < 22; j++) {
      const dx = frames[t+1][j][0] - frames[t][j][0];
      const dy = frames[t+1][j][1] - frames[t][j][1];
      const dz = frames[t+1][j][2] - frames[t][j][2];
      e += dx*dx + dy*dy + dz*dz;
    }
    energy[t] = e;
    if (e > maxE) maxE = e;
  }
  const padL = 36, padR = 12, padT = 22, padB = 28;
  const W = width - padL - padR, H = height - padT - padB;
  const xAt = (t) => padL + (t / (T - 1)) * W;
  const yAt = (e) => padT + H - (e / maxE) * H;
  let path = `M ${xAt(0)} ${padT + H} `;
  for (let t = 0; t < T; t++) path += `L ${xAt(t).toFixed(1)} ${yAt(energy[t]).toFixed(1)} `;
  path += `L ${xAt(T-1)} ${padT + H} Z`;
  let line = `M ${xAt(0)} ${yAt(energy[0]).toFixed(1)} `;
  for (let t = 1; t < T; t++) line += `L ${xAt(t).toFixed(1)} ${yAt(energy[t]).toFixed(1)} `;
  // current
  const cur = currentIdx != null && currentIdx > 0
    ? [xAt(Math.min(currentIdx-1, T-1)), yAt(energy[Math.min(currentIdx-1, T-1)])] : null;
  return (
    <svg width={width} height={height} style={{position: "absolute", inset: 0}}>
      <path d={path} fill="rgba(217,119,6,0.25)"/>
      <path d={line} fill="none" stroke="#d97706" strokeWidth="2"/>
      {cur && <circle cx={cur[0]} cy={cur[1]} r="4" fill="#fff8d8" stroke="#d97706" strokeWidth="1.5"/>}
      {/* axes */}
      <line x1={padL} y1={padT + H} x2={padL + W} y2={padT + H} stroke="#9aa5b1" strokeWidth="0.6"/>
      <line x1={padL} y1={padT} x2={padL} y2={padT + H} stroke="#9aa5b1" strokeWidth="0.6"/>
      <text x={padL + W/2} y={height - 8} fill="#6da3d6" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono">frame</text>
      <text x={padL/2 - 4} y={padT + H/2} fill="#6da3d6" fontSize="9" textAnchor="middle"
            transform={`rotate(-90 ${padL/2 - 4} ${padT + H/2})`} fontFamily="JetBrains Mono">Σ‖Δj‖²</text>
      <text x={padL + 4} y={padT - 6} fill="#d97706" fontSize="11" fontFamily="Kalam, cursive">kinetic energy</text>
    </svg>
  );
}


// ──────────────────────────────────────────────────────────────────────
// 主舞台组件
// ──────────────────────────────────────────────────────────────────────
function Stage({ action, prompt, transmitting }) {
  // viz 切换：stickfigure / jointmap / topdown / energy
  const [stageMode, setStageMode] = React.useState("stickfigure");
  // 当前播放到第几帧
  const [frameIdx, setFrameIdx] = React.useState(0);
  // 用 ref 存最新 action，避免 raf 闭包引用旧值
  const actionRef = React.useRef(action);
  const startRef = React.useRef(performance.now());

  React.useEffect(() => {
    actionRef.current = action;
    startRef.current = performance.now();
    setFrameIdx(0);
  }, [action]);

  // 播放循环：按 fps 推进帧
  React.useEffect(() => {
    let raf;
    const loop = (now) => {
      const a = actionRef.current;
      if (a && a.frames && a.frames.length > 0) {
        const elapsed = (now - startRef.current) / 1000;
        const idx = Math.floor(elapsed * (a.fps || 30)) % a.frames.length;
        setFrameIdx(idx);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  // 测量舞台容器尺寸，用来居中骨架
  const stageRef = React.useRef(null);
  const [stageSize, setStageSize] = React.useState({ w: 640, h: 360 });
  React.useEffect(() => {
    const m = () => {
      if (!stageRef.current) return;
      const r = stageRef.current.getBoundingClientRect();
      setStageSize({ w: r.width, h: r.height });
    };
    m();
    window.addEventListener("resize", m);
    return () => window.removeEventListener("resize", m);
  }, []);

  // 投影：让骨架居中 + 适配舞台高度
  const frame = action?.frames?.[frameIdx];
  let joints2d = null;
  if (frame) {
    // 自动 scale：让最大边对应到舞台高度的 ~60%
    const allX = frame.map((p) => p[0]);
    const allY = frame.map((p) => p[1]);
    const span = Math.max(
      Math.max(...allX) - Math.min(...allX),
      Math.max(...allY) - Math.min(...allY)
    ) || 1.6;
    const scale = (stageSize.h * 0.55) / span;
    joints2d = projectFrame(frame, {
      cx: stageSize.w / 2,
      cy: stageSize.h * 0.52,
      scale,
      zScale: 0.08,
    });
  }

  // 4 种 viz tab 数据
  const TABS = [
    {key: "stickfigure", label: "skeleton",  hint: "3D pose"},
    {key: "jointmap",    label: "heat-map",  hint: "joint × time"},
    {key: "topdown",     label: "top-down",  hint: "root trajectory"},
    {key: "energy",      label: "energy",    hint: "kinetic curve"},
  ];

  // 「黑屏舞台」只有 stickfigure 模式才需要；其他三种用白底配色更清晰
  const isSkeleton = stageMode === "stickfigure";

  return (
    <div className="stage-wrap">
      <div className="stage-tag annot">↓ render target</div>

      {/* viz 切换 tab —— 沿用蓝图便利贴风格 */}
      <div className="viz-tabs">
        {TABS.map(t => (
          <button
            key={t.key}
            className={`viz-tab ${stageMode === t.key ? "active" : ""}`}
            onClick={() => setStageMode(t.key)}
            title={t.hint}
          >
            <span className="viz-tab-label">{t.label}</span>
            <span className="viz-tab-hint">{t.hint}</span>
          </button>
        ))}
      </div>

      <div className={`stage scribble-border ${isSkeleton ? "" : "stage-light"}`} ref={stageRef}>
        <div className="stage-bg" style={isSkeleton ? {} : {background: "#faf3df"}}>
          {/* 仅 skeleton 模式显示舞台外壳（地板/太阳/坐标轴/扫描线） */}
          {isSkeleton && <>
            <div className="grid-floor"></div>
            <div className="grid-back"></div>
            <div className="sun"></div>
            <svg className="axes" viewBox="0 0 100 60">
              <g stroke="#c0392b" strokeWidth=".6" fill="none">
                <line x1="6" y1="50" x2="6" y2="20"/>
                <line x1="6" y1="50" x2="36" y2="50"/>
                <line x1="6" y1="50" x2="22" y2="40"/>
                <text x="2" y="18" fontSize="5" fill="#c0392b" fontFamily="JetBrains Mono">y</text>
                <text x="38" y="52" fontSize="5" fill="#c0392b" fontFamily="JetBrains Mono">x</text>
                <text x="22" y="38" fontSize="5" fill="#c0392b" fontFamily="JetBrains Mono">z</text>
              </g>
            </svg>
            <div className="scanlines"></div>
            <div className="vignette"></div>
          </>}

          {/* 4 种 viz 之一 */}
          {stageMode === "stickfigure" && joints2d && (
            <SkeletonSvg joints2d={joints2d} width={stageSize.w} height={stageSize.h}/>
          )}
          {stageMode === "jointmap" && action?.frames && (
            <JointHeatmap frames={action.frames} width={stageSize.w} height={stageSize.h}/>
          )}
          {stageMode === "topdown" && action?.frames && (
            <TopdownTrajectory frames={action.frames} width={stageSize.w} height={stageSize.h} currentIdx={frameIdx}/>
          )}
          {stageMode === "energy" && action?.frames && (
            <EnergyCurve frames={action.frames} width={stageSize.w} height={stageSize.h} currentIdx={frameIdx}/>
          )}

          {/* 角标 */}
          <div className="puppet-tag annot blue">
            agent_0
            <span style={{ display: "block", fontSize: 11, color: "#3a342a" }}>
              · {action?.name === "diffuse" ? "HY-Motion 1.0-Lite" : (action?.name || "idle")}
            </span>
            {action?.inference_seconds != null && (
              <span style={{ display: "block", fontSize: 10, color: "#6b6450" }}>
                · infer {action.inference_seconds.toFixed(2)}s · {action?.frames?.length || 0}f
              </span>
            )}
          </div>

          {/* 箭头 callout */}
          <svg className="callout-svg" viewBox="0 0 200 100">
            <path d="M 18 14 Q 60 10 100 32" fill="none" stroke="#c0392b" strokeWidth="1.5" strokeDasharray="3 3"/>
            <path d="M 100 32 l -8 -2 m 8 2 l -3 -7" fill="none" stroke="#c0392b" strokeWidth="1.5"/>
          </svg>
        </div>

        {/* 边角刻度 */}
        <span className="corner tl"></span>
        <span className="corner tr"></span>
        <span className="corner bl"></span>
        <span className="corner br"></span>

        {/* 底部状态条 */}
        <div className="stage-strip mono">
          <span><span className="rec-dot"/> REC · {action?.fps || 30}fps</span>
          <span>frame {String(frameIdx).padStart(3,'0')}/{(action?.frames?.length || 0)}</span>
          <span>action::{action?.name || "idle"}</span>
          <span style={{ color: transmitting ? "#16a34a" : "#888" }}>
            {transmitting ? "● live" : "○ idle"}
          </span>
        </div>
      </div>

      <style>{`
        .stage-wrap { position: relative; margin-bottom: 16px; }
        .viz-tabs {
          position: absolute; top: -42px; right: 8px;
          display: flex; gap: 6px; z-index: 5;
        }
        .viz-tab {
          font-family: "Kalam", cursive;
          background: #fff7a8;
          color: #1d1a14;
          border: 2px solid #1d1a14;
          border-radius: 6px 8px 5px 7px;
          padding: 4px 10px 5px;
          cursor: pointer;
          transition: transform .12s, background .15s, box-shadow .12s;
          box-shadow: 1.5px 2px 0 rgba(29,26,20,.4);
          transform: rotate(-1deg);
          line-height: 1.05;
          min-width: 70px;
        }
        .viz-tab:nth-child(2) { transform: rotate(1deg); background: #ffe6b3; }
        .viz-tab:nth-child(3) { transform: rotate(-1.5deg); background: #dbe9ff; }
        .viz-tab:nth-child(4) { transform: rotate(0.8deg); background: #d8f0c9; }
        .viz-tab:hover { transform: rotate(0) translateY(-1px); box-shadow: 2.5px 4px 0 rgba(29,26,20,.5); }
        .viz-tab.active {
          background: #1d4d8a;
          color: #fff8d8;
          box-shadow: inset 0 0 0 2px rgba(255,255,255,.18), 1.5px 2px 0 rgba(29,26,20,.55);
        }
        .viz-tab-label {
          display: block;
          font-size: 14px; font-weight: 700;
        }
        .viz-tab-hint {
          display: block;
          font-family: "JetBrains Mono", monospace;
          font-size: 8px; opacity: .65;
          margin-top: 1px;
        }
        .stage-tag {
          position: absolute;
          top: -34px; left: 24px;
          font-size: 22px; color: #2b6cb0;
          transform: rotate(-3deg);
        }
        .stage {
          position: relative;
          background: #faf3df;
          padding: 8px;
          height: 360px;
          overflow: hidden;
        }
        .stage-bg {
          position: relative;
          width: 100%; height: 100%;
          background:
            radial-gradient(ellipse at 50% 70%, #2c3e58 0%, #16202e 60%, #0a0f17 100%);
          border-radius: 6px;
          overflow: hidden;
          border: 1.5px solid #1d1a14;
          perspective: 700px;
          transform-style: preserve-3d;
        }
        .grid-floor {
          position: absolute;
          left: -10%; right: -10%;
          bottom: 0; height: 55%;
          background-image:
            linear-gradient(rgba(110,231,167,.45) 1px, transparent 1px),
            linear-gradient(90deg, rgba(110,231,167,.45) 1px, transparent 1px);
          background-size: 38px 38px, 38px 38px;
          transform: rotateX(70deg);
          transform-origin: 50% 100%;
          mask-image: linear-gradient(180deg, transparent 0%, #000 30%, #000 100%);
          -webkit-mask-image: linear-gradient(180deg, transparent 0%, #000 30%, #000 100%);
        }
        .grid-back {
          position: absolute;
          left: 0; right: 0; top: 0; height: 60%;
          background-image:
            linear-gradient(rgba(43,108,176,.18) 1px, transparent 1px),
            linear-gradient(90deg, rgba(43,108,176,.18) 1px, transparent 1px);
          background-size: 24px 24px;
        }
        .sun {
          position: absolute;
          left: 50%; top: 38%;
          width: 240px; height: 240px;
          margin: -120px 0 0 -120px;
          border-radius: 50%;
          background: radial-gradient(circle, rgba(255,180,90,.5), rgba(255,90,80,.18) 50%, transparent 70%);
          mix-blend-mode: screen;
        }
        .axes { position: absolute; left: 12px; bottom: 8px; width: 80px; height: 60px; opacity: .9; }
        .scanlines {
          position: absolute; inset: 0;
          background-image: repeating-linear-gradient(
            0deg, rgba(0,0,0,0) 0 2px, rgba(0,0,0,.18) 2px 3px
          );
          mix-blend-mode: multiply;
          pointer-events: none;
        }
        .vignette {
          position: absolute; inset: 0;
          box-shadow: inset 0 0 100px rgba(0,0,0,.55);
          pointer-events: none;
        }
        .puppet-tag {
          position: absolute;
          right: 12px; top: 36%;
          font-size: 20px;
          font-family: "Caveat", cursive;
          font-weight: 700;
          text-align: left;
        }
        .callout-svg {
          position: absolute;
          right: 0; top: 28%;
          width: 100px; height: 50px;
        }
        .corner {
          position: absolute;
          width: 14px; height: 14px;
          border: 2px solid #1d1a14;
        }
        .corner.tl { top: 4px; left: 4px; border-right: none; border-bottom: none; }
        .corner.tr { top: 4px; right: 4px; border-left: none; border-bottom: none; }
        .corner.bl { bottom: 4px; left: 4px; border-right: none; border-top: none; }
        .corner.br { bottom: 4px; right: 4px; border-left: none; border-top: none; }
        .stage-strip {
          position: absolute;
          left: 14px; right: 14px; bottom: 12px;
          display: flex; justify-content: space-between; align-items: center;
          padding: 4px 10px;
          background: rgba(12,26,20,.85);
          color: #6ee7a7;
          font-size: 10px;
          border: 1px solid #1d1a14;
          border-radius: 4px;
          letter-spacing: .04em;
        }
        .rec-dot {
          display: inline-block; width: 6px; height: 6px;
          background: #c0392b; border-radius: 50%;
          margin-right: 6px;
          animation: pulse 1.2s infinite;
          box-shadow: 0 0 4px #c0392b;
        }
      `}</style>
    </div>
  );
}

window.Stage = Stage;
