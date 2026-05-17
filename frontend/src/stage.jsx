// ────────────────────────────────────────────────────────────────────────
// stage.jsx —— 3D 舞台 + SVG 22-joint 真骨架渲染
//
// 输入：props.action = {
//   name, frames: number[T][22][3],  // 来自 HY-Motion 真模型
//   fps, kinematic_chain, duration (ms),
//   inference_seconds, prompt,
// }
//
// 工作流：
//   • 在 requestAnimationFrame 里维护当前帧索引 idx；
//   • 每收到新 action 重置 idx=0 并按 fps 推进；
//   • 把 22 个 3D 关节通过简单透视投影到 2D，
//     再用 SVG <polyline> 沿 kinematic_chain 画 5 条骨架线 + 22 个圆点。
//   • 保留原 demo 的"舞台外壳": 地板网格 + 太阳 + 坐标轴 + scanlines + REC 角标。
//
// 这一文件**完全替换**了原 CSS-3D puppet（没有那么多 limb/elbow 节点的 CSS），
// 但保留了舞台的视觉壳子和注释手感。
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
// 主舞台组件
// ──────────────────────────────────────────────────────────────────────
function Stage({ action, prompt, transmitting }) {
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

  return (
    <div className="stage-wrap">
      <div className="stage-tag annot">↓ render target</div>
      <div className="stage scribble-border" ref={stageRef}>
        <div className="stage-bg">
          {/* 地板网格 */}
          <div className="grid-floor"></div>
          <div className="grid-back"></div>
          {/* 太阳 */}
          <div className="sun"></div>
          {/* 坐标轴 */}
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

          {/* 真骨架 SVG */}
          {joints2d && (
            <SkeletonSvg joints2d={joints2d} width={stageSize.w} height={stageSize.h}/>
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
