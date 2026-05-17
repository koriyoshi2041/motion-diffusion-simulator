// Pipe: a hand-drawn, blueprint-styled tube that snakes from the phone's
// output port over to the right-side stage. Animated data tokens flow along
// the path; rate / glow tied to whether a prompt is "transmitting".

function Pipe({ from, to, transmitting, charPulses }) {
  const dx = (to && from) ? to.x - from.x : 0;
  const path = React.useMemo(() => {
    if (!from || !to) return "";
    const x0 = from.x, y0 = from.y;
    const x1 = to.x,   y1 = to.y;
    const midX = x0 + dx * 0.5;

    // arc up & loop — keep within ~80px above the higher endpoint
    const upY = Math.min(y0, y1) - 90;
    const loopX = x0 + dx * 0.22;
    const loopY = upY - 28;

    return [
      `M ${x0} ${y0}`,
      // initial right-up curve
      `C ${x0 + 40} ${y0 - 30}, ${loopX - 50} ${upY + 10}, ${loopX} ${upY}`,
      // little loop-de-loop
      `C ${loopX + 36} ${upY - 6}, ${loopX + 44} ${loopY + 8}, ${loopX + 26} ${loopY}`,
      `C ${loopX + 8} ${loopY - 4}, ${loopX - 6} ${upY - 8}, ${loopX + 22} ${upY + 4}`,
      // bridge across the top
      `C ${loopX + 70} ${upY + 12}, ${midX - 20} ${upY - 8}, ${midX + 40} ${upY + 12}`,
      // approach target (gentle curve down)
      `C ${midX + 120} ${upY + 30}, ${x1 - 80} ${y1 - 30}, ${x1} ${y1}`,
    ].join(" ");
  }, [from && from.x, from && from.y, to && to.x, to && to.y, dx]);

  const pathRef = React.useRef(null);
  const [len, setLen] = React.useState(1200);
  React.useEffect(() => {
    if (pathRef.current && path) setLen(pathRef.current.getTotalLength());
  }, [path]);

  const PARTICLE_COUNT = 14;
  const [tick, setTick] = React.useState(0);
  React.useEffect(() => {
    let raf;
    const loop = () => { setTick((t) => t + 1); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  if (!from || !to) return null;

  // sample points on path for particles
  let points = [];
  if (pathRef.current && len > 0) {
    const speed = transmitting ? 0.0012 : 0.0004;
    const phase = (tick * speed) % 1;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const u = ((i / PARTICLE_COUNT) + phase) % 1;
      try {
        const p = pathRef.current.getPointAtLength(u * len);
        points.push({ x: p.x, y: p.y, u });
      } catch (e) {}
    }
  }

  // tick marks
  let ticks = [];
  if (pathRef.current && len > 0) {
    const n = Math.floor(len / 70);
    for (let i = 0; i < n; i++) {
      const u = (i + 0.5) / n;
      try {
        const p = pathRef.current.getPointAtLength(u * len);
        const p2 = pathRef.current.getPointAtLength(Math.min(len, u * len + 1));
        const angle = Math.atan2(p2.y - p.y, p2.x - p.x);
        const nx = Math.cos(angle + Math.PI / 2);
        const ny = Math.sin(angle + Math.PI / 2);
        ticks.push({ x1: p.x + nx*10, y1: p.y + ny*10, x2: p.x - nx*10, y2: p.y - ny*10 });
      } catch (e) {}
    }
  }

  return (
    <svg
      className="pipe-svg"
      style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 3 }}
      width="100%" height="100%"
    >
      <defs>
        {/* rough hand-drawn filter */}
        <filter id="rough" x="-2%" y="-2%" width="104%" height="104%">
          <feTurbulence type="fractalNoise" baseFrequency="0.025" numOctaves="2" seed="3"/>
          <feDisplacementMap in="SourceGraphic" scale="2.2" />
        </filter>
        <linearGradient id="pipeGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#d6cdb1"/>
          <stop offset="50%" stopColor="#f1ead8"/>
          <stop offset="100%" stopColor="#d6cdb1"/>
        </linearGradient>
        <radialGradient id="particleGlow" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%" stopColor="#fff8d8" stopOpacity="1"/>
          <stop offset="40%" stopColor="#d97706" stopOpacity=".9"/>
          <stop offset="100%" stopColor="#d97706" stopOpacity="0"/>
        </radialGradient>
      </defs>

      <g filter="url(#rough)">
        {/* outer pipe — the tube wall */}
        <path d={path} fill="none" stroke="#1d1a14" strokeWidth="22" strokeLinecap="round" strokeLinejoin="round" opacity=".95"/>
        <path d={path} fill="none" stroke="url(#pipeGrad)" strokeWidth="18" strokeLinecap="round" strokeLinejoin="round"/>
        {/* inner channel hint */}
        <path d={path} fill="none" stroke="rgba(29,26,20,.35)" strokeWidth="11" strokeLinecap="round" strokeLinejoin="round"/>
        {/* center light streak */}
        <path
          d={path}
          fill="none"
          stroke={transmitting ? "#ffd66b" : "#9aa0aa"}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray="6 8"
          strokeDashoffset={-tick * (transmitting ? 1.4 : 0.4)}
          opacity={transmitting ? 0.95 : 0.5}
          style={{ filter: transmitting ? "drop-shadow(0 0 6px #ffb547)" : "none" }}
        />
      </g>

      {/* hidden reference path for sampling */}
      <path ref={pathRef} d={path} fill="none" stroke="none"/>

      {/* tick marks along the pipe (every ~70px) */}
      {ticks.map((t, i) => (
        <line key={i}
          x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
          stroke="rgba(29,26,20,.35)" strokeWidth=".8"/>
      ))}

      {/* particles */}
      {points.map((p, i) => {
        const pulse = charPulses && charPulses[i % charPulses.length];
        const r = transmitting ? 4 + (i % 3) * 0.6 : 2.2;
        return (
          <g key={i} transform={`translate(${p.x},${p.y})`}>
            <circle r={r * 3} fill="url(#particleGlow)" opacity={transmitting ? 0.55 : 0.18}/>
            <circle r={r} fill={transmitting ? "#ffd66b" : "#a8b3c0"}
              stroke="#1d1a14" strokeWidth=".7"/>
          </g>
        );
      })}

      {/* annotations along pipe */}
      <g style={{ font: "700 16px Caveat, cursive", fill: "#c0392b" }}>
        <text x={from.x + 60} y={from.y - 30} transform={`rotate(-12 ${from.x + 60} ${from.y - 30})`}>
          tokens →
        </text>
      </g>
      <g style={{ font: "700 14px Caveat, cursive", fill: "#2b6cb0" }}>
        <text x={(from.x + to.x) / 2} y={Math.min(from.y, to.y) - 160}>
          ~ neural hose · 1 Gbps ~
        </text>
      </g>
    </svg>
  );
}

window.Pipe = Pipe;
