// ────────────────────────────────────────────────────────────────────────
// app.jsx —— 主组件：连接 phone / pipe / stage / terminal。
//
// 与原模板的关键差异：
//   • sendNow 现在是 async：调 fetchAction("/api/generate") 拿 HY-Motion 真骨架。
//   • 推理在远程 A800 GPU 上跑 ~2.6s，期间 UI 仍可继续输入。
//   • 失败 / 超时 → 退回 fallback idle，前端不会黑屏。
// ────────────────────────────────────────────────────────────────────────

const { useState, useEffect, useRef, useMemo, useCallback } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "puppetScale": 1.6,
  "pipeStyle": "neural",
  "stageTheme": "dusk",
  "showAnnotations": true,
  "autoTransmit": true,
  "termSpeed": 1.0,
  "duration": 4.0,
  "cfg_scale": 5.0
}/*EDITMODE-END*/;

function nowTs() {
  const d = new Date();
  const pad = (n, w=2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

function App() {
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [value, setValue] = useState("a person waves hello");
  const [activeKey, setActiveKey] = useState(null);
  const [action, setAction] = useState(window.ACTIONS.idle);
  const [transmitting, setTransmitting] = useState(false);
  const [logLines, setLogLines] = useState(() => seedLog());
  const [pipePts, setPipePts] = useState({ from: null, to: null });
  const [phonePulse, setPhonePulse] = useState(0);
  const stageRef = useRef(null);
  const transmitTimer = useRef(null);

  // Layout: anchor pipe endpoints to DOM after layout settles
  useEffect(() => {
    const measure = () => {
      const src = document.getElementById("pipe-source");
      const tgt = document.getElementById("pipe-target");
      if (!src || !tgt) return;
      const sb = src.getBoundingClientRect();
      const tb = tgt.getBoundingClientRect();
      setPipePts({
        from: { x: sb.left + sb.width / 2, y: sb.top + sb.height / 2 },
        to:   { x: tb.left + tb.width / 2, y: tb.top + tb.height / 2 },
      });
    };
    measure();
    window.addEventListener("resize", measure);
    const obs = new MutationObserver(measure);
    obs.observe(document.body, { subtree: true, childList: true, attributes: true });
    const id = setInterval(measure, 600);
    return () => { window.removeEventListener("resize", measure); obs.disconnect(); clearInterval(id); };
  }, []);

  // Global keystroke -> append to value & flash key
  useEffect(() => {
    const onKey = (e) => {
      // ignore when modifier
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const k = e.key;
      if (k === "Backspace") { setValue(v => v.slice(0, -1)); flashKey("⌫"); e.preventDefault(); }
      else if (k === "Enter") { sendNow(); flashKey("↵"); e.preventDefault(); }
      else if (k === " ") { setValue(v => v + " "); flashKey("␣"); e.preventDefault(); }
      else if (k.length === 1 && /[a-zA-Z0-9 ?!.,'"]/.test(k)) {
        setValue(v => v + k);
        flashKey(k);
      }
      // bump pulse on every press for the pipe
      setPhonePulse(p => p + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Auto-transmit on value change (debounced)
  useEffect(() => {
    if (!tweaks.autoTransmit) return;
    if (!value.trim()) {
      setAction(window.ACTIONS.idle);
      return;
    }
    setTransmitting(true);
    if (transmitTimer.current) clearTimeout(transmitTimer.current);
    transmitTimer.current = setTimeout(() => {
      sendNow();
    }, 700);
    return () => clearTimeout(transmitTimer.current);
    // eslint-disable-next-line
  }, [value, tweaks.autoTransmit]);

  function flashKey(k) {
    setActiveKey(k.toLowerCase());
    setTimeout(() => setActiveKey((cur) => (cur === k.toLowerCase() ? null : cur)), 130);
  }

  function pushLog(lines) {
    setLogLines((cur) => {
      const next = [...cur, ...lines];
      return next.slice(-160);
    });
  }

  // ──────────────────────────────────────────────────────────────────
  // sendNow: 把当前 value 发给 HY-Motion 远程推理；拿回 (T, 22, 3) 骨架后
  // setAction 切到真模型动作；前后都打 [parser]/[infer]/[render] 日志。
  // ──────────────────────────────────────────────────────────────────
  async function sendNow() {
    const prompt = value;
    const ts = nowTs();
    const nTokens = prompt.split(/\s+/).filter(Boolean).length;

    pushLog([
      { ts, tag: "[parser]", msg: `tokenize("${prompt || "<empty>"}") → ${nTokens} tokens`, kind: "info" },
      { ts: nowTs(), tag: "[infer]", msg: `POST /api/generate · HY-Motion-Lite · cfg=${tweaks.cfg_scale} · dur=${tweaks.duration}s`, kind: "info" },
    ]);

    setTransmitting(true);
    try {
      const a = await window.fetchAction(prompt, {
        duration:  tweaks.duration,
        cfg_scale: tweaks.cfg_scale,
        seed:      0,
      });
      setAction(a);
      pushLog([
        ...a.log.map((m) => ({ ts: nowTs(), tag: "[engine]", msg: m, kind: "info" })),
        { ts: nowTs(), tag: "[ok]", msg: `✔ playing ${a.frames.length} frames @ ${a.fps}fps`, kind: "ok" },
      ]);
    } catch (e) {
      pushLog([
        { ts: nowTs(), tag: "[err]", msg: `fetch failed: ${e.message} — using fallback idle`, kind: "err" },
      ]);
      setAction(window.ACTIONS.idle);
    } finally {
      setTransmitting(false);
    }
  }

  return (
    <div className="app-root">
      {/* Hand-drawn title block */}
      <header className="title-block">
        <div className="title-line-1 marker">prompt → puppet</div>
        <div className="title-line-2 handwriting">
          a real-time diffusion playground · v0.3 · sketched by you
        </div>
        <div className="title-stamp typewriter">
          <div>FILE NO. 042</div>
          <div>2026 / 05 / 04</div>
          <div>REV. C</div>
        </div>
      </header>

      {/* Big crosshair / scribbles around */}
      <Annotations show={tweaks.showAnnotations}/>

      {/* Phone */}
      <Phone
        value={value}
        onChange={setValue}
        activeKey={activeKey}
        onSend={sendNow}
        status={transmitting ? `transmitting "${value || "..."}"` : "ready"}
      />

      {/* Pipe */}
      <Pipe
        from={pipePts.from}
        to={pipePts.to}
        transmitting={transmitting}
        charPulses={[phonePulse]}
      />

      {/* Right column: stage + terminal */}
      <div className="right-col">
        <div id="pipe-target" className="pipe-target">
          <div className="port-ring"></div>
          <div className="port-hole"></div>
        </div>

        <Stage action={action} prompt={value} transmitting={transmitting}/>
        <Terminal logLines={logLines}/>
      </div>

      {/* Tweaks */}
      {window.TweaksPanel && (
        <TweaksPanel title="Tweaks">
          <TweakSection label="puppet">
            <TweakSlider label="scale"  value={tweaks.puppetScale} min={0.8} max={2.4} step={0.1}
              onChange={(v) => setTweak("puppetScale", v)} />
            <TweakRadio label="theme" value={tweaks.stageTheme}
              options={["dusk","void","matrix"]}
              onChange={(v) => setTweak("stageTheme", v)}/>
          </TweakSection>
          <TweakSection label="pipe">
            <TweakRadio label="style" value={tweaks.pipeStyle}
              options={["neural","copper","glass"]}
              onChange={(v) => setTweak("pipeStyle", v)}/>
            <TweakToggle label="auto-transmit" value={tweaks.autoTransmit}
              onChange={(v) => setTweak("autoTransmit", v)}/>
          </TweakSection>
          <TweakSection label="terminal">
            <TweakSlider label="speed" value={tweaks.termSpeed} min={0.3} max={3} step={0.1}
              onChange={(v) => setTweak("termSpeed", v)}/>
          </TweakSection>
          <TweakSection label="canvas">
            <TweakToggle label="annotations" value={tweaks.showAnnotations}
              onChange={(v) => setTweak("showAnnotations", v)}/>
          </TweakSection>
          <TweakSection label="diffusion">
            <TweakSlider label="duration (s)" value={tweaks.duration} min={1.5} max={8} step={0.5}
              onChange={(v) => setTweak("duration", v)}/>
            <TweakSlider label="cfg scale" value={tweaks.cfg_scale} min={1} max={12} step={0.5}
              onChange={(v) => setTweak("cfg_scale", v)}/>
          </TweakSection>
        </TweaksPanel>
      )}

      <style>{`
        .app-root {
          position: relative;
          width: 100%; height: 100%;
          --puppet-scale: ${tweaks.puppetScale};
        }
        .puppet-3d { transform: translate(-50%, -50%) scale(${tweaks.puppetScale}) !important; }

        .title-block {
          position: absolute;
          top: 6px; left: 290px;
          z-index: 7;
          text-align: left;
          pointer-events: none;
          background: rgba(241,234,216,.85);
          padding: 4px 12px 6px;
          border-radius: 4px;
        }
        .title-line-1 {
          font-size: 32px;
          font-weight: 700;
          letter-spacing: -.02em;
          line-height: 1;
          color: #1d1a14;
          text-shadow: 2px 2px 0 rgba(43,108,176,.16);
        }
        .title-line-2 {
          font-size: 14px;
          color: #c0392b;
          margin-top: 2px;
          transform: rotate(-1deg);
        }
        .title-stamp {
          position: absolute;
          top: -2px; left: 280px;
          width: 150px;
          padding: 4px 8px;
          border: 2px solid #c0392b;
          color: #c0392b;
          font-size: 8px;
          line-height: 1.4;
          letter-spacing: .1em;
          transform: rotate(4deg);
          opacity: .65;
        }
        .title-stamp::before {
          content: "TOP SECRET";
          position: absolute;
          top: -16px; left: -6px;
          font-size: 14px;
          font-family: "Special Elite";
          letter-spacing: .15em;
        }

        .right-col {
          position: absolute;
          right: 2%;
          top: 56%;
          transform: translateY(-50%);
          width: 58%;
          max-width: 920px;
          z-index: 4;
        }
        .pipe-target {
          position: absolute;
          left: -16px; top: 168px;
          width: 32px; height: 32px;
          z-index: 6;
        }
        .pipe-target .port-ring {
          position: absolute; inset: 0;
          border-radius: 50%;
          border: 3px solid #1d1a14;
          background: #d6cdb1;
        }
        .pipe-target .port-hole {
          position: absolute; inset: 8px;
          border-radius: 50%;
          background: radial-gradient(circle at 30% 30%, #6b6450, #1d1a14 70%);
        }
      `}</style>
    </div>
  );
}

function seedLog() {
  return [
    { ts: "00:00:00.001", tag: "[boot]",  msg: "diffusion-puppet v0.3.0", kind: "info" },
    { ts: "00:00:00.043", tag: "[boot]",  msg: "loading model: agent_0.glb (4.2 MB)", kind: "info" },
    { ts: "00:00:00.219", tag: "[boot]",  msg: "skeleton: 14 joints · 22 dof", kind: "info" },
    { ts: "00:00:00.301", tag: "[net]",   msg: "neural-hose ↔ phone.local · OK", kind: "info" },
    { ts: "00:00:00.412", tag: "[ok]",    msg: "✔ ready · awaiting prompt", kind: "ok" },
  ];
}

function Annotations({ show }) {
  if (!show) return null;
  return (
    <svg className="annot-layer" viewBox="0 0 1600 1000" preserveAspectRatio="none"
      style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 2 }}>
      {/* big crosshair */}
      <g stroke="rgba(43,108,176,.4)" strokeWidth="1" fill="none" strokeDasharray="4 4">
        <line x1="0" y1="500" x2="1600" y2="500"/>
        <line x1="800" y1="0" x2="800" y2="1000"/>
      </g>
      {/* corner registration marks */}
      <g stroke="#1d1a14" strokeWidth="1.2" fill="none">
        <g transform="translate(20,20)">
          <line x1="0" y1="6" x2="12" y2="6"/><line x1="6" y1="0" x2="6" y2="12"/>
          <circle cx="6" cy="6" r="6" fill="none"/>
        </g>
        <g transform="translate(1568,20)">
          <line x1="0" y1="6" x2="12" y2="6"/><line x1="6" y1="0" x2="6" y2="12"/>
          <circle cx="6" cy="6" r="6" fill="none"/>
        </g>
        <g transform="translate(20,968)">
          <line x1="0" y1="6" x2="12" y2="6"/><line x1="6" y1="0" x2="6" y2="12"/>
          <circle cx="6" cy="6" r="6" fill="none"/>
        </g>
        <g transform="translate(1568,968)">
          <line x1="0" y1="6" x2="12" y2="6"/><line x1="6" y1="0" x2="6" y2="12"/>
          <circle cx="6" cy="6" r="6" fill="none"/>
        </g>
      </g>
      {/* coffee stain */}
      <ellipse cx="1480" cy="900" rx="55" ry="48" fill="rgba(120,60,20,.16)" />
      <ellipse cx="1480" cy="900" rx="48" ry="42" fill="none" stroke="rgba(120,60,20,.34)" strokeWidth="2"/>
      {/* scribbled arrow above pipe */}
      <g style={{ font: "700 17px Caveat, cursive", fill: "#1d1a14" }}>
        <text x="540" y="120" transform="rotate(-3 540 120)">// the magic happens here</text>
      </g>
      {/* label near terminal */}
      <g style={{ font: "700 15px Caveat, cursive", fill: "#2b6cb0" }}>
        <text x="60" y="980">scale 1:1 · paper grid Ø 22px · ink: black 0.5</text>
      </g>
    </svg>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App/>);
