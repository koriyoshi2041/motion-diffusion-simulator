// Left side: hand-drawn "phone" / pocket-input device.
// Tilted, with a screen showing live prompt & cursor, and a QWERTY keyboard
// where each pressed key flashes in sync with real keystrokes.

const KEY_ROWS = [
  ["q","w","e","r","t","y","u","i","o","p"],
  ["a","s","d","f","g","h","j","k","l"],
  ["⇧","z","x","c","v","b","n","m","⌫"],
  ["?123","␣","↵"],
];

function Phone({ value, onChange, activeKey, onSend, status }) {
  const screenRef = React.useRef(null);
  React.useEffect(() => {
    if (screenRef.current) screenRef.current.scrollTop = screenRef.current.scrollHeight;
  }, [value]);

  return (
    <div className="phone-wrap">
      {/* hand-drawn label above */}
      <div className="phone-label annot">
        ← input device
        <svg width="60" height="22" style={{ verticalAlign: "middle", marginLeft: 4 }}>
          <path d="M2,12 Q20,2 40,14 L36,8 M40,14 L34,18" fill="none" stroke="#c0392b" strokeWidth="1.6" strokeLinecap="round"/>
        </svg>
      </div>

      <div className="phone scribble-border">
        {/* speaker + camera detail */}
        <div className="phone-top">
          <div className="phone-speaker"></div>
          <div className="phone-camera"></div>
        </div>

        {/* screen */}
        <div className="phone-screen">
          <div className="phone-screen-bar">
            <span className="mono" style={{ fontSize: 9 }}>● ● ●</span>
            <span className="mono" style={{ fontSize: 9, opacity:.6 }}>prompt.app</span>
            <span className="mono" style={{ fontSize: 9, opacity:.6 }}>▮▮▮</span>
          </div>
          <div className="phone-screen-body" ref={screenRef}>
            <div className="screen-hint mono">/* type a verb. it animates. */</div>
            <div className="screen-prompt mono">
              <span className="screen-caret-prefix">&gt;</span>
              <span>{value}</span>
              <span className="screen-caret">▍</span>
            </div>
            {status && (
              <div className="screen-status mono">
                <span className="status-dot" /> {status}
              </div>
            )}
          </div>
          {/* output port — pipe attaches here */}
          <div id="pipe-source" className="pipe-source">
            <div className="port-ring"></div>
            <div className="port-ring port-ring-2"></div>
            <div className="port-hole"></div>
          </div>
        </div>

        {/* keyboard */}
        <div className="phone-kbd">
          {KEY_ROWS.map((row, ri) => (
            <div key={ri} className="kbd-row">
              {row.map((k) => {
                const isWide = k === "␣" ? "wide-space" : (k === "⇧" || k === "⌫" || k === "↵" || k === "?123") ? "wide" : "";
                const norm = k === "␣" ? " " : k.toLowerCase();
                const active = activeKey && (activeKey === norm || activeKey === k.toLowerCase());
                return (
                  <button
                    key={k}
                    className={`kbd-key ${isWide} ${active ? "kbd-active" : ""}`}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      if (k === "⌫") onChange(value.slice(0, -1));
                      else if (k === "↵") onSend && onSend();
                      else if (k === "␣") onChange(value + " ");
                      else if (k === "⇧" || k === "?123") {/* no-op */}
                      else onChange(value + k);
                    }}
                  >
                    {k}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* home indicator */}
        <div className="phone-home"></div>
      </div>

      {/* sticky note overlay */}
      <div className="sticky-note">
        <div className="handwriting" style={{ fontSize: 18, lineHeight: 1.1 }}>
          try: <i>"wave"</i>, <i>"jump"</i>,<br/>
          <i>"dance"</i>, <i>"think"</i>,<br/>
          <i>"sleep"</i> ...
        </div>
      </div>

      <style>{`
        .phone-wrap {
          position: absolute;
          left: 2%;
          top: 50%;
          transform: translateY(-50%) rotate(-4deg);
          width: 260px;
          z-index: 4;
        }
        .phone-label {
          position: absolute;
          top: -38px;
          left: 30px;
          font-size: 22px;
          transform: rotate(-6deg);
        }
        .phone {
          background: #faf6e9;
          border-radius: 36px 38px 34px 36px / 38px 34px 36px 36px;
          padding: 14px 14px 18px;
          position: relative;
          box-shadow:
            3px 5px 0 rgba(29,26,20,.18),
            inset 0 0 0 6px rgba(29,26,20,.06),
            inset 0 0 0 7px rgba(29,26,20,.85);
        }
        .phone::before {
          content: "";
          position: absolute; inset: -4px;
          border: 2px dashed rgba(29,26,20,.25);
          border-radius: inherit;
          pointer-events: none;
        }
        .phone-top {
          display: flex; align-items: center; justify-content: center;
          gap: 12px; height: 16px; margin-bottom: 6px;
        }
        .phone-speaker {
          width: 60px; height: 6px; border-radius: 4px;
          background: #1d1a14;
        }
        .phone-camera {
          width: 9px; height: 9px; border-radius: 50%;
          background: #1d1a14;
          box-shadow: inset 0 0 0 2px #6da3d6;
        }
        .phone-screen {
          background: #faf3df;
          border: 2px solid #1d1a14;
          border-radius: 14px 16px 12px 14px / 14px 12px 16px 14px;
          padding: 0;
          height: 180px;
          position: relative;
          overflow: hidden;
          background-image:
            repeating-linear-gradient(0deg, transparent 0 23px, rgba(43,108,176,.18) 23px 24px),
            repeating-linear-gradient(90deg, transparent 0 23px, rgba(43,108,176,.10) 23px 24px);
        }
        .phone-screen-bar {
          display: flex; justify-content: space-between; align-items: center;
          padding: 4px 8px;
          border-bottom: 1.5px solid #1d1a14;
          background: #ede4c5;
          color: #1d1a14;
        }
        .phone-screen-body {
          padding: 8px 10px;
          height: calc(100% - 22px);
          overflow-y: auto;
          font-size: 11px;
          color: #1d1a14;
          line-height: 1.55;
        }
        .screen-hint { color: #6b6450; font-size: 10px; margin-bottom: 8px; }
        .screen-prompt { font-size: 13px; word-break: break-word; }
        .screen-caret-prefix { color: var(--blueprint); margin-right: 6px; font-weight: 700; }
        .screen-caret { animation: blink 1s steps(1) infinite; color: var(--red-pen); }
        @keyframes blink { 50% { opacity: 0; } }
        .screen-status {
          margin-top: 10px;
          font-size: 10px;
          color: #2b6cb0;
          padding: 4px 6px;
          border: 1px dashed rgba(43,108,176,.55);
          border-radius: 6px;
          display: inline-block;
        }
        .status-dot {
          display: inline-block; width: 6px; height: 6px; border-radius: 50%;
          background: #16a34a; margin-right: 6px;
          box-shadow: 0 0 6px #16a34a;
          animation: pulse 1.2s ease-in-out infinite;
        }
        @keyframes pulse { 50% { transform: scale(1.6); opacity: .5; } }

        .pipe-source {
          position: absolute;
          right: -8px;
          top: 38%;
          width: 26px; height: 26px;
          z-index: 5;
        }
        .port-ring {
          position: absolute; inset: 0;
          border-radius: 50%;
          border: 2.5px solid #1d1a14;
          background: #d6cdb1;
        }
        .port-ring-2 {
          inset: 4px;
          background: #b8a980;
          border-color: #1d1a14;
          border-width: 2px;
        }
        .port-hole {
          position: absolute; inset: 9px;
          border-radius: 50%;
          background: radial-gradient(circle at 35% 30%, #6b6450, #1d1a14 70%);
        }

        .phone-kbd {
          margin-top: 12px;
          padding: 8px 4px 4px;
          background: #ede4c5;
          border: 1.5px solid #1d1a14;
          border-radius: 8px 10px 8px 10px;
        }
        .kbd-row {
          display: flex; justify-content: center; gap: 4px;
          margin-bottom: 5px;
        }
        .kbd-row:last-child { margin-bottom: 0; }
        .kbd-key {
          flex: 1;
          min-width: 0;
          height: 26px;
          font-family: "JetBrains Mono", monospace;
          font-size: 11px;
          font-weight: 600;
          color: #1d1a14;
          background: #faf6e9;
          border: 1.5px solid #1d1a14;
          border-radius: 5px 6px 5px 6px;
          cursor: pointer;
          padding: 0;
          transition: transform .08s, background .15s, color .15s;
          box-shadow: 1px 1.5px 0 rgba(29,26,20,.4);
        }
        .kbd-key.wide { flex: 1.5; }
        .kbd-key.wide-space { flex: 5.5; }
        .kbd-key:hover { background: #fff8d8; }
        .kbd-key.kbd-active {
          background: #2b6cb0;
          color: #faf6e9;
          transform: translate(1px, 1.5px);
          box-shadow: 0 0 0 rgba(0,0,0,0), 0 0 14px rgba(43,108,176,.7);
        }

        .phone-home {
          height: 4px; width: 38%; margin: 8px auto 0;
          background: #1d1a14;
          border-radius: 4px;
        }

        .sticky-note {
          position: absolute;
          right: -78px;
          bottom: -30px;
          width: 120px;
          padding: 10px 10px 14px;
          background: #fff7a8;
          color: #1d1a14;
          transform: rotate(7deg);
          box-shadow: 2px 4px 6px rgba(0,0,0,.18);
          z-index: 6;
        }
        .sticky-note::before {
          content: "";
          position: absolute; top: -8px; left: 50%;
          width: 30px; height: 16px;
          background: rgba(192,80,77,.4);
          transform: translateX(-50%) rotate(-6deg);
          border-radius: 2px;
        }
      `}</style>
    </div>
  );
}

window.Phone = Phone;
