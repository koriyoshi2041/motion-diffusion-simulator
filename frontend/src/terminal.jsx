// Terminal: streams synthetic compile/render logs whenever a prompt is sent.
// Hand-drawn frame around a CRT-styled console.

function Terminal({ logLines }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logLines]);

  return (
    <div className="term-wrap">
      <div className="term-tag annot orange">↓ runtime · stdout</div>

      <div className="term scribble-border">
        <div className="term-bar">
          <span className="dots">
            <i style={{ background: "#ff6058" }}/>
            <i style={{ background: "#ffbf2e" }}/>
            <i style={{ background: "#27c93f" }}/>
          </span>
          <span className="term-title mono">~/diffusion-puppet · ./run --watch</span>
          <span className="term-meta mono">PID 4271</span>
        </div>
        <div className="term-body mono" ref={ref}>
          {logLines.map((l, i) => (
            <div key={i} className={`term-line term-line-${l.kind || "info"}`}>
              <span className="term-ts">{l.ts}</span>
              <span className="term-tag-text">{l.tag}</span>
              <span className="term-msg">{l.msg}</span>
            </div>
          ))}
          <div className="term-line term-prompt">
            <span className="term-cursor">▍</span>
          </div>
        </div>

        {/* CRT scanlines */}
        <div className="term-scan"/>
      </div>

      <style>{`
        .term-wrap { position: relative; }
        .term-tag {
          position: absolute;
          top: -28px; left: 30px;
          font-size: 20px;
          transform: rotate(-2deg);
        }
        .term {
          background: var(--term-bg);
          padding: 0;
          height: 240px;
          overflow: hidden;
          position: relative;
          color: var(--green-term);
        }
        .term-bar {
          display: flex; align-items: center; justify-content: space-between;
          padding: 4px 10px;
          background: #16261d;
          border-bottom: 1.5px solid #1d1a14;
        }
        .dots { display: flex; gap: 5px; }
        .dots i {
          display: block; width: 9px; height: 9px;
          border-radius: 50%;
          border: 1px solid rgba(0,0,0,.3);
        }
        .term-title { font-size: 11px; color: #9bd9b8; }
        .term-meta { font-size: 10px; color: #688876; }

        .term-body {
          padding: 10px 14px;
          height: calc(100% - 28px);
          overflow-y: auto;
          font-size: 12px;
          line-height: 1.55;
          text-shadow: 0 0 6px rgba(110,231,167,.45);
        }
        .term-line { display: flex; gap: 8px; align-items: baseline; white-space: nowrap; }
        .term-ts { color: #5d8772; flex: 0 0 90px; font-size: 10px; }
        .term-tag-text { color: #6da3d6; flex: 0 0 78px; font-size: 11px; }
        .term-msg { color: #6ee7a7; word-break: break-word; flex: 1 1 auto; min-width: 0; white-space: normal; }
        .term-line-warn .term-msg { color: #ffd66b; }
        .term-line-err .term-msg { color: #ff8a7a; }
        .term-line-ok .term-tag-text { color: #ffd66b; }
        .term-line-ok .term-msg { color: #fff8d8; }

        .term-prompt { margin-top: 4px; }
        .term-cursor { animation: blink 1s steps(1) infinite; color: #6ee7a7; }

        .term-scan {
          position: absolute; inset: 28px 0 0 0;
          pointer-events: none;
          background-image: repeating-linear-gradient(
            0deg, rgba(0,0,0,0) 0 2px, rgba(0,0,0,.25) 2px 3px
          );
          mix-blend-mode: multiply;
        }
      `}</style>
    </div>
  );
}

window.Terminal = Terminal;
