"""
Run a set of baselines on a list of prompts (Chinese or English).

Usage:
    python scripts/02_run_baselines.py \
        --baselines hy_motion mdm momask \
        --prompts prompts/chinese_demo.txt \
        --translate \
        --out runs/2026-05-03/

Each baseline gets its own subfolder under <out>/<baseline>/
following the contract documented in src/pipeline.py.
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

# project import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline   import PRESETS, run_baseline   # noqa: E402
from translator import PromptTranslator, TranslatorConfig  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    choices=list(PRESETS.keys()),
                    default=["hy_motion", "mdm"])
    ap.add_argument("--prompts",   type=Path, required=True)
    ap.add_argument("--out",       type=Path,
                    default=Path("runs") / datetime.now().strftime("%Y%m%d_%H%M"))
    ap.add_argument("--translate", action="store_true",
                    help="If prompts are Chinese, route through DeepSeek/Qwen "
                         "and rewrite to HumanML3D-style English first.")
    ap.add_argument("--translator_backend", default="deepseek",
                    choices=("deepseek", "openai", "anthropic", "qwen-local", "offline"))
    return ap.parse_args()


def maybe_translate(prompts: list[str], cfg: TranslatorConfig) -> list[str]:
    t = PromptTranslator(cfg)
    print(f"[translator] backend={cfg.backend}, rewriting {len(prompts)} prompts ...")
    out = t.translate_batch(prompts)
    for zh, en in zip(prompts, out):
        print(f"  {zh}\n   -> {en}")
    return out


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw_prompts = args.prompts.read_text(encoding="utf-8").strip().splitlines()
    if args.translate:
        cfg = TranslatorConfig(backend=args.translator_backend)
        eng_prompts = maybe_translate(raw_prompts, cfg)
    else:
        eng_prompts = raw_prompts

    # save the english prompts side-by-side for reproducibility
    (args.out / "prompts_zh.txt").write_text("\n".join(raw_prompts), encoding="utf-8")
    (args.out / "prompts_en.txt").write_text("\n".join(eng_prompts), encoding="utf-8")

    summary = {}
    for b in args.baselines:
        cfg = PRESETS[b]
        sub = args.out / b
        print(f"\n=== {cfg.name} ===")
        try:
            res = run_baseline(cfg, eng_prompts, sub)
            summary[b] = {"ok": True, "duration_sec": res.duration_sec, "out": str(res.out_dir)}
        except Exception as e:
            print(f"!!! {cfg.name} failed: {e}")
            summary[b] = {"ok": False, "error": str(e)}

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:18s}  {v}")


if __name__ == "__main__":
    main()
