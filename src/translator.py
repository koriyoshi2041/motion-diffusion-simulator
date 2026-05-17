"""
─────────────────────────────────────────────────────────────────────────
translator.py —— 中文 prompt → 英文 prompt 改写
─────────────────────────────────────────────────────────────────────────

为什么需要这一步？
    HumanML3D / HY-Motion 训练时只见过英文 prompt。中文输入直接进 Qwen3
    虽然能编码，但生成出来的动作 prompt-conformance 会明显下降。
    所以在前端→模型之间加一层「中→英 + HumanML3D 风格化」改写。

输入：自由中文描述，如「做一个左勾拳并迅速侧步躲闪」
输出：HumanML3D 风格英文，如 "a person throws a left hook and quickly sidesteps"

支持的 backend（按优先级）：
    1. 本地 Qwen / DeepSeek （OpenAI 兼容端点）
    2. OpenAI / Anthropic / DeepSeek SaaS API
    3. 写死的 few-shot offline fallback —— 网络断了也能跑
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable


SYSTEM_PROMPT = (
    "You are a motion-prompt rewriter. Convert the user's Chinese motion "
    "description into a single English sentence that follows HumanML3D's "
    "annotation style. Rules:\n"
    "  - Start with 'a person'\n"
    "  - Use simple present tense verbs (walks, jumps, throws, bends)\n"
    "  - Mention body part if it matters (left arm, right leg)\n"
    "  - Keep under 25 words\n"
    "  - Output ONLY the English sentence, nothing else"
)

FEW_SHOT = [
    ("做一个左勾拳", "a person throws a left hook punch"),
    ("练一段太极云手", "a person performs slow tai chi cloud hands movement"),
    ("从蹲姿快速跳起", "a person jumps up quickly from a squatting position"),
    ("侧空翻并稳稳落地", "a person performs a side flip and lands steadily"),
    ("沿直线小跑然后急停", "a person jogs straight forward then stops abruptly"),
]


@dataclass
class TranslatorConfig:
    backend: str = "openai"        # openai | anthropic | deepseek | qwen-local | offline
    model: str = "deepseek-chat"
    base_url: str | None = None     # for local servers
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 64


class PromptTranslator:
    def __init__(self, cfg: TranslatorConfig | None = None):
        self.cfg = cfg or TranslatorConfig()
        self._client = None
        self._init_client()

    def _init_client(self):
        b = self.cfg.backend
        if b == "offline":
            return
        if b in ("openai", "deepseek", "qwen-local"):
            from openai import OpenAI
            api_key = os.getenv(self.cfg.api_key_env, "EMPTY")
            base_url = self.cfg.base_url
            if b == "deepseek" and base_url is None:
                base_url = "https://api.deepseek.com/v1"
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        elif b == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown backend: {b}")

    def translate(self, zh: str) -> str:
        if self.cfg.backend == "offline":
            return self._offline_lookup(zh)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[m for shot in FEW_SHOT for m in (
                {"role": "user", "content": shot[0]},
                {"role": "assistant", "content": shot[1]},
            )],
            {"role": "user", "content": zh},
        ]

        if self.cfg.backend == "anthropic":
            r = self._client.messages.create(
                model=self.cfg.model,
                system=SYSTEM_PROMPT,
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in messages if m["role"] != "system"],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return r.content[0].text.strip()

        r = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return r.choices[0].message.content.strip()

    def translate_batch(self, prompts: Iterable[str]) -> list[str]:
        return [self.translate(p) for p in prompts]

    @staticmethod
    def _offline_lookup(zh: str) -> str:
        for src, tgt in FEW_SHOT:
            if src in zh:
                return tgt
        return f"a person performs an action described as '{zh}'"


if __name__ == "__main__":
    cfg = TranslatorConfig(backend="offline")
    t = PromptTranslator(cfg)
    for zh in ["做一个左勾拳", "练一段太极云手", "跳科目三"]:
        print(f"{zh}\n  -> {t.translate(zh)}\n")
