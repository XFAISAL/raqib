"""Context-aware risk classification of transcripts with an LLM (local server or provider).

The model sees a short rolling window of the conversation (A = the speaker being assessed, B = anyone
else) and scores only A's newest lines. It tells direct speech from quoted / reported speech, jokes and
media, and non-direct speech is discounted. No keyword lists: the model judges meaning, in any dialect.
"""
from __future__ import annotations

from dataclasses import dataclass

TEXT_KINDS = ["harassment", "abuse", "threat", "hate", "sexual_solicitation", "extortion", "grooming", "scam", "self_harm"]
NON_DIRECT = {"quoted", "reported", "joking", "media", "hypothetical"}
NON_DIRECT_FACTOR = 0.4

SYSTEM = """You are a trust-and-safety classifier for live voice and video calls and streams.
You receive a short speech-to-text transcript (any language, Arabic dialects included; it may contain recognition errors).
Lines are labelled A (the person being assessed) or B (other participants). The input has a CONTEXT section
(earlier lines) and a NEW section (A's new lines).
Assess ONLY A's new lines, using the earlier lines as context. Risk that appears only in earlier lines or in B's lines
must NOT be scored again. If A's new lines are harmless on their own (greetings, small talk, jokes, quoting a movie),
return an empty risks list even when the earlier context was risky. Speech-to-text errors are common: read garbled
words as the phonetically closest plausible words in context, but do not invent risk that is not there.

Score each category 0-100 = how likely A's new lines ARE that behaviour, directed at a real person in this call:
- harassment: targeted insults, cursing at someone, intimidation, humiliation, repeated unwanted pressure
- abuse: severe verbal abuse, degrading slurs, wishing death on someone
- threat: intent to physically harm, kill, or damage someone or their property
- hate: attacks on a group (religion, ethnicity, nationality, gender, tribe)
- sexual_solicitation: asking for nudity, sexual acts, sexual images or paid sexual services
- extortion: demanding money or acts under threat (e.g. leaking photos), sextortion included
- grooming: building secret, private contact with someone who may be a minor (age or school questions combined with
  "don't tell your parents", "our secret", asking to be alone on camera, gifts for secrecy, sexualising)
- scam: impersonation (bank, government, support), asking for OTP codes, passwords, transfers, fake prizes or investments
- self_harm: A expresses intent or a plan to hurt or kill themself

speech_act, how A's new lines are spoken: direct (A says it to B and means it), quoted (repeating someone else's words),
reported (A describes what a third person said or did), joking (obvious banter between friends, gaming trash-talk),
media (lyrics, movie, game), hypothetical, none (nothing risky).
Judge meaning and intent, not keywords: if A insults, demeans, curses or humiliates B (or B's family), that is harassment
(severe or repeated degradation is also abuse), scored by how hostile it is. Do not assume friendly banter unless the
transcript itself clearly shows people joking together.
Reply with JSON only: {"intent": <what A's NEW lines say or do, max 8 words, English>, "act": <speech_act>,
"risks": [{"k": <category>, "s": <score>}]}, at most 3 categories with score >= 20 (empty list if harmless)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "act": {"type": "string", "enum": ["direct", "quoted", "reported", "joking", "media", "hypothetical", "none"]},
        "risks": {"type": "array", "items": {
            "type": "object",
            "properties": {"k": {"type": "string", "enum": TEXT_KINDS}, "s": {"type": "integer"}},
            "required": ["k", "s"]}},
    },
    "required": ["intent", "act", "risks"],
}


@dataclass
class TextResult:
    scores: dict[str, float]      # kind → confidence after the speech-act discount
    raw: dict[str, float]         # kind → model score 0..1
    act: str
    intent: str


def build_window(lines: list[tuple[str, str, bool]], max_chars: int = 1400) -> str:
    """lines: (speaker 'A'|'B', text, is_new), oldest → newest. Context and A's new lines go in separate
    sections so earlier risk is never re-attributed to harmless new lines."""
    new_txt = "\n".join(f"A: {t}" for spk, t, n in lines if n)[: max_chars // 2]
    budget = max_chars - len(new_txt)
    out, total = [], 0
    for spk, t in reversed([(s, t) for s, t, n in lines if not n]):
        line = f"{spk}: {t}"
        if total + len(line) > budget:
            break
        out.append(line)
        total += len(line) + 1
    ctx = "\n".join(reversed(out)) or "(none)"
    return f"CONTEXT (earlier lines, do NOT score):\n{ctx}\n\nA's NEW lines (score ONLY these):\n{new_txt}"


def parse(data: dict) -> TextResult:
    act = str(data.get("act", "none"))
    raw = {k: 0.0 for k in TEXT_KINDS}
    for it in data.get("risks") or []:
        k = it.get("k") if isinstance(it, dict) else None
        if k in raw:
            raw[k] = max(raw[k], max(0.0, min(1.0, float(it.get("s", 0)) / 100.0)))
    factor = NON_DIRECT_FACTOR if act in NON_DIRECT else 1.0
    scores = {k: v * factor for k, v in raw.items()}
    if act in NON_DIRECT:  # a joking or reported self-harm statement still deserves a look
        scores["self_harm"] = raw["self_harm"] * (0.7 if act == "joking" else 0.5)
    return TextResult(scores, raw, act, str(data.get("intent", ""))[:80])


class TextRisk:
    def __init__(self, provider):
        from .providers import client_for

        self.client = client_for(provider)
        self.name = provider.model

    def classify(self, window: str) -> TextResult:
        data = self.client.chat_json(SYSTEM, f"{window}\n\nReturn the JSON for A's NEW lines.", SCHEMA)
        return parse(data)
