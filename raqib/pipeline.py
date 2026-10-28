"""Analyzer: frames + speech in, findings out.

    video ─▶ sampler (schedule + near-duplicate skip + boost after a hit) ─▶ vision (batched)
                                                                  └─▶ sustained-signal rule ─▶ findings
    audio ─▶ VAD segmenter ─▶ speech-to-text ─▶ transcript lines ─▶ text-risk (rolling context) ─▶ findings

Works on whole files (`analyze`) and on live sources (`watch`): feed events, call `finish()`.
Flagged frames are kept (and blurred in the HTML report); everything else is dropped right after use.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from . import textrisk
from .audio import SR, LocalSTT, ProviderSTT, Segment, Segmenter, to_wav
from .config import Config
from .sources import to_rgb
from .vision import VIDEO_KINDS, LocalVision, ProviderVision

# Sustained CLIP signal: explicit footage the NSFW classifier misses (e.g. a screen filmed with a phone) keeps
# CLIP's nudity / sexual-act prompts high on almost every frame, while a single normal frame can spike.
SUSTAIN_THR = 0.6
SUSTAIN_WINDOW = 30.0
SUSTAIN_MIN_HITS = 3
SUSTAIN_CAP = 0.9
COOLDOWN = 10.0          # same kind + speaker: report again only after this many seconds (or if stronger)
MAX_SAVED_FRAMES = 300


@dataclass
class Finding:
    id: str
    kind: str
    modality: str                 # video | audio
    speaker: str
    t: float                      # seconds from the start of the source
    confidence: float
    margin: float = 0.0
    frame: str | None = None      # relative path of the flagged frame
    frame_blurred: str | None = None
    audio: str | None = None
    text: str | None = None       # the transcript line(s) assessed
    act: str | None = None
    intent: str | None = None
    rule: str = "single"          # single | sustained
    detail: dict = field(default_factory=dict)


@dataclass
class Line:
    speaker: str
    start: float
    duration: float
    text: str
    lang: str = ""
    flagged: list[str] = field(default_factory=list)


class _Sampler:
    def __init__(self, s):
        self.s = s
        self.next_t = 0.0
        self.boost_until = -1.0
        self.last_thumb: np.ndarray | None = None
        self.last_analysed = -1e9

    def due(self, t: float) -> bool:
        return t >= self.next_t

    def interval(self, t: float) -> float:
        if t < self.boost_until:
            return self.s.boost_interval
        return self.s.initial_interval if t < self.s.initial_window else self.s.interval

    def accept(self, t: float, rgb: np.ndarray) -> bool:
        """Due frame → analyse unless it is a near duplicate of the last analysed one."""
        self.next_t = t + self.interval(t)
        thumb = np.asarray(Image.fromarray(rgb).convert("L").resize((32, 32)), np.float32) / 255.0
        boosted = t < self.boost_until
        if (not boosted and self.s.dedup_threshold > 0 and self.last_thumb is not None
                and t - self.last_analysed < self.s.max_skip
                and float(np.abs(thumb - self.last_thumb).mean()) < self.s.dedup_threshold):
            return False
        self.last_thumb, self.last_analysed = thumb, t
        return True


class Analyzer:
    def __init__(self, cfg: Config, out_dir: Path | None = None, on_finding: Callable[[Finding], None] | None = None,
                 log: Callable[[str], None] = lambda m: None):
        self.cfg = cfg
        self.out = out_dir
        self.on_finding = on_finding
        self.log = log
        self.kinds = set(cfg.kinds or [])
        self.findings: list[Finding] = []
        self.lines: list[Line] = []
        self.models: list[str] = []
        self._last: dict[tuple[str, str], tuple[float, float]] = {}   # (speaker, kind) → (t, conf)
        self._samplers: dict[str, _Sampler] = {}
        self._segmenters: dict[str, Segmenter] = {}
        self._pending: list[tuple[str, float, np.ndarray]] = []
        self._clip: dict[str, deque] = {}
        self._saved = 0
        self.stats = {"frames_analysed": 0, "frames_skipped": 0, "segments": 0, "lines": 0, "assessments": 0}
        self._load()

    # ------------------------------------------------------------------ models
    def _load(self) -> None:
        c = self.cfg
        self.vision = None
        if c.vision.backend == "local":
            t0 = time.monotonic()
            self.vision = LocalVision(Path(c.models_dir).expanduser(), c.vision.device, c.vision.nsfw,
                                      c.vision.nudenet, c.vision.clip)
            used = [n for n, on in (("nsfw-vit-b16", c.vision.nsfw), ("nudenet-320n", c.vision.nudenet),
                                    ("clip-vit-b16", c.vision.clip)) if on]
            self.models += used
            self.log(f"vision: {', '.join(used)} on {self.vision.device} ({time.monotonic() - t0:.1f}s)")
        elif c.vision.backend == "provider":
            self.vision = ProviderVision(c.vision.provider)
            self.models.append(f"vision:{c.vision.provider.model}")
        self.stt = None
        if c.stt.backend == "local":
            t0 = time.monotonic()
            dev = "cpu" if c.stt.device == "auto" and not _cuda() else ("cuda" if c.stt.device == "auto" else c.stt.device)
            ct = c.stt.compute_type if c.stt.compute_type != "auto" else ("int8_float16" if dev == "cuda" else "int8")
            self.stt = LocalSTT(c.stt.model, dev, ct, tuple(c.stt.languages))
            self.models.append(self.stt.name)
            self.log(f"stt: {self.stt.name} on {dev} ({time.monotonic() - t0:.1f}s)")
        elif c.stt.backend == "provider":
            self.stt = ProviderSTT(c.stt.provider, tuple(c.stt.languages))
            self.models.append(self.stt.name)
        self.text = None
        if c.text.backend in ("local", "provider") and self.stt is not None:
            p = c.text.local if c.text.backend == "local" else c.text.provider
            self.text = textrisk.TextRisk(p)
            self.models.append(f"text:{p.model}")
            self.log(f"text-risk: {p.model} @ {p.base_url}")

    # ------------------------------------------------------------------ feeding
    def feed_video(self, speaker: str, t: float, frame) -> None:
        if self.vision is None:
            return
        sm = self._samplers.setdefault(speaker, _Sampler(self.cfg.sampling))
        if not sm.due(t):
            return
        rgb = to_rgb(frame, self.cfg.vision.max_width)
        if not sm.accept(t, rgb):
            self.stats["frames_skipped"] += 1
            return
        self._pending.append((speaker, t, rgb))
        if len(self._pending) >= self.cfg.vision.batch:
            self._run_vision()

    def feed_audio(self, speaker: str, chunk: np.ndarray) -> list[Line]:
        if self.stt is None:
            return []
        seg = self._segmenters.get(speaker)
        if seg is None:
            s = self.cfg.segmenter
            seg = self._segmenters[speaker] = Segmenter(s.on_threshold, s.off_threshold, s.hang, s.pad,
                                                         s.min_len, s.max_len, s.overlap)
        return [ln for sg in seg.push(chunk) if (ln := self._transcribe(speaker, sg))]

    def finish_source(self, speaker: str) -> list[Line]:
        self._run_vision()
        seg = self._segmenters.get(speaker)
        return [ln for sg in (seg.flush() if seg else []) if (ln := self._transcribe(speaker, sg))]

    # ------------------------------------------------------------------ vision
    def _run_vision(self) -> None:
        if not self._pending or self.vision is None:
            return
        batch, self._pending = self._pending, []
        results = self.vision.infer([b[2] for b in batch])
        self.stats["frames_analysed"] += len(batch)
        for (speaker, t, rgb), r in zip(batch, results):
            hit = False
            for kind in VIDEO_KINDS:
                conf = r.scores.get(kind, 0.0)
                if conf >= self.cfg.min_confidence and self._wanted(kind):
                    hit |= self._emit(Finding(_id(), kind, "video", speaker, t, round(conf, 4),
                                              round(r.margins.get(kind, 0.0), 4), detail=r.debug), rgb)
            hit |= self._sustained(speaker, t, rgb, r)
            if hit:
                sm = self._samplers[speaker]
                sm.boost_until = t + self.cfg.sampling.boost_window
                sm.next_t = min(sm.next_t, t + self.cfg.sampling.boost_interval)

    def _sustained(self, speaker: str, t: float, rgb: np.ndarray, r) -> bool:
        clip = (r.debug or {}).get("clip") or {}
        if not clip or not self._wanted("sexual_act"):
            return False
        win = self._clip.setdefault(speaker, deque(maxlen=64))
        win.append((t, float(clip.get("nudity", 0.0)) + float(clip.get("sexual_act", 0.0))))
        while win and t - win[0][0] > SUSTAIN_WINDOW:
            win.popleft()
        hits = [v for _, v in win if v >= SUSTAIN_THR]
        if len(hits) < SUSTAIN_MIN_HITS or len(hits) * 2 < len(win):
            return False
        conf = min(SUSTAIN_CAP, sum(hits) / len(hits))
        if conf < self.cfg.min_confidence:
            return False
        return self._emit(Finding(_id(), "sexual_act", "video", speaker, t, round(conf, 4), round(2 * conf - 1, 4),
                                  rule="sustained", detail={"frames": len(win), "hits": len(hits)}), rgb)

    # ------------------------------------------------------------------ audio
    def _transcribe(self, speaker: str, sg: Segment) -> Line | None:
        self.stats["segments"] += 1
        tr = self.stt.transcribe(sg.audio)
        audio = sg.audio
        sg.audio = None
        if tr is None:
            return None
        ln = Line(speaker, round(sg.start, 2), round(sg.duration, 2), tr.text, tr.lang)
        ln._audio = audio if self.cfg.output.save_audio else None  # type: ignore[attr-defined]
        self.lines.append(ln)
        self.stats["lines"] += 1
        return ln

    def assess(self, line: Line, context: list[Line]) -> None:
        """Score one new line of `line.speaker` with the conversation before it as context."""
        if self.text is None:
            return
        n = self.cfg.text.context_lines
        win = [("A" if l.speaker == line.speaker else "B", l.text, False) for l in context[-n:]]
        win.append(("A", line.text, True))
        try:
            res = self.text.classify(textrisk.build_window(win))
        except Exception as e:  # noqa: BLE001
            self.log(f"text-risk error: {type(e).__name__}: {e}")
            return
        self.stats["assessments"] += 1
        for kind, conf in sorted(res.scores.items(), key=lambda kv: kv[1], reverse=True):
            if conf < self.cfg.min_confidence or not self._wanted(kind):
                continue
            f = Finding(_id(), kind, "audio", line.speaker, line.start, round(conf, 4), round(abs(2 * conf - 1), 4),
                        text=line.text, act=res.act, intent=res.intent, detail={"raw": res.raw.get(kind)})
            if self._emit(f, None, getattr(line, "_audio", None)):
                line.flagged.append(kind)

    def assess_all(self) -> None:
        """Offline: every line in time order, each with the lines before it (all speakers) as context."""
        ordered = sorted(self.lines, key=lambda l: l.start)
        for i, ln in enumerate(ordered):
            self.assess(ln, ordered[:i])
        self.lines = ordered

    # ------------------------------------------------------------------ findings
    def _wanted(self, kind: str) -> bool:
        return not self.kinds or kind in self.kinds

    def _emit(self, f: Finding, rgb: np.ndarray | None, audio: np.ndarray | None = None) -> bool:
        key = (f.speaker, f.kind)
        last = self._last.get(key)
        if last and f.t - last[0] < COOLDOWN and f.confidence <= last[1]:
            return False
        self._last[key] = (f.t, f.confidence)
        if self.out is not None:
            if rgb is not None and self.cfg.output.save_frames and self._saved < MAX_SAVED_FRAMES:
                f.frame, f.frame_blurred = self._save_frame(f, rgb)
            if audio is not None and self.cfg.output.save_audio:
                (self.out / "audio").mkdir(parents=True, exist_ok=True)
                rel = f"audio/{_ts(f.t)}_{f.kind}_{f.id[:6]}.wav"
                (self.out / rel).write_bytes(to_wav(audio))
                f.audio = rel
        self.findings.append(f)
        if self.on_finding:
            self.on_finding(f)
        return True

    def _save_frame(self, f: Finding, rgb: np.ndarray) -> tuple[str, str]:
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        name = f"{_ts(f.t)}_{f.kind}_{f.id[:6]}"
        img = Image.fromarray(rgb)
        img.save(self.out / "frames" / f"{name}.jpg", quality=88)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=max(12, img.width // 28)))
        blurred.save(self.out / "frames" / f"{name}.blur.jpg", quality=80)
        self._saved += 1
        return f"frames/{name}.jpg", f"frames/{name}.blur.jpg"

    def result(self, sources: list[dict], seconds: float) -> dict:
        lines = []
        for l in sorted(self.lines, key=lambda x: x.start):
            d = asdict(l)
            lines.append(d)
        return {
            "tool": "raqib", "version": __import__("raqib").__version__, "id": uuid.uuid4().hex[:12],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": sources, "processing_sec": round(seconds, 1),
            "models": self.models, "min_confidence": self.cfg.min_confidence, "stats": self.stats,
            "findings": [asdict(f) for f in sorted(self.findings, key=lambda f: f.t)], "transcript": lines,
        }


def _id() -> str:
    return uuid.uuid4().hex


def _ts(t: float) -> str:
    return f"{int(t // 3600):02d}{int(t % 3600 // 60):02d}{int(t % 60):02d}"


def _cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
