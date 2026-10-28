"""Audio: streaming Silero VAD → speech segments → speech-to-text.

Segments live in memory only and are dropped after transcription (unless output.save_audio is on).
"""
from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512          # 32 ms @ 16 kHz (Silero v6 window)
CTX = 64


class SileroStream:
    """Streaming VAD state over a shared ONNX session (the Silero v6 model bundled with faster-whisper)."""

    _session = None

    @classmethod
    def session(cls):
        if cls._session is None:
            import faster_whisper
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            p = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
            cls._session = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        return cls._session

    def __init__(self) -> None:
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(CTX, np.float32)

    def probs(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32 mono 16 kHz, length a multiple of CHUNK → one speech probability per 32 ms."""
        n = len(audio) // CHUNK
        if n == 0:
            return np.zeros(0, np.float32)
        x = audio[: n * CHUNK].reshape(n, CHUNK)
        ctx = np.vstack([self.ctx[None, :], x[:-1, -CTX:]])
        inp = np.concatenate([ctx, x], axis=1).astype(np.float32)
        out, self.h, self.c = self.session().run(None, {"input": inp, "h": self.h, "c": self.c})
        self.ctx = x[-1, -CTX:].copy()
        return np.asarray(out, np.float32).reshape(-1)


@dataclass
class Segment:
    audio: np.ndarray | None   # float32 16 kHz
    start: float               # seconds from the start of the source
    duration: float


class Segmenter:
    """VAD hysteresis: speech starts at ≥ on_threshold, ends after `hang` s below off_threshold, is cut at
    max_len (the next segment starts `overlap` s earlier so a word on the cut is not lost).

    Long segments on purpose: with a short hang and little padding, Arabic dialect STT turns insults into
    unrelated words; with ~2 s hang and ~0.8 s padding it recovers them."""

    def __init__(self, on_threshold=0.5, off_threshold=0.35, hang=2.0, pad=0.8, min_len=0.6, max_len=12.0,
                 overlap=1.0, vad: bool = True):
        self.vad = SileroStream() if vad else None
        self.on, self.off, self.hang = on_threshold, off_threshold, hang
        self.pad, self.min_len, self.max_len, self.overlap = pad, min_len, max_len, overlap
        self._pending = np.zeros(0, np.float32)
        self._pre = np.zeros(0, np.float32)
        self._buf: list[np.ndarray] = []
        self._in = False
        self._silence = 0.0
        self._start = 0.0
        self._t = 0.0  # seconds of audio consumed (source clock)

    def _probs(self, a: np.ndarray) -> np.ndarray:
        if self.vad is not None:
            return self.vad.probs(a)
        rms = np.sqrt((a.reshape(-1, CHUNK) ** 2).mean(axis=1))  # energy gate fallback
        return (rms > 0.01).astype(np.float32)

    def push(self, audio: np.ndarray) -> list[Segment]:
        a = np.concatenate([self._pending, audio]) if len(self._pending) else audio
        n = (len(a) // CHUNK) * CHUNK
        self._pending = a[n:].copy()
        a = a[:n]
        if n == 0:
            return []
        out: list[Segment] = []
        for i, p in enumerate(self._probs(a)):
            ch = a[i * CHUNK:(i + 1) * CHUNK]
            t = self._t + i * CHUNK / SR
            if not self._in:
                self._pre = np.concatenate([self._pre, ch])[-int(self.pad * SR):]
                if p >= self.on:
                    self._in, self._silence = True, 0.0
                    self._buf = [self._pre.copy()]
                    self._start = t - (len(self._pre) - CHUNK) / SR
                continue
            self._buf.append(ch)
            self._silence = self._silence + CHUNK / SR if p < self.off else 0.0
            dur = sum(len(b) for b in self._buf) / SR
            if self._silence >= self.hang or dur >= self.max_len:
                seg = np.concatenate(self._buf)
                ended = self._silence >= self.hang
                if ended:
                    seg = seg[: max(0, len(seg) - int((self._silence - self.pad) * SR))]
                if len(seg) / SR >= self.min_len:
                    out.append(Segment(seg, max(0.0, self._start), len(seg) / SR))
                self._buf, self._pre = [], np.zeros(0, np.float32)
                self._in = not ended  # long speech continues in the next segment
                if self._in:
                    tail = seg[-int(self.overlap * SR):] if self.overlap > 0 else np.zeros(0, np.float32)
                    self._buf = [tail.copy()] if len(tail) else []
                    self._start = t + CHUNK / SR - len(tail) / SR
                self._silence = 0.0
        self._t += n / SR
        return out

    def flush(self) -> list[Segment]:
        """Source ended: the speech still open becomes a final segment."""
        out = []
        if self._in and self._buf:
            seg = np.concatenate(self._buf)
            if len(seg) / SR >= self.min_len:
                out.append(Segment(seg, max(0.0, self._start), len(seg) / SR))
        self._buf, self._in = [], False
        return out


# Well-known speech-to-text hallucinations on silence / noise (Arabic + English). Dropped, never classified.
_HALLU = [
    r"اشتركوا? في القناة", r"ترجمة نانسي قنقر", r"شكرا لكم على المشاهدة", r"شكرا على المشاهدة", r"لا تنسوا الاشتراك",
    r"thanks? for watching", r"subscribe to (my|the) channel", r"subtitles? by", r"amara\.org",
    r"^\W*نهاية الدرس\W*$", r"^\W*انتهى الدرس\W*$",
    # non-speech tags some models write for noise / music / laughter, e.g. «@@@ضوضاء», «*موسيقى*», «(ضحك)»
    r"^\W*(ضوضاء|ضجيج|موسيقى|ضحك|تصفيق|صمت|music|applause|noise|laughter|silence)\W*$",
    # subtitle-credit hallucinations seen on silence / far-off noise
    r"موقع الدراسة الجزائري", r"eddirasa", r"التفريغ والتدقيق", r"غير_واضح", r"\[\s*إلى أن ينتهي\s*\]",
]
HALLU_RE = re.compile("|".join(_HALLU), re.IGNORECASE)


def clean_transcript(text: str, avg_logprob: float = 0.0, no_speech: float = 0.0) -> str | None:
    """Collapse decoder loops, drop hallucinations and clearly bad decodes. None = nothing usable."""
    text = re.sub(r"(.)\1{4,}", r"\1\1\1", text or "")
    text = re.sub(r"\b(\w+)(\s+\1\b){3,}", r"\1 \1 \1", text).strip()[:600]
    if not text or HALLU_RE.search(text):
        return None
    # Phone-quality speech (and insults) often decodes with low confidence: only drop the clearly bad
    if (no_speech > 0.75 and avg_logprob < -1.0) or avg_logprob < -1.6:
        return None
    return text


def to_wav(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


@dataclass
class Transcript:
    text: str
    lang: str


class LocalSTT:
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto",
                 languages: tuple[str, ...] = ("ar", "en")):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.languages = tuple(languages)
        self.name = f"faster-whisper-{Path(model).name}"

    def transcribe(self, audio: np.ndarray) -> Transcript | None:
        lang = self.languages[0] if self.languages else None
        if len(self.languages) > 1:  # language ID restricted to the allowed set
            try:
                _, _, allp = self.model.detect_language(audio[: SR * 10])
                cand = [(l, p) for l, p in allp if l in self.languages]
                if cand:
                    lang = max(cand, key=lambda x: x[1])[0]
            except Exception:  # noqa: BLE001
                pass
        segs, _ = self.model.transcribe(
            audio, language=lang, beam_size=1, temperature=0.0, condition_on_previous_text=False,
            without_timestamps=True, vad_filter=False, no_speech_threshold=0.6, log_prob_threshold=-1.0)
        texts, lps, nsp = [], [], []
        for s in segs:
            texts.append(s.text.strip())
            lps.append(s.avg_logprob)
            nsp.append(s.no_speech_prob)
        text = clean_transcript(" ".join(t for t in texts if t),
                                float(np.mean(lps)) if lps else -9.0, float(np.max(nsp)) if nsp else 1.0)
        return Transcript(text, lang or "") if text else None


class ProviderSTT:
    """Any OpenAI-compatible /audio/transcriptions endpoint (OpenAI, Groq, a local whisper server …)."""

    def __init__(self, provider, languages: tuple[str, ...] = ("ar",)):
        from .providers import client_for

        self.client = client_for(provider)
        self.language = languages[0] if len(languages) == 1 else None
        self.name = f"stt-{provider.model}"

    def transcribe(self, audio: np.ndarray) -> Transcript | None:
        text = clean_transcript(self.client.transcribe(to_wav(audio), self.language))
        return Transcript(text, self.language or "") if text else None
