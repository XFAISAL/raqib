"""Settings: defaults < YAML file < environment (RAQIB_*) < CLI flags.

Every stage has a backend:

  vision  local    → ONNX models on this machine (NSFW ViT + NudeNet + CLIP zero-shot)
          provider → a multimodal model behind an API (OpenAI-compatible or Anthropic)
          off
  stt     local    → faster-whisper on this machine
          provider → any OpenAI-compatible /v1/audio/transcriptions endpoint
          off
  text    local    → an OpenAI-compatible server on this machine (llama.cpp server, Ollama, vLLM …)
          provider → OpenAI-compatible API or Anthropic
          off

API keys are only read from the environment (never from the YAML file).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HOME = Path(os.environ.get("RAQIB_HOME", Path.home() / ".raqib"))


@dataclass
class ProviderConfig:
    """One remote model endpoint. `kind`: openai (any OpenAI-compatible API) | anthropic."""
    kind: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"   # name of the env var that holds the key
    timeout: float = 30.0

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class VisionConfig:
    backend: str = "local"            # local | provider | off
    device: str = "auto"              # auto | cpu | cuda
    nsfw: bool = True                 # Falconsai NSFW ViT-B/16
    nudenet: bool = True              # NudeNet 320n body-part detector
    clip: bool = True                 # CLIP ViT-B/16 zero-shot (weapon, violence, dangerous, self-harm …)
    batch: int = 8
    max_width: int = 640              # frames are downscaled to this width before inference
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(model="gpt-4o-mini"))


@dataclass
class SamplingConfig:
    initial_interval: float = 2.0     # seconds between frames during the first `initial_window`
    initial_window: float = 60.0
    interval: float = 5.0             # steady-state interval
    boost_interval: float = 2.0       # interval after a finding …
    boost_window: float = 60.0        # … for this long
    dedup_threshold: float = 0.035    # skip near-identical frames (0 = off)
    max_skip: float = 30.0            # but analyse at least one frame per this many seconds


@dataclass
class STTConfig:
    backend: str = "local"            # local | provider | off
    model: str = "small"              # faster-whisper size or path (tiny … large-v3, turbo)
    device: str = "auto"
    compute_type: str = "auto"
    languages: list[str] = field(default_factory=lambda: ["ar", "en"])
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(model="whisper-1"))


@dataclass
class SegmenterConfig:
    """Voice-activity segmentation. Long-ish segments on purpose: speech recognisers (Arabic dialects
    especially) need the sentence around a word to get it right."""
    on_threshold: float = 0.5
    off_threshold: float = 0.35
    hang: float = 2.0
    pad: float = 0.8
    min_len: float = 0.6
    max_len: float = 12.0
    overlap: float = 1.0


@dataclass
class TextConfig:
    backend: str = "local"            # local | provider | off
    # local = an OpenAI-compatible server you run (llama.cpp `llama-server`, Ollama, vLLM, LM Studio …)
    local: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        kind="openai", base_url="http://127.0.0.1:8080/v1", model="local", api_key_env=""))
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(model="gpt-4o-mini"))
    context_lines: int = 12
    min_interval: float = 0.0         # seconds between two assessments of the same speaker (live mode)


@dataclass
class OutputConfig:
    dir: str = "reports"
    save_frames: bool = True          # keep the flagged frames (the report shows them)
    blur_frames: bool = True          # blur them in the HTML report (originals stay in frames/)
    save_audio: bool = False          # keep flagged speech clips as WAV
    html: bool = True


@dataclass
class Config:
    models_dir: str = str(DEFAULT_HOME / "models")
    min_confidence: float = 0.5       # findings below this are not reported
    kinds: list[str] = field(default_factory=list)  # empty = all
    vision: VisionConfig = field(default_factory=VisionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    text: TextConfig = field(default_factory=TextConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(obj: Any, data: dict, path: str = "") -> None:
    names = {f.name: f for f in fields(obj)}
    for k, v in (data or {}).items():
        if k not in names:
            raise ValueError(f"unknown setting: {path}{k}")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise ValueError(f"{path}{k} must be a mapping")
            _merge(cur, v, f"{path}{k}.")
        elif isinstance(cur, str) and isinstance(v, bool):
            setattr(obj, k, "on" if v else "off")  # YAML 1.1 reads a bare `off` as false
        else:
            setattr(obj, k, v)


def _coerce(cur: Any, raw: str) -> Any:
    if isinstance(cur, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(raw)
    if isinstance(cur, float):
        return float(raw)
    if isinstance(cur, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def set_path(cfg: Config, dotted: str, raw: str) -> None:
    """`vision.backend=provider`-style override."""
    parts = dotted.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], _coerce(getattr(obj, parts[-1]), raw))


def load(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else None
    if p is None:
        for cand in (Path("raqib.yaml"), DEFAULT_HOME / "config.yaml"):
            if cand.exists():
                p = cand
                break
    if p is not None:
        _merge(cfg, yaml.safe_load(p.read_text()) or {})
    # RAQIB_VISION__BACKEND=provider → vision.backend
    for k, v in os.environ.items():
        if k.startswith("RAQIB_") and "__" in k:
            try:
                set_path(cfg, k[6:].lower().replace("__", "."), v)
            except AttributeError:
                pass
    for o in overrides or []:
        key, _, val = o.partition("=")
        set_path(cfg, key.strip(), val.strip())
    return cfg
