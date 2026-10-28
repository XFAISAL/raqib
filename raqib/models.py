"""Model registry + downloader (`raqib models download`). Everything runs locally once downloaded.

| id       | what                                   | source                                      | license    |
|----------|----------------------------------------|---------------------------------------------|------------|
| nsfw     | ViT-B/16 normal/nsfw classifier (ONNX) | onnx-community/nsfw_image_detection-ONNX    | Apache-2.0 |
| nudenet  | YOLOv8n exposed/covered body parts     | notAI-tech/NudeNet v3.4 (320n.onnx)          | AGPL-3.0   |
| clip     | CLIP ViT-B/16 vision + text (ONNX)     | Xenova/clip-vit-base-patch16                 | MIT        |
| whisper  | faster-whisper (CTranslate2)           | Systran/faster-whisper-<size>                | MIT        |
| vad      | Silero VAD v6                          | bundled with faster-whisper                  | MIT        |
| llm      | small instruct model (GGUF, optional)  | unsloth/Qwen3-4B-Instruct-2507-GGUF          | Apache-2.0 |
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HFModel:
    repo: str
    files: tuple[str, ...]
    dest: str


NSFW = HFModel("onnx-community/nsfw_image_detection-ONNX",
               ("config.json", "preprocessor_config.json", "onnx/model_fp16.onnx", "onnx/model_quantized.onnx"),
               "nsfw")
CLIP = HFModel("Xenova/clip-vit-base-patch16",
               ("config.json", "tokenizer.json", "onnx/vision_model_fp16.onnx", "onnx/vision_model_quantized.onnx",
                "onnx/text_model_quantized.onnx"),
               "clip-b16")
# GitHub release asset (v3.4-weights/320n.onnx) through the API: the plain /releases/download/ link
# can answer with a login page for scripted clients.
NUDENET_URL = "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176831997"
NUDENET_SIZE = 12150158

# Suggested local text-risk models (GGUF for llama.cpp `llama-server`). Any OpenAI-compatible server works.
LLM_PRESETS = {
    "qwen3-4b": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),   # recommended
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
}


def _hf(m: HFModel, root: Path) -> None:
    from huggingface_hub import hf_hub_download

    out = root / m.dest
    for f in m.files:
        if (out / f).exists():
            continue
        print(f"  ↓ {m.repo}/{f}", file=sys.stderr)
        p = hf_hub_download(m.repo, f)
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out / f)


def _url(url: str, dest: Path, size: int | None = None) -> None:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "raqib"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed https URL)
        shutil.copyfileobj(r, f)
    if size and tmp.stat().st_size != size:
        tmp.unlink()
        raise SystemExit(f"download of {url} looks wrong (unexpected size); try again later")
    tmp.rename(dest)


def download(root: Path, which: list[str], whisper: str = "small", llm: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if "nsfw" in which:
        _hf(NSFW, root)
    if "clip" in which:
        _hf(CLIP, root)
    if "nudenet" in which:
        _url(NUDENET_URL, root / "nudenet" / "320n.onnx", NUDENET_SIZE)
    if "whisper" in which:
        from faster_whisper.utils import download_model

        print(f"  ↓ faster-whisper {whisper}", file=sys.stderr)
        download_model(whisper)
    if llm:
        from huggingface_hub import hf_hub_download

        repo, fname = LLM_PRESETS.get(llm, (llm, None))
        if fname is None:
            raise SystemExit(f"unknown LLM preset {llm!r}; choose one of {', '.join(LLM_PRESETS)}")
        print(f"  ↓ {repo}/{fname}", file=sys.stderr)
        p = hf_hub_download(repo, fname)
        (root / "llm").mkdir(exist_ok=True)
        dst = root / "llm" / fname
        if not dst.exists():
            shutil.copyfile(p, dst)
        print(f"\nStart it with llama.cpp:\n  llama-server -m {dst} --port 8080 -c 4096\n"
              f"(or point text.local.base_url at Ollama / vLLM / LM Studio)", file=sys.stderr)


def status(root: Path) -> dict[str, bool]:
    return {
        "nsfw": (root / NSFW.dest / "onnx" / "model_fp16.onnx").exists(),
        "nudenet": (root / "nudenet" / "320n.onnx").exists() and (root / "nudenet" / "320n.onnx").stat().st_size == NUDENET_SIZE,
        "clip": (root / CLIP.dest / "onnx" / "vision_model_fp16.onnx").exists(),
    }
