"""raqib command line.

  raqib models download [--whisper small] [--llm qwen2.5-3b] [--only nsfw,clip,nudenet,whisper]
  raqib models status
  raqib analyze call.mp4                       # local models, report in reports/<name>/
  raqib analyze alice=a.webm bob=b.webm        # one recording per participant (labelled speakers)
  raqib analyze talk.mp4 --text provider --set text.provider.model=gpt-4o-mini
  raqib watch rtsp://cam/stream                # live: prints findings as JSON lines
  raqib watch 0 --format avfoundation          # macOS webcam + mic ("0:0"), Linux: /dev/video0 --format v4l2
  raqib image photo.jpg                        # score images
  raqib text "line to check"                   # score one line of speech
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import __version__
from .config import load


def _log(msg: str) -> None:
    print(f"\033[2m· {msg}\033[0m", file=sys.stderr)


def _cfg(a):
    ov = list(a.set or [])
    for stage in ("vision", "stt", "text"):
        v = getattr(a, stage, None)
        if v:
            ov.append(f"{stage}.backend={v}")
    if getattr(a, "min_confidence", None) is not None:
        ov.append(f"min_confidence={a.min_confidence}")
    if getattr(a, "whisper", None):
        ov.append(f"stt.model={a.whisper}")
    if getattr(a, "lang", None):
        ov.append(f"stt.languages={a.lang}")
    if getattr(a, "no_blur", False):
        ov.append("output.blur_frames=false")
    return load(a.config, ov)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML config (default: ./raqib.yaml or ~/.raqib/config.yaml)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting, e.g. sampling.interval=3")
    p.add_argument("--vision", choices=["local", "provider", "off"])
    p.add_argument("--stt", choices=["local", "provider", "off"])
    p.add_argument("--text", choices=["local", "provider", "off"])
    p.add_argument("--min-confidence", type=float)
    p.add_argument("--whisper", help="faster-whisper model (tiny, base, small, medium, large-v3, turbo)")
    p.add_argument("--lang", help="allowed speech languages, comma separated (default ar,en)")


def _sources(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for i, it in enumerate(items):
        m = re.match(r"^([\w\-. ]{1,32})=(.+)$", it)
        if m and not Path(it).exists():
            out.append((m.group(1), m.group(2)))
        else:
            out.append((f"speaker {i + 1}" if len(items) > 1 else "speaker", it))
    return out


def cmd_analyze(a) -> int:
    from .api import analyze

    cfg = _cfg(a)
    srcs = _sources(a.inputs)
    name = a.name or re.sub(r"[^\w.-]+", "_", Path(srcs[0][1]).stem)[:60] or "run"
    out = Path(a.output or cfg.output.dir) / name
    res = analyze(dict(srcs), cfg, out, a.format, log=_log)
    print(json.dumps({"findings": len(res["findings"]), "report": str(out / "report.html"),
                      "json": str(out / "report.json")}, ensure_ascii=False))
    for f in res["findings"]:
        what = f.get("text") or f.get("frame") or ""
        print(f"  {f['t']:7.1f}s  {f['modality']:5}  {f['kind']:<20} {f['confidence']:.0%}  {f['speaker']}  {what}",
              file=sys.stderr)
    return 0


def cmd_watch(a) -> int:
    from dataclasses import asdict

    from . import report
    from .pipeline import Analyzer
    from .sources import events

    cfg = _cfg(a)
    out = Path(a.output or cfg.output.dir) / (a.name or time.strftime("live-%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    def emit(f):
        print(json.dumps(asdict(f), ensure_ascii=False), flush=True)

    an = Analyzer(cfg, out, on_finding=emit, log=_log)
    speaker = a.speaker
    t0 = time.monotonic()
    try:
        for kind, t, data in events(a.source, a.format, dict(o.split("=", 1) for o in (a.option or []))):
            if kind == "video":
                an.feed_video(speaker, t, data)
                if len(an._pending):  # live: no batching delay
                    an._run_vision()
            else:
                for ln in an.feed_audio(speaker, data):
                    print(json.dumps({"line": ln.text, "t": ln.start}, ensure_ascii=False), file=sys.stderr)
                    an.assess(ln, an.lines[:-1])
    except KeyboardInterrupt:
        pass
    an.finish_source(speaker)
    res = an.result([{"speaker": speaker, "path": a.source, "duration": None}], time.monotonic() - t0)
    _, hp = report.write(res, out, cfg.output.blur_frames)
    _log(f"report: {hp}")
    return 0


def cmd_image(a) -> int:
    from PIL import Image
    import numpy as np

    from .pipeline import Analyzer

    cfg = _cfg(a)
    cfg.stt.backend = "off"
    an = Analyzer(cfg, None, log=_log)
    for p in a.images:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        r = an.vision.infer([rgb])[0]
        flagged = {k: round(v, 3) for k, v in r.scores.items() if v >= cfg.min_confidence}
        print(json.dumps({"image": p, "flagged": flagged, "scores": {k: round(v, 3) for k, v in r.scores.items()}},
                         ensure_ascii=False))
    return 0


def cmd_text(a) -> int:
    from . import textrisk

    cfg = _cfg(a)
    p = cfg.text.local if cfg.text.backend == "local" else cfg.text.provider
    tr = textrisk.TextRisk(p)
    ctx = [("B", t, False) for t in (a.context or [])]
    res = tr.classify(textrisk.build_window(ctx + [("A", a.line, True)]))
    print(json.dumps({"act": res.act, "intent": res.intent,
                      "flagged": {k: round(v, 3) for k, v in res.scores.items() if v >= cfg.min_confidence},
                      "raw": {k: v for k, v in res.raw.items() if v}}, ensure_ascii=False))
    return 0


def cmd_models(a) -> int:
    from . import models

    cfg = load(a.config)
    root = Path(cfg.models_dir).expanduser()
    if a.action == "status":
        st = models.status(root)
        for k, v in st.items():
            print(f"{'✓' if v else '✗'} {k}")
        print(f"models dir: {root}")
        return 0 if all(st.values()) else 1
    which = [w.strip() for w in (a.only or "nsfw,nudenet,clip,whisper").split(",")]
    models.download(root, which, a.whisper or cfg.stt.model, a.llm)
    print(f"done → {root}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("raqib", description="Audio & video moderation: flags harmful frames and speech.")
    ap.add_argument("--version", action="version", version=f"raqib {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="analyse files (one per participant: name=path)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", help="output root (default: reports/)")
    p.add_argument("--name", help="report folder name")
    p.add_argument("--format", help="force container / device format (ffmpeg name)")
    p.add_argument("--no-blur", action="store_true", help="show flagged frames unblurred in the HTML report")
    _common(p)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("watch", help="analyse a live stream / device; prints findings as JSON lines")
    p.add_argument("source")
    p.add_argument("--format", help="e.g. avfoundation (macOS), v4l2 (Linux), dshow (Windows)")
    p.add_argument("--option", action="append", metavar="K=V", help="ffmpeg input option, e.g. rtsp_transport=tcp")
    p.add_argument("--speaker", default="live")
    p.add_argument("-o", "--output")
    p.add_argument("--name")
    p.add_argument("--no-blur", action="store_true")
    _common(p)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("image", help="score image files")
    p.add_argument("images", nargs="+")
    _common(p)
    p.set_defaults(fn=cmd_image)

    p = sub.add_parser("text", help="score one line of speech / chat")
    p.add_argument("line")
    p.add_argument("--context", action="append", help="earlier line from the other side (repeatable)")
    _common(p)
    p.set_defaults(fn=cmd_text)

    p = sub.add_parser("models", help="download / check local models")
    p.add_argument("action", choices=["download", "status"])
    p.add_argument("--only", help="comma list: nsfw,nudenet,clip,whisper")
    p.add_argument("--whisper", help="faster-whisper size to fetch (default: stt.model)")
    p.add_argument("--llm", help="also fetch a GGUF text-risk model: qwen3-4b | qwen2.5-3b")
    p.add_argument("-c", "--config")
    p.set_defaults(fn=cmd_models)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
