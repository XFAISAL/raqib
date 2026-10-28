"""Python API.

    import raqib
    result = raqib.analyze("call.mp4")                       # local models, defaults
    result = raqib.analyze({"alice": "a.webm", "bob": "b.webm"}, out="reports/call-42")
    for f in result["findings"]:
        print(f["t"], f["kind"], f["confidence"], f.get("frame") or f.get("text"))
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .config import Config, load


def analyze(sources: str | list[str] | dict[str, str], config: Config | str | None = None, out: str | Path | None = None,
            fmt: str | None = None, log: Callable[[str], None] = lambda m: None, html: bool = True) -> dict:
    """Analyse one recording, or one recording per participant ({speaker: path}). Returns the report dict;
    with `out`, also writes report.json / report.html and the flagged frames there."""
    from . import report
    from .pipeline import Analyzer
    from .sources import events, probe

    cfg = config if isinstance(config, Config) else load(config)
    if isinstance(sources, str):
        pairs = [("speaker", sources)]
    elif isinstance(sources, dict):
        pairs = list(sources.items())
    else:
        pairs = [(f"speaker {i + 1}", s) for i, s in enumerate(sources)]
    out_dir = Path(out) if out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    an = Analyzer(cfg, out_dir, log=log)
    meta = []
    for speaker, path in pairs:
        meta.append({"speaker": speaker, "path": str(path), **probe(str(path), fmt)})
        for kind, t, data in events(str(path), fmt):
            if kind == "video":
                an.feed_video(speaker, t, data)
            else:
                an.feed_audio(speaker, data)
        an.finish_source(speaker)
    an.assess_all()
    res = an.result(meta, time.monotonic() - t0)
    if out_dir:
        if html:
            report.write(res, out_dir, cfg.output.blur_frames)
        else:
            (out_dir / "report.json").write_text(__import__("json").dumps(res, ensure_ascii=False, indent=2))
    return res
