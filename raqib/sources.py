"""Decode a file, URL (rtsp / rtmp / http / hls) or capture device into time-ordered events:

    ("video", t_seconds, av.VideoFrame)   every decoded frame (the sampler decides which ones to analyse)
    ("audio", t_seconds, float32 16 kHz mono chunk)
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .audio import SR


def open_source(src: str, fmt: str | None = None, options: dict | None = None):
    import av

    return av.open(src, format=fmt, options=options or {})


def events(src: str, fmt: str | None = None, options: dict | None = None,
           video: bool = True, audio: bool = True) -> Iterator[tuple[str, float, object]]:
    import av

    c = open_source(src, fmt, options)
    try:
        vs = c.streams.video[0] if (video and c.streams.video) else None
        aus = c.streams.audio[0] if (audio and c.streams.audio) else None
        streams = [s for s in (vs, aus) if s is not None]
        if not streams:
            return
        if vs is not None:
            vs.thread_type = "AUTO"
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR) if aus is not None else None
        t0: float | None = None
        for frame in c.decode(*streams):
            if frame.time is None:
                continue
            if t0 is None:
                t0 = frame.time
            t = max(0.0, frame.time - t0)
            if isinstance(frame, av.VideoFrame):
                yield "video", t, frame
            else:
                for rf in resampler.resample(frame):
                    yield "audio", t, rf.to_ndarray().reshape(-1).astype(np.float32)
        if resampler is not None:
            for rf in resampler.resample(None):
                yield "audio", 0.0, rf.to_ndarray().reshape(-1).astype(np.float32)
    finally:
        c.close()


def probe(src: str, fmt: str | None = None) -> dict:
    c = open_source(src, fmt)
    try:
        dur = float(c.duration / 1_000_000) if c.duration else None
        v = c.streams.video[0] if c.streams.video else None
        return {"duration": dur, "has_video": v is not None, "has_audio": bool(c.streams.audio),
                "width": v.codec_context.width if v else None, "height": v.codec_context.height if v else None}
    finally:
        c.close()


def to_rgb(frame, max_width: int) -> np.ndarray:
    """av.VideoFrame → RGB uint8, downscaled to max_width (keeps aspect)."""
    w, h = frame.width, frame.height
    if w > max_width:
        h = int(h * max_width / w) // 2 * 2
        w = max_width
    return frame.to_ndarray(format="rgb24", width=w, height=h)
