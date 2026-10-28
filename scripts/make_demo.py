"""Build the demo clip used in the README (scripts/demo-assets → demo.mp4).

Four scenes (AI-generated stills) with synthetic speech (macOS `say`):
  0–14 s  video call, small talk                       → nothing flagged
  14–26 s staged threat with a prop pistol + insult     → weapon (video), harassment / threat (speech)
  26–38 s kitchen, a "bank" asks for an SMS code        → scam (speech)
  38–48 s kitchen fire                                  → dangerous (video)

usage: python scripts/make_demo.py [out.mp4]   (macOS: needs the Majed and Samantha voices)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

HERE = Path(__file__).parent / "demo-assets"
SR, FPS, W, H = 16000, 10, 1280, 720
SCENES = [("call.jpg", 0, 14), ("weapon.jpg", 14, 26), ("kitchen.jpg", 26, 38), ("fire.jpg", 38, 48)]
SPEECH = [  # (start s, voice, text)
    (1.0, "Majed", "هلا والله، كيف حالك؟ وش أخبار الشغل؟"),
    (6.5, "Samantha", "Hey, great to see you. How was your weekend?"),
    (15.0, "Majed", "اسمع يا غبي، إذا ما حولت الفلوس الليلة والله لأكسر راسك"),
    (27.0, "Samantha", "Hi, this is your bank's security team. Please read me the verification code we just sent to your phone."),
    (39.0, "Majed", "انتبهوا، المطبخ يحترق! اطلعوا بسرعة"),
]


def tts(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.wav"
        subprocess.run(["say", "-v", voice, "-o", str(p), "--file-format=WAVE", f"--data-format=LEI16@{SR}", text], check=True)
        with wave.open(str(p)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def main(out: str = "demo.mp4") -> None:
    total = SCENES[-1][2]
    audio = np.zeros(total * SR, np.float32)
    for start, voice, text in SPEECH:
        a = tts(voice, text) * 0.9
        i = int(start * SR)
        audio[i:i + len(a)] += a[: len(audio) - i]
    audio += np.random.default_rng(0).normal(0, 0.002, len(audio)).astype(np.float32)  # room noise

    c = av.open(out, "w")
    vs = c.add_stream("h264" if "h264" in av.codecs_available else "mpeg4", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    as_ = c.add_stream("aac", rate=SR)
    as_.layout = "mono"
    imgs = {f: np.asarray(Image.open(HERE / f).convert("RGB").resize((W, H))) for f, _, _ in SCENES}
    for n in range(total * FPS):
        t = n / FPS
        f = next(f for f, a, b in SCENES if a <= t < b)
        # slight camera drift so frames are not pixel-identical
        img = np.roll(imgs[f], int(3 * np.sin(t)), axis=1)
        for p in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(p)
    step = 1024
    for i in range(0, len(audio), step):
        chunk = audio[i:i + step]
        fr = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
        fr.sample_rate, fr.pts, fr.time_base = SR, i, Fraction(1, SR)
        for p in as_.encode(fr):
            c.mux(p)
    for s in (vs, as_):
        for p in s.encode(None):
            c.mux(p)
    c.close()
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
