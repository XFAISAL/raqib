"""Frame classification.

LocalVision (ONNX, no network):
  • NSFW classifier  ViT-B/16 normal/nsfw                     → nudity evidence
  • NudeNet 320n     YOLOv8n exposed / covered body parts      → nudity corroboration
  • CLIP ViT-B/16    zero-shot prompt groups                   → weapon, violence, dangerous, self_harm, sexual_act

ProviderVision sends the frame to a multimodal model behind an API and asks for the same scores.

Both return one FrameResult per frame: kind → confidence 0..1, plus the numbers behind it.
"""
from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

VIDEO_KINDS = ("nudity", "sexual_act", "weapon", "violence", "dangerous", "self_harm")

NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED",
    "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
]
# Parts that count as nudity (and how strongly). Male chest, belly and feet do not.
NUDENET_EXPLICIT = {
    "FEMALE_GENITALIA_EXPOSED": 1.0, "MALE_GENITALIA_EXPOSED": 1.0, "ANUS_EXPOSED": 1.0,
    "FEMALE_BREAST_EXPOSED": 0.9, "BUTTOCKS_EXPOSED": 0.8,
}

# CLIP zero-shot prompt groups. "neutral" is deliberately broad: real call / stream footage is mostly
# faces, rooms, ceilings, dark or blurry frames and everyday objects that look like weapons.
CLIP_GROUPS: dict[str, list[str]] = {
    "neutral": [
        "a person talking to the camera on a video call", "a selfie of a person", "a person sitting in a room",
        "a photo of a room interior", "a person in a car", "people eating food", "a blurry dark photo",
        "a person holding a smartphone", "a person holding a remote control", "a person brushing their hair",
        "a person cooking in a kitchen", "a photo of a computer screen", "a photo of a ceiling",
        "a photo of a cat or a dog", "a person laughing", "a person playing video games",
        "a person exercising at the gym", "a person wearing a costume", "a person drinking coffee",
        "a person holding a pen", "a cartoon or a drawing", "a landscape photo", "a street at night",
        "a person playing sports", "a toy", "a person putting on makeup", "a person wearing a headscarf",
        "people playing a board game", "a television showing a movie", "a sports match on television",
        # contextual negatives: measured false positives of the zero-shot heads
        "a campfire outdoors", "a small campfire with people sitting around it", "fireworks in the night sky",
        "a candle flame", "a fireplace", "a gas stove flame",
        "a colorful plastic toy water pistol", "a kitchen knife cutting vegetables on a cutting board",
        "a swimming pool", "a boxing match in a ring", "a martial arts class", "a fencing match",
        "a video game on a screen", "a movie scene on a television", "a barbecue grill",
    ],
    "weapon": [
        "a person holding a gun", "a handgun pointed at the camera", "a rifle", "a person holding a knife threateningly",
        "a knife pointed at the camera", "a person holding a firearm", "a machine gun",
    ],
    "violence": [
        "people fighting", "a person punching another person", "a person being beaten",
        "a bloody injured person", "a violent assault", "a person being choked",
    ],
    "dangerous": [
        "a building on fire", "a room engulfed in flames", "an explosion", "drugs and syringes",
        "a person doing a dangerous stunt on a rooftop edge", "a person driving recklessly while filming",
    ],
    "self_harm": [
        "a person cutting their own arm with a blade", "a noose hanging from the ceiling",
        "a person holding a handful of pills to overdose", "a wrist with fresh self-harm cuts",
    ],
    "sexual_act": [
        "people having sex", "a sexual act", "a person performing a sex act on camera", "a person masturbating",
    ],
    "nudity": ["a naked person", "a nude body", "a topless person"],
}

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


@dataclass
class FrameResult:
    scores: dict[str, float]                       # kind → confidence 0..1
    margins: dict[str, float] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)      # model internals (numbers only)


def _providers(device: str):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
    if device == "cuda":
        raise SystemExit("vision.device=cuda but onnxruntime has no CUDA provider (pip install raqib[gpu])")
    return ["CPUExecutionProvider"], "cpu"


class LocalVision:
    def __init__(self, models_dir: Path, device: str = "auto", nsfw: bool = True, nudenet: bool = True,
                 clip: bool = True, threads: int = 4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        prov, self.device = _providers(device)

        def load(p: Path):
            if self.device == "cpu":  # int8 variants are ~3x faster than fp16 on CPU
                q = p.parent / p.name.replace("_fp16", "_quantized")
                if q.exists():
                    p = q
            if not p.exists():
                raise SystemExit(f"model file missing: {p}\nrun: raqib models download")
            return ort.InferenceSession(str(p), sess_options=so, providers=prov)

        t0 = time.monotonic()
        self.nsfw = None
        if nsfw:
            d = models_dir / "nsfw"
            self.nsfw = load(d / "onnx" / "model_fp16.onnx")
            cfg = json.loads((d / "config.json").read_text())
            self.nsfw_size = int(cfg.get("image_size", 224))
            self.nsfw_idx = int({v: k for k, v in cfg["id2label"].items()}.get("nsfw", 1))
        self.nudenet = load(models_dir / "nudenet" / "320n.onnx") if nudenet else None
        self.clip = load(models_dir / "clip-b16" / "onnx" / "vision_model_fp16.onnx") if clip else None
        self.text_emb: dict[str, np.ndarray] = {}
        if clip:
            self._text_embeddings(models_dir / "clip-b16", ort, so)
        self.load_sec = round(time.monotonic() - t0, 2)

    # CLIP text side: embedded once, cached next to the model
    def _text_embeddings(self, d: Path, ort, so) -> None:
        cache = d / "prompt_embeddings.npz"
        allp = {f"{g}::{p}": p for g, ps in CLIP_GROUPS.items() for p in ps}
        if cache.exists():
            z = np.load(cache)
            if set(z.files) == set(allp):
                self.text_emb = {k: z[k] for k in z.files}
                return
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        tok.enable_padding(pad_id=49407, pad_token="<|endoftext|>", length=77)
        tok.enable_truncation(77)
        sess = ort.InferenceSession(str(d / "onnx" / "text_model_quantized.onnx"), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        keys = list(allp)
        ids = np.array([e.ids for e in tok.encode_batch([allp[k] for k in keys])], np.int64)
        emb = sess.run(None, {"input_ids": ids})[0]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.text_emb = {k: emb[i].astype(np.float32) for i, k in enumerate(keys)}
        np.savez(cache, **self.text_emb)

    # preprocessing
    @staticmethod
    def _square(img: Image.Image, size: int) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas.resize((size, size), Image.BICUBIC)

    def _prep_nsfw(self, imgs: list[Image.Image]) -> np.ndarray:
        arr = np.stack([np.asarray(i.resize((self.nsfw_size, self.nsfw_size), Image.BILINEAR), np.float32) for i in imgs])
        return ((arr / 255.0 - 0.5) / 0.5).transpose(0, 3, 1, 2).copy()

    def _prep_clip(self, imgs: list[Image.Image]) -> np.ndarray:
        arr = np.stack([np.asarray(self._square(i, 224), np.float32) for i in imgs]) / 255.0
        return ((arr - CLIP_MEAN) / CLIP_STD).transpose(0, 3, 1, 2).astype(np.float32).copy()

    @staticmethod
    def _prep_nudenet(imgs: list[Image.Image]) -> np.ndarray:
        out = []
        for i in imgs:  # pad bottom/right to square, 320 px, /255
            s = max(i.size)
            c = Image.new("RGB", (s, s))
            c.paste(i, (0, 0))
            out.append(np.asarray(c.resize((320, 320), Image.BILINEAR), np.float32) / 255.0)
        return np.stack(out).transpose(0, 3, 1, 2).copy()

    # inference
    def _nudenet_scores(self, x: np.ndarray) -> list[dict[str, float]]:
        out = self.nudenet.run(None, {"images": x})[0]  # [B, 22, N]
        res = []
        for b in range(out.shape[0]):
            cls = out[b].T[:, 4:]
            idx = np.argmax(cls, axis=1)
            sc = cls[np.arange(cls.shape[0]), idx]
            best: dict[str, float] = {}
            for ci, s in zip(idx[sc >= 0.2], sc[sc >= 0.2]):
                lab = NUDENET_LABELS[int(ci)]
                best[lab] = max(best.get(lab, 0.0), float(s))
            res.append(best)
        return res

    @staticmethod
    def _group_probs(img_emb: np.ndarray, emb: dict[str, np.ndarray]) -> dict[str, float]:
        logits = {}
        for g, ps in CLIP_GROUPS.items():
            vals = np.array([100.0 * float(img_emb @ emb[f"{g}::{p}"]) for p in ps])
            m = vals.max()
            logits[g] = m + np.log(np.mean(np.exp(vals - m)))  # log-mean-exp: neutral to group size
        mx = max(logits.values())
        ex = {g: np.exp(v - mx) for g, v in logits.items()}
        z = sum(ex.values())
        return {g: float(v / z) for g, v in ex.items()}

    def infer(self, frames: list[np.ndarray]) -> list[FrameResult]:
        """frames: HxWx3 uint8 RGB arrays."""
        imgs = [Image.fromarray(f) for f in frames]
        n = len(imgs)
        p_nsfw = [0.0] * n
        if self.nsfw is not None:
            logits = self.nsfw.run(None, {"pixel_values": self._prep_nsfw(imgs)})[0]
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            p_nsfw = list((ex / ex.sum(axis=1, keepdims=True))[:, self.nsfw_idx])
        nn = self._nudenet_scores(self._prep_nudenet(imgs)) if self.nudenet else [{} for _ in imgs]
        groups: list[dict[str, float]] = [{} for _ in imgs]
        if self.clip is not None:
            emb = self.clip.run(None, {"pixel_values": self._prep_clip(imgs)})[0]
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            groups = [self._group_probs(emb[i], self.text_emb) for i in range(n)]
        return [combine(float(p_nsfw[i]), nn[i], groups[i]) for i in range(n)]


def combine(p_nsfw: float, nn: dict[str, float], g: dict[str, float]) -> FrameResult:
    """Fuse the three models into per-kind confidences."""
    explicit = max((s * w for lab, s in nn.items() if (w := NUDENET_EXPLICIT.get(lab))), default=0.0)
    clip_nude = g.get("nudity", 0.0) + g.get("sexual_act", 0.0)
    # Nudity: the specialised classifier drives the score; an exposed explicit part (NudeNet) or CLIP
    # agreement raises it, and the absence of both damps it (common false positives: skin-coloured walls,
    # shirtless men, swimwear, dark close-ups).
    if explicit >= 0.45 or clip_nude >= 0.5:
        nud = 1 - (1 - p_nsfw) * (1 - 0.6 * max(explicit, clip_nude))
    else:
        nud = p_nsfw * 0.8
    scores = {"nudity": nud}
    margins = {"nudity": abs(2 * nud - 1)}
    if g:
        for kind in ("weapon", "violence", "dangerous", "self_harm"):
            p = g.get(kind, 0.0)
            scores[kind] = p
            margins[kind] = max(0.0, p - max(v for k, v in g.items() if k != kind))
        # sexual_act from a single frame only when the NSFW classifier agrees (CLIP alone is too noisy);
        # the sustained rule in the pipeline covers footage the classifier misses (e.g. filmed off a screen).
        sa = g.get("sexual_act", 0.0) * (p_nsfw if p_nsfw >= 0.5 else 0.0)
        scores["sexual_act"] = sa
        margins["sexual_act"] = max(0.0, sa - max(v for k, v in g.items() if k != "sexual_act"))
    return FrameResult(scores, margins, {"p_nsfw": round(p_nsfw, 4), "nudenet_explicit": round(explicit, 4),
                                         "clip": {k: round(v, 4) for k, v in g.items()}})


VISION_PROMPT = """You are a content-safety classifier for frames from live video calls and streams.
Score how clearly the image shows each category, 0-100:
nudity (exposed genitals, female breasts or buttocks), sexual_act, weapon (a real gun or a knife held threateningly),
violence (fighting, assault, blood), dangerous (fire, explosion, drugs, dangerous stunts), self_harm.
Ordinary scenes (faces, rooms, food, games, kitchen knives used for cooking, toys, TV screens) score 0.
Reply with JSON only: {"nudity":0,"sexual_act":0,"weapon":0,"violence":0,"dangerous":0,"self_harm":0}"""


class ProviderVision:
    """A multimodal model behind an API (one request per frame: use a longer sampling interval)."""

    def __init__(self, provider):
        from .providers import client_for

        self.client = client_for(provider)

    def infer(self, frames: list[np.ndarray]) -> list[FrameResult]:
        out = []
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, "JPEG", quality=85)
            data = self.client.vision_json(VISION_PROMPT, base64.b64encode(buf.getvalue()).decode())
            scores = {k: max(0.0, min(1.0, float(data.get(k, 0)) / 100.0)) for k in VIDEO_KINDS}
            out.append(FrameResult(scores, {k: abs(2 * v - 1) for k, v in scores.items()}, {"provider": data}))
        return out
