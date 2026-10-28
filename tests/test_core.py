import numpy as np
import pytest

from raqib import config, textrisk
from raqib.audio import SR, Segmenter, clean_transcript
from raqib.pipeline import _Sampler
from raqib.vision import combine


# ---------------------------------------------------------------- transcripts
@pytest.mark.parametrize("text", ["شكرا على المشاهدة", "Thanks for watching!", "@@@ضوضاء", "(ضحك)", "*music*",
                                  "موقع الدراسة الجزائري"])
def test_hallucinations_dropped(text):
    assert clean_transcript(text) is None


def test_repetition_loops_collapsed():
    assert clean_transcript("لا لا لا لا لا لا خلاص") == "لا لا لا خلاص"
    assert clean_transcript("هااااااااي") == "هاااي"


def test_low_confidence_dropped_but_phone_quality_kept():
    assert clean_transcript("كلام", avg_logprob=-1.7) is None
    assert clean_transcript("كلام", avg_logprob=-1.2, no_speech=0.5) == "كلام"


# ---------------------------------------------------------------- segmentation (energy gate, no model needed)
def _tone(sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_segmenter_splits_on_silence():
    seg = Segmenter(vad=False, hang=0.5, pad=0.2, min_len=0.3)
    audio = np.concatenate([_tone(1.0), np.zeros(SR, np.float32), _tone(1.5), np.zeros(SR, np.float32)])
    out = []
    for i in range(0, len(audio), 1600):
        out += seg.push(audio[i:i + 1600])
    out += seg.flush()
    assert len(out) == 2
    assert out[0].start == pytest.approx(0.0, abs=0.25)
    assert out[1].start == pytest.approx(2.0, abs=0.3)


def test_segmenter_cuts_long_speech_with_overlap():
    seg = Segmenter(vad=False, max_len=3.0, overlap=1.0, hang=0.5, pad=0.2)
    out = []
    a = _tone(7.0)
    for i in range(0, len(a), 1600):
        out += seg.push(a[i:i + 1600])
    out += seg.flush()
    assert len(out) >= 3
    assert out[1].start < out[0].start + out[0].duration  # overlaps the previous cut


# ---------------------------------------------------------------- text risk
def test_window_separates_context_from_new_lines():
    w = textrisk.build_window([("B", "hello", False), ("A", "you idiot", True)])
    ctx, new = w.split("A's NEW lines")
    assert "B: hello" in ctx and "you idiot" in new and "you idiot" not in ctx


def test_non_direct_speech_is_discounted():
    r = textrisk.parse({"act": "reported", "risks": [{"k": "threat", "s": 90}]})
    assert r.raw["threat"] == pytest.approx(0.9)
    assert r.scores["threat"] == pytest.approx(0.9 * textrisk.NON_DIRECT_FACTOR)
    d = textrisk.parse({"act": "direct", "risks": [{"k": "threat", "s": 90}, {"k": "bogus", "s": 99}]})
    assert d.scores["threat"] == pytest.approx(0.9) and "bogus" not in d.scores


# ---------------------------------------------------------------- vision fusion
def test_nudity_needs_corroboration():
    alone = combine(0.95, {}, {"neutral": 0.9, "nudity": 0.05})
    backed = combine(0.95, {"FEMALE_BREAST_EXPOSED": 0.8}, {"neutral": 0.9, "nudity": 0.05})
    assert alone.scores["nudity"] == pytest.approx(0.76)
    assert backed.scores["nudity"] > 0.95


def test_sexual_act_needs_nsfw_agreement():
    g = {"neutral": 0.1, "sexual_act": 0.85, "nudity": 0.05}
    assert combine(0.1, {}, g).scores["sexual_act"] == 0.0
    assert combine(0.9, {}, g).scores["sexual_act"] > 0.7


# ---------------------------------------------------------------- sampling
def test_sampler_skips_near_duplicates_but_not_forever():
    s = _Sampler(config.SamplingConfig(initial_interval=1, initial_window=999, dedup_threshold=0.05, max_skip=5))
    img = np.full((64, 64, 3), 120, np.uint8)
    assert s.accept(0.0, img)
    assert not s.accept(1.0, img)
    assert s.accept(6.0, img)  # max_skip reached
    other = np.zeros((64, 64, 3), np.uint8)
    assert s.accept(7.0, other)


# ---------------------------------------------------------------- config
def test_config_overrides(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("vision: {backend: off}\nstt: {languages: [en]}\n")
    monkeypatch.setenv("RAQIB_SAMPLING__INTERVAL", "7")
    c = config.load(p, ["text.provider.model=claude-haiku-5-5", "min_confidence=0.7"])
    assert c.vision.backend == "off" and c.stt.languages == ["en"]
    assert c.sampling.interval == 7.0 and c.min_confidence == 0.7
    assert c.text.provider.model == "claude-haiku-5-5"
    with pytest.raises(ValueError):
        config._merge(config.Config(), {"nope": 1})
