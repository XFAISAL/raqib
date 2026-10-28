"""report.json + a self-contained report.html (flagged frames blurred by default, click to reveal)."""
from __future__ import annotations

import html
import json
from pathlib import Path

KIND_LABEL = {
    "nudity": "Nudity", "sexual_act": "Sexual act", "weapon": "Weapon", "violence": "Violence",
    "dangerous": "Dangerous", "self_harm": "Self-harm", "harassment": "Harassment", "abuse": "Abuse",
    "threat": "Threat", "hate": "Hate", "sexual_solicitation": "Sexual solicitation", "extortion": "Extortion",
    "grooming": "Grooming", "scam": "Scam",
}


def _t(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:d}:{s % 60:02d}"


def _tier(c: float) -> str:
    return "high" if c >= 0.85 else "med" if c >= 0.65 else "low"


def write(result: dict, out: Path, blur: bool = True) -> tuple[Path, Path | None]:
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "report.json"
    jp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    hp = out / "report.html"
    hp.write_text(render(result, blur))
    return jp, hp


def render(r: dict, blur: bool = True) -> str:
    e = html.escape
    fs = r["findings"]
    dur = max([s.get("duration") or 0 for s in r["sources"]] + [f["t"] for f in fs] + [1])
    counts: dict[str, int] = {}
    for f in fs:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    chips = "".join(f'<span class="chip k-{e(k)}">{e(KIND_LABEL.get(k, k))} <b>{n}</b></span>'
                    for k, n in sorted(counts.items(), key=lambda kv: -kv[1])) or '<span class="ok">Nothing flagged</span>'
    marks = "".join(
        f'<a class="mark {f["modality"]} {_tier(f["confidence"])}" href="#f-{f["id"][:8]}" '
        f'style="left:{100 * f["t"] / dur:.2f}%" title="{_t(f["t"])} · {e(KIND_LABEL.get(f["kind"], f["kind"]))} '
        f'{f["confidence"]:.0%}"></a>' for f in fs)

    frames = [f for f in fs if f.get("frame")]
    cards = []
    for f in frames:
        img = f["frame_blurred"] if blur and f.get("frame_blurred") else f["frame"]
        rule = ' · sustained' if f.get("rule") == "sustained" else ""
        cards.append(
            f'<figure id="f-{f["id"][:8]}" class="frame {"blur" if blur else ""}" data-full="{e(f["frame"])}">'
            f'<img src="{e(img)}" alt="flagged frame at {_t(f["t"])}" loading="lazy">'
            f'<figcaption><span class="chip k-{e(f["kind"])}">{e(KIND_LABEL.get(f["kind"], f["kind"]))}</span>'
            f'<span class="conf {_tier(f["confidence"])}">{f["confidence"]:.0%}</span>'
            f'<span class="when">{_t(f["t"])} · {e(f["speaker"])}{rule}</span></figcaption></figure>')

    by_line: dict[tuple[str, float], list[dict]] = {}
    for f in fs:
        if f["modality"] == "audio":
            by_line.setdefault((f["speaker"], f["t"]), []).append(f)
    rows = []
    for ln in r["transcript"]:
        hits = by_line.get((ln["speaker"], ln["start"]), [])
        tags = "".join(f'<span id="f-{h["id"][:8]}" class="chip k-{e(h["kind"])}">{e(KIND_LABEL.get(h["kind"], h["kind"]))} '
                       f'{h["confidence"]:.0%}</span>' for h in hits)
        intent = f'<div class="intent">{e(hits[0].get("intent") or "")} · {e(hits[0].get("act") or "")}</div>' if hits else ""
        rows.append(f'<li class="{"hit" if hits else ""}"><span class="ts">{_t(ln["start"])}</span>'
                    f'<span class="spk">{e(ln["speaker"])}</span><div class="txt" dir="auto">{e(ln["text"])}{intent}</div>'
                    f'<div class="tags">{tags}</div></li>')

    src = " · ".join(e(Path(s["path"]).name) for s in r["sources"])
    st = r["stats"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raqib report</title>
<style>
:root{{--bg:#f6f5f2;--card:#fff;--ink:#1c2321;--muted:#68716d;--line:#e6e4de;--accent:#0f766e;
--high:#c2410c;--med:#b45309;--low:#64748b;--video:#7c3aed;--audio:#0f766e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111514;--card:#1a1f1e;--ink:#eef2f0;--muted:#9aa5a0;--line:#2a312f;--accent:#2dd4bf;
--high:#fb923c;--med:#fbbf24;--low:#94a3b8;--video:#a78bfa;--audio:#2dd4bf}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 64px}}
header h1{{margin:0;font-size:26px;letter-spacing:-.3px}}header .src{{color:var(--muted);margin-top:2px;word-break:break-all}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:20px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px}}
.stat b{{display:block;font-size:22px}}.stat span{{color:var(--muted);font-size:12px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 18px}}
.chip{{display:inline-flex;gap:6px;align-items:center;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;
background:color-mix(in srgb,var(--accent) 12%,transparent);color:var(--accent)}}
.k-nudity,.k-sexual_act,.k-sexual_solicitation,.k-grooming{{--accent:#db2777}}.k-weapon,.k-violence,.k-threat{{--accent:#dc2626}}
.k-dangerous,.k-self_harm{{--accent:#ea580c}}.k-harassment,.k-abuse,.k-hate{{--accent:#7c3aed}}.k-scam,.k-extortion{{--accent:#0369a1}}
.ok{{color:var(--muted)}}
h2{{font-size:17px;margin:30px 0 12px}}
.timeline{{position:relative;height:44px;background:var(--card);border:1px solid var(--line);border-radius:12px}}
.timeline .axis{{position:absolute;left:12px;right:12px;top:21px;height:2px;background:var(--line)}}
.mark{{position:absolute;top:10px;width:4px;height:24px;border-radius:2px;margin-left:10px;opacity:.9}}
.mark.video{{background:var(--video)}}.mark.audio{{background:var(--audio)}}.mark.low{{opacity:.45}}
.legend{{color:var(--muted);font-size:12px;margin-top:6px}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.frame{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;cursor:pointer}}
.frame img{{display:block;width:100%;aspect-ratio:16/10;object-fit:cover;background:#000}}
.frame figcaption{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:9px 10px}}
.conf{{font-weight:700;font-size:13px}}.conf.high{{color:var(--high)}}.conf.med{{color:var(--med)}}.conf.low{{color:var(--low)}}
.when{{color:var(--muted);font-size:12px;margin-left:auto}}
ol.tx{{list-style:none;padding:0;margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
ol.tx li{{display:grid;grid-template-columns:56px 90px 1fr auto;gap:10px;padding:10px 14px;border-top:1px solid var(--line);align-items:start}}
ol.tx li:first-child{{border-top:0}}ol.tx li.hit{{background:color-mix(in srgb,#dc2626 7%,transparent)}}
.ts{{color:var(--muted);font-variant-numeric:tabular-nums}}.spk{{color:var(--muted);font-size:13px}}
.intent{{color:var(--muted);font-size:12px;margin-top:2px}}.tags{{display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end}}
@media (max-width:640px){{ol.tx li{{grid-template-columns:48px 1fr}}.spk,.tags{{grid-column:2}}}}
footer{{color:var(--muted);font-size:12px;margin-top:36px}}
</style></head><body><main>
<header><h1>Raqib report</h1><div class="src">{src}</div></header>
<div class="stats">
<div class="stat"><b>{len(fs)}</b><span>findings</span></div>
<div class="stat"><b>{len(frames)}</b><span>flagged frames</span></div>
<div class="stat"><b>{sum(1 for l in r['transcript'] if l.get('flagged'))}</b><span>flagged lines</span></div>
<div class="stat"><b>{_t(dur)}</b><span>duration</span></div>
<div class="stat"><b>{st.get('frames_analysed', 0)}</b><span>frames analysed</span></div>
<div class="stat"><b>{st.get('lines', 0)}</b><span>speech lines</span></div>
</div>
<div class="chips">{chips}</div>
<h2>Timeline</h2>
<div class="timeline"><div class="axis"></div>{marks}</div>
<div class="legend"><i style="background:var(--video)"></i>video<i style="background:var(--audio)"></i>speech · faded = lower confidence</div>
<h2>Flagged frames</h2>
{'<div class="grid">' + ''.join(cards) + '</div>' if cards else '<p class="ok">No flagged frames.</p>'}
<h2>Transcript</h2>
{'<ol class="tx">' + ''.join(rows) + '</ol>' if rows else '<p class="ok">No speech.</p>'}
<footer>Raqib {e(r['version'])} · {e(', '.join(r['models']))} · min confidence {r['min_confidence']:.0%} ·
processed in {r['processing_sec']}s · automated scores, review before acting.</footer>
</main>
<script>
document.querySelectorAll('.frame.blur').forEach(f=>f.addEventListener('click',()=>{{
  const i=f.querySelector('img');i.src=f.dataset.full;f.classList.remove('blur');}}));
</script></body></html>"""
