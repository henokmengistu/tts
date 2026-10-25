#!/usr/bin/env python
"""Benchmark amharic-tts against Google TTS (gTTS, lang='am') and MMS-TTS.

Usage:
    python benchmark/run_benchmark.py --systems ours,google,mms --voice selam --out benchmark/out

Outputs per-system WAVs, a JSONL of per-item metrics, and a Markdown report.
Metrics: ASR round-trip CER (snapwre/hohe-asr-amharic), UTMOS, DNSMOS P.835,
duration, and wall-clock RTF.  Items of category 'norm' are scored against the
intended spoken form (``spoken``), so a system that reads numbers, dates or
abbreviations wrongly is penalised even if its audio is clean.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import metrics as M  # noqa: E402


def synth_google(text: str, out: Path) -> tuple[np.ndarray, int, float]:
    from gtts import gTTS
    import librosa

    mp3 = out.with_suffix(".mp3")
    t0 = time.time()
    gTTS(text, lang="am").save(str(mp3))
    el = time.time() - t0
    wav, sr = librosa.load(str(mp3), sr=None, mono=True)
    sf.write(str(out), wav, sr)
    return wav, sr, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(Path(__file__).parent / "testset.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--systems", default="ours,google,mms")
    ap.add_argument("--voice", default="selam")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--ids", default=None, help="comma-separated subset of item ids")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    items = json.loads(Path(args.testset).read_text(encoding="utf-8"))["items"]
    if args.ids:
        keep = set(args.ids.split(","))
        items = [i for i in items if i["id"] in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    systems = args.systems.split(",")

    engines = {}
    if "ours" in systems:
        from amharic_tts import AmharicTTS

        engines["ours"] = AmharicTTS(engine="omnivoice", num_steps=args.steps, quantize=args.quant)
    if "mms" in systems:
        from amharic_tts import AmharicTTS

        engines["mms"] = AmharicTTS(engine="mms")

    from amharic_tts.text import normalize

    rows = []
    jsonl = out / "results.jsonl"
    done = set()
    if args.skip_existing and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["system"], r["id"]))
            rows.append(r)
    with jsonl.open("a", encoding="utf-8") as fh:
        for it in items:
            ref_spoken = it.get("spoken") or normalize(it["text"])
            for sysname in systems:
                if (sysname, it["id"]) in done:
                    continue
                d = out / sysname
                d.mkdir(exist_ok=True)
                wavp = d / f"{it['id']}.wav"
                try:
                    if sysname == "google":
                        wav, sr, el = synth_google(it["text"], wavp)
                    else:
                        tts = engines[sysname]
                        res = tts.synthesize(it["text"], voice=args.voice if sysname == "ours" else None, seed=0)
                        wav, sr, el = res.audio, res.sample_rate, res.elapsed
                        sf.write(str(wavp), wav, sr)
                except Exception as e:  # noqa: BLE001
                    print(f"[{sysname}] {it['id']} FAILED: {e}", file=sys.stderr)
                    continue
                dur = len(wav) / sr
                hyp = M.transcribe(wav, sr)
                row = {
                    "system": sysname, "id": it["id"], "cat": it["cat"], "text": it["text"], "ref": ref_spoken,
                    "hyp": hyp, "cer": M.cer(ref_spoken, hyp), "utmos": M.utmos(wav, sr), **M.dnsmos(wav, sr),
                    "duration": dur, "elapsed": el, "rtf": el / dur if dur else None, "sr": sr,
                }
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{sysname}] {it['id']} cer={row['cer']:.3f} utmos={row['utmos']:.2f} ovrl={row['OVRL']:.2f} "
                      f"dur={dur:.1f}s rtf={row['rtf']:.2f}", flush=True)
    write_report(rows, out / "report.md")


def write_report(rows: list[dict], path: Path) -> None:
    import statistics as st

    systems = sorted({r["system"] for r in rows})
    cats = ["plain", "norm", "long", "all"]
    lines = ["# Amharic TTS benchmark", "", f"{len(rows)} utterances scored. CER = ASR round-trip character error rate "
             "(lower is better). UTMOS / DNSMOS OVRL = predicted MOS (higher is better).", ""]
    lines += ["| system | category | n | CER | UTMOS | DNSMOS SIG | DNSMOS BAK | DNSMOS OVRL | audio s | RTF |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in systems:
        for c in cats:
            sub = [r for r in rows if r["system"] == s and (c == "all" or r["cat"] == c)]
            if not sub:
                continue
            m = lambda k: st.mean(r[k] for r in sub if r.get(k) is not None)  # noqa: E731
            lines.append(f"| {s} | {c} | {len(sub)} | {m('cer')*100:.1f}% | {m('utmos'):.2f} | {m('SIG'):.2f} | "
                         f"{m('BAK'):.2f} | {m('OVRL'):.2f} | {sum(r['duration'] for r in sub):.0f} | {m('rtf'):.2f} |")
    lines += ["", "## Per-item CER (%)", ""]
    ids = []
    for r in rows:
        if r["id"] not in ids:
            ids.append(r["id"])
    lines.append("| id | cat | " + " | ".join(systems) + " |")
    lines.append("|---|---|" + "---:|" * len(systems))
    for i in ids:
        cells = []
        for s in systems:
            rr = [r for r in rows if r["id"] == i and r["system"] == s]
            cells.append(f"{rr[0]['cer']*100:.1f}" if rr else "-")
        cat = next(r["cat"] for r in rows if r["id"] == i)
        lines.append(f"| {i} | {cat} | " + " | ".join(cells) + " |")
    lines += ["", "## Normalisation items: what each system actually said (ASR transcript)", ""]
    for i in ids:
        rr = [r for r in rows if r["id"] == i and r["cat"] == "norm"]
        if not rr:
            continue
        lines.append(f"**{i}** `{rr[0]['text']}`  ")
        lines.append(f"- expected: {rr[0]['ref']}")
        for r in rr:
            lines.append(f"- {r['system']} ({r['cer']*100:.0f}%): {r['hyp']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
