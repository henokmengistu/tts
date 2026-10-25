#!/usr/bin/env python
"""Voice casting: rank candidate reference clips by how well the TTS performs with them.

Input: a JSON list of candidates ``[{"name":..., "path":..., "text":..., "gender":..., "source":...}, ...]``.
For each candidate the engine clones the voice and reads a fixed set of test
sentences; every output is scored (ASR CER, deletions, UTMOS, DNSMOS, energy
>4 kHz, syllable rate).  A composite score ranks the candidates.

Usage: python benchmark/diag/cast_voices.py candidates.json --out benchmark/diag/out/casting --speed 0.7
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))
import metrics as M  # noqa: E402
from diagnose import spectral, word_errors, syllables  # noqa: E402

TEST_SENTENCES = [
    "የኢትዮጵያ ብሔራዊ ባንክ ዛሬ አዲስ መመሪያ አወጣ። መመሪያው የውጭ ምንዛሬ ግብይትን የሚመለከት ሲሆን ከነገ ጀምሮ ተግባራዊ ይሆናል።",
    "ጤናማ አመጋገብ እና መደበኛ የሰውነት እንቅስቃሴ ለረጅም ዕድሜ መሠረት ናቸው።",
    "ዛሬ ጥቅምት 15 ቀን 2019 ዓ.ም. ነው። ዶ/ር አበበ በት/ቤቱ 1250 ብር ለገሱ።",
    "ልጆቹ ዝናቡ ሲቆም ወደ ሜዳ ወጥተው ኳስ መጫወት ጀመሩ፤ እናታቸው ግን ቶሎ እንዲመለሱ አሳሰበቻቸው።",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--out", default=str(ROOT / "benchmark/diag/out/casting"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-ref-seconds", type=float, default=10.0)
    args = ap.parse_args()

    cands = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from amharic_tts import AmharicTTS

    tts = AmharicTTS(engine="omnivoice", num_steps=args.steps)
    eng = tts.engine
    results = []
    done = {json.loads(l)["name"] for l in (out / "results.jsonl").open(encoding="utf-8")} if (out / "results.jsonl").exists() else set()
    fh = (out / "results.jsonl").open("a", encoding="utf-8")
    for c in cands:
        if c["name"] in done:
            continue
        t0 = time.time()
        ref_path = c["path"]
        ref_text = c["text"]
        y, sr = sf.read(ref_path)
        if y.ndim > 1:
            y = y.mean(1)
        y = y.astype(np.float32)
        if len(y) / sr > args.max_ref_seconds:
            # cut at the quietest 10 ms frame between 60 % and 100 % of the budget,
            # then let the recogniser transcribe exactly what is kept
            lo, hi = int(0.6 * args.max_ref_seconds * sr), int(args.max_ref_seconds * sr)
            frame = int(0.01 * sr)
            seg = y[lo:hi]
            n = len(seg) // frame
            e = (seg[: n * frame].reshape(n, frame) ** 2).mean(1)
            cut = lo + int(np.argmin(e)) * frame
            y = y[:cut]
            ref_path = str(out / f"{c['name']}_ref.wav")
            sf.write(ref_path, y, sr)
            ref_text = M.transcribe(y, sr)
            c["ref_text_used"] = ref_text
        try:
            prompt = eng.clone_prompt(ref_path, ref_text)
        except Exception as e:  # noqa: BLE001
            print(c["name"], "PROMPT FAILED", e, flush=True)
            continue
        rows = []
        for i, sent in enumerate(TEST_SENTENCES):
            res = tts.synthesize(sent, voice=None, speed=args.speed, seed=0, voice_prompt=prompt)
            wav, srr = res.audio, res.sample_rate
            sf.write(str(out / f"{c['name']}_{i}.wav"), wav, srr)
            hyp = M.transcribe(wav, srr)
            we = word_errors(res.text, hyp)
            d = M.dnsmos(wav, srr)
            rows.append({"cer": M.cer(res.text, hyp), "del": we.get("del", 0), "n_words": we.get("n_words", 1),
                         "utmos": M.utmos(wav, srr), "OVRL": d["OVRL"], "SIG": d["SIG"], "BAK": d["BAK"],
                         **spectral(wav, srr), "rate": syllables(sent) / (len(wav) / srr), "hyp": hyp})
        agg = {k: float(st.mean(r[k] for r in rows)) for k in ("cer", "utmos", "OVRL", "SIG", "BAK", "e4k", "rolloff95", "rate")}
        agg["del_rate"] = sum(r["del"] for r in rows) / max(1, sum(r["n_words"] for r in rows))
        # composite: intelligibility first, then predicted quality
        agg["score"] = (1 - 4 * agg["cer"] - 4 * agg["del_rate"]) + 0.15 * (agg["utmos"] - 3) + 0.15 * (agg["OVRL"] - 3)
        rec = {**c, **agg, "rows": rows, "elapsed": time.time() - t0}
        results.append(rec)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"{c['name']:14s} {c.get('gender',''):6s} cer={agg['cer']*100:4.1f} del={agg['del_rate']*100:3.1f}% utmos={agg['utmos']:.2f} "
              f"ovrl={agg['OVRL']:.2f} e4k={agg['e4k']*100:.2f}% rate={agg['rate']:.1f} score={agg['score']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    allr = [json.loads(l) for l in (out / "results.jsonl").open(encoding="utf-8")]
    allr.sort(key=lambda r: -r["score"])
    lines = ["| rank | name | gender | source | CER | del | UTMOS | DNSMOS OVRL | e>4k | syl/s | score |", "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(allr, 1):
        lines.append(f"| {i} | {r['name']} | {r.get('gender','')} | {r.get('source','')} | {r['cer']*100:.1f}% | {r['del_rate']*100:.1f}% | "
                     f"{r['utmos']:.2f} | {r['OVRL']:.2f} | {r['e4k']*100:.2f}% | {r['rate']:.1f} | {r['score']:.3f} |")
    (out / "ranking.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
