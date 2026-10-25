#!/usr/bin/env python
"""Deep diagnostic: word-level errors, spectral bandwidth and pitch dynamics.

Runs a configuration of the OmniVoice engine on N unseen FLEURS *test*
sentences and compares against the human recordings of the same sentences:

* ASR word alignment (jiwer): substitutions / deletions / insertions, and
  whether errors cluster at the final word (end-of-utterance truncation).
* CER (homophone-merged) vs the human recording's own CER (the ASR floor).
* Spectral roll-off (95 % / 99 %), energy above 4 kHz and above 7 kHz.
* log-F0 standard deviation (semitones) and voiced fraction: prosody dynamics.
* UTMOS, DNSMOS, speaking rate (syllables / s), RTF.

Usage: python benchmark/diag/diagnose.py --name default --n 40 [--voice selam --steps 16 --guidance 2.0 ...]
"""
from __future__ import annotations

import argparse
import csv
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

FLEURS = Path("/home/user/data/fleurs")


ADDIS = Path("/home/user/data/addis_bench")


def load_addis_items(n: int):
    """The 100 prompts of the public Addis AI Amharic TTS benchmark."""
    items = []
    for line in (ADDIS / "data/prompts.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        pid = r.get("case_id") or r.get("prompt_id") or r.get("id")
        text = r.get("text") or r.get("reference_text")
        wav = ADDIS / "audio/human-reference" / f"{pid}.wav"
        items.append(dict(id=pid, wav=str(wav), text=text, norm=text, gender="", dur=0.0, stratum=r.get("length_stratum"), source=r.get("source")))
    return items[:n]


def load_items(n: int, seed: int = 7):
    rows = [r for r in csv.reader(open(FLEURS / "test.tsv"), delimiter="\t") if len(r) >= 7]
    # one row per sentence id, prefer clips 4-12 s, deterministic shuffle
    seen = {}
    for r in rows:
        dur = int(r[5]) / 16000
        if 4 <= dur <= 12 and r[0] not in seen and (FLEURS / "test/test" / r[1]).exists():
            seen[r[0]] = dict(id=r[0], wav=str(FLEURS / "test/test" / r[1]), text=r[2], norm=r[3], gender=r[6], dur=dur)
    items = list(seen.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items[:n]


def spectral(wav: np.ndarray, sr: int) -> dict:
    import librosa

    S = np.abs(librosa.stft(wav.astype(np.float32), n_fft=2048)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    p = S.mean(1)
    p = p / (p.sum() + 1e-12)
    cum = np.cumsum(p)
    return {
        "rolloff95": float(f[min(len(f) - 1, np.searchsorted(cum, 0.95))]),
        "rolloff99": float(f[min(len(f) - 1, np.searchsorted(cum, 0.99))]),
        "e4k": float(p[f > 4000].sum()),
        "e7k": float(p[f > 7000].sum()),
    }


def pitch(wav: np.ndarray, sr: int) -> dict:
    import librosa

    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else wav.astype(np.float32)
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=16000, frame_length=1024)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    vflag = (rms > 0.25 * rms.max()) & (f0 > 65) & (f0 < 395)
    f0 = f0[vflag]
    if len(f0) < 10:
        return {"f0_mean": 0.0, "f0_std_st": 0.0, "voiced": 0.0}
    st_ = 12 * np.log2(f0 / np.median(f0))
    return {"f0_mean": float(np.mean(f0)), "f0_std_st": float(np.std(st_)), "voiced": float(np.mean(vflag))}


def word_errors(ref: str, hyp: str) -> dict:
    import jiwer

    r, h = M.canon(ref), M.canon(hyp)
    if not r.strip():
        return {}
    out = jiwer.process_words(r, h)
    rw = r.split()
    last_word_hit = False
    first_word_hit = False
    for chunk in out.alignments[0]:
        if chunk.type != "equal":
            if chunk.ref_end_idx >= len(rw):
                last_word_hit = True
            if chunk.ref_start_idx == 0:
                first_word_hit = True
    return {"wer": out.wer, "sub": out.substitutions, "del": out.deletions, "ins": out.insertions,
            "n_words": len(rw), "last_word_err": last_word_hit, "first_word_err": first_word_hit}


def syllables(text: str) -> int:
    return len(re.findall(r"[ሀ-፿]", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--voice", default="selam")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--t-shift", type=float, default=None)
    ap.add_argument("--position-temperature", type=float, default=None)
    ap.add_argument("--class-temperature", type=float, default=None)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--keep-ethiopic-punct", action="store_true")
    ap.add_argument("--merge-homophones", action="store_true")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--human", action="store_true", help="score the human recordings instead of synthesizing")
    ap.add_argument("--prompts", default="fleurs", choices=["fleurs", "addis"], help="prompt set")
    ap.add_argument("--audio-dir", default=None, help="score existing clips <dir>/<id>.wav instead of synthesizing")
    ap.add_argument("--eq", default=None, help="post EQ preset name (see amharic_tts.audio)")
    ap.add_argument("--out", default=str(ROOT / "benchmark/diag/out"))
    args = ap.parse_args()

    items = load_addis_items(args.n) if args.prompts == "addis" else load_items(args.n)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    tts = None
    if not args.human and not args.audio_dir:
        from amharic_tts import AmharicTTS, NormalizerOptions

        nopt = NormalizerOptions(map_punctuation=not args.keep_ethiopic_punct, merge_homophones=args.merge_homophones)
        kw = dict(num_steps=args.steps, guidance_scale=args.guidance)
        if args.model_path:
            kw["model_path"] = args.model_path
        tts = AmharicTTS(engine="omnivoice", normalizer=nopt, **kw)
    gen_kw = {}
    if args.t_shift is not None:
        gen_kw["t_shift"] = args.t_shift
    if args.position_temperature is not None:
        gen_kw["position_temperature"] = args.position_temperature
    if args.class_temperature is not None:
        gen_kw["class_temperature"] = args.class_temperature
    if args.no_denoise:
        gen_kw["denoise"] = False
    if args.no_postprocess:
        gen_kw["postprocess_output"] = False

    rows = []
    fh = (out / "rows.jsonl").open("w", encoding="utf-8")
    for it in items:
        t0 = time.time()
        if args.human or args.audio_dir:
            src = Path(args.audio_dir) / f"{it['id']}.wav" if args.audio_dir else Path(it["wav"])
            if not src.exists():
                continue
            wav, sr = sf.read(str(src))
            if wav.ndim > 1:
                wav = wav.mean(1)
            wav = wav.astype(np.float32)
            el = 0.0
            spoken = it["text"]
        else:
            res = tts.synthesize(it["text"], voice=args.voice, speed=args.speed, seed=0, **gen_kw)
            wav, sr, el, spoken = res.audio, res.sample_rate, res.elapsed, res.text
            if args.eq:
                from amharic_tts.audio import apply_eq

                wav = apply_eq(wav, sr, args.eq)
            sf.write(str(out / f"{it['id']}.wav"), wav, sr)
        hyp = M.transcribe(wav, sr)
        dur = len(wav) / sr
        row = {"id": it["id"], "text": it["text"], "hyp": hyp, "cer": M.cer(spoken, hyp), **word_errors(spoken, hyp),
               "dur": dur, "syl_per_s": syllables(it["text"]) / dur, "rtf": el / dur if dur else None,
               "utmos": M.utmos(wav, sr), **M.dnsmos(wav, sr), **spectral(wav, sr), **pitch(wav, sr)}
        rows.append(row)
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"{it['id']} cer={row['cer']*100:4.1f} del={row.get('del',0)} sub={row.get('sub',0)} last={row.get('last_word_err')} "
              f"utmos={row['utmos']:.2f} ro99={row['rolloff99']:.0f} f0sd={row['f0_std_st']:.1f} rate={row['syl_per_s']:.1f}", flush=True)
    summary = summarize(rows)
    summary["name"] = args.name
    summary["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, ensure_ascii=False))


def summarize(rows):
    m = lambda k: float(st.mean(r[k] for r in rows if r.get(k) is not None))  # noqa: E731
    s = {k: m(k) for k in ["cer", "wer", "utmos", "SIG", "BAK", "OVRL", "rolloff95", "rolloff99", "e4k", "e7k",
                           "f0_std_st", "voiced", "syl_per_s", "dur"]}
    s["rtf"] = m("rtf") if any(r.get("rtf") for r in rows) else None
    s["del_total"] = sum(r.get("del", 0) for r in rows)
    s["sub_total"] = sum(r.get("sub", 0) for r in rows)
    s["ins_total"] = sum(r.get("ins", 0) for r in rows)
    s["words_total"] = sum(r.get("n_words", 0) for r in rows)
    s["last_word_err_rate"] = float(np.mean([bool(r.get("last_word_err")) for r in rows]))
    s["first_word_err_rate"] = float(np.mean([bool(r.get("first_word_err")) for r in rows]))
    s["n"] = len(rows)
    return s


if __name__ == "__main__":
    main()
