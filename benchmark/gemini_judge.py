#!/usr/bin/env python
"""Use a Gemini audio model as an independent "ear" to evaluate TTS systems.

Four blind protocols, each with the bias controls recommended by the audio
LLM-as-judge literature (AudioJudge, EACL 2026; "Louder, Longer, Livelier";
protocol-shortcut audits):

  transcribe  - the model hears ONE clip, no text, and writes a verbatim Ethiopic
                transcript.  CER against the intended text = intelligibility as
                heard by this ear (independent of the wav2vec recogniser used
                elsewhere in this repo).
  rate        - ONE clip + the intended text; JSON scores for naturalness,
                intelligibility, voice quality, prosody, overall (1-5), plus a
                list of missed / mispronounced / repeated words and flags
                (robotic, cut off, noise).  Absolute scores are the least
                reliable protocol, so they are reported alongside the others.
  pair        - TWO clips of the same sentence, "A" and "B"; asked which is
                better overall and for naturalness.  Every pair is judged in
                BOTH orders; a preference only counts when the two orders
                agree, otherwise it is a tie.  Win rates vs. a reference system.
  rank        - ALL systems' clips of one sentence in ONE request, randomly
                shuffled and labelled with neutral letters; the model returns a
                full ranking.  Repeated with a second independent shuffle; the
                average rank (Borda) is reported.

Common controls: all clips are loudness-normalised to the same RMS and
converted to 16 kHz mono WAV before upload (removes the "louder = better"
shortcut), system names never appear in prompts, temperature is low, and every
response is cached in results.jsonl so runs are resumable and reproducible.

Usage (needs GEMINI_API_KEY in the environment):

  python benchmark/gemini_judge.py --list-models
  python benchmark/gemini_judge.py --n 30 --protocols transcribe,rate,pair,rank
  python benchmark/gemini_judge.py --n 30 --protocols rank --model gemini-3.8-flash
  python benchmark/gemini_judge.py --voices   # rank our own preset voices on the sample texts

Systems are read from --systems (name=dir); defaults cover the Addis AI
benchmark prompts: human, addis, google, gemini-tts, azure, mms, ours.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import statistics as st
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

ADDIS = Path(os.environ.get("ADDIS_BENCH_DIR", "/home/user/data/addis_bench"))
DEFAULT_SYSTEMS = {
    "human": str(ADDIS / "audio/human-reference"),
    "addis": str(ADDIS / "audio/addis"),
    "google": str(ADDIS / "audio/google-gtts"),
    "gemini-tts": str(ADDIS / "audio/gemini"),
    "azure": str(ADDIS / "audio/azure-ameha"),
    "mms": str(ADDIS / "audio/meta-mms"),
    "ours": str(ROOT / "benchmark/diag/out/addis_final"),
}
DEFAULT_MODEL = "gemini-4-argon"
LABELS = "ABCDEFGHJKLMNP"  # neutral labels (no I/O to avoid confusion)

# --------------------------------------------------------------------------- text
_HOMO = {}
for a, b in [(0x1210, 0x1200), (0x1280, 0x1200), (0x1220, 0x1230), (0x1340, 0x1338), (0x12D0, 0x12A0)]:
    for k in range(7):
        _HOMO[chr(a + k)] = chr(b + k)
for cp in range(0x1288, 0x1290):
    _HOMO.pop(chr(cp), None)
_HOMO_T = str.maketrans(_HOMO)
_HOMO2 = str.maketrans({"ሃ": "ሀ", "ኣ": "አ", "ዓ": "አ", "ዐ": "አ", "ሓ": "ሀ", "ኻ": "ሀ", "ሐ": "ሀ", "ኀ": "ሀ", "ኃ": "ሀ"})


def canon(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").translate(_HOMO_T).translate(_HOMO2)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = canon(ref), canon(hyp)
    return float(jiwer.cer(r, h)) if r else 0.0


# --------------------------------------------------------------------------- audio
def load_prepared(path: Path, target_dbfs: float = -20.0) -> bytes:
    """Load any audio file -> loudness-normalised 16 kHz mono 16-bit WAV bytes."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(path), sr=16000, mono=True)
    y = y.astype(np.float32)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if rms > 1e-6:
        y = y * (10 ** (target_dbfs / 20) / rms)
        peak = float(np.max(np.abs(y)))
        if peak > 0.97:
            y = y * (0.97 / peak)
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def find_clip(d: Path, pid: str) -> Path | None:
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        p = d / f"{pid}{ext}"
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- prompts
RUBRIC = (
    "You are an expert native Amharic listener evaluating text-to-speech quality. "
    "Judge only what you hear. Be strict and consistent. Scale 1-5: 1 = very bad, 2 = poor, "
    "3 = fair, 4 = good, 5 = excellent (indistinguishable from a professional human speaker)."
)

TRANSCRIBE_PROMPT = (
    "Transcribe this Amharic speech verbatim in Ethiopic (Ge'ez) script. Write numbers as the words "
    "you hear. Do not add punctuation, comments or translation. If a word is unintelligible write [?]."
)

RATE_SCHEMA = {
    "type": "object",
    "properties": {
        "naturalness": {"type": "integer"},
        "intelligibility": {"type": "integer"},
        "voice_quality": {"type": "integer"},
        "prosody": {"type": "integer"},
        "overall": {"type": "integer"},
        "robotic": {"type": "boolean"},
        "cut_off": {"type": "boolean"},
        "noisy_or_distorted": {"type": "boolean"},
        "missed_words": {"type": "array", "items": {"type": "string"}},
        "mispronounced_words": {"type": "array", "items": {"type": "string"}},
        "repeated_or_stuttered": {"type": "array", "items": {"type": "string"}},
        "comment": {"type": "string"},
    },
    "required": ["naturalness", "intelligibility", "voice_quality", "prosody", "overall", "robotic", "cut_off",
                 "noisy_or_distorted", "missed_words", "mispronounced_words", "repeated_or_stuttered", "comment"],
}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "better_overall": {"type": "string", "enum": ["A", "B", "tie"]},
        "more_natural": {"type": "string", "enum": ["A", "B", "tie"]},
        "more_intelligible": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["better_overall", "more_natural", "more_intelligible", "reason"],
}


def rank_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "ranking_overall": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n},
            "ranking_naturalness": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n},
            "sounds_human": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["ranking_overall", "ranking_naturalness", "sounds_human", "notes"],
    }


# --------------------------------------------------------------------------- client
class Judge:
    def __init__(self, model: str, temperature: float = 0.2, dry_run: bool = False, cache: Path | None = None):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.cache_path = cache
        self.cache: dict[str, dict] = {}
        if cache and cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
                except json.JSONDecodeError:
                    pass
        self.client = None
        if not dry_run:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise SystemExit("set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment")
            self.client = genai.Client(api_key=api_key)

    def list_models(self):
        for m in self.client.models.list():
            name = getattr(m, "name", "")
            if "gemini" in name:
                print(name)

    def _call(self, parts: list, schema: dict | None, key: str, tries: int = 6) -> dict:
        if key in self.cache:
            return self.cache[key]["response"]
        if self.dry_run:
            return {"dry_run": True}
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        )
        delay = 4.0
        last = None
        for attempt in range(tries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts, config=config)
                text = resp.text or ""
                out = json.loads(text) if schema else {"text": text.strip()}
                rec = {"key": key, "model": self.model, "response": out, "ts": time.time()}
                self.cache[key] = rec
                if self.cache_path:
                    with self.cache_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return out
            except Exception as e:  # noqa: BLE001 - rate limits, transient 5xx, JSON errors
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError(f"Gemini call failed after {tries} tries: {last}")

    @staticmethod
    def audio_part(wav_bytes: bytes):
        from google.genai import types

        return types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")

    # -- protocols ----------------------------------------------------------
    def transcribe(self, wav: bytes, key: str) -> str:
        out = self._call([TRANSCRIBE_PROMPT, self.audio_part(wav)], None, key)
        return out.get("text", "")

    def rate(self, wav: bytes, text: str, key: str) -> dict:
        prompt = (f"{RUBRIC}\n\nThe speaker was supposed to read exactly this Amharic text:\n\"{text}\"\n\n"
                  "Listen to the clip and return JSON with: scores (naturalness, intelligibility, voice_quality, "
                  "prosody, overall) from 1 to 5; booleans robotic, cut_off, noisy_or_distorted; lists of words "
                  "that were missed, mispronounced, or repeated/stuttered (copy them from the text); a short comment.")
        return self._call([prompt, self.audio_part(wav)], RATE_SCHEMA, key)

    def pair(self, wav_a: bytes, wav_b: bytes, text: str, key: str) -> dict:
        prompt = (f"{RUBRIC}\n\nBoth clips are meant to read this Amharic text:\n\"{text}\"\n\n"
                  "Clip A is first, clip B is second. Which clip is better overall, which is more natural, and which is "
                  "more intelligible? Answer 'tie' only if you really cannot tell. Return JSON.")
        return self._call([prompt, "Clip A:", self.audio_part(wav_a), "Clip B:", self.audio_part(wav_b)], PAIR_SCHEMA, key)

    def rank(self, wavs: list[bytes], labels: list[str], text: str, key: str) -> dict:
        prompt = (f"{RUBRIC}\n\nYou will hear {len(wavs)} different renderings of the same Amharic sentence:\n\"{text}\"\n\n"
                  f"They are labelled {', '.join(labels)} in the order played. Rank ALL of them from best to worst, "
                  "once for overall quality and once for naturalness (how human it sounds). Also list the labels that "
                  "could pass as a real human recording. Return JSON with the label lists.")
        parts: list = [prompt]
        for lab, w in zip(labels, wavs):
            parts.append(f"Clip {lab}:")
            parts.append(self.audio_part(w))
        return self._call(parts, rank_schema(len(wavs)), key)


# --------------------------------------------------------------------------- runner
def load_prompts(n: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in (ADDIS / "data/prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [{"id": r["case_id"], "text": r["text"], "stratum": r.get("length_stratum"), "source": r.get("source")} for r in rows[:n]]


def bootstrap_ci(vals: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = [float(np.mean(rng.choice(arr, len(arr)))) for _ in range(iters)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=30, help="number of prompts (random, fixed seed)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--protocols", default="transcribe,rate,pair,rank")
    ap.add_argument("--systems", default=None, help="comma list of name=dir; default: Addis benchmark systems + ours")
    ap.add_argument("--reference", default="ours", help="system used as the anchor in pairwise comparisons")
    ap.add_argument("--rank-repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "benchmark/out/gemini_judge"))
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build all requests, call nothing (checks data + prompts)")
    ap.add_argument("--voices", action="store_true", help="rank our own preset voices on samples/ instead of systems")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = Judge(args.model, args.temperature, args.dry_run, cache=out / "results.jsonl")
    if args.list_models:
        judge.list_models()
        return

    # ---- build the item list: prompt -> {system: wav bytes}
    if args.voices:
        meta = json.loads((ROOT / "samples/samples.json").read_text(encoding="utf-8"))
        texts = {}
        for m in meta:
            texts.setdefault(m["file"].split("_", 1)[1].rsplit(".", 1)[0], {"text": m["text"], "clips": {}})
            texts[m["file"].split("_", 1)[1].rsplit(".", 1)[0]]["clips"][m["voice"]] = ROOT / "samples" / m["file"]
        prompts = [{"id": k, "text": v["text"], "clips": v["clips"]} for k, v in texts.items()]
        systems = sorted({v for p in prompts for v in p["clips"]})
        args.reference = args.reference if args.reference in systems else systems[0]
    else:
        systems_map = dict(s.split("=", 1) for s in args.systems.split(",")) if args.systems else DEFAULT_SYSTEMS
        systems_map = {k: Path(v) for k, v in systems_map.items() if Path(v).exists()}
        systems = list(systems_map)
        prompts = []
        for p in load_prompts(args.n, args.seed):
            clips = {s: find_clip(d, p["id"]) for s, d in systems_map.items()}
            clips = {s: c for s, c in clips.items() if c}
            if len(clips) >= 2:
                prompts.append({**p, "clips": clips})
    print(f"{len(prompts)} prompts, systems: {systems}", flush=True)
    wavs: dict[tuple[str, str], bytes] = {}
    for p in prompts:
        for s, c in p["clips"].items():
            wavs[(p["id"], s)] = load_prepared(Path(c))
    total_mb = sum(len(b) for b in wavs.values()) / 1e6
    print(f"prepared {len(wavs)} clips ({total_mb:.0f} MB of 16 kHz wav)", flush=True)

    protocols = set(args.protocols.split(","))
    jobs = []
    rng = random.Random(args.seed)
    for p in prompts:
        for s in p["clips"]:
            if "transcribe" in protocols:
                jobs.append(("transcribe", p, s, None, None))
            if "rate" in protocols:
                jobs.append(("rate", p, s, None, None))
        if "pair" in protocols and args.reference in p["clips"]:
            for s in p["clips"]:
                if s == args.reference:
                    continue
                jobs.append(("pair", p, args.reference, s, "ref_first"))
                jobs.append(("pair", p, args.reference, s, "ref_second"))
        if "rank" in protocols and len(p["clips"]) >= 3:
            for rep in range(args.rank_repeats):
                order = list(p["clips"])
                rng.shuffle(order)
                jobs.append(("rank", p, None, None, (rep, tuple(order))))
    print(f"{len(jobs)} judge requests", flush=True)

    def run(job):
        kind, p, s, s2, extra = job
        text = p["text"]
        if kind == "transcribe":
            key = f"transcribe|{args.model}|{p['id']}|{s}"
            hyp = judge.transcribe(wavs[(p["id"], s)], key)
            return {"kind": kind, "id": p["id"], "system": s, "hyp": hyp, "cer": cer(text, hyp) if hyp else None}
        if kind == "rate":
            key = f"rate|{args.model}|{p['id']}|{s}"
            r = judge.rate(wavs[(p["id"], s)], text, key)
            return {"kind": kind, "id": p["id"], "system": s, **r}
        if kind == "pair":
            a, b = (s, s2) if extra == "ref_first" else (s2, s)
            key = f"pair|{args.model}|{p['id']}|{a}|{b}"
            r = judge.pair(wavs[(p["id"], a)], wavs[(p["id"], b)], text, key)
            return {"kind": kind, "id": p["id"], "ref": s, "other": s2, "order": extra, "A": a, "B": b, **r}
        if kind == "rank":
            rep, order = extra
            labels = list(LABELS[: len(order)])
            key = f"rank|{args.model}|{p['id']}|{rep}|{'/'.join(order)}"
            r = judge.rank([wavs[(p["id"], s_)] for s_ in order], labels, text, key)
            lab2sys = dict(zip(labels, order))
            return {"kind": kind, "id": p["id"], "rep": rep, "order": list(order), "labels": labels,
                    "ranking_overall": [lab2sys.get(l, l) for l in r.get("ranking_overall", [])],
                    "ranking_naturalness": [lab2sys.get(l, l) for l in r.get("ranking_naturalness", [])],
                    "sounds_human": [lab2sys.get(l, l) for l in r.get("sounds_human", [])], "notes": r.get("notes", "")}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print("ERROR", e, file=sys.stderr, flush=True)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    (out / "judgements.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.dry_run:
        print("dry run complete: data and prompts are ready")
        return
    report(results, systems, args, out / "report.md")


def report(results: list[dict], systems: list[str], args, path: Path) -> None:
    L = [f"# Gemini-as-an-ear evaluation ({args.model})", "",
         f"{len({r['id'] for r in results})} prompts, systems: {', '.join(systems)}. Clips loudness-normalised, "
         "names hidden, pairwise judged in both orders, rankings repeated with independent shuffles.", ""]
    # transcription
    tr = [r for r in results if r["kind"] == "transcribe" and r.get("cer") is not None]
    if tr:
        L += ["## Intelligibility as heard by Gemini (transcription CER, lower is better)", "",
              "| system | n | CER | 95 % CI |", "|---|---:|---:|---:|"]
        for s in systems:
            v = [r["cer"] for r in tr if r["system"] == s]
            if v:
                lo, hi = bootstrap_ci(v)
                L.append(f"| {s} | {len(v)} | {st.mean(v)*100:.1f} % | {lo*100:.1f}–{hi*100:.1f} |")
        L.append("")
    # ratings
    ra = [r for r in results if r["kind"] == "rate" and "overall" in r]
    if ra:
        L += ["## Blind absolute ratings (1-5, higher is better)", "",
              "| system | n | overall | naturalness | intelligibility | voice quality | prosody | robotic | cut off | missed words / clip | mispronounced / clip |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in systems:
            v = [r for r in ra if r["system"] == s]
            if not v:
                continue
            m = lambda k: st.mean(float(r.get(k, 0) or 0) for r in v)  # noqa: E731
            lo, hi = bootstrap_ci([float(r.get("overall", 0) or 0) for r in v])
            L.append(f"| {s} | {len(v)} | {m('overall'):.2f} ({lo:.2f}–{hi:.2f}) | {m('naturalness'):.2f} | {m('intelligibility'):.2f} | "
                     f"{m('voice_quality'):.2f} | {m('prosody'):.2f} | {st.mean(bool(r.get('robotic')) for r in v)*100:.0f} % | "
                     f"{st.mean(bool(r.get('cut_off')) for r in v)*100:.0f} % | {st.mean(len(r.get('missed_words') or []) for r in v):.2f} | "
                     f"{st.mean(len(r.get('mispronounced_words') or []) for r in v):.2f} |")
        L.append("")
    # pairwise
    pa = [r for r in results if r["kind"] == "pair"]
    if pa:
        L += [f"## Pairwise vs `{args.reference}` (both orders; counted only when both orders agree)", "",
              "| opponent | n pairs | ref wins | opponent wins | tie / inconsistent | ref win rate (decided) |", "|---|---:|---:|---:|---:|---:|"]
        by = {}
        for r in pa:
            by.setdefault((r["id"], r["other"]), {})[r["order"]] = r
        for s in systems:
            if s == args.reference:
                continue
            w = l = t = 0
            for (pid, other), d in by.items():
                if other != s or len(d) < 2:
                    continue
                verdicts = []
                for o, r in d.items():
                    v = r.get("better_overall")
                    verdicts.append("ref" if v == ("A" if r["A"] == args.reference else "B") else ("other" if v in ("A", "B") else "tie"))
                if verdicts == ["ref", "ref"]:
                    w += 1
                elif verdicts == ["other", "other"]:
                    l += 1
                else:
                    t += 1
            n = w + l + t
            if n:
                L.append(f"| {s} | {n} | {w} | {l} | {t} | {(w/(w+l)*100 if w+l else float('nan')):.0f} % |")
        L.append("")
    # ranking
    rk = [r for r in results if r["kind"] == "rank" and r.get("ranking_overall")]
    if rk:
        L += ["## Rank-all (average rank over prompts and shuffles; 1 = best)", "",
              "| system | mean rank overall | mean rank naturalness | ranked 1st (overall) | 'could pass as human' |", "|---|---:|---:|---:|---:|"]
        rows = {s: {"o": [], "n": [], "first": 0, "human": 0, "seen": 0} for s in systems}
        for r in rk:
            for i, s in enumerate(r["ranking_overall"], 1):
                if s in rows:
                    rows[s]["o"].append(i)
                    rows[s]["first"] += i == 1
            for i, s in enumerate(r.get("ranking_naturalness", []), 1):
                if s in rows:
                    rows[s]["n"].append(i)
            for s in r["order"]:
                if s in rows:
                    rows[s]["seen"] += 1
                    rows[s]["human"] += s in (r.get("sounds_human") or [])
        order = sorted((s for s in systems if rows[s]["o"]), key=lambda s: st.mean(rows[s]["o"]))
        for s in order:
            d = rows[s]
            L.append(f"| {s} | {st.mean(d['o']):.2f} | {st.mean(d['n']) if d['n'] else float('nan'):.2f} | "
                     f"{d['first']/len(d['o'])*100:.0f} % | {d['human']/max(1,d['seen'])*100:.0f} % |")
        L.append("")
    # qualitative: most frequent issues for ours
    if ra:
        L += ["## Issues Gemini reported most often", ""]
        for s in systems:
            v = [r for r in ra if r["system"] == s]
            miss = [w for r in v for w in (r.get("missed_words") or [])]
            mis = [w for r in v for w in (r.get("mispronounced_words") or [])]
            rep = [w for r in v for w in (r.get("repeated_or_stuttered") or [])]
            L.append(f"- **{s}**: missed {len(miss)}, mispronounced {len(mis)}, repeated {len(rep)} words over {len(v)} clips"
                     + (f"; e.g. {', '.join((mis+miss)[:6])}" if (mis or miss) else ""))
    path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
