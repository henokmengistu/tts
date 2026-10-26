#!/usr/bin/env python
"""Install the cast voices into amharic_tts/voices (clips + voices.json with attribution)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
VOICES = ROOT / "amharic_tts/voices"
CAST = ROOT / "benchmark/diag/out/casting"
CANDS = {c["name"]: c for c in json.loads(Path("/home/user/data/casting_candidates.json").read_text(encoding="utf-8"))}
RESULTS = {}
for line in (CAST / "results.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    RESULTS[r["name"]] = r

# name -> (candidate, gender, description, default speed)
LINEUP = {
    "selam": ("cur_selam", "female", "clear, neutral female news-reading voice (default)"),
    "meron": ("et_f21b5e", "female", "bright, warm female voice"),
    "liya": ("et_8b71aa", "female", "light, friendly female voice"),
    "hana": ("cur_hana", "female", "soft female voice, slightly slower delivery"),
    "yonas": ("et_58d897", "male", "clear, bright male voice"),
    "biruk": ("et_12b419", "male", "neutral male voice, steady delivery"),
    "nahom": ("et_dddcc0", "male", "deeper male voice, calm"),
    "tesfa": ("wx_am-066", "male", "low male narrator voice (studio read speech)"),
}


def main():
    out = {"license": "Reference clips: FLEURS (CC-BY-4.0), Dataset.ET / snapwre/amharic-speech (CC-BY-4.0), "
                      "WaxalNLP via addisai/amharic-tts-benchmark (CC-BY-SA-4.0). See each voice's 'source'.",
           "voices": []}
    for name, (cand, gender, desc) in LINEUP.items():
        c = CANDS[cand]
        r = RESULTS.get(cand, {})
        src = Path(c["path"])
        ref_text = c["text"]
        trimmed = CAST / f"{cand}_ref.wav"
        if trimmed.exists():  # long clip was cut at a silence boundary during casting
            src = trimmed
            ref_text = r.get("ref_text_used") or ref_text
        y, sr = sf.read(str(src))
        if y.ndim > 1:
            y = y.mean(1)
        dst = VOICES / f"{name}.wav"
        sf.write(str(dst), y.astype(np.float32), sr, subtype="PCM_16")
        out["voices"].append({
            "name": name, "audio": dst.name, "gender": gender, "description": desc, "ref_text": ref_text,
            "source": c.get("source", ""), "license": c.get("license", ""),
            "casting": {k: round(r[k], 3) for k in ("cer", "utmos", "OVRL", "e4k", "rate", "score") if k in r},
        })
        print(name, cand, f"{len(y)/sr:.1f}s", r.get("score"))
    # remove retired clips
    keep = {v["audio"] for v in out["voices"]}
    for f in VOICES.glob("*.wav"):
        if f.name not in keep:
            f.unlink()
            print("removed", f.name)
    (VOICES / "voices.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
