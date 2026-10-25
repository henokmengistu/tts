#!/usr/bin/env python
"""Generate showcase samples for every preset voice into samples/."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from amharic_tts import AmharicTTS  # noqa: E402

SAMPLES = {
    "greeting": "ሰላም፣ እንኳን ደህና መጣችሁ። ይህ በነጻ ሶፍትዌር የተሠራ የአማርኛ የንግግር ድምፅ ነው።",
    "news": "የኢትዮጵያ ብሔራዊ ባንክ ዛሬ አዲስ መመሪያ አወጣ። መመሪያው የውጭ ምንዛሬ ግብይትን የሚመለከት ሲሆን፣ ከነገ ጀምሮ ተግባራዊ ይሆናል።",
    "numbers_dates": "ዛሬ ጥቅምት 15 ቀን 2019 ዓ.ም. ነው። ዶ/ር አበበ በት/ቤቱ 1250 ብር ለገሱ፤ ይህም ከአምናው በ25% ብልጫ አለው።",
    "story": "ድሮ ድሮ በአንድ ትንሽ መንደር ውስጥ አንድ ብልህ አዛውንት ይኖሩ ነበር። ልጆቹ በየቀኑ ማታ ማታ ተሰብስበው ተረት ያዳምጡ ነበር።",
}
OUT = Path(__file__).parent.parent / "samples"


def main():
    OUT.mkdir(exist_ok=True)
    tts = AmharicTTS(engine="omnivoice", num_steps=16)
    meta = []
    for voice in tts.list_voices():
        for key, text in SAMPLES.items():
            path = OUT / f"{voice}_{key}.mp3"
            t0 = time.time()
            res = tts.save(text, path, voice=voice, seed=0)
            meta.append({"file": path.name, "voice": voice, "text": text, "spoken": res.text,
                         "seconds": round(res.duration, 2), "elapsed": round(time.time() - t0, 1)})
            print(meta[-1], flush=True)
    (OUT / "samples.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
