# amharic-tts — open, high-quality Amharic text-to-speech

`amharic-tts` turns Amharic (አማርኛ) text into natural speech, fully offline, with
open weights and open code. It combines:

* **A real Amharic text front-end** (the part commercial systems get wrong):
  numbers, ordinals, decimals, Ethiopic numerals (፳፻፲፱), Ethiopian **and**
  Gregorian dates, times, currency, percentages, ranges, phone numbers, units,
  slash abbreviations (ዶ/ር, ወ/ሮ, ት/ቤት, ጠ/ሚ, ክ/ከተማ, ገ/ማርያም …), Latin acronyms and
  Ethiopic punctuation — 96 unit tests, zero heavy dependencies.
* **OmniVoice-Amharic** ([`Lab-et/omnivoice-amharic`](https://huggingface.co/Lab-et/omnivoice-amharic), Apache-2.0):
  a 0.6 B-parameter discrete-diffusion TTS fine-tuned on ~331 h of Amharic
  speech, 24 kHz output, zero-shot **voice cloning** from a 3–10 s clip.
* **Seven ready-made voices** (4 female, 3 male), chosen by an evidence-based
  casting run over 25 candidate clips and re-verified after synthesis (see DIAGNOSIS.md), plus `auto` for a
  random voice and one-line cloning of your own.
* **Meta MMS-TTS** (`facebook/mms-tts-amh`) as a real-time CPU fallback.
* One-call Python API, a CLI, and a FastAPI server with an
  **OpenAI-compatible `/v1/audio/speech`** endpoint and a web demo.

Benchmark results against Google TTS are in [BENCHMARK.md](BENCHMARK.md)
(summary below).

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[all]"

# speak (models download automatically on first use, ~3.3 GB)
amharic-tts say "ሰላም፣ እንኳን ደህና መጣችሁ። ዛሬ ጥቅምት 15 ቀን 2019 ዓ.ም. ነው።" -o hello.wav
amharic-tts say -f article.txt -o article.mp3 --voice yonas
amharic-tts say "ሰላም" -e mms -o fast.wav              # real-time fallback voice

# see exactly what will be spoken
amharic-tts normalize "ዶ/ር አበበ 1250 ብር በ25% ቅናሽ ከፈለ።"
# → ዶክተር አበበ አንድ ሺህ ሁለት መቶ ሃምሳ ብር በሃያ አምስት በመቶ ቅናሽ ከፈለ.

# voices
amharic-tts voices list
amharic-tts voices add myvoice ref.wav "የሪፈረንስ ድምፁ የሚናገረው ጽሑፍ በትክክል።"

# HTTP API + web demo on http://localhost:8080
amharic-tts serve --port 8080
```

### Python

```python
from amharic_tts import AmharicTTS

tts = AmharicTTS()                                   # OmniVoice-Amharic, auto device
tts.save("የኢትዮጵያ ብሔራዊ ባንክ ዛሬ አዲስ መመሪያ አወጣ።", "news.wav", voice="selam")

res = tts.synthesize("ሰላም ዓለም", voice="meron")
res.audio, res.sample_rate, res.text   # numpy float32, 24000, normalised text

for chunk in tts.stream(long_text):     # progressive playback, chunk by chunk
    player.write(chunk)

tts.to_bytes("ሰላም", fmt="mp3")        # wav / flac / ogg / mp3 / opus
```

### Web studio

```bash
pip install -e ".[omnivoice,server]"      # plus `gtts` / `google-genai` for the compare panel
amharic-tts serve --port 8080             # then open http://localhost:8080
```

The page served at `/` is a small studio: type Amharic, pick one of the seven
preset voices (each card plays its reference clip), set speed / diffusion steps
/ seed, and press **Speak**. Tick **Google TTS** or **Gemini TTS** to hear the
same sentence from those services next to ours (Gemini needs `GEMINI_API_KEY`;
both are fetched live and never required). Every take lands in a history list
with a waveform player and a download link.

### HTTP (OpenAI-compatible)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input": "እንኳን ደህና መጡ።", "voice": "liya", "response_format": "mp3", "speed": 1.0}' \
  -o welcome.mp3
```

`GET /v1/voices`, `POST /v1/normalize`, `GET /health` are also available.

---

## Why it beats Google's Amharic voice

| | Google TTS (Translate / gTTS `am`) | amharic-tts |
|---|---|---|
| Acoustic model | single proprietary voice | 0.6 B diffusion model trained on 331 h Amharic, 24 kHz |
| Voices | 1 | 7 presets + unlimited zero-shot cloning + random |
| Slash abbreviations (ት/ቤት, ጠ/ሚ, ክ/ከተማ) | spelled letter-by-letter or **dropped** | expanded correctly |
| Ethiopic numerals above ፻ (፳፻፲፱ = 2019) | wrong (“ሁለት መቶ አስራ ዘጠኝ”) | correct |
| Date order (ጥቅምት 15 ቀን 2019) | reorders to *day month* and drops ቀን | kept in Amharic order |
| Ethiopian vs Gregorian calendar | guesses | decided from ዓ.ም. / እ.ኤ.አ. context |
| Numeric dates 15/02/2016 ዓ.ም. | — | ጥቅምት አስራ አምስት ቀን ሁለት ሺህ አስራ ስድስት … |
| Currency 12.50 ብር | ok | አስራ ሁለት ብር ከሃምሳ ሳንቲም |
| Speed / pronunciation control | none | speed, diffusion steps, seed, custom lexicon via `extra_abbreviations` |
| Offline / private | no | yes |
| License | proprietary | Apache-2.0 code, Apache-2.0 model |

Measured on 41 benchmark utterances (details in [BENCHMARK.md](BENCHMARK.md)),
ASR round-trip character error rate with an independent Amharic recogniser:

| | plain prose | numbers / dates / abbreviations | all |
|---|---:|---:|---:|
| Google TTS | 1.4 % | 14.3 % | 7.7 % |
| MMS-TTS | 8.3 % | 4.5 % | 6.3 % |
| **amharic-tts** | 2.2 % | **2.6 %** | **2.3 %** |

On Addis AI's own public 100-prompt benchmark, scored with the same
independent recogniser, amharic-tts reaches **CER 3.6 % vs 5.6 % for Addis
Voice 2** (better on 58 prompts, tied on 17) with half the sentence-end
truncations. Delivery was then calibrated to Addis Voice 2's calm pace
(5.5 syllables/s, natural phrase pauses) and the voices re-cast by measurement;
the full diagnosis and every number are in [DIAGNOSIS.md](DIAGNOSIS.md).

What the literature says about Amharic gemination / epenthesis, OmniVoice-class
models and audio-LLM judges, with a ranked plan: [RESEARCH.md](RESEARCH.md).

A second, independent "ear": `benchmark/gemini_judge.py` has a Gemini audio
model transcribe, rate, A/B-compare (both orders) and rank all systems blind;
see [benchmark/GEMINI_JUDGE.md](benchmark/GEMINI_JUDGE.md).

Listen for yourself: [`samples/compare/`](samples/compare) has the same
sentences spoken by amharic-tts, Google and MMS, and [`samples/`](samples)
has every preset voice on four texts.

---

## Architecture

```
text ──► AmharicNormalizer ──► chunk_text ──► engine.synthesize ──► post-process ──► wav/mp3
          (numbers, dates,      (sentence       OmniVoice-Amharic       (fade, pauses,
           abbrevs, acronyms,    groups ≤220     or MMS-VITS             loudness, resample)
           punctuation)          chars)
```

* `amharic_tts/text/` — normaliser (`normalizer.py`), number verbaliser
  (`numbers.py`), tables (`abbreviations.py`), segmentation (`segment.py`).
* `amharic_tts/engines/` — `OmniVoiceEngine` (primary) and `MMSEngine` (fallback).
* `amharic_tts/voices/` — preset reference clips + `voices.json`; user voices live
  in `~/.amharic_tts/voices` (override with `AMHARIC_TTS_HOME`). Encoded voice
  prompts are cached so each voice is analysed only once.
* `amharic_tts/tts.py` — `AmharicTTS` façade, `encode_audio`, `write_audio`.
* `amharic_tts/server.py` — FastAPI app; `amharic_tts/cli.py` — CLI.
* `benchmark/` — reproducible evaluation vs Google TTS and MMS.

### Normaliser options

```python
from amharic_tts import AmharicTTS, NormalizerOptions
tts = AmharicTTS(normalizer=NormalizerOptions(
    default_calendar="gregorian",        # month names for bare numeric dates
    merge_homophones=False,              # ሐ/ኀ→ሀ, ሠ→ሰ, ፀ→ጸ, ዐ→አ
    spell_acronyms=True,                 # BBC → ቢ ቢ ሲ
    extra_abbreviations={"አ.ዩ": "አዲስ አበባ ዩኒቨርሲቲ"},
))
```

### Performance

OmniVoice-Amharic is a 600 M-parameter model. Measured on this project's
4-core CPU box: real-time factor ≈ 7 at 16 diffusion steps and the calibrated
tempo (≈ 70 s for a 10 s sentence); a single T4 GPU gives well under 1.
32 steps measured no better than 16 (see BENCHMARK.md), and `--steps 8` is
noticeably faster with a small quality cost; MMS (`-e mms`) runs faster than
real time on CPU. Long inputs are synthesised chunk by chunk,
so memory stays flat for arbitrarily long documents.

---

## Licenses & attribution

* Code: Apache-2.0.
* OmniVoice-Amharic weights: Apache-2.0 (Lab-et / african-low-resource; base
  model k2-fsa/OmniVoice, audio tokenizer HiggsAudio-v2).
* Preset voice clips: Google FLEURS `am_et` (CC-BY-4.0) and Dataset.ET
  (`snapwre/amharic-speech`, CC-BY-4.0); attribution per voice in
  `amharic_tts/voices/voices.json`.
* MMS-TTS-amh weights: CC-BY-NC-4.0 (non-commercial) — only used if you pick `-e mms`.
* Benchmark ASR: `snapwre/hohe-asr-amharic`, CC-BY-4.0.

Clone only voices you have permission to use.
