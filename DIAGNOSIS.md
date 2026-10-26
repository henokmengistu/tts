# Quality diagnosis: "sometimes robotic, sometimes misses words"

This document records what was measured, what was confirmed, and what was
changed. Everything is reproducible with `benchmark/diag/diagnose.py`
(per-utterance JSON in `benchmark/diag/out/<run>/rows.jsonl`).

## Method

* **Unseen text with human references.** 40 random sentences from the FLEURS
  Amharic *test* split (never used for voice prompts). The human recording of
  each sentence is scored with exactly the same pipeline, which gives the
  recogniser's own error floor.
* **Addis AI's public benchmark.** The 100 prompts of
  [`addisai/amharic-tts-benchmark`](https://huggingface.co/datasets/addisai/amharic-tts-benchmark)
  (FLEURS / Waxal / Horn-ASR, short/medium/long). Their archived clips of
  Addis Voice 2, the human references, Meta MMS, Gemini and Azure are scored with
  our metrics, and our system is run on the same 100 prompts.
* **Metrics per utterance**
  * ASR (`snapwre/hohe-asr-amharic`) word alignment: substitutions,
    **deletions** (missed words), insertions, CER/WER, whether the first or
    last word is wrong (truncation).
  * Spectral roll-off (95/99 %), energy above 4 kHz and 7 kHz (dull/muffled
    sound).
  * log-F0 standard deviation in semitones (flat = monotone), voiced fraction.
  * speaking rate in Ethiopic syllables per second, duration ratio to the human
    recording of the same sentence.
  * UTMOS and DNSMOS P.835 predicted MOS; RTF.

## Confirmed findings (40 unseen sentences, voice `selam`, 16 steps)

| | human recordings | amharic-tts default |
|---|---:|---:|
| CER / WER | 6.9 % / 20.0 % | **2.6 % / 11.4 %** |
| deleted words (of ~590) | 13 | 20 |
| substituted words | 83 | 46 |
| last-word error rate | 2.5 % | 2.5 % |
| energy above 4 kHz | 4.5 % | **0.3 %** |
| spectral roll-off 95 % | 2.1 kHz | **1.0 kHz** |
| log-F0 std (semitones) | 3.5 | 3.4 |
| speaking rate (syl/s, edge silence removed) | 7.6 | **8.1–8.3** |
| duration vs human, same sentence | 1.00 | **0.73** |
| UTMOS / DNSMOS OVRL | 2.95 / 2.84 | 3.14 / 3.21 |

1. **Intelligibility is not the problem.** By an independent recogniser our
   output is *more* intelligible than the FLEURS human recordings; end-of-sentence
   truncation is at the human floor (2.5 %).
2. **"Missing words" are real but rare (≈3 % of words) and specific.** Reading the
   alignments: (a) the conjunction እና is merged into the previous word as the
   clitic -ና (`ከአለቶች እና` → `ከአለቶችና`), which is how Amharic is actually spoken
   but counts as a deletion; (b) an occasional word-final consonant is swallowed
   (`ክረስት` → `ክረስ`); (c) rare foreign proper nouns are mangled
   (`ዱንላፕ` → `ዱላ`). All three are symptoms of fast delivery.
3. **Delivery is too fast.** The model's duration estimate yields utterances
   27 % shorter than the human reading of the same sentence; even after removing
   the humans' edge silence we are ~8 % faster. Fast speech is what listeners
   describe as rushed/robotic and is where the swallowed syllables come from.
   → tempo ablations (speed 0.85 / 0.75) are measured below.
4. **The sound is muffled, not noisy.** Energy above 4 kHz is 15× lower than in
   real speech (−9 dB at 5–6 kHz, −15 dB at 6–7 kHz, −29 dB at 7–8 kHz
   relative to human). Fricatives (ስ, ሽ, ፍ, ጽ) lose their brightness, which is
   the "robotic/telephone" character. Pitch dynamics are *not* flat.
5. **The muffled spectrum is intrinsic to the Amharic checkpoint.** It was
   trained on 16 kHz (WaxalNLP, FLEURS-like) and 8 kHz (Leyu) corpora. A 24 kHz
   full-band reference clip restores only the band above 7 kHz (0.001 % →
   0.087 % energy) and leaves 4–7 kHz unchanged; the Chatterbox-Amharic adapter
   (also trained on WaxalNLP) shows the same roll-off. No prompt, step count or
   guidance setting can add detail the model never learned.
6. **int8 quantisation is not an option** (CER 0 % → 16 %, UTMOS 3.3 → 1.4).

## Alternative checkpoints and engines (same 20 unseen sentences)

| system | CER | WER | deleted words | UTMOS | DNSMOS OVRL | energy >4 kHz | log-F0 std | RTF (CPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Lab-et/omnivoice-amharic** (ours, default) | 2.6 % | 11.4 % | 20 / 595 | 3.14 | 3.21 | 0.3 % | 3.36 st | 6.7 |
| gheero-Leyu/amharic-omnivoice-tts | 2.9 % | 13.5 % | 10 / ~300 | 3.11 | 3.18 | 0.3 % | 3.38 st | 6.9 |
| k2-fsa/OmniVoice base (12.8 h Amharic) | 4.5 % | 18.9 % | 12 / ~300 | 3.13 | 3.16 | 0.3 % | **2.54 st** (monotone) | 15.1 |
| natidotnet/chatterbox-amharic (LoRA on Chatterbox v3) | 0–6 % on 2 test sentences | – | – | 3.2–3.5 | 3.0–3.4 | 0.1 % | – | 17–26 |

The Leyu checkpoint is the same training run re-uploaded (identical numbers
within noise). The base model is less intelligible and noticeably flatter in
pitch. Chatterbox-Amharic is intelligible and its stock 24 kHz decoder does
not help: the output is just as band-limited (the LoRA was trained on the same
16 kHz WaxalNLP data), and autoregressive decoding is 3× slower on CPU. The
Amharic OmniVoice fine-tune stays the engine; the Chatterbox loader is kept in
`amharic_tts/engines/chatterbox_engine.py` as an optional GPU-oriented candidate.

## Ablations

ABLATION_TABLE

## Comparison on Addis AI's benchmark prompts

All 100 prompts of `addisai/amharic-tts-benchmark`; Addis Voice 2 clips are
the ones Addis AI archived; "human" = the 67 original recordings that ship as
WAV. Same recogniser and metrics for every row.

| | amharic-tts (selam, 16 steps, default tempo) | Addis Voice 2 | human recordings |
|---|---:|---:|---:|
| CER / WER | **3.6 % / 13.4 %** | 5.6 % / 17.1 % | 9.5 % / 23.8 % |
| prompts won on CER (of 100) | **58** (17 ties) | 25 | – |
| deleted words | 60 / 2456 | 52 / 2436 | 76 / 1545 |
| last-word error rate | **5 %** | 10 % | 16 % |
| UTMOS | 3.06 | **3.21** | 2.44 |
| DNSMOS OVRL / SIG / BAK | 3.28 / 3.59 / 3.98 | **3.41 / 3.67 / 4.14** | 2.75 / 3.26 / 3.47 |
| energy above 4 kHz | 0.3 % | 0.1 % | 1.1 % |
| spectral roll-off 95 % | 1015 Hz | 652 Hz | 1333 Hz |
| log-F0 std (semitones) | 3.38 | 3.28 | 3.94 |
| speaking rate (syl/s, whole clip) | 8.0 | **5.0** | 6.4 |
| pauses per 10 s / mean pause | 0.8 / 0.09 s | **4.9 / 0.40 s** | 2.6 / 0.34 s |
| CER by length: short / medium / long | 6.1 / 2.8 / 1.9 % | 8.0 / 5.7 / 3.0 % | – |
| CER by source: FLEURS / Horn-ASR / Waxal | 4.6 / 4.0 / 2.2 % | 6.8 / 7.3 / 2.6 % | – |

Reading: we are already *more intelligible* than Addis Voice 2 on their own
prompts and truncate sentence endings half as often. Addis Voice 2 is **not**
brighter (it is even more band-limited), and its pitch range is no wider. What
it does differently is *delivery*: 40 % slower, with a 0.4 s pause roughly every
two seconds, and a slightly cleaner predicted signal quality. Those are the
parameters the fixes below target.

## Changes made as a result

CHANGES
