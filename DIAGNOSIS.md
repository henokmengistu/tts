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

## Ablations

_(filled in from `benchmark/diag/out/*/summary.json` as runs complete)_

ABLATION_TABLE

## Comparison on Addis AI's benchmark prompts

ADDIS_TABLE

## Changes made as a result

CHANGES
