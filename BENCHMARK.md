# Benchmark: amharic-tts vs Google TTS vs MMS-TTS

All numbers in this file are produced by `benchmark/run_benchmark.py` on the
sentences in `benchmark/testset.json` and can be reproduced with:

```bash
pip install -e ".[all]"
python benchmark/run_benchmark.py --systems ours,google,mms --voice selam --steps 16
```

## Systems

| system | what it is |
|---|---|
| **ours** | amharic-tts: Amharic normaliser + OmniVoice-Amharic (0.6 B, 16 diffusion steps, voice `selam`), CPU, 24 kHz |
| **google** | Google Translate's Amharic voice via `gTTS(lang="am")`, 24 kHz MP3 |
| **mms** | amharic-tts normaliser + Meta `facebook/mms-tts-amh` (VITS, 16 kHz) |

## Metrics

* **CER** — character error rate of an independent Amharic recogniser
  ([`snapwre/hohe-asr-amharic`](https://huggingface.co/snapwre/hohe-asr-amharic), 5–6 % CER on
  clean read speech) transcribing the synthesized audio, compared with the
  intended reading. Homophone letters (ሀ/ሐ/ኀ, ሰ/ሠ, ጸ/ፀ, አ/ዐ) are merged and
  punctuation ignored. For `norm` items the reference is the intended *spoken*
  form, so a system that reads “ጠ/ሚ” as nothing, or ፳፻፲፱ as 219, is penalised.
  This is the closest objective proxy to intelligibility.
* **UTMOS** — UTMOS22-strong predicted naturalness MOS. It was trained on
  English listening tests and is a weak signal for Amharic (it rates the
  human FLEURS recordings at only 3.4), reported for completeness.
* **DNSMOS P.835** — Microsoft's language-independent signal / background /
  overall quality predictors (1–5).
* **RTF** — wall-clock synthesis time ÷ audio duration on a 4-core CPU.

## Results

_(filled in from `benchmark/out/report.md`)_

RESULTS_TABLE

## Text normalisation: what each system actually said

The ASR transcript of each system on the normalisation sentences (digits are
always written out by the recogniser):

NORM_TABLE

## Qualitative differences observed

* Google drops or spells slash abbreviations (`ጠ/ሚ` disappeared, `ት/ቤቱ` → “በት ቤቱ”),
  misreads Ethiopic numerals above ፻ (፳፻፲፱ → “ሁለት መቶ አስራ ዘጠኝ”), reorders dates
  to English *day-month* order and drops ቀን, and reads `1990ዎቹ` as “nineteen nineties”.
* Google's number verbalisation for plain cardinals, percentages, times and
  currency is good; our normaliser matches it on those and adds Ethiopian /
  Gregorian calendar awareness, units, ranges, fiscal years and acronyms.
* MMS-TTS is fast but noticeably less intelligible (CER several times higher)
  and has one bible-reader voice.
* OmniVoice-Amharic produces the most natural prosody and is the only system
  with multiple voices and zero-shot cloning; its cost is CPU speed.

## Notes on int8 quantisation

Dynamic int8 quantisation of the Qwen3 backbone gives ~1.6× speed-up on CPU
but destroyed intelligibility in our test (CER 100 %), so it stays opt-in
(`--quant`) and is not recommended.
