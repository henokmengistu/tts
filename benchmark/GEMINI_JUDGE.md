# Gemini as an independent ear

`benchmark/gemini_judge.py` asks a Gemini audio model to listen to the same
sentences rendered by every system (human recording, Addis Voice 2, Google
Translate TTS, Gemini TTS, Azure, Meta MMS, amharic-tts) and to transcribe,
rate, compare and rank them blind.

## Why the protocol looks the way it does

Audio-LLM judges are useful but have documented shortcuts (AudioJudge, EACL
2026; "Louder, Longer, Livelier" 2026; protocol-shortcut audits 2026):
position bias in A/B tests, a preference for louder and longer clips, and
"reference following" when the text is shown. The script therefore:

* loudness-normalises every clip to the same RMS and 16 kHz mono WAV;
* never puts a system name in a prompt (clips are "A", "B", …);
* judges every pair in **both orders** and counts a win only when the two
  orders agree (disagreement = tie);
* ranks all systems per sentence in a **random order, twice**, and reports the
  mean rank;
* transcribes **without** the text (pure listening) and rates **with** the text
  (so missed / mispronounced / repeated words can be named);
* uses low temperature, JSON-schema outputs, and caches every answer in
  `results.jsonl` (resumable, reproducible).

Five protocols: `transcribe` (CER of Gemini's own transcript), `diagnose`
(**where and what**: a time-stamped defect list per clip), `rate` (1–5
naturalness / intelligibility / voice quality / prosody / overall, plus flags
and word lists), `pair` (win rate vs. a reference system), `rank` (full
ranking per sentence). Report: `benchmark/out/gemini_judge/report.md`.

### The `diagnose` protocol (default on)

For every clip Gemini gets the intended text and must return, in time order,
each defect it hears with:

* `type` — one of: missing_word, extra_word, repeated_or_stutter,
  mispronounced, wrong_gemination_or_stress, unnatural_pause,
  rushed_or_slurred, robotic_or_metallic, noise_or_artifact, cut_off,
  wrong_intonation, wrong_number_or_abbreviation_reading, other;
* `text_word` — the exact word(s) from the text where it happens;
* `heard_as` — what was actually said (Ethiopic);
* `start` / `end` — position in the clip (MM:SS.s);
* `severity` 1–3 and a one-sentence explanation;

plus the transcript it heard, a two-sentence native-listener impression, a
"would a native notice problems" flag and an overall 1–5.

The script turns this into `issues_<system>.md` (one row per defect with
prompt, time, type, severity, word, heard-as, explanation) and cuts a
**snippet WAV** around every flagged spot into `snippets/<system>/` so you can
listen to the exact moment. The main report adds a per-system summary
(issues per clip, severe issues per clip, how often a native would notice, the
most frequent defect types). Running it on `--voices` gives the same defect
map for each of our preset voices, which is the fastest way to see what is
wrong with a specific voice.

## Running it

```bash
export GEMINI_API_KEY=...            # or store it as an environment secret
python benchmark/gemini_judge.py --list-models            # confirm the model id
python benchmark/gemini_judge.py --n 20 --protocols rank,transcribe   # cheap first look (≈180 requests)
python benchmark/gemini_judge.py --n 20 --protocols diagnose          # where/what defect maps + snippets (140 requests)
python benchmark/gemini_judge.py --n 30                   # transcribe + diagnose + pair + rank (≈840 requests)
python benchmark/gemini_judge.py --voices                 # rank our 7 preset voices on samples/
python benchmark/gemini_judge.py --model gemini-3.8-flash --n 50       # cheaper model, more prompts
```

Default model: `gemini-4-argon` (flagship, 1M context, audio input). Audio
costs 32 tokens per second; 30 prompts × 7 systems ≈ 200 clips ≈ 10 minutes
of audio ≈ 20k audio tokens per protocol pass, so the whole run is cheap; the
request count, not tokens, is what rate limits bound. Use `--workers` to
adjust concurrency.

`--dry-run` builds every request without calling the API (checks that all
audio sets are present). Google's own clips for the 100 Addis prompts are
generated with `gtts` into `audio/google-gtts` so Google Translate's voice can
be ranked alongside the rest.

## Reading the result

* Treat the **rank-all** and **pairwise** tables as the main evidence; absolute
  1–5 scores from LLM judges are the least calibrated.
* The human recordings are the anchor: a system ranked at or above them, or
  often marked "could pass as human", is doing well regardless of the number.
* Cross-check Gemini's transcription CER against the wav2vec-based CER in
  BENCHMARK.md; two independent ears agreeing is strong evidence.
