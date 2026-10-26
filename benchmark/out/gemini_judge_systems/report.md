# Gemini-as-an-ear evaluation (gemini-3.5-flash-lite)

20 prompts, systems: human, addis, google, gemini-tts, azure, mms, ours. Clips loudness-normalised, names hidden, pairwise judged in both orders, rankings repeated with independent shuffles.

## Intelligibility as heard by Gemini (transcription CER, lower is better)

| system | n | CER | 95 % CI |
|---|---:|---:|---:|
| human | 20 | 18.1 % | 11.5–26.0 |
| addis | 20 | 12.0 % | 9.1–15.2 |
| google | 20 | 10.4 % | 6.5–15.5 |
| gemini-tts | 20 | 7.4 % | 4.3–11.5 |
| azure | 20 | 8.5 % | 5.4–12.5 |
| mms | 20 | 14.4 % | 9.7–19.6 |
| ours | 20 | 7.7 % | 4.3–12.0 |

## Where and what goes wrong (time-stamped defect reports; full lists in issues_<system>.md)

| system | clips | issues / clip | severe (3) / clip | overall 1-5 | native would notice | top issue types |
|---|---:|---:|---:|---:|---:|---|
| human | 20 | 1.05 | 0.50 | 3.95 | 55 % | mispronounced ×10, repeated_or_stutter ×3, wrong_gemination_or_stress ×2, other ×2 |
| addis | 20 | 1.00 | 0.35 | 4.15 | 55 % | mispronounced ×7, repeated_or_stutter ×4, wrong_gemination_or_stress ×3, extra_word ×2 |
| google | 20 | 0.70 | 0.35 | 4.25 | 45 % | mispronounced ×4, repeated_or_stutter ×3, other ×2, wrong_gemination_or_stress ×2 |
| gemini-tts | 20 | 0.55 | 0.30 | 4.40 | 25 % | mispronounced ×3, other ×3, repeated_or_stutter ×1, missing_word ×1 |
| azure | 20 | 0.60 | 0.45 | 4.25 | 35 % | mispronounced ×5, repeated_or_stutter ×3, other ×2, cut_off ×1 |
| mms | 20 | 1.05 | 0.70 | 4.10 | 40 % | mispronounced ×9, repeated_or_stutter ×3, noise_or_artifact ×2, wrong_gemination_or_stress ×1 |
| ours | 20 | 0.75 | 0.40 | 4.10 | 55 % | repeated_or_stutter ×5, mispronounced ×4, other ×2, missing_word ×1 |
