# Gemini-as-an-ear evaluation (gemini-3.5-flash-lite)

4 prompts, systems: biruk, dawit, liya, meron, sara, selam, yonas. Clips loudness-normalised, names hidden, pairwise judged in both orders, rankings repeated with independent shuffles.

## Intelligibility as heard by Gemini (transcription CER, lower is better)

| system | n | CER | 95 % CI |
|---|---:|---:|---:|
| biruk | 4 | 23.5 % | 0.6–67.4 |
| dawit | 4 | 22.1 % | 0.6–63.0 |
| liya | 4 | 21.2 % | 0.0–62.7 |
| meron | 4 | 28.2 % | 3.4–70.7 |
| sara | 4 | 25.7 % | 2.8–64.1 |
| selam | 4 | 11.9 % | 0.0–33.9 |
| yonas | 4 | 25.5 % | 0.6–73.6 |

## Where and what goes wrong (time-stamped defect reports; full lists in issues_<system>.md)

| system | clips | issues / clip | severe (3) / clip | overall 1-5 | native would notice | top issue types |
|---|---:|---:|---:|---:|---:|---|
| biruk | 4 | 0.25 | 0.00 | 5.00 | 0 % | wrong_number_or_abbreviation_reading ×1 |
| dawit | 4 | 0.25 | 0.00 | 4.75 | 25 % | wrong_gemination_or_stress ×1 |
| liya | 4 | 0.00 | 0.00 | 5.00 | 0 % |  |
| meron | 4 | 1.75 | 0.00 | 4.00 | 50 % | wrong_number_or_abbreviation_reading ×3, repeated_or_stutter ×2, wrong_gemination_or_stress ×1, unnatural_pause ×1 |
| sara | 4 | 0.75 | 0.00 | 4.75 | 25 % | wrong_number_or_abbreviation_reading ×2, repeated_or_stutter ×1 |
| selam | 4 | 0.00 | 0.00 | 5.00 | 0 % |  |
| yonas | 4 | 0.75 | 0.25 | 4.25 | 50 % | robotic_or_metallic ×1, repeated_or_stutter ×1, wrong_number_or_abbreviation_reading ×1 |

## Pairwise vs `biruk` (both orders; counted only when both orders agree)

| opponent | n pairs | ref wins | opponent wins | tie / inconsistent | ref win rate (decided) |
|---|---:|---:|---:|---:|---:|
| dawit | 4 | 3 | 0 | 1 | 100 % |
| liya | 4 | 0 | 0 | 4 | nan % |
| meron | 4 | 3 | 1 | 0 | 75 % |
| sara | 4 | 1 | 1 | 2 | 50 % |
| selam | 4 | 0 | 1 | 3 | 0 % |
| yonas | 4 | 1 | 0 | 3 | 100 % |

## Rank-all (average rank over prompts and shuffles; 1 = best)

| system | mean rank overall | mean rank naturalness | ranked 1st (overall) | 'could pass as human' |
|---|---:|---:|---:|---:|
| selam | 3.00 | 3.00 | 12 % | 25 % |
| yonas | 3.62 | 3.62 | 25 % | 25 % |
| biruk | 4.12 | 4.12 | 12 % | 25 % |
| liya | 4.12 | 4.12 | 0 % | 12 % |
| meron | 4.12 | 4.12 | 25 % | 38 % |
| sara | 4.25 | 4.25 | 12 % | 12 % |
| dawit | 4.75 | 4.75 | 12 % | 12 % |
