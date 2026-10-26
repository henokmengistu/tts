# Gemini-as-an-ear evaluation (gemini-3.5-flash-lite)

20 prompts, systems: human, addis, google, gemini-tts, azure, mms, ours. Clips loudness-normalised, names hidden, pairwise judged in both orders, rankings repeated with independent shuffles.

## Rank-all (average rank over prompts and shuffles; 1 = best)

| system | mean rank overall | mean rank naturalness | ranked 1st (overall) | 'could pass as human' |
|---|---:|---:|---:|---:|
| azure | 2.41 | 2.41 | 38 % | 55 % |
| gemini-tts | 3.54 | 3.54 | 13 % | 45 % |
| mms | 3.87 | 3.85 | 13 % | 28 % |
| google | 3.92 | 3.92 | 15 % | 30 % |
| addis | 3.95 | 3.97 | 18 % | 30 % |
| ours | 4.92 | 4.92 | 0 % | 10 % |
| human | 5.38 | 5.38 | 3 % | 12 % |
