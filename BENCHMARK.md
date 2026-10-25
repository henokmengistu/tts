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

| system | category | n | CER | UTMOS | DNSMOS SIG | DNSMOS BAK | DNSMOS OVRL | audio s | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| google | plain | 20 | 1.4% | 3.40 | 3.46 | 4.01 | 3.16 | 125 | 0.11 |
| google | norm | 20 | 14.3% | 3.52 | 3.43 | 4.04 | 3.15 | 86 | 0.16 |
| google | long | 1 | 2.6% | 2.95 | 3.54 | 4.13 | 3.29 | 31 | 0.07 |
| google | all | 41 | 7.7% | 3.45 | 3.45 | 4.03 | 3.16 | 243 | 0.13 |
| mms | plain | 20 | 8.3% | 3.39 | 3.54 | 4.07 | 3.27 | 125 | 0.39 |
| mms | norm | 20 | 4.5% | 3.36 | 3.52 | 3.98 | 3.21 | 93 | 0.25 |
| mms | long | 1 | 3.5% | 3.16 | 3.58 | 4.12 | 3.34 | 32 | 0.29 |
| mms | all | 41 | 6.3% | 3.37 | 3.53 | 4.03 | 3.24 | 250 | 0.32 |
| ours | plain | 20 | 2.2% | 3.26 | 3.49 | 3.79 | 3.11 | 91 | 8.70 |
| ours | norm | 20 | 2.6% | 3.48 | 3.46 | 3.84 | 3.10 | 69 | 10.75 |
| ours | long | 1 | 0.0% | 2.63 | 3.58 | 4.07 | 3.30 | 23 | 5.69 |
| ours | all | 41 | 2.3% | 3.35 | 3.48 | 3.82 | 3.11 | 182 | 9.62 |

**Headline:** over all 41 utterances our system's ASR character error rate is
**2.3 %**, Google's is **7.7 %** and MMS's is **6.3 %**. On plain prose Google
and ours are within ASR noise of each other (1.4 % vs 2.2 %; the residual
errors on both are single-letter confusions such as ሆኖ/ሁኖ); on text with
numbers, dates and abbreviations Google's error rate is **5.5× higher**
(14.3 % vs 2.6 %). Predicted quality scores are on par (UTMOS 3.45 vs 3.35,
DNSMOS OVRL 3.16 vs 3.11). The price is speed: the 0.6 B model runs at
RTF ≈ 9 on this 4-core CPU box (well below 1 on a GPU), versus a cloud
round-trip for Google.

## Text normalisation: what each system actually said

The ASR transcript of each system on the normalisation sentences (digits are
always written out by the recogniser):

**n01** `ዛሬ ጥቅምት 15 ቀን 2019 ዓ.ም. ነው።`  
- expected: ዛሬ ጥቅምት አስራ አምስት ቀን ሁለት ሺህ አስራ ዘጠኝ ዓመተ ምህረት ነው
- google (20%): ዛሬ አስራ አምስት ጥቅምት ሁለት ሺህ አስራ ዘጠኝ ዓመተ ምህረት ነው
- mms (2%): ዛሬ ጥቅምት አስራ አምስት ቀን ሁለት ሺህ አስራ ዘጠኝ ዓመተ ምረት ነው
- ours (0%): ዛሬ ጥቅምት አስራ አምስት ቀን ሁለት ሺህ አስራ ዘጠኝ ዓመተ ምህረት ነው

**n02** `ዶ/ር አበበ እና ወ/ሮ አልማዝ በት/ቤቱ ተገኙ።`  
- expected: ዶክተር አበበ እና ወይዘሮ አልማዝ በትምህርት ቤቱ ተገኙ
- google (11%): ዶክተር አበበ እና ወይዘሮ አልማዝ በት ቤቱ ተገኙ
- mms (9%): ሎክተር አበበና ወይዘሮ አልማዝ በትምህርት ቤቱ ተገኙ
- ours (0%): ዶክተር አበበ እና ወይዘሮ አልማዝ በትምህርት ቤቱ ተገኙ

**n03** `ጠ/ሚ ዓብይ አህመድ በአዲስ አበባ ክ/ከተማ ተናገሩ።`  
- expected: ጠቅላይ ሚኒስትር ዓብይ አህመድ በአዲስ አበባ ክፍለ ከተማ ተናገሩ
- google (37%): አብይ አህመድ በአዲስ አበባ  ከተማ ተናገሩ
- mms (2%): ጠቅላይ ሚኒስትር ዐቢይ አሕመድ በአዲስ አበባ ክፍለ ከተማ ተናገሩ
- ours (2%): ጠቅላይ ሚኒስትር ዐቢይ አሕመድ በአዲስ አበባ ክፍለ ከተማ ተናገሩ

**n04** `እሱ 3ኛ ሆኖ አጠናቀቀ፤ እሷ ደግሞ 21ኛ ነበረች።`  
- expected: እሱ ሶስተኛ ሆኖ አጠናቀቀ እሷ ደግሞ ሃያ አንደኛ ነበረች
- google (0%): እሱ ሦስተኛ ሆኖ አጠናቀቀ እሷ ደግሞ ሀያ አንደኛ ነበረች
- mms (0%): እሱ ሶስተኛ ሆኖ አጠናቀቀ እሷ ደግሞ ሃያ አንደኛ ነበረች
- ours (3%): እሱ ሶስተኛ ሁኖ አጠናቀቀ እሷ ደግሞ ሃያ አንደኛ ነበረች

**n05** `በ፳፻፲፱ ዓ.ም. ፻ ሰዎች መጡ።`  
- expected: በሁለት ሺህ አስራ ዘጠኝ ዓመተ ምህረት መቶ ሰዎች መጡ
- google (18%): በሁለት መቶ አስራ ዘጠኝ ዓመተ ምህረት አንድ መቶ ሰዎች መጡ
- mms (0%): በሁለት ሺህ አስራ ዘጠኝ ዓመተ ምህረት መቶ ሰዎች መጡ
- ours (9%): በሁለት ሺህ አስራ ዘጠኝ አመተ ምህረት መሶዎች መጡ

**n06** `ዋጋው በ25% ጨመረ።`  
- expected: ዋጋው በሃያ አምስት በመቶ ጨመረ
- google (25%): ዋጋው በሀያ አምስት ፐርሰንት ጨመረ
- mms (0%): ዋጋው በሀያ አምስት በመቶ ጨመረ
- ours (0%): ዋጋው በሃያ አምስት በመቶ ጨመረ

**n07** `ስብሰባው 3:30 ላይ ይጀምራል።`  
- expected: ስብሰባው ሶስት ሰዓት ተኩል ላይ ይጀምራል
- google (15%): ስብሰባው ሶስት ተኩል ላይ ይጀምራል
- mms (8%): ስብሰባው ሶስት ሰዓተ እኩል ላይ ይጀምራል
- ours (0%): ስብሰባው ሶስት ሰዓት ተኩል ላይ ይጀምራል

**n08** `የሙቀት መጠኑ 36.6 ዲግሪ ነው።`  
- expected: የሙቀት መጠኑ ሰላሳ ስድስት ነጥብ ስድስት ዲግሪ ነው
- google (0%): የሙቀት መጠኑ ሰላሳ ስድስት ነጥብ ስድስት ዲግሪ ነው
- mms (3%): የሙቀት መተኑ ሰላሳ ስድስት ነጥብ ስድስት ዲግሪ ነው
- ours (0%): የሙቀት መጠኑ ሰላሳ ስድስት ነጥብ ስድስት ዲግሪ ነው

**n09** `ከ10-20 ሰዎች መጡ።`  
- expected: ከአስር እስከ ሃያ ሰዎች መጡ
- google (0%): ከአስር እስከ ሀያ ሰዎች መጡ
- mms (6%): ከአስር እስከ አያ ሰዎች መጡ
- ours (0%): ከአስር እስከ ሃያ ሰዎች መጡ

**n10** `ለስብሰባው 1250 ብር ከፍያለሁ።`  
- expected: ለስብሰባው አንድ ሺህ ሁለት መቶ ሃምሳ ብር ከፍያለሁ
- google (3%): ለስብሰባው አንድ ሺህ ሁለት መቶ ሀምሳ ብር ከፍ ያለሁ
- mms (6%): ለስብሰባው አንድ ሺህ ሁለት መቶ ሀምሳ ብር ክፍያ ለሁ
- ours (6%): ለስብሰባው አንድ ሺህ ሁለት መቶ ሃምሳ ብር ከፍ እያለሁ

**n11** `$50 እና 12.50 ብር ከፈለ።`  
- expected: ሃምሳ ዶላር እና አስራ ሁለት ብር ከሃምሳ ሳንቲም ከፈለ
- google (9%): ሃምሳ ዶላር እና አሥራ ሁለት ብር እና ሃምሳ ሳንቲም ከፈለ
- mms (0%): ሃምሳ ዶላር እና አሥራ ሁለት ብር ከሃምሳ ሳንቲም ከፈለ
- ours (3%): ሃምሳ ዶላርእና አስራ ሁለት ብር ከሃምሳ ሳንቲም ከፈለ

**n12** `የኢትዮጵያ ህዝብ ብዛት 120 ሚሊዮን ደርሷል።`  
- expected: የኢትዮጵያ ህዝብ ብዛት መቶ ሃያ ሚሊዮን ደርሷል
- google (10%): የኢትዮጵያ ህዝብ ብዛት አንድመቶ ሃያ ሚሊዮን ደርሷል
- mms (3%): የኢትዮጵያ ሕዝቡ ብዛት መቶ ሀያ ሚሊዮን ደርሷል
- ours (0%): የኢትዮጵያ ህዝብ ብዛት መቶ ሀያ ሚሊዮን ደርሷል

**n13** `ርቀቱ 25 ኪ.ሜ. ሲሆን ክብደቱ 3 ኪ.ግ ነው።`  
- expected: ርቀቱ ሃያ አምስት ኪሎ ሜትር ሲሆን ክብደቱ ሶስት ኪሎ ግራም ነው
- google (22%): ርቀቱ ሀያ አምስት ኪሚ ሲሆን ክብደቱ ሶስት ኪግ ነው
- mms (2%): እቀቱ ሀያ አምስት ኪሎ ሜትር ሲሆን ክብደቱ ሦስት ኪሎ ግራም ነው
- ours (0%): ርቀቱ ሃያ አምስት ኪሎ ሜትር ሲሆን ክብደቱ ሶስት ኪሎ ግራም ነው

**n14** `ገ/ማርያም እና ኃ/ሥላሴ ተገናኙ።`  
- expected: ገብረ ማርያም እና ኃይለ ሥላሴ ተገናኙ
- google (21%): ገ ማርያም እናሃ ስላሴ ተገናኙ
- mms (4%): ደብረ ማርያም እና ሀይለ ስላሴ ተገናኙ
- ours (12%): ገብረማርያም እና ሀይለስላሴ ተገናኘ

**n15** `የ2016/17 በጀት ዓመት ሪፖርት ቀረበ።`  
- expected: የሁለት ሺህ አስራ ስድስት አስራ ሰባት በጀት ዓመት ሪፖርት ቀረበ
- google (0%): የሁለት ሺህ አስራ ስድስት አስራ ሰባት በጀት ዓመት ሪፖርት ቀረበ
- mms (5%): የሁለት ሺህ አስራ ስድስት አስራ ሰባት በጀት መት ሪኮርት ቀረበ
- ours (0%): የሁለት ሺህ አስራ ስድስት አስራ ሰባት በጀት አመት ሪፖርት ቀረበ

**n16** `በ1990ዎቹ ተወለደ።`  
- expected: በአንድ ሺህ ዘጠኝ መቶ ዘጠናዎቹ ተወለደ
- google (32%): በአስራ ዘጠኝ ዘጠናዎቹ ተወለደ
- mms (0%): በአንድ ሺህ ዘጠኝ መቶ ዘጠናዎቹ ተወለደ
- ours (0%): በአንድ ሺህ ዘጠኝ መቶ ዘጠናዎቹ ተወለደ

**n17** `ቢቢሲ እና ESAT ዘገቡ።`  
- expected: ቢቢሲ እና ኢሳት ዘገቡ
- google (29%): ቢቢሲ እና ኢኤስኤት ይዘገቡ
- mms (29%): ኢጊሲህ እና ኢሳት ዘጋቡ
- ours (14%): ቢቢሲና ኢሳት ዘገቡ

**n18** `ስልክ 0911 234567 ይደውሉ።`  
- expected: ስልክ ዜሮ ዘጠኝ አንድ አንድ ሁለት ሶስት አራት አምስት ስድስት ሰባት ይደውሉ
- google (0%): ስልክ ዜሮ ዘጠኝ አንድ አንድ ሁለት ሶስት አራት አምስት ስድስት ሰባት ይደውሉ
- mms (0%): ስልክ ዜሮ ዘጠኝ አንድ አንድ ሁለት ሶስት አራት አምስት ስድስት ሰባት ይደውሉ
- ours (2%): ስልክ ዜሮ ዘጠኝ አንዳ አንድ ሁለት ሶስት አራት አምስት ስድስት ሰባት ይደውሉ

**n19** `ውሉ 15/02/2016 ዓ.ም. ተፈረመ።`  
- expected: ውሉ ጥቅምት አስራ አምስት ቀን ሁለት ሺህ አስራ ስድስት ዓመተ ምህረት ተፈረመ
- google (18%): ውሉ አስራ አምስት ጥቅምት ሁለት ሺህ አስራ ስድስት ዓመተ ምህረት ተፈረመ
- mms (10%): ውሉ ጥቅምጥ አስራ አምስት ቀን ሁለት ሺህ አስራ ስድስት ዓመተ መሬት ተፈረነ
- ours (0%): ውሉ ጥቅምት አስራ አምስት ቀን ሁለት ሺህ አስራ ስድስት ዓመተ ምህረት ተፈረመ

**n20** `ብር 1,250,000 ለፕሮጀክቱ ተመደበ።`  
- expected: አንድ ሚሊዮን ሁለት መቶ ሃምሳ ሺህ ብር ለፕሮጀክቱ ተመደበ
- google (16%): ብር አንድ ሚሊዮን ሁለት መቶ ሃምሳ ሺህ ለፕሮጀክቱ ተመደበ
- mms (0%): አንድ ሚሊዮን ሁለት መቶ ሃምሳ ሺህ ብር ለፕሮጀክቱ ተመደበ
- ours (0%): አንድ ሚሊዮን ሁለት መቶ ሃምሳ ሺህ ብር ለፕሮጀክቱ ተመደበ

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

## Diffusion steps and voice choice (plain-prose subset, 20 sentences)

| run | CER | UTMOS | DNSMOS SIG | DNSMOS BAK | DNSMOS OVRL | RTF (4-core CPU) |
|---|---:|---:|---:|---:|---:|---:|
| `selam`, 16 steps (default) | 2.2 % | 3.26 | 3.49 | 3.79 | 3.11 | 8.7 |
| `selam`, 32 steps | 2.6 % | 3.33 | 3.52 | 3.84 | 3.15 | 16.5 |
| `abebe` (male), 16 steps, speed 1.15 | **1.5 %** | 3.15 | 3.43 | 3.81 | 3.05 | 6.3 |
| `hana`, 16 steps | 2.3 % | 2.88 | 3.44 | 3.74 | 3.01 | 9.5 |

Doubling the diffusion steps costs 2× time for no measurable gain, so 16 is
the default. All presets are intelligible at the level of the recogniser's
own error floor; `abebe` is the most intelligible, `selam` scores highest on
predicted naturalness. Raw data: `benchmark/out_steps32/`, `benchmark/out_abebe/`,
`benchmark/out_hana/`.

## Notes on int8 quantisation

Dynamic int8 quantisation of the Qwen3 backbone gives a 1.65× speed-up on CPU
(34 s vs 57 s for an 8 s sentence) but degrades the output badly: CER rose
from 0 % to 16 % and predicted MOS collapsed (UTMOS 3.3 → 1.4) on the same
sentence; quantising the audio heads as well made the output unintelligible.
It therefore stays opt-in (`--quant`) and is not recommended. The practical
speed knobs are `--steps 8` and a GPU.
