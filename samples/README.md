# Samples

* `compare/` — the same benchmark sentences synthesized by **ours** (amharic-tts,
  voice `selam`), **google** (gTTS `am`) and **mms** (facebook/mms-tts-amh):
  * `p02` plain news sentence
  * `n01` ዛሬ ጥቅምት 15 ቀን 2019 ዓ.ም. ነው። (date order, ቀን, ዓ.ም.)
  * `n03` ጠ/ሚ ዓብይ አህመድ በአዲስ አበባ ክ/ከተማ ተናገሩ። (slash abbreviations — Google drops ጠ/ሚ)
  * `n05` በ፳፻፲፱ ዓ.ም. ፻ ሰዎች መጡ። (Ethiopic numerals — Google reads 2019 as 219)
  * `n19` ውሉ 15/02/2016 ዓ.ም. ተፈረመ። (numeric Ethiopian date)
  * `l01` a four-sentence paragraph
* `<voice>_<topic>.mp3` — every preset voice (selam, hana, abebe, dawit) on four
  texts; `samples.json` holds the text and timing for each file.
