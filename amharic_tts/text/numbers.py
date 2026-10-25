"""Amharic number verbalisation.

Converts integers, decimals, ordinals, digit strings and Ethiopic numerals
into their spoken Amharic form. All output uses the modern, most common
spellings (ሶስት, አስር, ሃያ, ሃምሳ, ስልሳ ...), which are also the forms a
neural TTS model has seen most often in training transcripts.
"""
from __future__ import annotations

import re

ZERO = "ዜሮ"
ONES = ["", "አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ"]
TEENS_PREFIX = "አስራ"  # 11-19: አስራ አንድ ... አስራ ዘጠኝ
TEN = "አስር"
TENS = ["", TEN, "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "ስልሳ", "ሰባ", "ሰማንያ", "ዘጠና"]
HUNDRED = "መቶ"
THOUSAND = "ሺህ"
# scale words for 10^6, 10^9, 10^12, 10^15
SCALES = [(10**15, "ኳድሪሊዮን"), (10**12, "ትሪሊዮን"), (10**9, "ቢሊዮን"), (10**6, "ሚሊዮን")]
POINT = "ነጥብ"
MINUS = "ሲቀነስ"

# Ordinal forms of the final number word.  The rule for Amharic ordinals is
# "stem + ኛ" where a final 6th-order (vowel-less) syllable moves to the
# 1st order (ä): ሁለት -> ሁለተኛ, አራት -> አራተኛ, አስር -> አስረኛ.  A few words are
# irregular, so an explicit table is safer than a rule.
ORDINAL_LAST_WORD = {
    "አንድ": "አንደኛ",
    "ሁለት": "ሁለተኛ",
    "ሶስት": "ሶስተኛ",
    "አራት": "አራተኛ",
    "አምስት": "አምስተኛ",
    "ስድስት": "ስድስተኛ",
    "ሰባት": "ሰባተኛ",
    "ስምንት": "ስምንተኛ",
    "ዘጠኝ": "ዘጠነኛ",
    "አስር": "አስረኛ",
    "ሃያ": "ሃያኛ",
    "ሰላሳ": "ሰላሳኛ",
    "አርባ": "አርባኛ",
    "ሃምሳ": "ሃምሳኛ",
    "ስልሳ": "ስልሳኛ",
    "ሰባ": "ሰባኛ",
    "ሰማንያ": "ሰማንያኛ",
    "ዘጠና": "ዘጠናኛ",
    "መቶ": "መቶኛ",
    "ሺህ": "ሺኛ",
    "ሚሊዮን": "ሚሊዮነኛ",
    "ቢሊዮን": "ቢሊዮነኛ",
    "ትሪሊዮን": "ትሪሊዮነኛ",
    "ዜሮ": "ዜሮኛ",
}

# Ethiopic numerals (U+1369 .. U+137C)
ETHIOPIC_DIGITS = {chr(0x1369 + i): i + 1 for i in range(9)}      # ፩..፱ = 1..9
ETHIOPIC_TENS = {chr(0x1372 + i): (i + 1) * 10 for i in range(9)}  # ፲..፺ = 10..90
ETHIOPIC_HUNDRED = "፻"   # 100
ETHIOPIC_TEN_THOUSAND = "፼"  # 10 000
ETHIOPIC_NUMERAL_RE = re.compile(r"[፩-፼]+")


def _below_thousand(n: int) -> list[str]:
    """Words for 1 <= n <= 999."""
    words: list[str] = []
    h, rem = divmod(n, 100)
    if h:
        if h == 1:
            words.append(HUNDRED)
        else:
            words.extend([ONES[h], HUNDRED])
    if rem:
        t, o = divmod(rem, 10)
        if t == 0:
            words.append(ONES[o])
        elif t == 1:
            words.append(TEN if o == 0 else f"{TEENS_PREFIX} {ONES[o]}")
        else:
            words.append(TENS[t])
            if o:
                words.append(ONES[o])
    return words


def cardinal(n: int) -> str:
    """Spoken Amharic for an integer (any size up to 10^18)."""
    if n < 0:
        return f"{MINUS} {cardinal(-n)}"
    if n == 0:
        return ZERO
    words: list[str] = []
    for value, name in SCALES:
        q, n = divmod(n, value)
        if q:
            words.append(cardinal(q))
            words.append(name)
    thousands, rem = divmod(n, 1000)
    if thousands:
        # "አንድ ሺህ" is the explicit, formal reading used by broadcasters.
        words.extend(_below_thousand(thousands))
        words.append(THOUSAND)
    if rem:
        words.extend(_below_thousand(rem))
    return " ".join(words)


def ordinal(n: int) -> str:
    """Spoken Amharic ordinal: 1 -> አንደኛ, 21 -> ሃያ አንደኛ, 100 -> መቶኛ."""
    words = cardinal(n).split()
    last = words[-1]
    words[-1] = ORDINAL_LAST_WORD.get(last, last + "ኛ")
    return " ".join(words)


def digits(s: str, group_sep: str = "") -> str:
    """Read a digit string digit by digit (phone numbers, codes, IDs)."""
    out = []
    for ch in s:
        if ch.isdigit():
            d = int(ch)
            out.append(ZERO if d == 0 else ONES[d])
        elif ch in " -" and group_sep:
            out.append(group_sep)
    return " ".join(out)


def decimal(int_part: str, frac_part: str) -> str:
    """3.14 -> ሶስት ነጥብ አንድ አራት (fraction digits are read one by one)."""
    whole = cardinal(int(int_part)) if int_part else ZERO
    if not frac_part:
        return whole
    return f"{whole} {POINT} {digits(frac_part)}"


def ethiopic_to_int(s: str) -> int:
    """Convert an Ethiopic numeral string (e.g. ፳፻፲፱ = 2019) to an int.

    Ethiopic numerals are written in groups of 10^2 (፻) and 10^4 (፼):
    ፼ multiplies everything before it by 10 000, ፻ multiplies the
    preceding 1-99 group by 100.
    """
    total = 0      # completed ፼-groups
    current = 0    # value within the current ፼ group
    pending = 0    # digits/tens accumulated for the next ፻ or group end
    for ch in s:
        if ch in ETHIOPIC_DIGITS:
            pending += ETHIOPIC_DIGITS[ch]
        elif ch in ETHIOPIC_TENS:
            pending += ETHIOPIC_TENS[ch]
        elif ch == ETHIOPIC_HUNDRED:
            current += (pending if pending else 1) * 100
            pending = 0
        elif ch == ETHIOPIC_TEN_THOUSAND:
            current += pending
            pending = 0
            total = (total + (current if current else 1)) * 10000
            current = 0
        else:
            raise ValueError(f"not an Ethiopic numeral: {ch!r}")
    return total + current + pending


def ethiopic_numeral_words(s: str) -> str:
    return cardinal(ethiopic_to_int(s))


__all__ = [
    "cardinal", "ordinal", "digits", "decimal", "ethiopic_to_int",
    "ethiopic_numeral_words", "ETHIOPIC_NUMERAL_RE", "ZERO", "ONES",
]
