"""Amharic text normaliser for TTS.

Pipeline (each step is individually switchable through :class:`NormalizerOptions`):

1. Unicode clean-up (NFC, zero-width characters, emoji, odd whitespace)
2. Abbreviation expansion (ዶ/ር -> ዶክተር, ዓ.ም. -> ዓመተ ምህረት, ...)
3. Ethiopic numerals -> Arabic digits (፳፻፲፱ -> 2019)
4. Semiotic classes in priority order:
   phone numbers, dates, times, currency, percentages, ranges,
   ordinals, decimals, plain cardinals with attached prefixes/suffixes
5. Units and symbols (25 ኪ.ሜ -> ሃያ አምስት ኪሎ ሜትር, & -> እና)
6. Latin acronyms (optional spelling with Amharic letter names)
7. Optional homophone merging (ሐ/ኀ -> ሀ, ሠ -> ሰ, ፀ -> ጸ, ዐ -> አ)
8. Punctuation mapping (። -> .  ፣ -> ,  ...) and whitespace clean-up

The output is still Ethiopic-script text that a human would read aloud
exactly as intended; it is engine-agnostic.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import numbers as num
from .abbreviations import (
    ABBREVIATIONS, CURRENCIES, LATIN_ACRONYMS, LATIN_LETTER_NAMES, SYMBOLS,
    UNIT_ABBREVIATIONS,
)

# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
ETHIOPIAN_MONTHS = [
    "መስከረም", "ጥቅምት", "ህዳር", "ታህሳስ", "ጥር", "የካቲት", "መጋቢት",
    "ሚያዝያ", "ግንቦት", "ሰኔ", "ሐምሌ", "ነሐሴ", "ጳጉሜ",
]
GREGORIAN_MONTHS = [
    "ጃንዋሪ", "ፌብሩዋሪ", "ማርች", "ኤፕሪል", "ሜይ", "ጁን", "ጁላይ",
    "ኦገስት", "ሴፕቴምበር", "ኦክቶበር", "ኖቬምበር", "ዲሴምበር",
]
# Spelling variants of month names found in the wild -> canonical index
_MONTH_LOOKUP: dict[str, int] = {}
for _i, _m in enumerate(ETHIOPIAN_MONTHS):
    _MONTH_LOOKUP[_m] = _i
for _i, _m in enumerate(GREGORIAN_MONTHS):
    _MONTH_LOOKUP[_m] = _i
for _alias, _i in {
    "ኅዳር": 2, "ሕዳር": 2, "ታኅሣሥ": 3, "ታኅሳስ": 3, "ታህሣሥ": 3, "ጥቅምት": 1,
    "ሃምሌ": 10, "ሀምሌ": 10, "ነሀሴ": 11, "ነሃሴ": 11, "ጳጉሜን": 12, "ጳጉሜ": 12,
    "ሚያዚያ": 7, "መጋብት": 6,
}.items():
    _MONTH_LOOKUP[_alias] = _i

DAY_WORD = "ቀን"
ERA_EC = "ዓመተ ምህረት"
ERA_MARKERS_EC = re.compile(r"ዓመተ ምህረት|ዓ\.?ም|ዓ/ም|በኢትዮጵያ አቆጣጠር")
ERA_MARKERS_GC = re.compile(r"እንደ አውሮፓውያን አቆጣጠር|እ\.?ኤ\.?አ|እ\.?አ\.?አ|ጎርጎሮሳውያን|ፈረንጆች")

# --------------------------------------------------------------------------
# Character classes
# --------------------------------------------------------------------------
ETHIOPIC_LETTER = r"[ሀ-ፚᎀ-᎙ⶀ-ⷞꬁ-ꬮ]"
_ETH = ETHIOPIC_LETTER
# Prefix particles that attach to numbers: በ, ከ, ለ, የ, ወደ, እስከ, ስለ, እንደ, ና, ም
NUM_PREFIX = r"(?:(?<!" + _ETH + r")(?:እስከ|ወደ|ስለ|እንደ|በ|ከ|ለ|የ))?"

# Homophone variant merging (identical pronunciation in modern Amharic)
_VARIANT_MAP: dict[str, str] = {}


def _add_series(src_base: int, dst_base: int, n: int = 7) -> None:
    for k in range(n):
        _VARIANT_MAP[chr(src_base + k)] = chr(dst_base + k)


_add_series(0x1210, 0x1200)  # ሐ-series -> ሀ-series
_add_series(0x1280, 0x1200)  # ኀ-series -> ሀ-series
_add_series(0x1220, 0x1230)  # ሠ-series -> ሰ-series
_add_series(0x1340, 0x1338)  # ፀ-series -> ጸ-series
_add_series(0x12D0, 0x12A0)  # ዐ-series -> አ-series
# labialised ኈ-series (U+1288..128D) sounds like ኋ-series (U+12B8..) only
# partially; leave them alone except the common ኋ/ኍ spellings which are
# already in the ኀ block and were remapped above.  Undo that for the
# labialised ones to keep ኋ distinct from ሃ:
for _cp in range(0x1288, 0x1290):
    _VARIANT_MAP.pop(chr(_cp), None)
_VARIANT_TABLE = str.maketrans(_VARIANT_MAP)

# Ethiopic punctuation -> ASCII equivalents (silent/structural marks)
PUNCT_MAP = {
    "።": ".", "፡": " ", "፣": ",", "፤": ";", "፥": ":", "፦": ":", "፧": "?",
    "፠": ".", "፨": ".\n", "‹": "", "›": "", "«": "", "»": "", "“": "", "”": "",
    "„": "", "‘": "", "’": "", "`": "", "´": "", "…": ", ", "—": ", ", "–": ", ",
    "―": ", ", "·": " ", "•": " ", "|": ", ", "_": " ", "*": "", "^": "",
    "<": "", ">": "", "{": "", "}": "", "[": ",", "]": ",", "(": ", ", ")": ", ",
    "\\": " ", " ": " ", " ": " ", "　": " ",
}
_PUNCT_TABLE = str.maketrans(PUNCT_MAP)

_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠﻿]")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF⭐⭕‼⁉️]"
)
_REPEATED_PUNCT = re.compile(r"([!?.,;:])(?:\s*\1)+")
_SPACES = re.compile(r"[ \t\f\v]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


@dataclass
class NormalizerOptions:
    expand_abbreviations: bool = True
    expand_numbers: bool = True
    expand_units: bool = True
    expand_symbols: bool = True
    spell_acronyms: bool = True
    merge_homophones: bool = False
    map_punctuation: bool = True
    default_calendar: str = "ethiopian"  # month names for ambiguous numeric dates
    extra_abbreviations: dict[str, str] = field(default_factory=dict)


class AmharicNormalizer:
    """Rule-based, dependency-free Amharic text normaliser."""

    def __init__(self, options: NormalizerOptions | None = None, **kwargs):
        self.opt = options or NormalizerOptions(**kwargs)
        abbr = dict(ABBREVIATIONS)
        abbr.update(self.opt.extra_abbreviations)
        # longest first so "ጠ/ሚ/ር" wins over "ጠ/ሚ"
        keys = sorted(abbr, key=len, reverse=True)
        self._abbr = abbr
        # Only ASCII/word boundaries are unreliable with Ethiopic, so build
        # explicit boundary lookarounds.
        parts = []
        for k in keys:
            esc = re.escape(k)
            if k.endswith("/"):
                # name prefixes like ገ/ must be followed by Ethiopic letters
                parts.append(rf"{esc}(?={_ETH})")
            elif k == "ቁ":
                parts.append(rf"ቁ(?=\s*\d)")
            else:
                parts.append(rf"{esc}(?!{_ETH})")
        # An abbreviation may carry an attached particle (በት/ቤቱ = በ + ት/ቤቱ).
        self._abbr_re = re.compile(
            rf"(?P<pre>(?<!{_ETH})(?:እስከ|ወደ|ስለ|እንደ|በ|ከ|ለ|የ|ና|ከነ)?)(?P<abbr>" + "|".join(parts) + ")"
        )

        unit_keys = sorted(UNIT_ABBREVIATIONS, key=len, reverse=True)
        self._unit_re = re.compile(
            r"(?<=\d)\s*(" + "|".join(re.escape(u) for u in unit_keys) + r")(?![A-Za-z" + _ETH[1:-1] + r"])"
        )
        cur_keys = sorted(CURRENCIES, key=len, reverse=True)
        cur_alt = "|".join(re.escape(c) for c in cur_keys)
        # $50 / USD 50 / 50$ / 50 ብር / ብር 50 / 12.50 ብር
        self._cur_prefix_re = re.compile(
            rf"(?P<cur>{cur_alt})\s?(?P<num>\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.(?P<frac>\d{{1,2}}))?(?!\d)"
        )
        self._cur_suffix_re = re.compile(
            rf"(?<![\d.])(?P<num>\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.(?P<frac>\d{{1,2}}))?\s?(?P<cur>{cur_alt})(?![A-Za-z{_ETH[1:-1]}])"
        )
        self._acronym_re = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z]{0,6})(?![A-Za-z])")

    # ------------------------------------------------------------------ API
    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        t = self._clean(text)
        if self.opt.expand_abbreviations:
            t = self._abbr_re.sub(lambda m: m.group("pre") + self._abbr[m.group("abbr")], t)
        t = self._ethiopic_numerals(t)
        if self.opt.expand_numbers:
            t = self._numbers(t)
        if self.opt.expand_symbols:
            t = self._symbols(t)
        if self.opt.spell_acronyms:
            t = self._acronyms(t)
        if self.opt.merge_homophones:
            t = t.translate(_VARIANT_TABLE)
        if self.opt.map_punctuation:
            t = t.translate(_PUNCT_TABLE)
        return self._tidy(t)

    __call__ = normalize

    # ------------------------------------------------------------ pipeline
    @staticmethod
    def _clean(t: str) -> str:
        t = unicodedata.normalize("NFC", t)
        t = _ZERO_WIDTH.sub("", t)
        t = _EMOJI.sub(" ", t)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        # Arabic-Indic / fullwidth digits -> ASCII
        t = "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" and not c.isascii() else c for c in t)
        return t

    @staticmethod
    def _ethiopic_numerals(t: str) -> str:
        return num.ETHIOPIC_NUMERAL_RE.sub(lambda m: str(num.ethiopic_to_int(m.group(0))), t)

    # -- numbers ------------------------------------------------------------
    def _numbers(self, t: str) -> str:
        if self.opt.expand_units:
            t = self._unit_re.sub(lambda m: " " + UNIT_ABBREVIATIONS[m.group(1)], t)
        t = self._phone_numbers(t)
        t = self._dates(t)
        t = self._times(t)
        t = self._currency(t)
        t = self._percent(t)
        t = self._ranges(t)
        t = self._fractions(t)
        t = self._ordinals(t)
        t = self._plain_numbers(t)
        return t

    _PHONE_RE = re.compile(
        r"(?<!\d)(?P<plus>\+)?(?P<body>(?:251|0)[ -]?[1-9]\d{1,2}[ -]?\d{3}[ -]?\d{3,4})(?!\d)"
    )
    _LONG_DIGITS_RE = re.compile(r"(?<![\d.,])\d{10,}(?![\d.,])")

    def _phone_numbers(self, t: str) -> str:
        def rep(m):
            body = m.group("body")
            words = num.digits(body, group_sep=",")
            return ("ፕላስ " if m.group("plus") else "") + words
        t = self._PHONE_RE.sub(rep, t)
        t = self._LONG_DIGITS_RE.sub(lambda m: num.digits(m.group(0)), t)
        return t

    _DATE_NUMERIC_RE = re.compile(
        r"(?<![\d/.-])(?P<a>\d{1,4})[/.-](?P<b>\d{1,2})[/.-](?P<c>\d{1,4})(?![\d/.-])"
    )
    _MONTH_ALT = "|".join(sorted(map(re.escape, _MONTH_LOOKUP), key=len, reverse=True))
    _DATE_MONTH_FIRST_RE = re.compile(
        rf"(?P<prefix>{NUM_PREFIX})(?P<month>{_MONTH_ALT})\s+(?P<day>\d{{1,2}})(?!\d)(?P<rest>\s*(?:ቀን)?\s*[,/]?\s*(?P<year>\d{{4}})(?!\d))?"
    )
    _DATE_DAY_FIRST_RE = re.compile(
        rf"(?<!\d)(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_ALT})(?:\s+(?P<year>\d{{4}})(?!\d))?"
    )

    _SENT_BREAK = re.compile(r"[.!?።፧\n]")

    def _calendar_for(self, t: str, pos: int) -> str:
        """Decide Ethiopian vs Gregorian month names from era markers in the
        same sentence (the marker normally follows the year directly)."""
        after = t[pos: pos + 40]
        brk = self._SENT_BREAK.search(after)
        if brk:
            after = after[: brk.start()]
        if ERA_MARKERS_EC.search(after):
            return "ethiopian"
        if ERA_MARKERS_GC.search(after):
            return "gregorian"
        before = t[max(0, pos - 40): pos]
        brk = list(self._SENT_BREAK.finditer(before))
        if brk:
            before = before[brk[-1].end():]
        if ERA_MARKERS_GC.search(before):
            return "gregorian"
        if ERA_MARKERS_EC.search(before):
            return "ethiopian"
        return self.opt.default_calendar

    def _dates(self, t: str) -> str:
        def numeric(m):
            a, b, c = int(m.group("a")), int(m.group("b")), int(m.group("c"))
            cal = self._calendar_for(t, m.end())
            if len(m.group("a")) == 4:         # YYYY-MM-DD (ISO) -> Gregorian unless EC marker
                year, month, day = a, b, c
                if cal != "ethiopian" or not ERA_MARKERS_EC.search(t[m.end(): m.end() + 12]):
                    cal = "gregorian"
            elif len(m.group("c")) == 4:       # DD/MM/YYYY
                day, month, year = a, b, c
            else:                               # DD/MM/YY or not a date
                day, month, year = a, b, None
                if c > 31:
                    year = 2000 + c if c < 100 else c
            months = ETHIOPIAN_MONTHS if cal == "ethiopian" else GREGORIAN_MONTHS
            if not (1 <= month <= len(months)) or not (1 <= day <= 31):
                return m.group(0)
            out = f"{months[month - 1]} {num.cardinal(day)}"
            if cal == "ethiopian":
                out += f" {DAY_WORD}"
            if year is not None:
                out += f" {num.cardinal(year)}"
            return out

        t = self._DATE_NUMERIC_RE.sub(numeric, t)

        def month_first(m):
            day = int(m.group("day"))
            if not 1 <= day <= 31:
                return m.group(0)
            out = f"{m.group('prefix') or ''}{m.group('month')} {num.cardinal(day)}"
            if m.group("rest"):
                rest = m.group("rest")
                if "ቀን" not in rest:
                    out += f" {DAY_WORD}"
                else:
                    out += f" {DAY_WORD}"
                out += f" {num.cardinal(int(m.group('year')))}"
            return out

        t = self._DATE_MONTH_FIRST_RE.sub(month_first, t)

        def day_first(m):
            day = int(m.group("day"))
            if not 1 <= day <= 31:
                return m.group(0)
            out = f"{m.group('month')} {num.cardinal(day)} {DAY_WORD}"
            if m.group("year"):
                out += f" {num.cardinal(int(m.group('year')))}"
            return out

        t = self._DATE_DAY_FIRST_RE.sub(day_first, t)
        return t

    _TIME_RE = re.compile(r"(?<!\d)(?P<h>\d{1,2}):(?P<m>[0-5]\d)(?::(?P<s>[0-5]\d))?(?!\d)")

    def _times(self, t: str) -> str:
        def rep(m):
            h, mi = int(m.group("h")), int(m.group("m"))
            if h > 24:
                return m.group(0)
            out = f"{num.cardinal(h)} ሰዓት"
            if mi == 30:
                out += " ተኩል"
            elif mi:
                out += f" ከ{num.cardinal(mi)}"
            if m.group("s"):
                s = int(m.group("s"))
                if s:
                    out += f" ከ{num.cardinal(s)} ሴኮንድ"
            return out
        return self._TIME_RE.sub(rep, t)

    def _currency(self, t: str) -> str:
        def rep(m):
            major, minor = CURRENCIES[m.group("cur")]
            whole = int(m.group("num").replace(",", ""))
            frac = m.group("frac")
            out = f"{num.cardinal(whole)} {major}"
            if frac and int(frac):
                cents = int(frac) * (10 if len(frac) == 1 else 1)
                out += f" ከ{num.cardinal(cents)} {minor}"
            return out
        t = self._cur_prefix_re.sub(rep, t)
        t = self._cur_suffix_re.sub(rep, t)
        return t

    _PCT_RE = re.compile(r"(?<![\d.])(?P<num>\d+(?:\.\d+)?)\s?(?:%|ፐርሰንት|ፐርሰንት|በመቶ)")

    def _percent(self, t: str) -> str:
        return self._PCT_RE.sub(lambda m: f"{self._number_word(m.group('num'))} በመቶ", t)

    _RANGE_RE = re.compile(r"(?<!\d)(?P<a>\d+(?:\.\d+)?)\s?[-–—]\s?(?P<b>\d+(?:\.\d+)?)(?!\d)")

    def _ranges(self, t: str) -> str:
        return self._RANGE_RE.sub(
            lambda m: f"{self._number_word(m.group('a'))} እስከ {self._number_word(m.group('b'))}", t
        )

    _FRACTION_RE = re.compile(r"(?<![\d/])(?P<a>\d{1,2})/(?P<b>\d{1,3})(?![\d/])")

    def _fractions(self, t: str) -> str:
        def rep(m):
            a, b = int(m.group("a")), int(m.group("b"))
            if 0 < a < b <= 100:
                if a == 1 and b == 2:
                    return "ግማሽ"
                if a == 1 and b == 4:
                    return "ሩብ"
                return f"{num.cardinal(a)} {num.ordinal(b)}"
            if len(m.group("b")) == 2 and len(m.group("a")) <= 2 and 1900 < a <= 2200:
                return f"{num.cardinal(a)} {num.cardinal(b)}"  # unreachable, kept for clarity
            return f"{num.cardinal(a)} ስላሽ {num.cardinal(b)}"
        # fiscal years like 2016/17
        t = re.sub(r"(?<!\d)(?P<y>\d{4})/(?P<y2>\d{2})(?!\d)",
                   lambda m: f"{num.cardinal(int(m.group('y')))} {num.cardinal(int(m.group('y2')))}", t)
        return self._FRACTION_RE.sub(rep, t)

    _ORDINAL_RE = re.compile(rf"(?P<prefix>{NUM_PREFIX})(?P<num>\d+)(?:ተ)?ኛ(?P<suffix>{_ETH}*)")

    def _ordinals(self, t: str) -> str:
        return self._ORDINAL_RE.sub(
            lambda m: f"{m.group('prefix') or ''}{num.ordinal(int(m.group('num')))}{m.group('suffix')}", t
        )

    _NUMBER_RE = re.compile(
        rf"(?P<prefix>{NUM_PREFIX})(?P<neg>(?<![\w])-)?(?P<int>\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.(?P<frac>\d+))?(?P<suffix>{_ETH}*)"
    )

    def _plain_numbers(self, t: str) -> str:
        def rep(m):
            int_part = m.group("int").replace(",", "")
            frac = m.group("frac")
            if frac is not None:
                words = num.decimal(int_part, frac)
            elif int_part.startswith("0") and len(int_part) > 1:
                words = num.digits(int_part)   # codes like 007, 0911
            else:
                words = num.cardinal(int(int_part))
            if m.group("neg"):
                words = f"{num.MINUS} {words}"
            prefix = m.group("prefix") or ""
            suffix = m.group("suffix") or ""
            return f"{prefix}{words}{suffix}"
        return self._NUMBER_RE.sub(rep, t)

    @staticmethod
    def _number_word(s: str) -> str:
        if "." in s:
            a, b = s.split(".", 1)
            return num.decimal(a, b)
        return num.cardinal(int(s.replace(",", "")))

    # -- units / symbols / acronyms ------------------------------------------
    def _symbols(self, t: str) -> str:
        for sym, word in SYMBOLS.items():
            if sym in t:
                t = t.replace(sym, f" {word} ")
        return t

    def _acronyms(self, t: str) -> str:
        def rep(m):
            tok = m.group(1)
            if tok in LATIN_ACRONYMS:
                return LATIN_ACRONYMS[tok]
            if tok.isupper() and len(tok) <= 4:
                return " ".join(LATIN_LETTER_NAMES[c] for c in tok)
            return tok
        return self._acronym_re.sub(rep, t)

    # -- final tidy --------------------------------------------------------------
    @staticmethod
    def _tidy(t: str) -> str:
        t = _REPEATED_PUNCT.sub(r"\1", t)
        t = re.sub(r"\s*,\s*,\s*", ", ", t)
        t = re.sub(r",\s*\.", ".", t)
        t = re.sub(r"\s*\.\s*,", ".", t)
        t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
        t = re.sub(r"([,.;:!?])(?=[^\s\d])", r"\1 ", t)
        t = _SPACES.sub(" ", t)
        t = re.sub(r"\n\s*\n+", "\n", t)
        t = re.sub(r" *\n *", "\n", t)
        t = re.sub(r"^[\s,.;:]+", "", t)
        return t.strip()


_default = None


def normalize(text: str, **kwargs) -> str:
    """Module-level convenience wrapper with a cached default normaliser."""
    global _default
    if kwargs:
        return AmharicNormalizer(**kwargs).normalize(text)
    if _default is None:
        _default = AmharicNormalizer()
    return _default.normalize(text)


__all__ = ["AmharicNormalizer", "NormalizerOptions", "normalize", "ETHIOPIAN_MONTHS", "GREGORIAN_MONTHS"]
