"""amharic-tts: high-quality, fully open Amharic text-to-speech.

Components
----------
* ``amharic_tts.text``   – rule-based Amharic text normaliser (numbers, dates,
  currency, abbreviations, Ethiopic numerals, acronyms, punctuation)
* ``amharic_tts.engines`` – acoustic back-ends: OmniVoice-Amharic (0.6B,
  zero-shot voice cloning, 24 kHz) and Meta MMS-TTS (fast CPU fallback)
* ``amharic_tts.AmharicTTS`` – one-call API with chunking, pauses and
  loudness normalisation
"""
from .text import AmharicNormalizer, NormalizerOptions, normalize
from .tts import AmharicTTS, encode_audio, write_audio

__version__ = "0.1.0"
__all__ = ["AmharicTTS", "AmharicNormalizer", "NormalizerOptions", "normalize", "encode_audio", "write_audio", "__version__"]
