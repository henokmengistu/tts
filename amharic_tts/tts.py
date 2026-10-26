"""High-level Amharic TTS API.

    >>> from amharic_tts import AmharicTTS
    >>> tts = AmharicTTS()                      # OmniVoice-Amharic, CPU
    >>> tts.save("ሰላም፣ እንኳን ደህና መጣችሁ።", "hello.wav", voice="selam")

The class owns the text front-end (normalisation + chunking), the acoustic
engine, and audio post-processing (loudness, pauses, resampling, encoding).
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from . import audio as A
from .engines.base import Engine, SynthesisResult
from .text import AmharicNormalizer, NormalizerOptions, chunk_text

logger = logging.getLogger(__name__)

ENGINES = {"omnivoice": "amharic_tts.engines.omnivoice_engine:OmniVoiceEngine",
           "mms": "amharic_tts.engines.mms_engine:MMSEngine"}

DEFAULT_VOICE = "selam"


def _load_engine(name: str, **kwargs) -> Engine:
    if isinstance(name, Engine):
        return name
    spec = ENGINES.get(name)
    if spec is None:
        raise ValueError(f"unknown engine {name!r}; choose from {list(ENGINES)}")
    mod, cls = spec.split(":")
    import importlib

    return getattr(importlib.import_module(mod), cls)(**kwargs)


class AmharicTTS:
    def __init__(
        self,
        engine: str | Engine = "omnivoice",
        normalizer: AmharicNormalizer | NormalizerOptions | None = None,
        max_chunk_chars: int = 220,
        sentence_pause_ms: int = 0,
        sentence_mode: bool = False,
        paragraph_pause_ms: int = 500,
        target_dbfs: float = -20.0,
        **engine_kwargs,
    ):
        if isinstance(normalizer, NormalizerOptions):
            normalizer = AmharicNormalizer(normalizer)
        self.normalizer = normalizer or AmharicNormalizer()
        self.engine = _load_engine(engine, **engine_kwargs)
        self.max_chunk_chars = max_chunk_chars
        self.sentence_pause_ms = sentence_pause_ms
        self.sentence_mode = sentence_mode  # one engine call per sentence + explicit pauses
        self.paragraph_pause_ms = paragraph_pause_ms
        self.target_dbfs = target_dbfs

    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    def list_voices(self) -> list[str]:
        return self.engine.list_voices()

    def normalize(self, text: str) -> str:
        return self.normalizer.normalize(text)

    def default_voice(self) -> str | None:
        voices = self.list_voices()
        if not voices:
            return None
        return DEFAULT_VOICE if DEFAULT_VOICE in voices else voices[0]

    # ------------------------------------------------------------------
    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0, normalize: bool = True,
                   **kwargs) -> SynthesisResult:
        """Synthesize ``text`` (any length) to a single waveform."""
        t0 = time.time()
        norm = self.normalizer.normalize(text) if normalize else text
        if not norm:
            return SynthesisResult(np.zeros(0, dtype=np.float32), self.sample_rate, "", self.engine.name, voice)
        if voice is None:
            voice = self.default_voice()
        paragraphs = [p for p in norm.split("\n") if p.strip()]
        sr = self.sample_rate
        pieces: list[np.ndarray] = []
        for pi, para in enumerate(paragraphs):
            if pi:
                pieces.append(A.silence(sr, self.paragraph_pause_ms))
            chunks = self._chunks(para)
            for ci, chunk in enumerate(chunks):
                if ci and self.sentence_pause_ms:
                    pieces.append(A.silence(sr, self.sentence_pause_ms))
                wav = self.engine.synthesize(chunk, voice=voice, speed=speed, **kwargs)
                pieces.append(A.fade(np.asarray(wav, dtype=np.float32), sr))
        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        wav = A.rms_normalize(wav, self.target_dbfs)
        return SynthesisResult(wav, sr, norm, self.engine.name, voice, elapsed=time.time() - t0,
                               meta={"chunks": sum(len(self._chunks(p)) for p in paragraphs)})

    def _chunks(self, para: str) -> list[str]:
        if self.sentence_mode:
            from .text import split_sentences

            return [s for s in split_sentences(para) if s.strip()]
        return chunk_text(para, max_chars=self.max_chunk_chars)

    def stream(self, text: str, voice: str | None = None, speed: float = 1.0, **kwargs) -> Iterator[np.ndarray]:
        """Yield audio chunk by chunk (sentence groups) for progressive playback."""
        norm = self.normalizer.normalize(text)
        if voice is None:
            voice = self.default_voice()
        for para in (p for p in norm.split("\n") if p.strip()):
            for chunk in self._chunks(para):
                wav = self.engine.synthesize(chunk, voice=voice, speed=speed, **kwargs)
                yield A.rms_normalize(A.fade(np.asarray(wav, dtype=np.float32), self.sample_rate), self.target_dbfs)
            yield A.silence(self.sample_rate, self.paragraph_pause_ms)

    # ------------------------------------------------------------------
    def save(self, text: str, path: str | Path, voice: str | None = None, speed: float = 1.0,
             sample_rate: int | None = None, **kwargs) -> SynthesisResult:
        res = self.synthesize(text, voice=voice, speed=speed, **kwargs)
        write_audio(res.audio, res.sample_rate, path, sample_rate)
        return res

    def to_bytes(self, text: str, fmt: str = "wav", voice: str | None = None, speed: float = 1.0,
                 sample_rate: int | None = None, **kwargs) -> bytes:
        res = self.synthesize(text, voice=voice, speed=speed, **kwargs)
        return encode_audio(res.audio, res.sample_rate, fmt, sample_rate)


# ----------------------------------------------------------------------
def encode_audio(wav: np.ndarray, sr: int, fmt: str = "wav", out_sr: int | None = None) -> bytes:
    import soundfile as sf

    if out_sr and out_sr != sr:
        wav, sr = A.resample(wav, sr, out_sr), out_sr
    fmt = fmt.lower().lstrip(".")
    if fmt in ("wav", "flac", "ogg"):
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format=fmt.upper(), subtype="PCM_16" if fmt == "wav" else None)
        return buf.getvalue()
    if fmt in ("mp3", "m4a", "opus"):
        import subprocess
        import tempfile

        # Lossy formats are written to a real file, not a pipe, so ffmpeg can seek back and write the
        # Xing/duration header (players such as Windows Media Player, QuickTime and many phones refuse
        # or mis-seek a headerless stream).  MP3 is produced as MPEG-1 Layer III at 44.1 kHz, the one
        # profile every player decodes; 24 kHz input would otherwise become MPEG-2 LSF.
        codec = {"mp3": ["-codec:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-write_xing", "1", "-id3v2_version", "3"],
                 "m4a": ["-codec:a", "aac", "-b:a", "128k", "-ar", "44100", "-movflags", "+faststart"],
                 "opus": ["-codec:a", "libopus", "-b:a", "48k"]}[fmt]
        suffix = {"m4a": ".m4a", "opus": ".opus", "mp3": ".mp3"}[fmt]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            out_path = tmp.name
        try:
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                   *codec, out_path]
            subprocess.run(cmd, input=A.to_int16(wav).tobytes(), capture_output=True, check=True)
            return Path(out_path).read_bytes()
        finally:
            Path(out_path).unlink(missing_ok=True)
    raise ValueError(f"unsupported format {fmt!r}")


def write_audio(wav: np.ndarray, sr: int, path: str | Path, out_sr: int | None = None) -> None:
    path = Path(path)
    data = encode_audio(wav, sr, path.suffix.lstrip(".") or "wav", out_sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


__all__ = ["AmharicTTS", "encode_audio", "write_audio", "ENGINES", "DEFAULT_VOICE"]
