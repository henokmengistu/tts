"""Audio post-processing helpers (pure numpy, no heavy dependencies)."""
from __future__ import annotations

import numpy as np


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m < 1e-6:
        return wav
    return (wav * (peak / m)).astype(np.float32)


def rms_normalize(wav: np.ndarray, target_dbfs: float = -20.0, peak_ceiling: float = 0.97) -> np.ndarray:
    """Scale to a target RMS level (dBFS) with a hard peak ceiling."""
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0
    if rms < 1e-6:
        return wav
    gain = 10 ** (target_dbfs / 20.0) / rms
    out = wav * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -45.0, keep_ms: int = 60) -> np.ndarray:
    """Trim leading/trailing silence, keeping ``keep_ms`` of padding."""
    if wav.size == 0:
        return wav
    frame = max(1, int(sr * 0.01))
    n = len(wav) // frame
    if n == 0:
        return wav
    frames = wav[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    loud = np.where(db > threshold_db)[0]
    if loud.size == 0:
        return wav
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * frame - keep)
    end = min(len(wav), (loud[-1] + 1) * frame + keep)
    return wav[start:end]


def fade(wav: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    n = min(len(wav) // 2, int(sr * ms / 1000))
    if n <= 0:
        return wav
    out = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


def concat_with_pauses(parts: list[np.ndarray], sr: int, pause_ms: int = 350) -> np.ndarray:
    out: list[np.ndarray] = []
    for i, p in enumerate(parts):
        if i:
            out.append(silence(sr, pause_ms))
        out.append(p.astype(np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


def to_int16(wav: np.ndarray) -> np.ndarray:
    return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)


__all__ = [
    "peak_normalize", "rms_normalize", "trim_silence", "fade", "silence",
    "concat_with_pauses", "resample", "to_int16",
]
