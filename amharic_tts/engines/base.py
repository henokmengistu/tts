from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SynthesisResult:
    audio: np.ndarray            # float32 mono in [-1, 1]
    sample_rate: int
    text: str                    # normalised text that was spoken
    engine: str
    voice: str | None = None
    elapsed: float = 0.0         # wall-clock seconds spent in the engine
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def rtf(self) -> float:
        return self.elapsed / self.duration if self.duration else float("inf")


class Engine:
    """Minimal interface every acoustic back-end implements."""

    name: str = "base"
    sample_rate: int = 24000

    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def list_voices(self) -> list[str]:
        return []

    def warmup(self) -> None:  # pragma: no cover - optional
        pass
