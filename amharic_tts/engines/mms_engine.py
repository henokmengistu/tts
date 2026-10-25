"""Meta MMS-TTS Amharic back-end (fast fallback, CPU real-time).

Model: ``facebook/mms-tts-amh`` (VITS, 36M params, 16 kHz, CC-BY-NC-4.0).
Input is romanised with ``uroman`` exactly as during training.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .base import Engine

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/mms-tts-amh"


class MMSEngine(Engine):
    name = "mms"
    sample_rate = 16000

    def __init__(self, model_path: str | None = None, device: str = "cpu", threads: int | None = None,
                 noise_scale: float = 0.667, noise_scale_duration: float = 0.8, cache_dir: str | None = None):
        import torch
        import uroman
        from transformers import AutoTokenizer, VitsModel

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.device = device
        src = model_path or DEFAULT_MODEL_ID
        t0 = time.time()
        self.model = VitsModel.from_pretrained(src, cache_dir=cache_dir).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(src, cache_dir=cache_dir)
        self.uroman = uroman.Uroman()
        self.sample_rate = int(self.model.config.sampling_rate)
        self.model.noise_scale = noise_scale
        self.model.noise_scale_duration = noise_scale_duration
        logger.info("MMS-TTS loaded in %.1fs", time.time() - t0)

    def list_voices(self) -> list[str]:
        return ["default"]

    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0, seed: int | None = None, **kwargs) -> np.ndarray:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        rom = self.uroman.romanize_string(text, lcode="amh")
        inputs = self.tokenizer(rom, return_tensors="pt").to(self.device)
        self.model.speaking_rate = float(speed)
        with torch.inference_mode():
            wav = self.model(**inputs).waveform[0]
        return wav.float().cpu().numpy()


__all__ = ["MMSEngine", "DEFAULT_MODEL_ID"]
