"""OmniVoice-Amharic back-end (primary, highest quality).

Model: ``Lab-et/omnivoice-amharic`` (Apache-2.0) — a 0.6B non-autoregressive
discrete-diffusion TTS fine-tuned from k2-fsa/OmniVoice on ~331 h of Amharic.
The HiggsAudio-v2 tokenizer (``k2-fsa/OmniVoice/audio_tokenizer``) is shared
with the base model and is fetched automatically if missing.

Runs on CPU (float32, optional int8 dynamic quantisation) or any
accelerator supported by PyTorch.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Engine
from ..voices import VoiceRegistry, resolve_voice_dir

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Lab-et/omnivoice-amharic"
TOKENIZER_MODEL_ID = "k2-fsa/OmniVoice"


def _ensure_model(model_path: str | None, cache_dir: str | None) -> str:
    """Return a local directory holding the Amharic model + audio tokenizer."""
    from huggingface_hub import snapshot_download

    if model_path and os.path.isdir(model_path):
        local = model_path
    else:
        local = snapshot_download(model_path or DEFAULT_MODEL_ID, cache_dir=cache_dir)
    tok_dir = os.path.join(local, "audio_tokenizer")
    if not os.path.isdir(tok_dir):
        base = snapshot_download(TOKENIZER_MODEL_ID, allow_patterns=["audio_tokenizer/*"], cache_dir=cache_dir)
        src = os.path.join(base, "audio_tokenizer")
        try:
            os.symlink(src, tok_dir)
        except OSError:
            import shutil

            shutil.copytree(src, tok_dir)
    return local


class OmniVoiceEngine(Engine):
    name = "omnivoice"
    sample_rate = 24000

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        quantize: bool = False,
        num_steps: int = 16,
        guidance_scale: float = 2.0,
        threads: int | None = None,
        voices_dir: str | os.PathLike | None = None,
        cache_dir: str | None = None,
    ):
        import torch

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        if dtype == "auto":
            dtype = "float16" if str(device).startswith(("cuda", "xpu")) else "float32"
        torch_dtype = getattr(torch, dtype)
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale

        from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

        self._GenCfg = OmniVoiceGenerationConfig
        self._VoiceClonePrompt = VoiceClonePrompt

        local = _ensure_model(model_path, cache_dir)
        t0 = time.time()
        self.model = OmniVoice.from_pretrained(local, device_map=device, dtype=torch_dtype)
        self.model.eval()
        # int8 dynamic quantisation is opt-in: it is ~1.6x faster on CPU but
        # was found to degrade output badly in our tests (see README).
        self.quantized = False
        if quantize and str(device) == "cpu":
            self._quantize()
        self.sample_rate = int(self.model.sampling_rate)
        logger.info("OmniVoice loaded on %s in %.1fs (quantized=%s)", device, time.time() - t0, self.quantized)

        self.voices = VoiceRegistry(resolve_voice_dir(voices_dir))
        self._prompt_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _quantize(self) -> None:
        """int8 dynamic quantisation of the LLM linear layers (CPU only)."""
        torch = self.torch
        try:
            self.model.llm = torch.ao.quantization.quantize_dynamic(self.model.llm, {torch.nn.Linear}, dtype=torch.qint8)
            self.quantized = True
        except Exception as e:  # pragma: no cover
            logger.warning("int8 quantisation failed, continuing in float32: %s", e)

    def list_voices(self) -> list[str]:
        return self.voices.names()

    def _voice_prompt(self, voice: str | None):
        if voice in (None, "", "auto", "random"):
            return None
        if voice in self._prompt_cache:
            return self._prompt_cache[voice]
        spec = self.voices.get(voice)
        if spec is None:
            # treat as a path to reference audio for ad-hoc cloning
            p = Path(voice)
            if p.exists():
                prompt = self.model.create_voice_clone_prompt(ref_audio=str(p), ref_text=None) if self.model._asr_pipe else None
                if prompt is None:
                    raise ValueError(
                        "Ad-hoc cloning from a bare audio path needs a transcript: use clone(ref_audio, ref_text) or "
                        "register the voice."
                    )
                self._prompt_cache[voice] = prompt
                return prompt
            raise KeyError(f"unknown voice {voice!r}; available: {self.list_voices()}")
        cached = spec.prompt_path
        if cached and cached.exists():
            prompt = self._VoiceClonePrompt.load(str(cached))
        else:
            prompt = self.model.create_voice_clone_prompt(ref_audio=str(spec.audio_path), ref_text=spec.ref_text)
            if cached:
                try:
                    prompt.save(str(cached))
                except Exception:  # pragma: no cover
                    pass
        self._prompt_cache[voice] = prompt
        return prompt

    def clone_prompt(self, ref_audio: str, ref_text: str):
        """Build a reusable voice prompt from a 3-10 s reference clip."""
        return self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

    def register_voice(self, name: str, ref_audio: str, ref_text: str, gender: str = "", description: str = "") -> None:
        self.voices.add(name, ref_audio, ref_text, gender=gender, description=description)
        self._prompt_cache.pop(name, None)

    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        voice_prompt: Any = None,
        seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        prompt = voice_prompt if voice_prompt is not None else self._voice_prompt(voice)
        cfg = self._GenCfg(
            num_step=num_steps or self.num_steps,
            guidance_scale=self.guidance_scale if guidance_scale is None else guidance_scale,
            audio_chunk_duration=kwargs.pop("audio_chunk_duration", 12.0),
            audio_chunk_threshold=kwargs.pop("audio_chunk_threshold", 20.0),
            **kwargs,
        )
        with torch.inference_mode():
            audios = self.model.generate(
                text=text,
                language="Amharic",
                voice_clone_prompt=prompt,
                speed=None if abs(speed - 1.0) < 1e-3 else float(speed),
                generation_config=cfg,
            )
        return np.asarray(audios[0], dtype=np.float32)


__all__ = ["OmniVoiceEngine", "DEFAULT_MODEL_ID"]
