"""Chatterbox-Amharic back-end (candidate engine).

Base: ResembleAI Chatterbox Multilingual v3 (MIT) — a 0.5 B autoregressive
text-to-speech-token model (T3) plus the S3Gen flow-matching decoder and
HiFi-GAN vocoder at 24 kHz.  Amharic comes from the LoRA adapter published as
``natidotnet/chatterbox-amharic`` (CC-BY-SA-4.0, trained on ~190 h WaxalNLP +
Common Voice + studio speech).  Only T3 is adapted; the acoustic decoder is the
stock full-band one, so the output keeps Chatterbox's signal quality.

This module is an independent re-implementation of the adapter's loading
recipe (enlarged text vocabulary, PEFT LoRA on the attention/MLP projections,
full-rank ``text_emb``/``text_head`` from ``modules_to_save``, Fidel tokenizer
without a language token), written against the public ``chatterbox-tts``
package API.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from .base import Engine
from ..voices import VoiceRegistry, resolve_voice_dir

logger = logging.getLogger(__name__)

ADAPTER_REPO = "natidotnet/chatterbox-amharic"
BASE_REPO = "ResembleAI/chatterbox"
BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_FILE = "t3_mtl23ls_v3.safetensors"
BASE_FILES = ["ve.pt", T3_FILE, "s3gen.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"]
ADAPTER_FILES = ["new_lang_adapter/adapter_config.json", "new_lang_adapter/adapter_model.safetensors",
                 "tokenizer/tokenizer.json", "training_config.json"]

SPACE = "[SPACE]"
_SENT_SPLIT = re.compile(r"(?<=[።?!])\s+")
_KEEP_PUNCT = set("።፣፤?! ")


def _download(repo: str, patterns: list[str], revision: str | None = None, cache_dir: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=patterns, revision=revision, cache_dir=cache_dir))


class _FidelTokenizer:
    """Plain character/BPE tokenizer over the extended vocabulary: no language
    token, no lower-casing, no NFKD, spaces become ``[SPACE]``."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text.replace(" ", SPACE)).ids


class ChatterboxEngine(Engine):
    name = "chatterbox"
    sample_rate = 24000

    def __init__(self, adapter_path: str | None = None, base_path: str | None = None, device: str = "cpu",
                 threads: int | None = None, temperature: float = 0.6, cfg_weight: float = 0.5,
                 repetition_penalty: float = 1.2, min_p: float = 0.05, top_p: float = 1.0, exaggeration: float = 0.5,
                 voices_dir: str | os.PathLike | None = None, cache_dir: str | None = None, watermark: bool = False):
        import torch
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.models.t3 import T3
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.voice_encoder import VoiceEncoder
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
        from peft import PeftModel
        from safetensors.torch import load_file

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.device = device
        self.gen = dict(temperature=temperature, cfg_weight=cfg_weight, repetition_penalty=repetition_penalty,
                        min_p=min_p, top_p=top_p, exaggeration=exaggeration)
        t0 = time.time()
        adapter = Path(adapter_path) if adapter_path and os.path.isdir(adapter_path) else _download(ADAPTER_REPO, ADAPTER_FILES, cache_dir=cache_dir)
        base = Path(base_path) if base_path and os.path.isdir(base_path) else _download(BASE_REPO, BASE_FILES, BASE_REVISION, cache_dir)

        import json

        tcfg = json.loads((adapter / "training_config.json").read_text(encoding="utf-8"))
        new_vocab = int(tcfg.get("new_vocab_size", 2698))
        base_vocab = int(tcfg.get("base_vocab_size", 2454))

        # --- T3 with enlarged text vocabulary -------------------------------
        hp = T3Config(text_tokens_dict_size=new_vocab)
        t3 = T3(hp)
        state = load_file(str(base / T3_FILE))
        if "model" in state:
            state = state["model"][0]
        own = t3.state_dict()
        for k, v in state.items():
            if k in ("text_emb.weight", "text_head.weight"):
                own[k][: v.shape[0]].copy_(v)
            elif k in own:
                own[k].copy_(v)
        t3.load_state_dict(own)
        t3 = PeftModel.from_pretrained(t3, str(adapter / "new_lang_adapter"))
        t3 = t3.merge_and_unload()
        t3.to(device).eval()

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(base / "ve.pt", map_location="cpu", weights_only=True))
        ve.to(device).eval()
        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(base / "s3gen.pt", map_location="cpu", weights_only=True))
        s3gen.to(device).eval()
        conds = Conditionals.load(base / "conds.pt", map_location="cpu").to(device) if (base / "conds.pt").exists() else None

        # Reuse the library's conditioning/inference plumbing with our tokenizer.
        self.tts = ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer=None, device=device, conds=conds)
        self.tokenizer = _FidelTokenizer(str(adapter / "tokenizer/tokenizer.json"))
        self.watermark = watermark
        self.sample_rate = int(self.tts.sr)
        self.voices = VoiceRegistry(resolve_voice_dir(voices_dir))
        self._cond_cache: dict[str, object] = {}
        logger.info("Chatterbox-Amharic loaded in %.1fs (vocab %d -> %d)", time.time() - t0, base_vocab, new_vocab)

    # ------------------------------------------------------------------
    def list_voices(self) -> list[str]:
        return self.voices.names()

    def _conditionals(self, voice: str | None):
        key = voice or "__builtin__"
        if key in self._cond_cache:
            return self._cond_cache[key]
        spec = self.voices.get(voice) if voice else None
        if spec is None and voice and Path(voice).exists():
            path = voice
        elif spec is not None:
            path = str(spec.audio_path)
        else:
            path = None
        if path is None:
            if self.tts.conds is None:
                raise KeyError(f"unknown voice {voice!r}; available: {self.list_voices()}")
            conds = self.tts.conds
        else:
            self.tts.prepare_conditionals(path, exaggeration=self.gen["exaggeration"])
            conds = self.tts.conds
        self._cond_cache[key] = conds
        return conds

    @staticmethod
    def _clean(text: str) -> str:
        # keep letters, digits (already verbalised by the normaliser), and the five trained marks
        text = re.sub(r"[.]", "።", text)
        text = re.sub(r"[,;:]", "፣", text)
        text = "".join(ch for ch in text if ch.isalnum() or ch in _KEEP_PUNCT)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in "።?!":
            text += "።"
        return text

    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0, seed: int | None = None, **kwargs) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.models.s3tokenizer import S3_TOKEN_RATE
        from chatterbox.models.s3tokenizer import drop_invalid_tokens

        if seed is not None:
            torch.manual_seed(seed)
        conds = self._conditionals(voice)
        self.tts.conds = conds
        text = self._clean(text)
        pieces: list[np.ndarray] = []
        gap = np.zeros(int(0.25 * self.sample_rate), dtype=np.float32)
        gen = dict(self.gen)
        gen.update({k: v for k, v in kwargs.items() if k in gen})
        for sent in [s for s in _SENT_SPLIT.split(text) if s.strip()]:
            ids = self.tokenizer.encode(sent)
            toks = torch.tensor([ids], dtype=torch.long, device=self.device)
            toks = torch.cat([toks, toks], dim=0)
            toks = F.pad(toks, (1, 0), value=self.tts.t3.hp.start_text_token)
            toks = F.pad(toks, (0, 1), value=self.tts.t3.hp.stop_text_token)
            with torch.inference_mode():
                speech_tokens = self.tts.t3.inference(
                    t3_cond=conds.t3, text_tokens=toks, max_new_tokens=1000, temperature=gen["temperature"],
                    cfg_weight=gen["cfg_weight"], repetition_penalty=gen["repetition_penalty"], min_p=gen["min_p"], top_p=gen["top_p"],
                )[0]
                speech_tokens = drop_invalid_tokens(speech_tokens).to(self.device)
                wav, _ = self.tts.s3gen.inference(speech_tokens=speech_tokens, ref_dict=conds.gen)
                wav = wav.squeeze(0).detach().cpu().numpy()
                n_tokens = int(speech_tokens.shape[-1])
                wav = wav[: max(1, n_tokens - 1) * (S3GEN_SR // S3_TOKEN_RATE)]
            if self.watermark:
                wav = self.tts.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
            if pieces:
                pieces.append(gap)
            pieces.append(wav.astype(np.float32))
        out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        if abs(speed - 1.0) > 1e-3:
            import librosa

            out = librosa.effects.time_stretch(out, rate=float(speed))
        return out


__all__ = ["ChatterboxEngine"]
