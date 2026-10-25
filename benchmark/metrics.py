"""Objective TTS metrics used by the benchmark.

* ASR round-trip CER with ``snapwre/hohe-asr-amharic`` (w2v-BERT CTC, 5-6 % CER on read speech)
* UTMOS (UTMOS22 strong; k2-fsa/TTS_eval_models checkpoint) — predicted naturalness MOS
* DNSMOS P.835 (Microsoft, ONNX) — SIG / BAK / OVRL predicted quality, language independent
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = Path(os.environ.get("AMHARIC_TTS_BENCH_MODELS", "/home/user/models"))

# ------------------------------------------------------------------ text
_HOMO = {}
for src_base, dst_base in [(0x1210, 0x1200), (0x1280, 0x1200), (0x1220, 0x1230), (0x1340, 0x1338), (0x12D0, 0x12A0)]:
    for k in range(7):
        _HOMO[chr(src_base + k)] = chr(dst_base + k)
for cp in range(0x1288, 0x1290):
    _HOMO.pop(chr(cp), None)
_HOMO_TABLE = str.maketrans(_HOMO)
# first-order ሀ/ኣ ~ fourth-order ሃ/አ are pronounced the same in modern Amharic
_HOMO2 = str.maketrans({"ሃ": "ሀ", "ኣ": "አ", "ዓ": "አ", "ዐ": "አ", "ሓ": "ሀ", "ኻ": "ሀ", "ሐ": "ሀ", "ኀ": "ሀ", "ኃ": "ሀ"})


def canon(text: str) -> str:
    """Canonical form for CER: homophones merged, punctuation removed."""
    t = unicodedata.normalize("NFC", text).translate(_HOMO_TABLE).translate(_HOMO2)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def cer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = canon(ref), canon(hyp)
    if not r:
        return 0.0
    return float(jiwer.cer(r, h))


# ------------------------------------------------------------------ ASR
@lru_cache(maxsize=1)
def _asr():
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    path = MODELS / "hohe-asr-amharic"
    src = str(path) if path.exists() else "snapwre/hohe-asr-amharic"
    proc = AutoProcessor.from_pretrained(src)
    model = AutoModelForCTC.from_pretrained(src).eval()
    return proc, model


def transcribe(wav: np.ndarray, sr: int) -> str:
    import torch
    import librosa

    proc, model = _asr()
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    inp = proc(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        ids = model(**inp).logits.argmax(-1)
    return proc.batch_decode(ids)[0].replace("[AMH]", "").strip()


# ------------------------------------------------------------------ UTMOS
@lru_cache(maxsize=1)
def _utmos():
    import torch
    from omnivoice.eval.models.utmos import UTMOS22Strong

    ckpt = MODELS / "tts_eval" / "mos" / "utmos22_strong_step7459_v1.pt"
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download

        ckpt = Path(hf_hub_download("k2-fsa/TTS_eval_models", "mos/utmos22_strong_step7459_v1.pt"))
    model = UTMOS22Strong()
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval()


def utmos(wav: np.ndarray, sr: int) -> float:
    import torch
    import librosa

    model = _utmos()
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    with torch.inference_mode():
        out = model(x, 16000)
    return float(out.squeeze().item())


# ------------------------------------------------------------------ DNSMOS
_DNS_INPUT_LEN = 9.01


@lru_cache(maxsize=1)
def _dnsmos():
    import onnxruntime as ort

    p = MODELS / "dnsmos" / "sig_bak_ovr.onnx"
    if not p.exists():
        import urllib.request

        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx", p)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])


def dnsmos(wav: np.ndarray, sr: int) -> dict[str, float]:
    """DNSMOS P.835 (personalized=False): returns SIG, BAK, OVRL (1-5)."""
    import librosa

    sess = _dnsmos()
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    wav = wav.astype(np.float32)
    seg_len = int(sr * _DNS_INPUT_LEN)
    if len(wav) < seg_len:
        wav = np.tile(wav, int(np.ceil(seg_len / max(1, len(wav)))))[:seg_len]
    hop = int(sr * 1.0)
    sigs, baks, ovrs = [], [], []
    for start in range(0, len(wav) - seg_len + 1, hop):
        seg = wav[start:start + seg_len][None, :].astype(np.float32)
        raw_sig, raw_bak, raw_ovr = sess.run(None, {"input_1": seg})[0][0]
        sigs.append(raw_sig); baks.append(raw_bak); ovrs.append(raw_ovr)
    rs, rb, ro = float(np.mean(sigs)), float(np.mean(baks)), float(np.mean(ovrs))
    # polynomial mapping from the official dnsmos_local.py (non-personalized)
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return {"SIG": float(p_sig(rs)), "BAK": float(p_bak(rb)), "OVRL": float(p_ovr(ro))}


__all__ = ["canon", "cer", "transcribe", "utmos", "dnsmos"]
