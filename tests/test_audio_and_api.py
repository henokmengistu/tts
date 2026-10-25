import numpy as np

from amharic_tts import audio as A
from amharic_tts.engines.base import Engine
from amharic_tts.tts import AmharicTTS, encode_audio


class DummyEngine(Engine):
    name = "dummy"
    sample_rate = 16000

    def __init__(self):
        self.calls = []

    def list_voices(self):
        return ["v1"]

    def synthesize(self, text, voice=None, speed=1.0, **kw):
        self.calls.append((text, voice))
        n = int(0.05 * self.sample_rate * max(1, len(text)))
        return (0.1 * np.sin(np.arange(n) * 0.1)).astype(np.float32)


def test_audio_helpers():
    sr = 16000
    x = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr), np.zeros(sr)]).astype(np.float32)
    t = A.trim_silence(x, sr)
    assert sr * 0.9 < len(t) < sr * 1.3
    y = A.rms_normalize(x, -20.0)
    assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) + 20) < 1.0 or np.max(np.abs(y)) <= 0.97
    assert np.max(np.abs(A.peak_normalize(x, 0.9))) <= 0.9 + 1e-6
    f = A.fade(np.ones(sr, dtype=np.float32), sr, ms=10)
    assert f[0] == 0 and abs(f[sr // 2] - 1) < 1e-6
    c = A.concat_with_pauses([np.ones(10, dtype=np.float32)] * 2, sr, pause_ms=10)
    assert len(c) == 20 + int(sr * 0.01)


def test_api_chunks_and_normalises():
    eng = DummyEngine()
    tts = AmharicTTS(engine=eng, paragraph_pause_ms=100)
    text = "ዶ/ር አበበ 25% አለ።\n\nሁለተኛ አንቀጽ።"
    res = tts.synthesize(text)
    assert res.text == "ዶክተር አበበ ሃያ አምስት በመቶ አለ.\nሁለተኛ አንቀጽ."
    assert [c[0] for c in eng.calls] == ["ዶክተር አበበ ሃያ አምስት በመቶ አለ.", "ሁለተኛ አንቀጽ."]
    assert res.voice == "v1" and res.sample_rate == 16000 and res.duration > 1.0
    wav = encode_audio(res.audio, res.sample_rate, "wav")
    assert wav[:4] == b"RIFF"
    assert len(tts.to_bytes("ሰላም", fmt="flac")) > 100


def test_long_text_is_split():
    eng = DummyEngine()
    tts = AmharicTTS(engine=eng, max_chunk_chars=60)
    text = " ".join(["ይህ በጣም ረጅም ዓረፍተ ነገር ነው፣ እና መከፋፈል አለበት።"] * 6)
    tts.synthesize(text)
    assert len(eng.calls) >= 4
    assert all(len(c[0]) <= 72 for c in eng.calls)
