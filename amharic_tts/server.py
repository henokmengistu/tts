"""FastAPI server with an OpenAI-compatible speech endpoint and a tiny web demo.

    amharic-tts serve --port 8080
    curl -X POST localhost:8080/v1/audio/speech -H 'content-type: application/json' \
         -d '{"input":"ሰላም ዓለም","voice":"selam","response_format":"mp3"}' -o hi.mp3
"""

import os
import threading
from pathlib import Path
from typing import Optional

from .tts import AmharicTTS, encode_audio

_WEB = Path(__file__).with_name("web") / "index.html"


def external_backends() -> dict:
    """Which third-party voices the compare panel can fetch live (never required)."""
    out = {}
    try:
        import gtts  # noqa: F401
        out["google"] = True
    except ImportError:
        out["google"], out["google_reason"] = False, "pip install gtts"
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    try:
        import google.genai  # noqa: F401
        out["gemini"] = bool(key)
        if not key:
            out["gemini_reason"] = "set GEMINI_API_KEY"
    except ImportError:
        out["gemini"], out["gemini_reason"] = False, "pip install google-genai and set GEMINI_API_KEY"
    return out



def create_app(engine: str = "omnivoice", **engine_kwargs):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from pydantic import BaseModel

    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Amharic TTS", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Audio-Seconds", "X-RTF"])
    state = {"tts": None, "genai": {}}
    lock = threading.Lock()
    lock_ext = threading.Lock()

    def get_tts() -> AmharicTTS:
        with lock:
            if state["tts"] is None:
                state["tts"] = AmharicTTS(engine=engine, **engine_kwargs)
            return state["tts"]

    class SpeechRequest(BaseModel):
        input: str
        voice: Optional[str] = None
        speed: float = 1.0
        response_format: str = "wav"
        steps: Optional[int] = None
        sample_rate: Optional[int] = None
        normalize: bool = True
        seed: Optional[int] = None
        model: Optional[str] = None  # ignored; OpenAI compatibility

    class TextRequest(BaseModel):
        input: str

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_WEB.read_text(encoding="utf-8"))

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": engine, "loaded": state["tts"] is not None}

    @app.get("/v1/voices")
    def voices():
        tts = get_tts()
        reg = getattr(tts.engine, "voices", None)
        if reg is None:
            return {"voices": [{"name": n} for n in tts.list_voices()]}
        default = tts.default_voice()
        return {"voices": [{"name": v.name, "gender": v.gender, "description": v.description, "speed": v.speed,
                            "default": v.name == default} for v in reg.all()]}

    @app.get("/v1/voices/{name}/reference")
    def voice_reference(name: str):
        reg = getattr(get_tts().engine, "voices", None)
        spec = reg.get(name) if reg is not None else None
        if spec is None or not spec.audio_path.exists():
            raise HTTPException(404, f"unknown voice {name!r}")
        return FileResponse(str(spec.audio_path), media_type="audio/wav")

    @app.get("/v1/external")
    def external():
        return external_backends()

    @app.post("/v1/external/google")
    def external_google(req: TextRequest):
        """Google Translate's Amharic voice (gTTS), for side-by-side listening only."""
        try:
            from gtts import gTTS
        except ImportError:
            raise HTTPException(501, "pip install gtts")
        import io

        buf = io.BytesIO()
        try:
            gTTS(req.input, lang="am", lang_check=False).write_to_fp(buf)
        except Exception as e:  # noqa: BLE001 - network / quota
            raise HTTPException(502, f"Google TTS failed: {e}")
        return Response(content=buf.getvalue(), media_type="audio/mpeg")

    GEMINI_VOICES = [
        ("Kore", "female", "firm, clear"), ("Aoede", "female", "breezy"), ("Leda", "female", "youthful"),
        ("Zephyr", "female", "bright"), ("Autonoe", "female", "bright"), ("Despina", "female", "smooth"),
        ("Erinome", "female", "clear"), ("Sulafat", "female", "warm"), ("Vindemiatrix", "female", "gentle"),
        ("Achernar", "female", "soft"), ("Puck", "male", "upbeat"), ("Charon", "male", "informative"),
        ("Fenrir", "male", "excitable"), ("Orus", "male", "firm"), ("Iapetus", "male", "clear"),
        ("Algieba", "male", "smooth"), ("Schedar", "male", "even"), ("Gacrux", "male", "mature"),
        ("Achird", "male", "friendly"), ("Sadaltager", "male", "knowledgeable"),
    ]
    GEMINI_MODELS = [("gemini-3.8-flash-tts", "Gemini 3.8 Flash TTS"), ("gemini-3.8-flash-lite-tts", "Gemini 3.8 Flash-Lite TTS"),
                     ("gemini-3.1-flash-tts-preview", "Gemini 3.1 Flash TTS (preview)")]

    class GeminiRequest(BaseModel):
        input: str
        voice: Optional[str] = None
        model: Optional[str] = None
        style: Optional[str] = None   # natural-language delivery instruction, e.g. "read like a news anchor"

    @app.get("/v1/models")
    def models():
        """Every engine the studio can speak with, and its voices."""
        tts = get_tts()
        reg = getattr(tts.engine, "voices", None)
        default = tts.default_voice()
        ours = [{"name": v.name, "gender": v.gender, "description": v.description, "default": v.name == default,
                 "reference": f"/v1/voices/{v.name}/reference"} for v in (reg.all() if reg is not None else [])]
        ext = external_backends()
        return {"models": [
            {"id": "ours", "name": "amharic-tts", "detail": "OmniVoice-Amharic 0.6B, open source, runs here",
             "available": True, "voices": ours, "controls": ["speed", "steps", "seed"]},
            *[{"id": mid, "name": label, "detail": "Google Gemini, prompt-steerable, Amharic in preview",
               "available": bool(ext.get("gemini")), "reason": ext.get("gemini_reason", ""), "key_optional": True,
               "voices": [{"name": n, "gender": g, "description": d} for n, g, d in GEMINI_VOICES], "controls": ["style"]}
              for mid, label in GEMINI_MODELS],
            {"id": "google", "name": "Google Translate", "detail": "the voice behind translate.google.com (gTTS)",
             "available": bool(ext.get("google")), "reason": ext.get("google_reason", ""),
             "voices": [{"name": "default", "gender": "female", "description": "single Amharic voice"}], "controls": []},
        ]}

    @app.post("/v1/external/gemini")
    def external_gemini(req: GeminiRequest, request: Request):
        """Gemini TTS, for side-by-side listening. Key: GEMINI_API_KEY on the server or an X-Gemini-Key header."""
        key = request.headers.get("x-gemini-key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise HTTPException(501, "set GEMINI_API_KEY on the server or paste your own key in the studio")
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise HTTPException(501, "pip install google-genai")
        model = req.model or os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
        if model not in {m for m, _ in GEMINI_MODELS}:
            raise HTTPException(400, f"unknown Gemini TTS model {model!r}")
        voice = req.voice or os.environ.get("GEMINI_TTS_VOICE", "Kore")
        text = f"{req.style.strip()}:\n{req.input}" if req.style and req.style.strip() else req.input
        with lock_ext:
            clients = state.setdefault("genai", {})
            if key not in clients:
                clients[key] = genai.Client(api_key=key)
            client = clients[key]
        try:
            resp = client.models.generate_content(
                model=model, contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
            part = resp.candidates[0].content.parts[0].inline_data
            pcm, mime = part.data, part.mime_type or ""
        except Exception as e:  # noqa: BLE001 - quota, model availability
            raise HTTPException(502, f"Gemini TTS failed: {e}")
        if mime.startswith("audio/wav") or pcm[:4] == b"RIFF":
            return Response(content=pcm, media_type="audio/wav")
        import re as _re

        import numpy as np

        sr = int(m.group(1)) if (m := _re.search(r"rate=(\d+)", mime)) else 24000  # raw audio/L16 PCM
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return Response(content=encode_audio(wav, sr, "wav"), media_type="audio/wav")

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(400, "empty input")
        if len(req.input) > 20000:
            raise HTTPException(413, "input too long (max 20000 chars)")
        tts = get_tts()
        kw = {}
        if req.steps and tts.engine.name == "omnivoice":
            kw["num_steps"] = max(4, min(64, req.steps))
        if req.seed is not None and tts.engine.name == "omnivoice":
            kw["seed"] = req.seed
        with lock:  # one synthesis at a time on CPU
            try:
                res = tts.synthesize(req.input, voice=req.voice or None, speed=req.speed, normalize=req.normalize, **kw)
            except KeyError as e:
                raise HTTPException(400, str(e))
        fmt = req.response_format.lower()
        data = encode_audio(res.audio, res.sample_rate, fmt, req.sample_rate)
        media = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg", "opus": "audio/ogg",
                 "m4a": "audio/mp4"}.get(fmt, "application/octet-stream")
        return Response(content=data, media_type=media, headers={"X-Audio-Seconds": f"{res.duration:.2f}", "X-RTF": f"{res.rtf:.2f}"})

    return app


def run(host: str = "0.0.0.0", port: int = 8080, engine: str = "omnivoice", preload: bool = False, **engine_kwargs):
    import uvicorn

    app = create_app(engine, **{k: v for k, v in engine_kwargs.items() if v is not None})
    if preload or os.environ.get("AMHARIC_TTS_PRELOAD") == "1":
        app.state.preload = True

        @app.on_event("startup")
        def _warm():  # load weights before the first request so hosted instances answer /health truthfully
            threading.Thread(target=lambda: app.router.routes and next(r for r in app.router.routes if r.path == "/v1/voices").endpoint(), daemon=True).start()

    uvicorn.run(app, host=host, port=port)


__all__ = ["create_app", "run"]
