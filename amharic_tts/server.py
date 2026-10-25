"""FastAPI server with an OpenAI-compatible speech endpoint and a tiny web demo.

    amharic-tts serve --port 8080
    curl -X POST localhost:8080/v1/audio/speech -H 'content-type: application/json' \
         -d '{"input":"ሰላም ዓለም","voice":"selam","response_format":"mp3"}' -o hi.mp3
"""

import threading
from typing import Optional

from .tts import AmharicTTS, encode_audio

_HTML = """<!doctype html><html lang="am"><head><meta charset="utf-8"><title>Amharic TTS</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,'Noto Sans Ethiopic',sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;background:#0f1115;color:#e8e8e8}
textarea{width:100%;min-height:9rem;font-size:1.15rem;padding:.8rem;border-radius:8px;border:1px solid #333;background:#181b22;color:#fff}
select,button,input{font-size:1rem;padding:.5rem .8rem;border-radius:8px;border:1px solid #333;background:#181b22;color:#fff;margin:.3rem .3rem .3rem 0}
button{background:#2b6cb0;cursor:pointer}button:disabled{opacity:.5}audio{width:100%;margin-top:1rem}small{color:#999}pre{white-space:pre-wrap;background:#181b22;padding:.6rem;border-radius:8px}</style></head>
<body><h1>አማርኛ ድምፅ ማሰሚያ <small>Amharic TTS</small></h1>
<textarea id="t">ሰላም፣ እንኳን ደህና መጣችሁ። ዛሬ ጥቅምት 15 ቀን 2019 ዓ.ም. ነው። ዶ/ር አበበ በት/ቤቱ 1250 ብር ለገሱ።</textarea><br>
<label>ድምፅ <select id="v"></select></label>
<label>ፍጥነት <input id="s" type="number" step="0.05" min="0.7" max="1.4" value="1.0" style="width:5rem"></label>
<label>ጥራት <select id="q"><option value="8">ፈጣን (8)</option><option value="16" selected>መደበኛ (16)</option><option value="32">ከፍተኛ (32)</option></select></label>
<button id="b">አሰማ ▶</button> <span id="st"></span>
<audio id="a" controls></audio>
<details><summary>የተለመደ ጽሑፍ (normalized)</summary><pre id="n"></pre></details>
<script>
const V=document.getElementById('v');fetch('/v1/voices').then(r=>r.json()).then(j=>{for(const x of j.voices){const o=document.createElement('option');o.value=x.name;o.textContent=x.name+(x.gender?' ('+x.gender+')':'');V.appendChild(o)}const o=document.createElement('option');o.value='auto';o.textContent='auto (random)';V.appendChild(o)});
document.getElementById('b').onclick=async()=>{const b=document.getElementById('b'),st=document.getElementById('st');b.disabled=true;st.textContent='እየተዘጋጀ ነው…';const t0=Date.now();
const txt=document.getElementById('t').value;fetch('/v1/normalize',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({input:txt})}).then(r=>r.json()).then(j=>document.getElementById('n').textContent=j.normalized);
const r=await fetch('/v1/audio/speech',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({input:txt,voice:V.value,speed:parseFloat(document.getElementById('s').value),steps:parseInt(document.getElementById('q').value),response_format:'wav'})});
if(!r.ok){st.textContent='ስህተት: '+await r.text();b.disabled=false;return}
const blob=await r.blob();const a=document.getElementById('a');a.src=URL.createObjectURL(blob);a.play();st.textContent=((Date.now()-t0)/1000).toFixed(1)+' s';b.disabled=false};
</script></body></html>"""


def create_app(engine: str = "omnivoice", **engine_kwargs):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
    from pydantic import BaseModel

    app = FastAPI(title="Amharic TTS", version="0.1.0")
    state = {"tts": None}
    lock = threading.Lock()

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
        model: Optional[str] = None  # ignored; OpenAI compatibility

    class TextRequest(BaseModel):
        input: str

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _HTML

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": engine, "loaded": state["tts"] is not None}

    @app.get("/v1/voices")
    def voices():
        tts = get_tts()
        reg = getattr(tts.engine, "voices", None)
        if reg is None:
            return {"voices": [{"name": n} for n in tts.list_voices()]}
        return {"voices": [{"name": v.name, "gender": v.gender, "description": v.description} for v in reg.all()]}

    @app.post("/v1/normalize")
    def normalize(req: TextRequest):
        return {"normalized": get_tts().normalize(req.input)}

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


def run(host: str = "0.0.0.0", port: int = 8080, engine: str = "omnivoice", **engine_kwargs):
    import uvicorn

    uvicorn.run(create_app(engine, **{k: v for k, v in engine_kwargs.items() if v is not None}), host=host, port=port)


__all__ = ["create_app", "run"]
