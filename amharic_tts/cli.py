"""Command-line interface: ``amharic-tts``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _read_text(args) -> str:
    if args.text:
        return " ".join(args.text)
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("no text given (pass TEXT, --file, or pipe on stdin)")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="amharic-tts", description="Open-source Amharic text-to-speech")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("say", help="synthesize text to an audio file")
    s.add_argument("text", nargs="*", help="Amharic text (or use --file / stdin)")
    s.add_argument("-f", "--file", help="read text from a UTF-8 file")
    s.add_argument("-o", "--out", default="out.wav", help="output file (.wav/.flac/.ogg/.mp3)")
    s.add_argument("-v", "--voice", default=None, help="voice name, 'auto' for a random voice")
    s.add_argument("-e", "--engine", default="omnivoice", choices=["omnivoice", "mms"])
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--steps", type=int, default=16, help="diffusion steps (omnivoice): 8 fast .. 32 best")
    s.add_argument("--device", default=None, help="cpu / cuda:0 / mps (default: auto)")
    s.add_argument("--quant", action="store_true", help="experimental int8 CPU quantisation (faster, lower quality)")
    s.add_argument("--sr", type=int, default=None, help="output sample rate")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--raw", action="store_true", help="skip text normalisation")
    s.add_argument("--print-normalized", action="store_true")
    s.add_argument("--ref-audio", help="one-off voice cloning: 3-10 s reference clip (omnivoice)")
    s.add_argument("--ref-text", help="exact transcript of --ref-audio")

    n = sub.add_parser("normalize", help="print the normalised (spoken-form) text")
    n.add_argument("text", nargs="*")
    n.add_argument("-f", "--file")
    n.add_argument("--merge-homophones", action="store_true")

    v = sub.add_parser("voices", help="list or add voices")
    vs = v.add_subparsers(dest="vcmd")
    vs.add_parser("list")
    va = vs.add_parser("add", help="register a reference clip as a named voice")
    va.add_argument("name")
    va.add_argument("audio", help="3-10 s clean mono clip of the speaker")
    va.add_argument("text", help="exact transcript of the clip")
    va.add_argument("--gender", default="")
    va.add_argument("--description", default="")

    sv = sub.add_parser("serve", help="run the HTTP API / web demo")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("-e", "--engine", default="omnivoice", choices=["omnivoice", "mms"])
    sv.add_argument("--device", default=None)
    sv.add_argument("--steps", type=int, default=16)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "normalize":
        from .text import normalize

        print(normalize(_read_text(args), merge_homophones=args.merge_homophones))
        return 0

    if args.cmd == "voices":
        from .voices import VoiceRegistry, resolve_voice_dir

        reg = VoiceRegistry(resolve_voice_dir(None))
        if args.vcmd == "add":
            spec = reg.add(args.name, args.audio, args.text, gender=args.gender, description=args.description)
            print(f"registered voice {spec.name!r} -> {spec.audio_path}")
            return 0
        for spec in reg.all():
            print(f"{spec.name:12s} {spec.gender:7s} {spec.description}")
        return 0

    if args.cmd == "serve":
        from .server import run

        run(host=args.host, port=args.port, engine=args.engine, device=args.device, num_steps=args.steps)
        return 0

    if args.cmd != "say":
        p.print_help()
        return 1

    text = _read_text(args)
    from .tts import AmharicTTS

    kw = {}
    if args.engine == "omnivoice":
        kw.update(num_steps=args.steps, device=args.device, quantize=args.quant)
    t0 = time.time()
    tts = AmharicTTS(engine=args.engine, **kw)
    logging.getLogger("amharic-tts").info("engine ready in %.1fs", time.time() - t0)
    if args.print_normalized:
        print(tts.normalize(text))
    extra = {}
    if args.ref_audio:
        if args.engine != "omnivoice":
            raise SystemExit("--ref-audio needs the omnivoice engine")
        if not args.ref_text:
            raise SystemExit("--ref-text (transcript of the clip) is required with --ref-audio")
        extra["voice_prompt"] = tts.engine.clone_prompt(args.ref_audio, args.ref_text)
    res = tts.save(text, args.out, voice=args.voice, speed=args.speed, sample_rate=args.sr,
                   normalize=not args.raw, seed=args.seed, **extra)
    print(json.dumps({"out": args.out, "seconds": round(res.duration, 2), "elapsed": round(res.elapsed, 1),
                      "rtf": round(res.rtf, 2), "voice": res.voice, "engine": res.engine}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
