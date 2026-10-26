"""Preset voice registry.

A voice is a short (3-10 s) clean reference clip plus its exact transcript.
Presets shipped with the package come from the CC-BY-4.0 FLEURS Amharic
corpus; users can register their own with ``amharic-tts voices add``.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

PACKAGE_VOICES = Path(__file__).parent
USER_VOICES = Path(os.environ.get("AMHARIC_TTS_HOME", Path.home() / ".amharic_tts")) / "voices"


@dataclass
class VoiceSpec:
    name: str
    audio: str                   # file name relative to the voice dir (or absolute)
    ref_text: str
    gender: str = ""
    description: str = ""
    source: str = ""
    license: str = ""
    speed: float = 1.0           # tempo factor equalising this voice to the calibrated default rate
    _dir: Path | None = None

    @property
    def audio_path(self) -> Path:
        p = Path(self.audio)
        return p if p.is_absolute() else (self._dir or PACKAGE_VOICES) / p

    @property
    def prompt_path(self) -> Path | None:
        base = self._dir or PACKAGE_VOICES
        cache = USER_VOICES / "_prompts"
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return cache / f"{self.name}.pt"

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("_dir", None)
        return d


def resolve_voice_dir(path: str | os.PathLike | None) -> list[Path]:
    dirs = [PACKAGE_VOICES]
    if USER_VOICES.exists():
        dirs.append(USER_VOICES)
    if path:
        dirs.append(Path(path))
    return dirs


class VoiceRegistry:
    def __init__(self, dirs: list[Path]):
        self.dirs = dirs
        self._voices: dict[str, VoiceSpec] = {}
        for d in dirs:
            self._load_dir(d)

    def _load_dir(self, d: Path) -> None:
        f = d / "voices.json"
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for item in data.get("voices", []):
            spec = VoiceSpec(**{k: v for k, v in item.items() if k in VoiceSpec.__dataclass_fields__})
            spec._dir = d
            if spec.audio_path.exists():
                self._voices[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._voices)

    def get(self, name: str) -> VoiceSpec | None:
        return self._voices.get(name)

    def all(self) -> list[VoiceSpec]:
        return list(self._voices.values())

    def add(self, name: str, ref_audio: str, ref_text: str, gender: str = "", description: str = "") -> VoiceSpec:
        USER_VOICES.mkdir(parents=True, exist_ok=True)
        dst = USER_VOICES / f"{name}{Path(ref_audio).suffix.lower() or '.wav'}"
        shutil.copyfile(ref_audio, dst)
        f = USER_VOICES / "voices.json"
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"voices": []}
        data["voices"] = [v for v in data["voices"] if v.get("name") != name]
        spec = VoiceSpec(name=name, audio=dst.name, ref_text=ref_text, gender=gender, description=description, source="user")
        data["voices"].append(spec.to_json())
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        spec._dir = USER_VOICES
        self._voices[name] = spec
        pp = spec.prompt_path
        if pp and pp.exists():
            pp.unlink()
        return spec


__all__ = ["VoiceRegistry", "VoiceSpec", "resolve_voice_dir", "PACKAGE_VOICES", "USER_VOICES"]
