"""Optional local speech-to-text with faster-whisper. Never calls a cloud service.

    transcribe("talk.mp4")  ->  {"model": "base", "language": "en", "has_audio": True,
                                 "segments": [{"start": 0.0, "end": 2.4, "text": "..."}]}

ffmpeg decodes the audio track to 16 kHz mono floats, and a Whisper model runs
on the CPU (int8). The model downloads from the Hugging Face Hub the first time
it is used (~145 MB for the default "base") and loads from the local Hugging
Face cache after that, so later runs work offline.

Needs the optional dependency: pip install 'video-digest[transcript]'.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from functools import lru_cache

import numpy as np

from .sample import FFmpegError, _require

# "base" is ~145 MB, multilingual, and transcribes a few times faster than real
# time on a laptop CPU. "tiny" is smaller and rougher; "small" is ~3x slower and
# better. Any faster-whisper model name or a local CTranslate2 model folder works.
DEFAULT_MODEL = "base"
SAMPLE_RATE = 16_000


class TranscriptUnavailable(RuntimeError):
    """faster-whisper isn't installed."""


def default_model() -> str:
    return os.environ.get("VIDEO_DIGEST_WHISPER_MODEL") or DEFAULT_MODEL


def require() -> None:
    """Raise TranscriptUnavailable unless faster-whisper can be imported. Imports nothing."""
    if importlib.util.find_spec("faster_whisper") is None:
        raise TranscriptUnavailable(
            "transcript needs the 'transcript' extra: pip install 'video-digest[transcript]'")


def has_audio(path: str) -> bool:
    proc = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    return bool(proc.stdout.strip())


def load_audio(path: str) -> np.ndarray:
    """The first audio track as 16 kHz mono float32 samples."""
    proc = subprocess.run(
        [_require("ffmpeg"), "-loglevel", "error", "-i", str(path), "-vn", "-map", "0:a:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed decoding audio from {path}: "
                          f"{proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, np.float32)


@lru_cache(maxsize=2)
def _load_model(name: str):
    require()
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(path: str, model: str | None = None) -> dict:
    """Timestamped speech segments of the video's first audio track.

    A video without an audio track returns no segments and loads no model.
    """
    require()
    model = model or default_model()
    out = {"model": model, "language": None, "has_audio": has_audio(path), "segments": []}
    if not out["has_audio"]:
        return out
    audio = load_audio(path)
    if not audio.size:
        return out
    # The VAD filter skips silence, which keeps Whisper from inventing text there.
    segments, info = _load_model(model).transcribe(audio, vad_filter=True)
    out["segments"] = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": text}
                       for s in segments if (text := s.text.strip())]
    out["language"] = info.language
    return out
