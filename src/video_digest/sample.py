"""Video I/O: probe metadata, decode tiny thumbnails, grab exact frames.

Everything goes through the ffmpeg/ffprobe CLIs via subprocess, so any codec
ffmpeg supports works and there are no native bindings to break.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

# Size of the thumbnails all change detection runs on.
THUMB_SIZE = (128, 72)
# Decode width before the final thumbnail resize (matches the prototype, whose
# thresholds were tuned on 320px decodes downscaled with PIL bilinear).
DECODE_WIDTH = 320


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise FFmpegError(f"{tool} not found on PATH; install ffmpeg to use video-digest")
    return exe


def probe(path: str) -> VideoInfo:
    """Width, height and duration of the first video stream."""
    proc = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    if not info.get("streams"):
        raise FFmpegError(f"no video stream in {path}")
    s = info["streams"][0]
    return VideoInfo(width=int(s["width"]), height=int(s["height"]),
                     duration=float(info["format"]["duration"]))


def iter_thumbnails(path: str, fps: float, info: VideoInfo | None = None
                    ) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, 72x128x3 uint8 thumbnail) at `fps` samples per second."""
    info = info or probe(path)
    width = DECODE_WIDTH
    height = max(2, int(round(info.height * width / info.width / 2) * 2))
    proc = subprocess.Popen(
        [_require("ffmpeg"), "-loglevel", "error", "-i", str(path),
         "-vf", f"fps={fps},scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            thumb = Image.fromarray(frame).resize(THUMB_SIZE, Image.BILINEAR)
            yield i / fps, np.asarray(thumb)
            i += 1
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
    if rc != 0 and i == 0:
        raise FFmpegError(f"ffmpeg failed decoding {path}: {err.strip()}")


def grab_frame(path: str, t: float, width: int | None = None,
               duration: float | None = None) -> Image.Image:
    """The frame shown at `t` seconds as an RGB image, optionally scaled to `width`.

    Timestamps past the end are clamped to the last frame.
    """
    if duration is not None:
        t = min(max(t, 0.0), max(duration - 0.05, 0.0))
    t = max(t, 0.0)
    vf = ["-vf", f"scale={width}:-2"] if width else []
    for attempt in (t, max(t - 0.5, 0.0), max(t - 2.0, 0.0)):
        # -ss before -i seeks fast; modern ffmpeg still decodes to the exact frame.
        proc = subprocess.run(
            [_require("ffmpeg"), "-loglevel", "error", "-ss", f"{attempt:.3f}", "-i", str(path),
             "-frames:v", "1", *vf, "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            return Image.open(BytesIO(proc.stdout)).convert("RGB")
    raise FFmpegError(f"could not grab a frame at {t:.3f}s from {path}: "
                      f"{proc.stderr.decode(errors='replace').strip()}")
