"""Benchmark fixtures: a synthetic video with a known answer key, and the showreel.

A fixture is a video plus a question set (bench/fixtures/<name>/questions.json).
The synthetic video is generated on demand and cached; the showreel is a 22 MB
file that stays out of git, so its path comes from an environment variable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from video_digest.sheets import font

BENCH_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BENCH_DIR / "fixtures"
CACHE_DIR = Path(os.environ.get("VIDEO_DIGEST_BENCH_CACHE", BENCH_DIR / ".cache"))
SHOWREEL_ENV = "VIDEO_DIGEST_SHOWREEL"


@dataclass(frozen=True)
class Fixture:
    name: str
    video: Path
    questions: list[dict]
    sha256: str | None = None  # expected file hash, when the fixture pins one


def load_questions(path: str | os.PathLike) -> list[dict]:
    qs = json.loads(Path(path).read_text())
    if isinstance(qs, dict):
        qs = qs["questions"]
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids in {path}")
    return qs


def resolve(name_or_path: str, questions: str | None = None) -> Fixture:
    """A fixture by name ("synthetic", "showreel") or a video path + --questions."""
    if name_or_path == "synthetic":
        return Fixture("synthetic", synthetic_video(),
                       load_questions(questions or FIXTURES_DIR / "synthetic" / "questions.json"))
    if name_or_path == "showreel":
        meta = json.loads((FIXTURES_DIR / "showreel" / "questions.json").read_text())
        return Fixture("showreel", showreel_video(), load_questions(
            questions or FIXTURES_DIR / "showreel" / "questions.json"), meta.get("sha256"))
    video = Path(name_or_path)
    if not video.is_file():
        raise SystemExit(f"{video}: no such file (fixtures: synthetic, showreel)")
    if not questions:
        raise SystemExit("--questions is required for your own video")
    return Fixture(video.stem, video, load_questions(questions))


def showreel_video() -> Path:
    candidates = [os.environ.get(SHOWREEL_ENV), FIXTURES_DIR / "showreel" / "benchmarket-showreel.mp4"]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser()
    raise SystemExit(
        "The showreel video is not in git (22 MB). Point the bench at your copy:\n"
        f"  export {SHOWREEL_ENV}=/path/to/benchmarket-showreel.mp4\n"
        "or place it at bench/fixtures/showreel/benchmarket-showreel.mp4 (gitignored).")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------- synthetic
# Mimics a "slides + screen recording" video: long static stretches, one
# motion segment, one slow fade, a 2-second flash that 2-second sampling can
# land on or miss, and a running clock in the corner. Ported from the
# prototype's make_test_video.py; the answer key is bench/fixtures/synthetic.
SYNTHETIC_VERSION = 1
FPS = 15
# (start, end, kind, text, background)
SCENES = [
    (0, 20, "slide", "Q3 Product Review", (30, 40, 70)),
    (20, 45, "slide", "Revenue grew 18% to $4.2M", (20, 70, 50)),
    (45, 60, "motion", "Live demo", (15, 15, 15)),
    (60, 93, "slide", "Launch date: March 14", (70, 30, 30)),
    (93, 95, "slide", "Secret code: BLUE-42", (200, 160, 0)),  # 2-second flash
    (95, 130, "slide", "Top risk: supplier delays in Vietnam", (60, 30, 80)),
    (130, 145, "fade", "Customer NPS: 61", None),
    (145, 180, "slide", "Next steps: hire 3 engineers", (30, 60, 90)),
]
DURATION = SCENES[-1][1]


def _draw_centered(img: Image.Image, text: str, size: int) -> None:
    d = ImageDraw.Draw(img)
    f = font(size)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((img.width - (r - l)) // 2 - l, (img.height - (b - t)) // 2 - t), text,
           font=f, fill=(255, 255, 255))


def synthetic_frame(t: float, width: int = 1280, height: int = 720) -> Image.Image:
    s = width / 1280
    start, end, kind, text, bg = next(sc for sc in SCENES if sc[0] <= t < sc[1])
    if kind == "fade":
        p = (t - start) / (end - start)
        bg = (int(20 + 120 * p), int(90 - 40 * p), int(120 - 60 * p))
    img = Image.new("RGB", (width, height), bg)
    if kind == "motion":
        x = width / 2 + 450 * s * math.sin(t * 1.3)
        y = height / 2 + 200 * s * math.cos(t * 2.1)
        r = 60 * s
        ImageDraw.Draw(img).ellipse([x - r, y - r, x + r, y + r], fill=(240, 90, 60))
        _draw_centered(img, text, round(40 * s))
    else:
        _draw_centered(img, text, round(56 * s))
    # small running clock, like a screen recording would have
    ImageDraw.Draw(img).text((20 * s, 20 * s), f"{t:6.1f}s", font=font(max(10, round(18 * s))),
                             fill=(180, 180, 180))
    return img


def make_synthetic(path: str | os.PathLike, width: int = 1280, height: int = 720,
                   fps: int = FPS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.mp4")
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)],
        stdin=subprocess.PIPE)
    try:
        for i in range(DURATION * fps):
            proc.stdin.write(np.asarray(synthetic_frame(i / fps, width, height)).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg failed to encode the synthetic video")
    os.replace(tmp, path)
    return path


def synthetic_video(width: int = 1280, height: int = 720) -> Path:
    """The cached synthetic video, generated on first use (~10 s)."""
    p = CACHE_DIR / f"synthetic-v{SYNTHETIC_VERSION}-{width}x{height}.mp4"
    return p if p.is_file() else make_synthetic(p, width, height)
