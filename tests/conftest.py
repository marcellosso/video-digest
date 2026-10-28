"""Shared fixtures: a short synthetic video with a known answer key."""
import random
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw

from video_digest.sheets import font

W, H, FPS = 320, 180, 15
DURATION = 9
WORD = "EVERYTHING"

# (start, end, background, text) — scene 2 opens with 0.75 s of scrambling
# letters (like a text-reveal animation) before WORD settles.
SCENES = [
    (0.0, 3.0, (30, 40, 70), "INTRO"),
    (3.0, 6.0, (20, 110, 50), WORD),
    (6.0, 9.0, (140, 30, 30), "THE END"),
]
SCRAMBLE_END = 3.75


def frame_at(t, rng):
    start, _, bg, text = next(s for s in SCENES if s[0] <= t < s[1])
    if text == WORD and t < SCRAMBLE_END:
        text = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in WORD)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    f = font(40)
    l, tp, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((W - (r - l)) // 2 - l, (H - (b - tp)) // 2 - tp), text, font=f, fill=(255, 255, 255))
    return img


def make_video(path):
    rng = random.Random(0)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    for i in range(DURATION * FPS):
        proc.stdin.write(np.asarray(frame_at(i / FPS, rng)).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    p = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    make_video(p)
    return p


@pytest.fixture(scope="session")
def tone_video(tmp_path_factory):
    """4 s of solid color with a 440 Hz tone: a video with an audio track, but no speech."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    p = tmp_path_factory.mktemp("video") / "tone.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=navy:s={W}x{H}:r={FPS}:d=4",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)],
        check=True)
    return p


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel: fixed segments, records each call."""

    SEGMENTS = [(0.0, 1.5, " Hello there."), (1.5, 2.0, "  "), (2.0, 3.25, " Click save now.")]

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        from types import SimpleNamespace
        self.calls.append((audio, kwargs))
        segs = (SimpleNamespace(start=s, end=e, text=t) for s, e, t in self.SEGMENTS)
        return segs, SimpleNamespace(language="en")


@pytest.fixture
def fake_whisper(monkeypatch):
    """Transcribe with FakeWhisper instead of a real model; works without the extra."""
    from video_digest import transcript
    model = FakeWhisper()
    loaded = []
    monkeypatch.setattr(transcript, "require", lambda: None)
    monkeypatch.setattr(transcript, "_load_model", lambda name: loaded.append(name) or model)
    model.loaded = loaded
    return model
