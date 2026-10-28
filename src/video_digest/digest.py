"""The Digest object, its cache, and the on-disk digest format.

A digest is a plain folder, so any tool or model can read it:

    .video-digest/<key>/
      digest.json      # metadata, settings, keyframes, transcript
      digest.md        # the text timeline sent to the model
      sheets/          # sheet_01.jpg, sheet_02.jpg, ...
      frames/          # zoomed frames fetched later via Digest.frames()

<key> is "<file-hash>-<settings-hash>": the first 16 hex chars of the video's
SHA-256, then 8 of a hash over the selection settings and format version.
Building the same file with the same settings reuses the folder.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import __version__
from .sample import grab_frame, iter_thumbnails, probe
from .select import SelectSettings, select_keyframes
from .sheets import PER_SHEET, TILE_WIDTH, fmt_ts, tile_sheet, write_sheets

# Bump when the folder layout, digest.json schema or selection output changes,
# so stale caches are rebuilt instead of reused.
FORMAT_VERSION = 1
DEFAULT_CACHE_DIR = ".video-digest"
MAX_FRAMES = PER_SHEET


def file_hash(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def settings_hash(settings: SelectSettings) -> str:
    payload = json.dumps({"format": FORMAT_VERSION, "tile_width": TILE_WIDTH,
                          "per_sheet": PER_SHEET, **dataclasses.asdict(settings)},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def default_cache_dir() -> Path:
    return Path(os.environ.get("VIDEO_DIGEST_CACHE", DEFAULT_CACHE_DIR))


def _image_block(img: Image.Image, quality: int = 85) -> dict:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def _check_provider(provider: str) -> None:
    if provider != "anthropic":
        raise ValueError(f"unsupported provider {provider!r}; supported: 'anthropic'")


# ------------------------------------------------------------------ zooming
@dataclass
class Frames:
    """The answer to a zoom request: one full frame, or one labeled contact sheet."""

    timestamps: list[float]
    image: Image.Image
    label: str
    path: Path | None = None

    def to_content(self, provider: str = "anthropic") -> list[dict]:
        """Content blocks (a text label, then the image) for a model message."""
        _check_provider(provider)
        return [{"type": "text", "text": self.label}, _image_block(self.image, 92)]


def _normalize_timestamps(timestamps: float | Iterable[float]) -> list[float]:
    if isinstance(timestamps, (int, float)):
        timestamps = [timestamps]
    ts = sorted({round(float(t), 2) for t in timestamps})
    if not ts:
        raise ValueError("need at least one timestamp")
    if len(ts) > MAX_FRAMES:
        raise ValueError(f"at most {MAX_FRAMES} timestamps per request, got {len(ts)}")
    if any(t < 0 for t in ts):
        raise ValueError("timestamps must be >= 0")
    return ts


def get_frames(video: str | os.PathLike, timestamps: float | Iterable[float],
               out_dir: str | os.PathLike | None = None,
               duration: float | None = None) -> Frames:
    """Zoom into exact moments of `video`.

    One timestamp returns the full-resolution frame. Two to nine return one
    contact sheet with each tile labeled by its timestamp. Timestamps past the
    end are clamped to the last frame. If `out_dir` is given the image is also
    saved there as JPEG.
    """
    ts = _normalize_timestamps(timestamps)
    video = str(video)
    if duration is None:
        duration = probe(video).duration
    if len(ts) == 1:
        image = grab_frame(video, ts[0], duration=duration)
        label = f"Frame at {fmt_ts(ts[0])} ({ts[0]:g}s)"
    else:
        tiles = [grab_frame(video, t, TILE_WIDTH, duration=duration) for t in ts]
        image = tile_sheet(tiles, [fmt_ts(t) for t in ts])
        label = "Frames at " + ", ".join(f"{fmt_ts(t)} ({t:g}s)" for t in ts)
    path = None
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / ("at_" + "_".join(f"{t:.2f}" for t in ts) + ".jpg")
        image.save(path, quality=92)
    return Frames(timestamps=ts, image=image, label=label, path=path)


# ------------------------------------------------------------------ digest
def timeline_text(d: dict) -> str:
    """The text part of the digest that goes to the model with the sheets."""
    kfs, n_sheets = d["keyframes"], len(d["sheets"])
    lines = [f"VIDEO DIGEST — {fmt_ts(d['duration'])} long, {d['width']}x{d['height']}.",
             f"{len(kfs)} keyframes were kept (a new one only when the picture changed). "
             f"They are shown in {n_sheets} contact sheet(s), labeled #N and timestamp. "
             "Each keyframe represents the video until the next one.",
             "If you need more detail, ask for a full-resolution frame at a timestamp.",
             ""]
    if n_sheets > 1:
        for s in range(n_sheets):
            first, last = s * PER_SHEET + 1, min((s + 1) * PER_SHEET, len(kfs))
            lines.append(f"Sheet {s + 1}: keyframes #{first}–#{last}")
        lines.append("")
    lines.append("Keyframe timeline:")
    for i, k in enumerate(kfs):
        notes = [f"{k['until'] - k['t']:.0f}s"]
        if k["change"] is not None:
            notes.append(f"change {k['change']}")
        if "settled_from" in k:
            notes.append("settled after animation")
        if k.get("scene_end"):
            notes.append("final state of scene")
        lines.append(f"  #{i + 1:<3} {fmt_ts(k['t'])} – {fmt_ts(k['until'])}"
                     f"  ({', '.join(notes)})")
    if d.get("transcript"):
        lines += ["", "Transcript:"] + [f"  [{fmt_ts(s['start'])}] {s['text']}"
                                        for s in d["transcript"]]
    else:
        lines += ["", "Transcript: not available (no speech-to-text run)."]
    return "\n".join(lines) + "\n"


class Digest:
    """A built, cached digest of one video.

    >>> d = Digest.build("bug-repro.mp4")
    >>> print(d.timeline)
    >>> d.sheets                      # contact sheet image paths
    >>> d.frames([12.5, 13.0, 14.2])  # zoom: one labeled sheet
    >>> d.to_content("anthropic")     # ready-made content blocks
    """

    def __init__(self, path: Path, data: dict, video: str | None = None):
        self.path = Path(path)
        self.data = data
        self.video = video or data["video"]

    # ---------------------------------------------------------- build/load
    @classmethod
    def build(cls, video: str | os.PathLike, *, cache_dir: str | os.PathLike | None = None,
              force: bool = False, settings: SelectSettings | None = None,
              **overrides) -> Digest:
        """Build (or reuse from cache) the digest of a local video file.

        Selection knobs can be passed as a SelectSettings or as keyword
        overrides, e.g. Digest.build("v.mp4", threshold=2.0, min_gap=1.0).
        """
        video = os.path.abspath(video)
        if not os.path.isfile(video):
            raise FileNotFoundError(video)
        settings = dataclasses.replace(settings or SelectSettings(), **overrides)
        fhash = file_hash(video)
        key = f"{fhash[:16]}-{settings_hash(settings)[:8]}"
        root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        target = root / key

        if not force and (target / "digest.json").is_file():
            try:
                return cls.load(target, video=video)
            except (ValueError, KeyError, json.JSONDecodeError):
                pass  # unreadable or stale cache: rebuild

        root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
        try:
            data = _build_into(tmp, video, fhash, key, settings)
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return cls(target, data, video)

    @classmethod
    def load(cls, path: str | os.PathLike, video: str | None = None) -> Digest:
        """Open an existing digest folder."""
        path = Path(path)
        data = json.loads((path / "digest.json").read_text())
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"digest format {data.get('format_version')} != {FORMAT_VERSION}")
        return cls(path, data, video)

    # ---------------------------------------------------------- accessors
    @property
    def key(self) -> str:
        return self.data["key"]

    @property
    def duration(self) -> float:
        return self.data["duration"]

    @property
    def width(self) -> int:
        return self.data["width"]

    @property
    def height(self) -> int:
        return self.data["height"]

    @property
    def keyframes(self) -> list[dict]:
        return self.data["keyframes"]

    @property
    def settings(self) -> SelectSettings:
        return SelectSettings(**self.data["settings"])

    @property
    def timeline(self) -> str:
        return (self.path / "digest.md").read_text()

    @property
    def sheets(self) -> list[Path]:
        return [self.path / p for p in self.data["sheets"]]

    def __repr__(self) -> str:
        return (f"Digest({self.video!r}, {len(self.keyframes)} keyframes, "
                f"{len(self.sheets)} sheets, path={str(self.path)!r})")

    # ---------------------------------------------------------- use
    def frames(self, timestamps: float | Sequence[float]) -> Frames:
        """Zoom: one full frame, or one labeled sheet for up to 9 timestamps.

        The image is also saved under <digest>/frames/.
        """
        return get_frames(self.video, timestamps, out_dir=self.path / "frames",
                          duration=self.duration)

    def to_content(self, provider: str = "anthropic") -> list[dict]:
        """Timeline text followed by each contact sheet, as message content blocks."""
        _check_provider(provider)
        blocks: list[dict] = [{"type": "text", "text": self.timeline}]
        n = len(self.sheets)
        for s, p in enumerate(self.sheets):
            first = s * PER_SHEET + 1
            last = min((s + 1) * PER_SHEET, len(self.keyframes))
            blocks.append({"type": "text",
                           "text": f"Contact sheet {s + 1} of {n} (keyframes #{first}–#{last}):"})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(p.read_bytes()).decode()}})
        return blocks


def _build_into(out: Path, video: str, fhash: str, key: str, settings: SelectSettings) -> dict:
    t0 = time.time()
    info = probe(video)
    samples = list(iter_thumbnails(video, settings.sample_fps, info))
    if not samples:
        raise ValueError(f"no frames decoded from {video}")
    keyframes = select_keyframes(samples, info.duration, settings)
    tiles = [grab_frame(video, k["t"], TILE_WIDTH, duration=info.duration) for k in keyframes]
    labels = [f"#{i + 1}  {fmt_ts(k['t'])}" for i, k in enumerate(keyframes)]
    sheet_paths = write_sheets(tiles, labels, out / "sheets")
    data = {
        "format_version": FORMAT_VERSION,
        "engine_version": __version__,
        "key": key,
        "video": video,
        "file_sha256": fhash,
        "width": info.width,
        "height": info.height,
        "duration": round(info.duration, 3),
        "settings": dataclasses.asdict(settings),
        "keyframes": keyframes,
        "sheets": [p.relative_to(out).as_posix() for p in sheet_paths],
        "transcript": None,
        "build_seconds": round(time.time() - t0, 2),
    }
    (out / "digest.json").write_text(json.dumps(data, indent=2))
    (out / "digest.md").write_text(timeline_text(data))
    return data
