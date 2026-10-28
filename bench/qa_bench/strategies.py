"""Strategies: different ways of showing the same video to a model.

  baseline       one full-resolution frame every --baseline-every seconds
                 (what an agent does when it dumps frames itself)
  digest_nozoom  the engine's timeline + contact sheets, no zoom tool
  digest         the same, plus a batched get_frames zoom tool

Comparing digest_nozoom with digest shows what the zoom tool adds.
"""
from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from video_digest import Digest
from video_digest.sample import grab_frame, probe

from .tokens import DEFAULT_PROFILE, image_tokens, text_tokens

ALL = ("baseline", "digest_nozoom", "digest")
ZOOM_LINE = "If you need more detail, ask for a full-resolution frame at a timestamp.\n"
DIGEST_HINT = (
    "If a question needs detail the contact sheets don't show clearly (small text, exact "
    "moments, something between keyframes), call get_frames. Put EVERY timestamp you need "
    "into ONE call (up to 9); each extra call costs much more than extra timestamps.")


@dataclass
class Strategy:
    name: str
    content: list[dict]          # message content blocks: the video material
    image_sizes: list[tuple[int, int]]
    times: list[float]           # timestamps of the frames the model is shown
    zoom: bool = False
    digest: Digest | None = None
    prep_seconds: float = 0.0

    def estimate_tokens(self, profile: str = DEFAULT_PROFILE) -> int:
        text = "".join(b["text"] for b in self.content if b["type"] == "text")
        return sum(image_tokens(w, h, profile) for w, h in self.image_sizes) + text_tokens(text)


def jpeg_block(img: Image.Image, quality: int = 85) -> dict:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def baseline(video: Path, every: float = 2.0, max_images: int = 100) -> Strategy:
    t0 = time.time()
    duration = probe(str(video)).duration
    times = [round(i * every, 3) for i in range(int(duration / every))]
    if len(times) > max_images:
        raise SystemExit(f"The baseline would need {len(times)} images, above the {max_images}-"
                         "image per-request limit. Raise --baseline-every.")
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(lambda t: grab_frame(str(video), t, duration=duration), times))
    content = [{"type": "text", "text": f"{len(times)} frames, one every {every:g}s:"}]
    for t, img in zip(times, frames):
        content += [{"type": "text", "text": f"Frame at {t:.1f}s"}, jpeg_block(img)]
    return Strategy("baseline", content, [f.size for f in frames], times,
                    prep_seconds=time.time() - t0)


def build_digest(video: Path, cache_dir: Path, **settings) -> tuple[Digest, float]:
    """Build from scratch (never from cache) so the build time is real."""
    t0 = time.time()
    d = Digest.build(video, cache_dir=cache_dir, force=True, **settings)
    return d, time.time() - t0


def digest(d: Digest, zoom: bool, build_seconds: float) -> Strategy:
    content = d.to_content("anthropic")
    if zoom:
        content[0] = {"type": "text", "text": d.timeline + "\n" + DIGEST_HINT}
    else:
        content[0] = {"type": "text", "text": d.timeline.replace(ZOOM_LINE, "")}
    sizes = []
    for p in d.sheets:
        with Image.open(p) as im:
            sizes.append(im.size)
    return Strategy("digest" if zoom else "digest_nozoom", content, sizes,
                    [k["t"] for k in d.keyframes], zoom=zoom, digest=d,
                    prep_seconds=build_seconds)


def make(names: list[str], video: Path, out_dir: Path, baseline_every: float = 2.0,
         max_images: int = 100, settings: dict | None = None) -> dict[str, Strategy]:
    unknown = set(names) - set(ALL)
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(sorted(unknown))} (known: {', '.join(ALL)})")
    out: dict[str, Strategy] = {}
    if "baseline" in names:
        out["baseline"] = baseline(video, baseline_every, max_images)
    if {"digest", "digest_nozoom"} & set(names):
        d, secs = build_digest(video, out_dir / "digest-cache", **(settings or {}))
        for name in ("digest_nozoom", "digest"):
            if name in names:
                out[name] = digest(d, name == "digest", secs)
    return {n: out[n] for n in names}
