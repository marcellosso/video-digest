"""Frame selection: which moments of the video are worth a frame.

Works on (timestamp, thumbnail) samples only, so it is pure and testable
without ffmpeg. The steps, in order:

1. Change detection: keep a sample when it differs enough (block-wise, in
   color) from the last *kept* one, so static stretches collapse and slow
   fades accumulate until they register.
2. Settling: a keyframe caught mid-animation (scrambling text, a half-drawn
   line) moves forward to the first moment the picture holds still, but never
   across a scene cut.
3. Scene ends: each scene's final calm state (and the video's last frame) is
   kept when it shows something the scene's earlier keyframe doesn't.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectSettings:
    """Frame-selection knobs. Defaults are the values proven on the prototype."""

    sample_fps: float = 4.0
    # Percent of 4x4 blocks that must change vs. the last kept frame.
    threshold: float = 1.5
    # During continuous motion keep at most one frame per min_gap seconds.
    min_gap: float = 2.0
    # Always keep a frame at least this often, even if nothing changed.
    max_gap: float = 60.0
    settle: bool = True
    # Sample-to-sample change at or below this counts as "still".
    settle_threshold: float = 3.0
    # How far forward (s) a keyframe may move to find a still picture.
    settle_window: float = 1.5
    # How long (s) the picture must hold still to count as settled.
    settle_hold: float = 0.5
    # Sample-to-sample change at or above this is a scene cut.
    cut_threshold: float = 50.0
    scene_ends: bool = True


def change_score(a: np.ndarray, b: np.ndarray, block: int = 4,
                 pixel_delta: float = 24.0) -> float:
    """Percent of block x block cells whose color changed noticeably.

    Block-based, not a whole-frame average, so a small moving object or a new
    line of text registers while noise and compression shimmer don't. Uses the
    max over RGB channels, so a hue shift at constant brightness still counts.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    h, w = a.shape[:2]
    d = np.abs(a - b).max(axis=2)[: h // block * block, : w // block * block]
    blocks = d.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    return float((blocks > pixel_delta).mean() * 100)


def motion_profile(thumbs: list[np.ndarray]) -> list[float]:
    """motion[i] = change from sample i to sample i + 1 (last entry is 0)."""
    return [change_score(thumbs[i], thumbs[i + 1]) for i in range(len(thumbs) - 1)] + [0.0]


def scene_cuts(times: list[float], motion: list[float], cut_threshold: float) -> list[float]:
    """Timestamps where a new scene starts (the sample right after a cut)."""
    return [times[i + 1] for i in range(len(motion) - 1) if motion[i] >= cut_threshold]


def select_keyframes(samples: list[tuple[float, np.ndarray]], duration: float,
                     s: SelectSettings = SelectSettings()) -> list[dict]:
    """Pick keyframes from (timestamp, thumbnail) samples.

    Returns a list of dicts sorted by time:
      t             timestamp of the frame to show
      until         when the next keyframe starts (or the video ends)
      change        percent changed vs. the previous kept frame (None for the first)
      settled_from  original timestamp, if the frame was moved past an animation
      scene_end     True if kept as the final state of a scene
    """
    if not samples:
        return []
    times = [t for t, _ in samples]
    thumbs = [th for _, th in samples]

    kept: list[dict] = []  # entries hold sample indices in "i"
    last_i: int | None = None
    pending: tuple[int, float] | None = None
    for i, t in enumerate(times):
        if last_i is None:
            kept.append({"i": i, "change": None})
            last_i = i
            continue
        last_t = times[last_i]
        diff = change_score(thumbs[i], thumbs[last_i])
        if diff >= s.threshold or t - last_t >= s.max_gap:
            if t - last_t >= s.min_gap:
                kept.append({"i": i, "change": round(diff, 1)})
                last_i, pending = i, None
            else:
                pending = (i, diff)
        elif pending and t - last_t >= s.min_gap:
            # A change happened inside min_gap and the picture has calmed since:
            # keep the moment it changed rather than losing it.
            kept.append({"i": pending[0], "change": round(pending[1], 1)})
            last_i, pending = pending[0], None

    motion = motion_profile(thumbs) if (s.settle or s.scene_ends) else []

    if s.settle:
        hold = max(1, round(s.settle_hold * s.sample_fps))

        def calm(j: int) -> bool:
            return all(motion[m] <= s.settle_threshold
                       for m in range(j, min(j + hold, len(motion))))

        for n, k in enumerate(kept):
            i = k["i"]
            if calm(i) or motion[i] >= s.cut_threshold:
                continue  # already still, or the clean last frame before a cut
            stop = kept[n + 1]["i"] if n + 1 < len(kept) else len(samples)
            for j in range(i + 1, stop):
                if times[j] - times[i] > s.settle_window or motion[j - 1] >= s.cut_threshold:
                    break  # too far, or we'd cross into another scene
                if calm(j):
                    k["settled_from"] = times[i]
                    k["i"] = j
                    break

    if s.scene_ends:
        ends = [i for i in range(len(samples) - 1) if motion[i] >= s.cut_threshold]
        ends.append(len(samples) - 1)
        have = {k["i"] for k in kept}
        for e in ends:
            # last frame of the scene that arrived steadily (not mid-fade-out)
            j = next((j for j in range(e, max(e - 4, 0), -1)
                      if motion[j - 1] <= s.settle_threshold), e)
            prev = max((k["i"] for k in kept if k["i"] <= j), default=None)
            if prev is None or j in have:
                continue
            gap = times[j] - times[prev]
            diff = change_score(thumbs[j], thumbs[prev])
            if diff >= s.threshold and gap >= 0.75:
                kept.append({"i": j, "change": round(diff, 1), "scene_end": True})
                have.add(j)
        kept.sort(key=lambda k: k["i"])

    out = []
    for k in kept:
        e = {"t": round(times[k["i"]], 3), "change": k["change"]}
        if "settled_from" in k:
            e["settled_from"] = round(k["settled_from"], 3)
        if k.get("scene_end"):
            e["scene_end"] = True
        out.append(e)
    for n, e in enumerate(out):
        e["until"] = out[n + 1]["t"] if n + 1 < len(out) else round(duration, 3)
    return out
