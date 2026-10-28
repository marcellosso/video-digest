"""Input-token estimates, for dry runs and for CI (live runs report real usage).

Two image profiles:
  highres  current Claude models (Opus 4.7+, Sonnet 5+): images up to 2576 px
           on the long edge, about one token per 28x28 patch, capped at 4784.
           A 1920x1080 frame comes to ~2,700 (the PRD measured ~2,560).
  legacy   older models: 1568 px long edge, ~w*h/750, capped at 1600. This is
           what the prototype estimated with; its live runs came out ~1.6x higher.
"""
from __future__ import annotations

import math

PROFILES = {
    "highres": {"max_edge": 2576, "max_tokens": 4784},
    "legacy": {"max_edge": 1568, "max_tokens": 1600},
}
DEFAULT_PROFILE = "highres"


def image_tokens(width: int, height: int, profile: str = DEFAULT_PROFILE) -> int:
    p = PROFILES[profile]
    scale = min(1.0, p["max_edge"] / max(width, height))
    w, h = width * scale, height * scale
    if profile == "legacy":
        est = w * h / 750
        if est > p["max_tokens"]:
            s = math.sqrt(p["max_tokens"] * 750 / (w * h))
            w, h = w * s, h * s
        return math.ceil(w * h / 750)
    patches = math.ceil(w / 28) * math.ceil(h / 28)
    if patches > p["max_tokens"]:
        s = math.sqrt(p["max_tokens"] / patches)
        patches = math.ceil(w * s / 28) * math.ceil(h * s / 28)
    return min(patches, p["max_tokens"])


def text_tokens(s: str) -> int:
    return math.ceil(len(s) / 3.5)  # rough; live runs report the real number
