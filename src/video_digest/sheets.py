"""Contact sheets: tile frames into labeled grids, up to 9 frames per image.

One 3x3 sheet costs a vision model about the same as one full frame and stays
legible, so this is where most of the token savings come from.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 3, 3
PER_SHEET = COLS * ROWS
TILE_WIDTH = 512

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",     # macOS
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",                          # Windows
]


@lru_cache(maxsize=8)
def font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = next((f for f in _FONT_CANDIDATES if os.path.exists(f)), None)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)  # Pillow's bundled font


def fmt_ts(t: float) -> str:
    """Seconds as mm:ss.s (e.g. 94.25 -> '01:34.2')."""
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def tile_sheet(tiles: Sequence[Image.Image], labels: Sequence[str],
               cols: int = COLS) -> Image.Image:
    """Grid of same-sized tiles, each labeled in its bottom-left corner."""
    if not tiles:
        raise ValueError("no tiles")
    if len(tiles) != len(labels):
        raise ValueError("need one label per tile")
    tw, th = tiles[0].size
    cols = min(cols, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    f = font(20)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * tw, (i // cols) * th
        if tile.size != (tw, th):
            tile = tile.resize((tw, th), Image.BILINEAR)
        sheet.paste(tile, (x, y))
        l, t, r, b = draw.textbbox((0, 0), label, font=f)
        pad = 6
        box_h = (b - t) + 2 * pad
        draw.rectangle([x, y + th - box_h, x + (r - l) + 2 * pad, y + th], fill=(0, 0, 0))
        draw.text((x + pad - l, y + th - box_h + pad - t), label, font=f, fill=(255, 255, 0))
        draw.rectangle([x, y, x + tw - 1, y + th - 1], outline=(80, 80, 80))
    return sheet


def write_sheets(tiles: Sequence[Image.Image], labels: Sequence[str], out_dir: Path,
                 per_sheet: int = PER_SHEET, cols: int = COLS) -> list[Path]:
    """Write sheet_01.jpg, sheet_02.jpg, ... with up to `per_sheet` tiles each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(math.ceil(len(tiles) / per_sheet)):
        chunk = slice(s * per_sheet, (s + 1) * per_sheet)
        p = out_dir / f"sheet_{s + 1:02d}.jpg"
        tile_sheet(tiles[chunk], labels[chunk], cols).save(p, quality=85)
        paths.append(p)
    return paths
