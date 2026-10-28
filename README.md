# video-digest
Open-source engine: video → digest (keyframes / contact sheets + timeline) so AI models read fewer tokens with better accuracy

> Early development. The engine works locally; the CLI, MCP server and benchmark are not built yet.

## Install (local, editable)

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on your `PATH`.

```bash
uv venv && uv pip install -e '.[dev]'      # or: python -m venv .venv && pip install -e '.[dev]'
```

## Usage

```python
from video_digest import Digest

d = Digest.build("bug-repro.mp4")        # cached by file hash + settings
print(d.timeline)                         # text timeline for the model
d.sheets                                  # contact sheet image paths (≤ 9 frames each)
zoom = d.frames([12.5, 13.0, 14.2])       # one labeled sheet (one timestamp → full frame)
zoom.image, zoom.path                     # PIL image, and where it was saved
blocks = d.to_content("anthropic")        # timeline + sheets as content blocks
```

Selection knobs are keyword overrides, for example `Digest.build(path, threshold=2.0, min_gap=1.0)`. See `video_digest.SelectSettings` for all of them. Each distinct setting combination gets its own cache folder.

`get_frames(path, timestamps)` zooms without building a digest first.

## Digest format

```
.video-digest/<file-hash>-<settings-hash>/
  digest.json      # metadata, settings, keyframes, transcript (null for now)
  digest.md        # the text timeline sent to the model
  sheets/          # sheet_01.jpg, sheet_02.jpg, …
  frames/          # zoomed frames saved by Digest.frames()
```

The cache root is `./.video-digest` by default. Override it with `cache_dir=` or the `VIDEO_DIGEST_CACHE` environment variable.

## Tests

```bash
.venv/bin/pytest
```

The tests build a 9-second synthetic video with ffmpeg and never call a model.
