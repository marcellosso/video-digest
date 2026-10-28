# video-digest
Open-source engine: video → digest (keyframes / contact sheets + timeline) so AI models read fewer tokens with better accuracy

> Early development. The engine, CLI, MCP server and benchmark work locally; nothing is published yet.

## Install (local, editable)

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on your `PATH`.

```bash
uv venv && uv pip install -e '.[dev]'      # or: python -m venv .venv && pip install -e '.[dev]'
```

## Command line

```bash
video-digest build bug-repro.mp4               # digest folder, keyframe timeline, sheet paths
video-digest build bug-repro.mp4 --timeline    # print the full text timeline (digest.md)
video-digest build bug-repro.mp4 --json        # machine-readable summary
video-digest frames bug-repro.mp4 12.5         # full-resolution frame -> ./at_12.50.jpg
video-digest frames bug-repro.mp4 12.5 0:13 14.2 -o zoom/   # one labeled sheet (2-9 timestamps)
```

`python -m video_digest ...` is the same thing. `build` takes `--cache-dir`, `--force` and `--set KNOB=VALUE` (selection overrides, repeatable). Timestamps are seconds or `[hh:]mm:ss[.f]`.

Exit status: 0 success, 1 ffmpeg failed on the file (not a video, corrupt), 2 usage error or missing video, 3 ffmpeg/ffprobe not on `PATH`.

## Python

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

`get_frames(path, timestamps)` zooms without building a digest first. `Digest.cached(path)` returns the cached digest, or `None` if there isn't one. It never builds.

## MCP server

Gives coding agents (Claude Code, Claude Desktop, Cursor, Codex) two tools over stdio. The server never calls a model.

- `digest_video(path, transcript=false, force=false, cache_dir=null)`: builds or reuses the digest. Returns a summary (digest folder, duration, keyframe and sheet counts), the text timeline and the contact sheets as JPEG images. `transcript` is accepted but has no effect until speech-to-text lands (#7).
- `get_frames(path, timestamps, cache_dir=null)`: 1 timestamp returns the full-resolution frame, and 2 to 9 return one labeled sheet. It never builds a digest. If the video already has one, the image is also saved in its `frames/` folder.

Digests are cached on disk just as they are for the library and CLI. An agent can digest a video once and then zoom in as often as it likes without a rebuild.

```bash
uv pip install -e '.[mcp]'        # the core install stays without the MCP SDK
video-digest-mcp                  # or: python -m video_digest.mcp_server (stdio)
```

Claude Code:

```bash
claude mcp add video-digest --env VIDEO_DIGEST_CACHE=$HOME/.cache/video-digest -- /path/to/video-digest/.venv/bin/video-digest-mcp
```

Claude Desktop (`claude_desktop_config.json`) and other clients that use the `mcpServers` format:

```json
{
  "mcpServers": {
    "video-digest": {
      "command": "/path/to/video-digest/.venv/bin/video-digest-mcp",
      "env": { "VIDEO_DIGEST_CACHE": "/Users/you/.cache/video-digest" }
    }
  }
}
```

Use absolute paths for the command and for videos. Set `VIDEO_DIGEST_CACHE` because clients may start the server in a directory it can't write to. Otherwise the default `./.video-digest` is resolved from wherever the server was started. `ffmpeg` and `ffprobe` must be on the server's `PATH`. If your client launches it with a minimal `PATH`, add one in `env`.

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

The tests build synthetic videos with ffmpeg and never call a model.

## Benchmark

```bash
python -m bench.qa_bench synthetic                 # offline: digest vs a frame every 2 s
python -m bench.qa_bench showreel --live           # real model (needs '.[bench]' and credentials)
```

Changes to frame selection must pass the quality gate. See [bench/README.md](bench/README.md).
