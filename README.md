# video-digest

Turn a video into a compact digest that an AI model can read cheaply: a short list of keyframes, laid out on labeled contact sheets, plus a text timeline. The model answers from the digest and asks for a full-resolution frame only where it needs a closer look.

It ships as a Python library, a CLI, an MCP server for coding agents, and a Claude Code plugin. It runs locally on top of `ffmpeg` and never calls a model itself.

> Early development (`0.1.0.dev0`). Everything below works from a clone. Nothing is on PyPI yet.

## Why

The usual way to show a model a video is to dump a frame every second or two. That is expensive (every frame costs about as much as a full image) and still misses things: a 2-second toast between samples, the end state of an animation, text caught mid-transition.

video-digest picks frames by what changed on screen instead of by the clock:

1. **Change detection.** A frame is kept when enough of the picture differs from the last kept frame, so static stretches collapse to one frame and slow fades still register.
2. **Settling.** A frame caught mid-animation moves forward to the first moment the picture holds still (never across a scene cut).
3. **Scene ends.** Each scene's final state is kept when it shows something the scene's earlier keyframe doesn't.

The kept frames are tiled into labeled sheets of up to 9 (one sheet costs a vision model about as much as one frame), and the timeline tells the model what each tile covers. On the 32 s showreel in the benchmark, the digest answered 15/15 questions on every run against 11/15 for a frame every 2 s, with 16% of the input tokens (6,591 vs 41,049; 3 live runs on `claude-sonnet-5-5`, see [`bench/README.md`](https://github.com/marcellosso/video-digest/blob/main/bench/README.md)).

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` on your `PATH` (`brew install ffmpeg`, `apt install ffmpeg`, …; `ffprobe` comes with it)

## Install

From a clone, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/marcellosso/video-digest && cd video-digest
uv venv && uv pip install -e '.[mcp]'      # drop [mcp] if you only need the library and CLI
source .venv/bin/activate
video-digest --help
```

With pip: `python -m venv .venv && .venv/bin/pip install -e '.[mcp]'`.

Run the CLI once without installing anything, straight from GitHub:

```bash
uvx --from git+https://github.com/marcellosso/video-digest video-digest build demo.mp4
```

Extras: `mcp` (MCP server), `bench` (live benchmark runs, Anthropic SDK), `dev` (pytest).

After a PyPI release this becomes `uvx video-digest build demo.mp4`, `pip install video-digest` and `uvx --from 'video-digest[mcp]' video-digest-mcp`. The package name isn't final yet, see [PACKAGING.md](https://github.com/marcellosso/video-digest/blob/main/PACKAGING.md).

## Quickstart: command line

```bash
video-digest build bug-repro.mp4               # digest folder, keyframe timeline, sheet paths
video-digest build bug-repro.mp4 --timeline    # print the full text timeline (digest.md)
video-digest build bug-repro.mp4 --json        # machine-readable summary
video-digest frames bug-repro.mp4 12.5         # full-resolution frame -> ./at_12.50.jpg
video-digest frames bug-repro.mp4 12.5 0:13 14.2 -o zoom/   # one labeled sheet (2-9 timestamps)
```

`build` prints something like:

```
.video-digest/2c6d9fb18fb5a0a4-58d19347  (built in 0.24s)
clip.mp4: 00:12.0, 640x360, 5 keyframes in 1 sheet(s)
  #1   00:00.0 – 00:04.0
  #2   00:04.0 – 00:08.0
  ...
Timeline: .video-digest/2c6d9fb18fb5a0a4-58d19347/digest.md
Sheet:    .video-digest/2c6d9fb18fb5a0a4-58d19347/sheets/sheet_01.jpg
```

Hand `digest.md` and the sheet images to any model. `python -m video_digest ...` is the same CLI. `build` takes `--cache-dir`, `--force` and `--set KNOB=VALUE` (selection overrides, repeatable). Timestamps are seconds or `[hh:]mm:ss[.f]`.

Exit status: 0 success, 1 ffmpeg failed on the file (not a video, corrupt), 2 usage error or missing video, 3 ffmpeg/ffprobe not on `PATH`.

## Quickstart: Python

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

Gives coding agents (Claude Code, Claude Desktop, Cursor, Codex) two tools over stdio. Needs the `mcp` extra. The server never calls a model.

- `digest_video(path, transcript=false, force=false, cache_dir=null)`: builds or reuses the digest. Returns a summary (digest folder, duration, keyframe and sheet counts), the text timeline and the contact sheets as JPEG images. `transcript` is accepted but has no effect until speech-to-text lands (#7).
- `get_frames(path, timestamps, cache_dir=null)`: 1 timestamp returns the full-resolution frame, and 2 to 9 return one labeled sheet. It never builds a digest. If the video already has one, the image is also saved in its `frames/` folder.

An agent can digest a video once and then zoom in as often as it likes without a rebuild.

```bash
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

## Claude Code plugin

`plugin/` is a Claude Code plugin with a `digest` skill. The skill starts the MCP server above and tells the model to digest the video once, answer from the timeline and sheets, and call `get_frames` only for moments it needs a closer look at.

```
/digest ./bug-repro.mp4 why does the save button stop working?
```

The full name is `/video-digest:digest`. Plain `/digest` works too unless another plugin or skill also has a `digest`. With no question, the skill summarizes the video with timestamps. Claude also picks up the skill on its own when you ask about a video file. Follow-up questions in the same session reuse the digest without rebuilding it.

Install (after the clone install above, with the `mcp` extra):

```bash
claude --plugin-dir ./plugin                  # load the plugin for this session
```

The plugin runs `video-digest-mcp` from your `PATH`. If it isn't on your `PATH` (for example, because the venv isn't activated), point the plugin at it with `VIDEO_DIGEST_MCP` before you start Claude Code:

```bash
export VIDEO_DIGEST_MCP=/path/to/video-digest/.venv/bin/video-digest-mcp
```

Digests are cached in `~/.cache/video-digest`. `ffmpeg` and `ffprobe` must be on the `PATH` that Claude Code starts with. Run `/mcp` to check that `plugin:video-digest:video-digest` is connected.

The skill pre-approves the two tools for the turn that runs `/digest`. For follow-ups without permission prompts, add `"mcp__plugin_video-digest_video-digest"` to `permissions.allow` in your Claude Code settings.

To check the plugin layout, run `claude plugin validate ./plugin`. The plugin isn't in a marketplace yet. Load it with `--plugin-dir`.

## Digest format

```
.video-digest/<file-hash>-<settings-hash>/
  digest.json      # metadata, settings, keyframes, transcript (null for now)
  digest.md        # the text timeline sent to the model
  sheets/          # sheet_01.jpg, sheet_02.jpg, …
  frames/          # zoomed frames saved by Digest.frames()
```

The cache root is `./.video-digest` by default. Override it with `cache_dir=` or the `VIDEO_DIGEST_CACHE` environment variable.

## Benchmark

`qa_bench` asks the same questions about a video of two or three strategies (a frame every 2 s, the digest, the digest with zoom), grades the answers and compares accuracy against input tokens. It lives in the repo, not in the package, so run it from the root of a clone:

```bash
uv pip install -e '.[dev,bench]'

# Offline, no model calls (~10 s): generates a 180 s synthetic video on first use,
# estimates tokens and checks that the shown frames cover each answer
python -m bench.qa_bench synthetic

# A real video of your own. Offline gives token estimates only; --live asks a real
# model (default claude-opus-5-5) 3 times per strategy and grades the answers
python -m bench.qa_bench my.mp4 --questions my-questions.json
ANTHROPIC_API_KEY=... python -m bench.qa_bench my.mp4 --questions my-questions.json --live
```

`my-questions.json` is a list of questions (or `{"questions": [...]}`):

```json
[
  {"id": "error", "question": "What error message appears after clicking Save?", "check": "judge"},
  {"id": "count", "question": "How many rows are in the table at the end?", "check": "number", "answer": 12}
]
```

`check` is `contains`, `number`, `yesno` or `judge`. Without `answer`, the baseline's answer becomes the reference. The showreel fixture (`python -m bench.qa_bench showreel`) needs the video file, which isn't in git. Each run writes `bench/out/<fixture>-<mode>/qa_report.md` and `results.json`, and exits 1 if a quality-gate check fails.

Changes to frame selection must pass the quality gate. Fixtures, offline vs live scoring and the gate are in [`bench/README.md`](https://github.com/marcellosso/video-digest/blob/main/bench/README.md).

## Development

```bash
uv venv && uv pip install -e '.[dev,bench,mcp]'
.venv/bin/pytest
```

The tests build synthetic videos with ffmpeg and never call a model. CI runs them plus the offline synthetic benchmark.

## License

[Apache-2.0](https://github.com/marcellosso/video-digest/blob/main/LICENSE). Source: <https://github.com/marcellosso/video-digest>.
