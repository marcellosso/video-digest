---
name: digest
description: Answer a question about a local video file (screen recording, bug repro, demo, tutorial) by reading its digest, a keyframe timeline plus contact sheets, instead of raw frames. Use when the user points at a .mp4/.mov/.webm/.mkv file and asks what happens in it, when something breaks, or what is on screen at some moment.
argument-hint: <video> <question…>
allowed-tools: mcp__plugin_video-digest_video-digest__digest_video, mcp__plugin_video-digest_video-digest__get_frames
---

# Answer from a video digest

The user wants an answer about a video. The request is:

> $ARGUMENTS

The first argument is the video path. Everything after it is the question. If the path is missing, ask for it. If the question is missing, give a short summary of what happens in the video, with timestamps.

## Steps

1. **Resolve the path.** Make it absolute against the current working directory (for example `./bug-repro.mp4` becomes `<cwd>/bug-repro.mp4`). Paths with spaces may arrive quoted, so strip the quotes.

2. **Digest it once.** Call `digest_video` with the absolute path. It returns a summary line, a text timeline of keyframes (`#N  mm:ss.s` plus what changed) and one or more contact sheets, each tile labeled `#N` and its timestamp. Each keyframe stands for the video until the next one. Don't pass `force` unless the user says the file changed.

3. **Answer from the digest.** Read the timeline and the sheets, then answer the question directly. Cite moments as timestamps (`0:12.5`) and keyframe numbers (`#7`) so the user can scrub to them.

4. **Zoom only when you need to.** Call `get_frames` with 1 to 9 timestamps (seconds) when:
   - a tile is too small to read (text, an error message, a value in a form),
   - the question is about a precise moment that falls between keyframes, or
   - you are unsure and a closer look would settle it.

   One timestamp returns a full-resolution frame. Two to nine return one labeled sheet, so batch the moments you want in a single call instead of calling it repeatedly. Don't zoom when the sheets already answer the question.

5. **Say what you can't see.** The digest has no audio or transcript yet. If the answer depends on sound or on something the frames don't show, say so instead of guessing.

## Follow-ups

For later questions about the same video in this session, reuse the timeline and sheets you already have and call `get_frames` for anything new. Don't call `digest_video` again for that video. Digests are cached on disk by file hash, so a new session gets the same digest back without a rebuild.

## If a tool fails

- The `video-digest` MCP server is missing or won't start: the Python package isn't installed with its MCP extra, or `video-digest-mcp` isn't on `PATH`. Point the user to the plugin's install steps: `uv pip install -e '.[mcp]'` in a clone of https://github.com/marcellosso/video-digest, then `VIDEO_DIGEST_MCP=/path/to/.venv/bin/video-digest-mcp` if the command isn't on `PATH`.
- `ffmpeg`/`ffprobe` not found: they must be on the `PATH` Claude Code was started with.
- Not a video, or the file doesn't exist: report the error and the path you tried.
