"""MCP server: the engine as two tools for coding agents, over stdio.

    digest_video(path)             build (or reuse) the digest: timeline text + contact sheets
                                   (transcript=True adds local speech-to-text)
    get_frames(path, timestamps)   zoom: one full frame, or one labeled sheet for 2-9 timestamps

Digests are cached on disk exactly like the library and CLI ($VIDEO_DIGEST_CACHE
or ./.video-digest), so an agent digests a video once and zooms in later
without a rebuild. The server never calls a model.

Run with `video-digest-mcp` or `python -m video_digest.mcp_server`.
Needs the optional dependency: pip install 'video-digest[mcp]'. Transcripts
also need the 'transcript' extra; speech-to-text runs locally (faster-whisper).
"""
from __future__ import annotations

from io import BytesIO
from typing import Annotated

try:
    from pydantic import Field  # installed with mcp
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as e:  # pragma: no cover
    raise ImportError("the MCP server needs the 'mcp' extra: "
                      "pip install 'video-digest[mcp]'") from e

from . import __version__
from . import digest as _digest
from .cli import CliError, _preflight
from .digest import MAX_FRAMES, Digest
from .sample import FFmpegError
from .sheets import PER_SHEET, fmt_ts
from .transcript import TranscriptUnavailable

INSTRUCTIONS = """\
Read videos (screen recordings, bug repros, demos) cheaply.
Call digest_video first: it returns a text timeline of keyframes and contact
sheets with each tile labeled #N and its timestamp. To look closer, call
get_frames with 1 timestamp (full-resolution frame) or 2-9 timestamps (one
labeled sheet). Digests are cached on disk, so repeat calls are fast.
Pass transcript=true only when the answer may depend on what is said: it runs
local speech-to-text, which is slower and downloads a model on first use.
Pass absolute video paths."""

server = MCPServer("video-digest", instructions=INSTRUCTIONS, version=__version__)


def _check_video(path: str) -> None:
    try:
        _preflight(path)
    except CliError as e:
        raise ToolError(str(e)) from e


def _jpeg(img, quality: int = 92) -> Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image(data=buf.getvalue(), format="jpeg")


@server.tool()
def digest_video(
    path: Annotated[str, Field(description="Local video file (absolute path recommended)")],
    transcript: Annotated[bool, Field(
        description="Add a timestamped speech transcript (local speech-to-text, slower; "
        "needs the server's 'transcript' extra and downloads a model on first use)")] = False,
    force: Annotated[bool, Field(description="Rebuild even if a cached digest exists")] = False,
    cache_dir: Annotated[str | None, Field(
        description="Cache root (default $VIDEO_DIGEST_CACHE or ./.video-digest)")] = None,
) -> list:
    """Digest a video: text timeline of keyframes plus contact sheet images.

    Builds once and reuses the cached digest afterwards (by file hash). Each
    keyframe stands for the video until the next one; use get_frames to zoom.
    """
    _check_video(path)
    existing = None if force else Digest.cached(path, cache_dir=cache_dir, transcript=transcript)
    try:
        d = existing or Digest.build(path, cache_dir=cache_dir, force=force,
                                     transcript=transcript)
    except (TranscriptUnavailable, ValueError, FFmpegError) as e:  # not a video, no extra
        raise ToolError(str(e)) from e
    how = "reused from cache" if existing else f"built in {d.data['build_seconds']:g}s"
    meta = (f"Digest: {d.path.absolute()} ({how})\n"
            f"Video: {d.video}\n"
            f"Duration: {fmt_ts(d.duration)} ({d.duration:g}s), {d.width}x{d.height}; "
            f"{len(d.keyframes)} keyframes in {len(d.sheets)} contact sheet(s)")
    if (tr := d.transcript) is not None:
        meta += (f"\nTranscript: {len(tr['segments'])} segment(s), whisper {tr['model']}"
                 + ("" if tr["has_audio"] else " (no audio track)"))
    out: list = [meta, d.timeline]
    for s, p in enumerate(d.sheets):
        first, last = s * PER_SHEET + 1, min((s + 1) * PER_SHEET, len(d.keyframes))
        out.append(f"Contact sheet {s + 1} of {len(d.sheets)} (keyframes #{first}–#{last}):")
        out.append(Image(path=p))
    return out


@server.tool()
def get_frames(
    path: Annotated[str, Field(description="Local video file (same path given to digest_video)")],
    timestamps: Annotated[list[float], Field(
        min_length=1, max_length=MAX_FRAMES,
        description=f"Seconds into the video, 1 to {MAX_FRAMES} of them")],
    cache_dir: Annotated[str | None, Field(
        description="Cache root (default $VIDEO_DIGEST_CACHE or ./.video-digest)")] = None,
) -> list:
    """Zoom into exact moments: 1 timestamp gives the full-resolution frame,
    2-9 give one contact sheet with each tile labeled by its timestamp.

    Never builds a digest. If the video was digested, the image is saved in its
    folder under frames/; otherwise it is only returned.
    """
    _check_video(path)
    d = (Digest.cached(path, cache_dir=cache_dir)
         or Digest.cached(path, cache_dir=cache_dir, transcript=True))
    try:
        z = d.frames(timestamps) if d else _digest.get_frames(path, timestamps)
    except (ValueError, FFmpegError) as e:  # bad timestamps, not a video
        raise ToolError(str(e)) from e
    label = z.label + (f"\nSaved: {z.path.absolute()}" if z.path else "")
    return [label, Image(path=z.path) if z.path else _jpeg(z.image)]


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
