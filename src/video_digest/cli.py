"""Command line: video-digest {build,frames} ...

    video-digest build VIDEO                build (or reuse) the digest, print a summary
    video-digest build VIDEO --transcript   ... with local speech-to-text (needs the extra)
    video-digest frames VIDEO T1 [T2 ...]   zoom: one full frame, or one sheet for 2-9 timestamps

Timestamps are seconds (12.5) or [hh:]mm:ss[.f] (1:02.5). Pass --json for
machine-readable output on stdout. `python -m video_digest` works the same.

Exit status: 0 on success, 1 when ffmpeg fails on the video (not a video,
corrupt, no video stream), 2 on usage errors (bad arguments, missing video
file, --transcript without the 'transcript' extra), 3 when ffmpeg/ffprobe
are not installed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .digest import MAX_FRAMES, Digest, get_frames
from .sample import FFmpegError
from .select import SelectSettings
from .sheets import fmt_ts
from .transcript import TranscriptUnavailable

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NO_FFMPEG = 0, 1, 2, 3


class CliError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="video-digest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    b = sub.add_parser("build", help="build (or reuse from cache) the digest of a video",
                       description="Build the digest of VIDEO, or reuse it from the cache, "
                       "and print its folder and a short timeline.")
    b.add_argument("video", help="local video file")
    b.add_argument("--cache-dir", help="cache root (default $VIDEO_DIGEST_CACHE or ./.video-digest)")
    b.add_argument("--force", action="store_true", help="rebuild even if a cached digest exists")
    b.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE",
                   help="selection setting override, e.g. --set threshold=2.0 (repeatable)")
    b.add_argument("--transcript", action="store_true",
                   help="add a local speech-to-text transcript (needs the 'transcript' extra; "
                   "downloads the Whisper model on first use)")
    b.add_argument("--transcript-model", metavar="MODEL",
                   help="Whisper model for --transcript: tiny, base, small, ... or a local "
                   "model folder (default $VIDEO_DIGEST_WHISPER_MODEL or base)")
    b.add_argument("--timeline", action="store_true",
                   help="print the full text timeline (digest.md) instead of the summary")
    b.add_argument("--json", action="store_true", help="print a JSON summary instead")

    f = sub.add_parser("frames", help="zoom into exact timestamps of a video",
                       description="One timestamp saves the full-resolution frame; 2 to "
                       f"{MAX_FRAMES} save one labeled contact sheet.")
    f.add_argument("video", help="local video file")
    f.add_argument("timestamps", nargs="+", type=timestamp, metavar="T",
                   help="seconds (12.5) or [hh:]mm:ss[.f] (1:02.5)")
    f.add_argument("-o", "--out", default=".", help="folder to save the JPEG in (default: .)")
    f.add_argument("--json", action="store_true", help="print a JSON summary instead")
    return ap


def timestamp(text: str) -> float:
    """'12.5' -> 12.5, '1:02.5' -> 62.5, '1:00:00' -> 3600.0."""
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts) or any(p >= 60 for p in parts[1:]):
        raise argparse.ArgumentTypeError(f"invalid timestamp {text!r} (use seconds or mm:ss.f)")
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def _settings(pairs: list[str]) -> dict:
    known = {f.name for f in dataclasses.fields(SelectSettings)}
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if not v:
            raise CliError(f"--set expects KNOB=VALUE, got {p!r}", EXIT_USAGE)
        if k not in known:
            raise CliError(f"unknown setting {k!r}; known: {', '.join(sorted(known))}", EXIT_USAGE)
        try:
            out[k] = json.loads(v)  # numbers and true/false
        except json.JSONDecodeError:
            raise CliError(f"--set {k}: {v!r} is not a number or true/false", EXIT_USAGE)
    return out


def _preflight(video: str) -> None:
    if not os.path.exists(video):
        raise CliError(f"video not found: {video}", EXIT_USAGE)
    if not os.path.isfile(video):
        raise CliError(f"not a file: {video}", EXIT_USAGE)
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise CliError(f"{' and '.join(missing)} not found on PATH; install ffmpeg "
                       "(e.g. brew install ffmpeg, apt install ffmpeg)", EXIT_NO_FFMPEG)


def _rel(p: Path) -> str:
    """A path relative to the cwd when it's under it, else absolute."""
    p = Path(p).absolute()
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def build(args) -> int:
    _preflight(args.video)
    overrides = _settings(args.set)
    if args.transcript_model and not args.transcript:
        raise CliError("--transcript-model needs --transcript", EXIT_USAGE)
    t0 = time.time()
    try:
        d = Digest.build(args.video, cache_dir=args.cache_dir, force=args.force,
                         transcript=args.transcript, transcript_model=args.transcript_model,
                         **overrides)
    except TranscriptUnavailable as e:
        raise CliError(str(e), EXIT_USAGE)
    except ValueError as e:  # nothing decodable in the file
        raise CliError(str(e), EXIT_FAILED)
    # A fresh build writes digest.json after t0; a cache hit leaves the old one.
    cached = (d.path / "digest.json").stat().st_mtime < t0
    if args.json:
        print(json.dumps({
            "path": str(d.path.absolute()), "key": d.key, "video": d.video,
            "duration": d.duration, "width": d.width, "height": d.height,
            "keyframes": [k["t"] for k in d.keyframes],
            "sheets": [str(p.absolute()) for p in d.sheets],
            "timeline": str((d.path / "digest.md").absolute()),
            "transcript": d.transcript,
            "build_seconds": d.data["build_seconds"], "cached": cached,
        }, indent=2))
        return EXIT_OK
    if args.timeline:
        print(d.timeline, end="")
        return EXIT_OK
    how = "reused from cache" if cached else f"built in {d.data['build_seconds']:g}s"
    print(f"{_rel(d.path)}  ({how})")
    print(f"{os.path.basename(d.video)}: {fmt_ts(d.duration)}, {d.width}x{d.height}, "
          f"{len(d.keyframes)} keyframes in {len(d.sheets)} sheet(s)")
    for i, k in enumerate(d.keyframes):
        print(f"  #{i + 1:<3} {fmt_ts(k['t'])} – {fmt_ts(k['until'])}")
    if (tr := d.transcript) is not None:
        print(f"Transcript: {len(tr['segments'])} segment(s), whisper {tr['model']}"
              + ("" if tr["has_audio"] else " (no audio track)"))
    print(f"Timeline: {_rel(d.path / 'digest.md')}")
    for p in d.sheets:
        print(f"Sheet:    {_rel(p)}")
    return EXIT_OK


def frames(args) -> int:
    _preflight(args.video)
    try:
        z = get_frames(args.video, args.timestamps, out_dir=args.out)
    except ValueError as e:  # too many timestamps
        raise CliError(str(e), EXIT_USAGE)
    if args.json:
        print(json.dumps({"path": str(z.path.absolute()), "timestamps": z.timestamps,
                          "label": z.label, "width": z.image.width,
                          "height": z.image.height}, indent=2))
    else:
        print(z.label)
        print(_rel(z.path))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"build": build, "frames": frames}[args.command](args)
    except CliError as e:
        print(f"video-digest: error: {e}", file=sys.stderr)
        return e.code
    except FFmpegError as e:
        print(f"video-digest: error: {e}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
