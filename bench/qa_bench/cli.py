"""Command line: python -m bench.qa_bench FIXTURE_OR_VIDEO [options]

FIXTURE_OR_VIDEO is "synthetic", "showreel", or a path to your own video
(then --questions is required). Without --live nothing calls a model.

Exit status: 0 when the quality gate passes, 1 when a check fails
(N/A checks don't fail), 2 on usage errors.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import fixtures
from .gate import Targets, check
from .report import render
from .run import run
from .tokens import DEFAULT_PROFILE, PROFILES


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bench.qa_bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help='"synthetic", "showreel", or a video path')
    ap.add_argument("--questions", help="question set JSON (defaults to the fixture's)")
    ap.add_argument("--strategies", help="comma-separated; default baseline,digest offline and "
                    "baseline,digest_nozoom,digest live")
    ap.add_argument("--live", action="store_true", help="ask a real model (needs Anthropic credentials)")
    ap.add_argument("--runs", type=int, default=3, help="live runs per strategy (default 3)")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--judge-model", default="claude-haiku-5-5")
    ap.add_argument("--baseline-every", type=float, default=2.0, help="seconds between baseline frames")
    ap.add_argument("--max-images", type=int, default=100)
    ap.add_argument("--max-zoom-rounds", type=int, default=3)
    ap.add_argument("--token-profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE)
    ap.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE",
                    help="selection setting override, e.g. --set threshold=2.0 (repeatable)")
    ap.add_argument("--against", help="previous results.json: also require accuracy held and "
                    "tokens equal or lower")
    ap.add_argument("--margin", type=float, default=Targets.accuracy_margin,
                    help="accuracy points the digest may trail the baseline")
    ap.add_argument("--out", help="output folder (default bench/out/<fixture>-<mode>)")
    return ap


def _settings(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if not v:
            raise SystemExit(f"--set expects KNOB=VALUE, got {p!r}")
        out[k] = json.loads(v)  # numbers and true/false
    return out


def main(argv: list[str] | None = None, client=None) -> int:
    args = parser().parse_args(argv)
    fixture = fixtures.resolve(args.target, args.questions)
    default = "baseline,digest_nozoom,digest" if args.live else "baseline,digest"
    names = [n.strip() for n in (args.strategies or default).split(",") if n.strip()]
    mode = "live" if args.live else "offline"
    out = Path(args.out or fixtures.BENCH_DIR / "out" / f"{fixture.name}-{mode}")
    out.mkdir(parents=True, exist_ok=True)

    if args.live and client is None:
        try:
            import anthropic
        except ImportError:
            raise SystemExit("--live needs the anthropic package: pip install -e '.[bench]'")
        client = anthropic.Anthropic()

    results = run(fixture, names, out, client=client if args.live else None, model=args.model,
                  judge_model=args.judge_model, runs=args.runs, baseline_every=args.baseline_every,
                  max_images=args.max_images, max_zoom_rounds=args.max_zoom_rounds,
                  token_profile=args.token_profile, settings=_settings(args.set))
    against = json.loads(Path(args.against).read_text()) if args.against else None
    gate = check(results, Targets(accuracy_margin=args.margin), against)
    results["gate"] = gate

    report = render(results, gate)
    (out / "qa_report.md").write_text(report)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(report)
    print(f"Wrote {os.path.relpath(out / 'qa_report.md')} and results.json")
    return 0 if gate["passed"] else 1
