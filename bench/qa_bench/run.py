"""Run a benchmark: build each strategy's input, ask or estimate, grade, summarize.

Offline (default) makes no model calls: input tokens are estimated and each
question with evidence spans is "answerable" when the strategy shows a frame
inside them. Live (`client` given) asks the model every question `runs` times
and grades the answers.
"""
from __future__ import annotations

import statistics
import uuid
from pathlib import Path

from video_digest.sample import probe

from . import strategies as strat_mod
from .ask import ask
from .fixtures import Fixture, sha256_of
from .grading import evidence_covered, grade
from .tokens import DEFAULT_PROFILE

PRICES = {  # $ per million input / output tokens
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def summarize(runs: list[dict], price: tuple[float, float] | None = None) -> dict:
    """Accuracy (mean, range), consistency, tokens, zooms and cost over runs.

    Only graded questions count; accuracy is None when nothing was gradable.
    """
    ids = sorted(set().union(*(r.get("grades", {}).keys() for r in runs))) if runs else []
    per_run = [sum(r["grades"][i][0] for i in ids) / len(ids) for r in runs] if ids else []
    consistent = sum(len({r["grades"][i][0] for r in runs}) == 1 for i in ids)
    out = {
        "graded": len(ids),
        "accuracy": statistics.mean(per_run) if per_run else None,
        "accuracy_min": min(per_run) if per_run else None,
        "accuracy_max": max(per_run) if per_run else None,
        "consistent": consistent,
        "consistency": consistent / len(ids) if ids else None,
        "input_tokens": statistics.mean(r["input_tokens"] for r in runs),
        "zoom_rounds_mean": None, "zoom_rounds_max": None, "cost_usd": None,
    }
    if runs and "zoom_rounds" in runs[0]:
        out["zoom_rounds_mean"] = statistics.mean(r["zoom_rounds"] for r in runs)
        out["zoom_rounds_max"] = max(r["zoom_rounds"] for r in runs)
    if price and runs and "billed_input_tokens" in runs[0]:
        out["cost_usd"] = statistics.mean(r["billed_input_tokens"] * price[0] / 1e6
                                          + r["output_tokens"] * price[1] / 1e6 for r in runs)
    return out


def run(fixture: Fixture, names: list[str], out_dir: Path, *, client=None,
        model: str = "claude-opus-5-5", judge_model: str = "claude-haiku-5-5", runs: int = 3,
        baseline_every: float = 2.0, max_images: int = 100, max_zoom_rounds: int = 3,
        token_profile: str = DEFAULT_PROFILE, settings: dict | None = None,
        log=print) -> dict:
    live = client is not None
    questions = fixture.questions
    info = probe(str(fixture.video))
    sha = sha256_of(fixture.video)
    if fixture.sha256 and sha != fixture.sha256:
        log(f"warning: {fixture.video} does not match the fixture's pinned sha256; "
            "results are not comparable to the reference.")
    reference_mode = all("answer" not in q for q in questions)
    if reference_mode and "baseline" not in names:
        raise SystemExit("Questions have no answers, so the baseline is needed as the reference.")
    if not live and "digest_nozoom" in names and "digest" in names:
        names = [n for n in names if n != "digest_nozoom"]  # identical offline

    log(f"preparing {', '.join(names)} for {fixture.video.name} ({info.duration:.1f}s) …")
    strategies = strat_mod.make(names, fixture.video, out_dir, baseline_every, max_images, settings)

    results = {
        "fixture": fixture.name, "video": str(fixture.video), "video_sha256": sha,
        "duration": info.duration, "width": info.width, "height": info.height,
        "mode": "live" if live else "offline", "model": model if live else None,
        "judge_model": judge_model if live else None, "runs": runs if live else 1,
        "token_profile": token_profile, "baseline_every": baseline_every,
        "reference_mode": reference_mode, "questions": questions, "build": None,
        "strategies": {},
    }
    d = next((s.digest for s in strategies.values() if s.digest), None)
    if d is not None:
        secs = next(s.prep_seconds for s in strategies.values() if s.digest)
        results["build"] = {"seconds": round(secs, 2), "ratio": round(secs / info.duration, 3),
                            "keyframes": len(d.keyframes), "sheets": len(d.sheets),
                            "settings": d.data["settings"], "engine_version": d.data["engine_version"]}

    for name, s in strategies.items():
        results["strategies"][name] = {
            "images": len(s.image_sizes), "frames_shown": len(s.times),
            "estimated_input_tokens": s.estimate_tokens(token_profile),
            "prep_seconds": round(s.prep_seconds, 2), "runs": []}

    if not live:
        for name, s in strategies.items():
            grades = {}
            for q in questions:
                ok = evidence_covered(q, s.times)
                if ok is not None:
                    grades[q["id"]] = [ok, "evidence shown" if ok else "evidence not shown"]
            r = results["strategies"][name]
            r["runs"] = [{"input_tokens": r["estimated_input_tokens"], "grades": grades}]
            r["summary"] = summarize(r["runs"])
        return results

    price = PRICES.get(model)
    for i in range(runs):
        run_id = uuid.uuid4().hex[:8]
        for name, s in strategies.items():
            log(f"run {i + 1}/{runs}: {name} …")
            results["strategies"][name]["runs"].append(
                ask(client, model, s, questions, f"{run_id}-{name}", max_zoom_rounds))
        ref = results["strategies"]["baseline"]["runs"][-1]["answers"] if reference_mode else {}
        for name in strategies:
            r = results["strategies"][name]["runs"][-1]
            if reference_mode and name == "baseline":
                r["grades"] = {}
                continue
            r["grades"] = {q["id"]: list(grade(q, r["answers"].get(q["id"], ""), client,
                                               judge_model, ref.get(q["id"])))
                           for q in questions}
    for name in strategies:
        r = results["strategies"][name]
        r["summary"] = summarize(r["runs"], price)
    return results
