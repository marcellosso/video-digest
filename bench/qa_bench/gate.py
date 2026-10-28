"""The selection quality gate (PRD "Quality bar", issue #2).

No frame-selection change ships unless it holds or improves accuracy at equal
or lower tokens. Absolute targets, measured against the fixed-rate baseline:

  accuracy      digest no worse than 10 points below the baseline
  tokens        digest input tokens <= 25% of the baseline's
  consistency   same verdict on >= 90% of questions across >= 3 runs
  zoom          at most 3 zoom rounds in any run
  build         digest build time under 1x the video's length

With --against PREVIOUS_RESULTS.json it also checks the change itself:
accuracy held or improved, tokens equal or lower, versus the earlier run.

A check that the run can't measure (consistency with one run, zooms offline,
accuracy on questions with no answer key) is N/A, never a silent pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

PASS, FAIL, NA = "PASS", "FAIL", "N/A"


@dataclass(frozen=True)
class Targets:
    accuracy_margin: float = 10.0     # points below the baseline still allowed
    token_ratio: float = 0.25         # digest / baseline input tokens
    consistency: float = 0.90         # share of questions with one verdict across runs
    consistency_min_runs: int = 3
    zoom_rounds: int = 3              # max rounds in any run
    build_ratio: float = 1.0          # build seconds / video seconds
    # --against: allowed slack vs the previous run (0 = strictly hold or improve)
    accuracy_slack: float = 0.0       # points
    token_slack: float = 0.02         # fraction; live token counts jitter a little


def _row(name, target, value, status, detail=""):
    return {"check": name, "target": target, "value": value, "status": status, "detail": detail}


def gate_strategy(results: dict) -> str | None:
    s = results["strategies"]
    return next((n for n in ("digest", "digest_nozoom") if n in s), None)


def check(results: dict, t: Targets = Targets(), against: dict | None = None) -> dict:
    rows = []
    name = gate_strategy(results)
    strategies = results["strategies"]
    dig = strategies[name]["summary"] if name else None
    base = strategies["baseline"]["summary"] if "baseline" in strategies else None
    word = "accuracy" if results["mode"] == "live" else "evidence coverage"

    # accuracy vs baseline
    if dig is None or base is None:
        rows.append(_row("accuracy vs baseline", f">= baseline - {t.accuracy_margin:g} pts",
                         None, NA, "needs both a digest strategy and the baseline"))
    else:
        b_acc = 1.0 if results.get("reference_mode") else base["accuracy"]
        d_acc = dig["accuracy"]
        if d_acc is None or b_acc is None:
            rows.append(_row("accuracy vs baseline", f">= baseline - {t.accuracy_margin:g} pts",
                             None, NA, "no gradable questions (add evidence spans or run --live)"))
        else:
            diff = (d_acc - b_acc) * 100
            rows.append(_row(f"{word} vs baseline", f">= baseline - {t.accuracy_margin:g} pts",
                             f"{d_acc * 100:.0f}% vs {b_acc * 100:.0f}% ({diff:+.0f} pts)",
                             PASS if diff >= -t.accuracy_margin else FAIL))

    # tokens vs baseline
    if dig is None or base is None:
        rows.append(_row("input tokens vs baseline", f"<= {t.token_ratio:.0%}", None, NA))
    else:
        ratio = dig["input_tokens"] / max(base["input_tokens"], 1)
        rows.append(_row("input tokens vs baseline", f"<= {t.token_ratio:.0%}",
                         f"{dig['input_tokens']:,.0f} vs {base['input_tokens']:,.0f} ({ratio:.1%})",
                         PASS if ratio <= t.token_ratio else FAIL,
                         "" if results["mode"] == "live" else "estimated"))

    # consistency
    runs = results["runs"]
    if dig is None or results["mode"] != "live" or runs < t.consistency_min_runs:
        rows.append(_row("consistency across runs", f">= {t.consistency:.0%} of questions", None, NA,
                         f"needs --live and >= {t.consistency_min_runs} runs"))
    else:
        c = dig["consistency"]
        rows.append(_row("consistency across runs", f">= {t.consistency:.0%} of questions",
                         f"{c:.0%} ({dig['consistent']}/{dig['graded']})",
                         PASS if c >= t.consistency else FAIL))

    # zoom rounds
    zr = dig.get("zoom_rounds_max") if dig else None
    if zr is None or name != "digest" or results["mode"] != "live":
        rows.append(_row("zoom rounds", f"usually 0, max {t.zoom_rounds}", None, NA,
                         "needs --live with the digest strategy"))
    else:
        rows.append(_row("zoom rounds", f"usually 0, max {t.zoom_rounds}",
                         f"mean {dig['zoom_rounds_mean']:.1f}, max {zr}",
                         PASS if zr <= t.zoom_rounds else FAIL))

    # build time
    b = results.get("build")
    if not b:
        rows.append(_row("digest build time", f"< {t.build_ratio:g}x video length", None, NA))
    else:
        rows.append(_row("digest build time", f"< {t.build_ratio:g}x video length",
                         f"{b['seconds']:.1f}s for {results['duration']:.1f}s ({b['ratio']:.2f}x)",
                         PASS if b["ratio"] < t.build_ratio else FAIL))

    if against is not None:
        rows += _regression(results, against, name, t)

    failed = [r["check"] for r in rows if r["status"] == FAIL]
    return {"strategy": name, "targets": asdict(t), "checks": rows,
            "passed": not failed, "failed": failed}


def _regression(results: dict, prev: dict, name: str | None, t: Targets) -> list[dict]:
    label = "vs previous run"
    if prev.get("mode") != results["mode"] or prev.get("fixture") != results["fixture"]:
        return [_row(label, "same fixture and mode", None, NA,
                     f"previous was {prev.get('fixture')}/{prev.get('mode')}")]
    if prev.get("model") != results.get("model"):
        return [_row(label, "same model", None, NA,
                     f"previous used {prev.get('model')}, this run {results.get('model')}")]
    old = prev["strategies"].get(name, {}).get("summary") if name else None
    new = results["strategies"][name]["summary"] if name else None
    if not old or not new:
        return [_row(label, "same strategy in both runs", None, NA)]
    rows = []
    if old["accuracy"] is not None and new["accuracy"] is not None:
        diff = (new["accuracy"] - old["accuracy"]) * 100
        rows.append(_row(f"accuracy {label}", "held or improved",
                         f"{new['accuracy'] * 100:.0f}% vs {old['accuracy'] * 100:.0f}% ({diff:+.0f} pts)",
                         PASS if diff >= -t.accuracy_slack else FAIL))
    ratio = new["input_tokens"] / max(old["input_tokens"], 1)
    rows.append(_row(f"input tokens {label}", "equal or lower",
                     f"{new['input_tokens']:,.0f} vs {old['input_tokens']:,.0f} ({ratio - 1:+.1%})",
                     PASS if ratio <= 1 + t.token_slack else FAIL))
    return rows
