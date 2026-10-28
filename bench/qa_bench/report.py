"""Markdown report for a benchmark run (printed and written to qa_report.md)."""
from __future__ import annotations

import os


def _pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def render(results: dict, gate: dict | None = None) -> str:
    live = results["mode"] == "live"
    ref_mode = results["reference_mode"]
    strategies = results["strategies"]
    questions = results["questions"]
    word = ("Agreement" if ref_mode else "Accuracy") if live else "Evidence coverage"
    head = (f"{results['duration']:.1f}s video, {results['width']}x{results['height']} · "
            f"{len(questions)} questions · ")
    head += (f"{results['runs']} run(s) per strategy · model {results['model']} · "
             f"judge {results['judge_model']}" if live else
             f"offline (no model calls) · token profile {results['token_profile']}")
    L = [f"# QA benchmark — {results['fixture']} ({os.path.basename(results['video'])})", head]
    if live:
        L.append("Grading: " + ("the baseline's answers are the reference; the judge checks "
                                "agreement, not correctness." if ref_mode else "answer key.")
                 + " Each run starts uncached.")
    else:
        L.append("Offline: tokens are estimates, and coverage counts a question as answerable "
                 "when a shown frame falls inside its evidence span (an upper bound, not "
                 "accuracy). Add --live for real accuracy and tokens.")
    b = results.get("build")
    if b:
        L.append(f"Digest: {b['keyframes']} keyframes in {b['sheets']} sheet(s), built in "
                 f"{b['seconds']:.1f}s ({b['ratio']:.2f}x video length).")
    L.append("")

    base_tok = strategies["baseline"]["summary"]["input_tokens"] if "baseline" in strategies else None
    if not live:
        L += ["## Summary", "",
              f"| Strategy | Images | Frames shown | {word} | Est. input tokens | vs baseline |",
              "|---|---|---|---|---|---|"]
        for name, s in strategies.items():
            tok = s["summary"]["input_tokens"]
            L.append(f"| {name} | {s['images']} | {s['frames_shown']} | "
                     f"{_pct(s['summary']['accuracy'])} | {tok:,.0f} | "
                     f"{'—' if not base_tok else f'{tok / base_tok:.1%}'} |")
    else:
        L += ["## Summary", "",
              f"| Strategy | Images | {word} | Range over runs | Consistent | Input tokens / run "
              "| vs baseline | Zoom rounds mean (max) | Cost / run |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, s in strategies.items() if live else ():
        m = s["summary"]
        acc = "reference" if (live and ref_mode and name == "baseline") else _pct(m["accuracy"])
        rng = ("—" if m["accuracy"] is None
               else f"{_pct(m['accuracy_min'])}–{_pct(m['accuracy_max'])}")
        cons = "—" if not m["graded"] else f"{m['consistent']}/{m['graded']}"
        zoom = "—" if name != "digest" else f"{m['zoom_rounds_mean']:.1f} ({m['zoom_rounds_max']})"
        cost = "—" if m["cost_usd"] is None else f"${m['cost_usd']:.4f}"
        vs = "—" if not base_tok else f"{m['input_tokens'] / base_tok:.1%}"
        L.append(f"| {name} | {s['images']} | {acc} | {rng} | {cons} | {m['input_tokens']:,.0f} | "
                 f"{vs} | {zoom} | {cost} |")

    if gate:
        L += ["", f"## Quality gate ({gate['strategy']}) — "
              f"{'PASSED' if gate['passed'] else 'FAILED: ' + ', '.join(gate['failed'])}", "",
              "| Check | Target | Value | Status |", "|---|---|---|---|"]
        for r in gate["checks"]:
            val = r["value"] or "—"
            if r["detail"]:
                val += f" ({r['detail']})" if r["value"] else f" {r['detail']}"
            L.append(f"| {r['check']} | {r['target']} | {val} | {r['status']} |")

    scored = [n for n in strategies if not (live and ref_mode and n == "baseline")]
    graded_ids = [q["id"] for q in questions
                  if all(q["id"] in strategies[n]["runs"][0].get("grades", {}) for n in scored)]
    if "baseline" in strategies and not ref_mode and len(scored) > 1 and graded_ids:
        maj = {n: {i: sum(r["grades"][i][0] for r in strategies[n]["runs"]) * 2
                   > len(strategies[n]["runs"]) for i in graded_ids} for n in scored}
        L += ["", "## Head to head vs baseline (majority verdict per question)", ""]
        for n in scored:
            if n == "baseline":
                continue
            won = [i for i in graded_ids if maj[n][i] and not maj["baseline"][i]]
            lost = [i for i in graded_ids if maj["baseline"][i] and not maj[n][i]]
            both = [i for i in graded_ids if not maj[n][i] and not maj["baseline"][i]]
            L.append(f"- **{n}** — right where baseline was wrong: {', '.join(won) or 'none'}; "
                     f"wrong where baseline was right: {', '.join(lost) or 'none'}; "
                     f"both wrong: {', '.join(both) or 'none'}")

    L += ["", "## Per question", "", "| Question | Type | " + " | ".join(scored) + " |",
          "|---|---|" + "---|" * len(scored)]
    for q in questions:
        cells = []
        for n in scored:
            runs = strategies[n]["runs"]
            if q["id"] not in runs[0].get("grades", {}):
                cells.append("—")
            else:
                cells.append(f"{sum(r['grades'][q['id']][0] for r in runs)}/{len(runs)}")
        L.append(f"| {q['id']} | {q.get('type', '')} | " + " | ".join(cells) + " |")

    if live:
        L += ["", "## Answers (run 1)", ""]
        for q in questions:
            L.append(f"**{q['id']}** — {q['question']}")
            if "answer" in q:
                L.append(f"- expected: {q['answer']}")
            for n, s in strategies.items():
                r = s["runs"][0]
                g = r["grades"].get(q["id"])
                mark = "" if g is None else (" ✅" if g[0] else " ❌")
                note = f" _({g[1]})_" if g and g[1] else ""
                L.append(f"- {n}{mark}: {r['answers'].get(q['id'], '(none)')}{note}")
            L.append("")

    L += ["", "## Caveats",
          f"- {len(questions)} questions is a small sample; a few points can be noise.",
          "- Input tokens are processed tokens summed over every turn, before caching "
          "discounts; cost applies cache pricing." if live else
          "- Token estimates assume the image formula in bench/qa_bench/tokens.py; live runs "
          "report real usage, which has come out higher than estimates before."]
    return "\n".join(L) + "\n"
