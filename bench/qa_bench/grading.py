"""Grading answers, and the offline evidence proxy.

Question fields:
  id, question, type     what is asked, and which failure mode it probes
  check                  contains | number | yesno | judge
  answer                 terms (contains), a number (number), "yes"/"no", or a
                         reference sentence (judge). Omit it on your own videos:
                         the baseline's answer becomes the reference.
  tol, reject            number tolerance; terms that make a contains-answer wrong
  evidence               optional: [{"start": s, "end": e, "min_frames": n}, ...].
                         The answer is visible only in those moments; offline,
                         a strategy "can answer" when every span has at least
                         min_frames (default 1) shown frames inside [start, end].
"""
from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are grading an answer about a video.
Question: {q}
Reference answer: {ref}
Answer to grade: {a}
Does the answer to grade convey the same essential information as the reference? Minor wording
differences and extra correct detail are fine; missing or contradicting key facts are not.
Reply with ONLY JSON: {{"verdict": "correct" or "incorrect", "reason": "<one sentence>"}}"""


def norm(s) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", str(s).lower()).strip()


def parse_seconds(s: str) -> float | None:
    m = re.search(r"(\d+):(\d{1,2}(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def judge(client, model: str, q: str, ref: str, a: str) -> tuple[bool, str]:
    r = client.messages.create(model=model, max_tokens=4000, messages=[
        {"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, a=a)}])
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(m.group(0))
        return j.get("verdict") == "correct", j.get("reason", "")
    except (AttributeError, json.JSONDecodeError):
        return False, "judge output unparseable"


def grade(q: dict, a: str, client=None, judge_model: str | None = None,
          reference: str | None = None) -> tuple[bool, str]:
    """(correct, note). `reference` is the baseline's answer when q has no answer."""
    if not a or norm(a) in ("", "unknown"):
        return False, "no answer"
    if "answer" not in q:
        return judge(client, judge_model, q["question"], reference, a)
    c, exp, n = q["check"], q["answer"], norm(a)
    if c == "contains":
        ok = any(norm(t) in n for t in exp) and not any(norm(t) in n for t in q.get("reject", []))
        return ok, ""
    if c == "number":
        v = parse_seconds(a)
        return v is not None and abs(v - exp) <= q.get("tol", 0), f"parsed {v}"
    if c == "yesno":
        first = n.split()[0].strip(".") if n else ""
        return first == exp, ""
    if c == "judge":
        return judge(client, judge_model, q["question"], exp, a)
    raise ValueError(f"unknown check {c!r} in question {q['id']}")


def evidence_covered(q: dict, times: list[float]) -> bool | None:
    """Offline proxy: were the frames needed for this answer shown at all?

    None when the question has no evidence spans (cannot be judged offline).
    A question with an empty list needs no frame (e.g. a hallucination check).
    This is an upper bound: a frame inside the span may still be unreadable.
    """
    spans = q.get("evidence")
    if spans is None:
        return None
    return all(sum(s["start"] <= t <= s["end"] for t in times) >= s.get("min_frames", 1)
               for s in spans)
