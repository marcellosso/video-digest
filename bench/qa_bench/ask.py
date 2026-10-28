"""Asking a model: all questions in one conversation, with an optional zoom loop.

`client` is anything with `.messages.create(...)` shaped like anthropic.Anthropic,
so tests can pass a fake and never call the API.
"""
from __future__ import annotations

import json
import re
import time

from video_digest import Digest

ANSWER_SYSTEM = (
    "You are answering questions about a video. You cannot watch it directly; you are given "
    "material extracted from it. Answer every question concisely. If the material does not "
    "let you answer, say \"unknown\" rather than guessing. Reply with ONLY a JSON object "
    "mapping each question id to your answer string.")

ZOOM_TOOL = {
    "name": "get_frames",
    "description": ("Look at the video at specific timestamps. One timestamp returns a full "
                    "frame; several return one labeled contact sheet. Batch all the "
                    "timestamps you need into a single call."),
    "input_schema": {"type": "object", "properties": {
        "timestamps": {"type": "array", "items": {"type": "number"}, "minItems": 1,
                       "maxItems": 9, "description": "Seconds from the start."}},
        "required": ["timestamps"]}}


def _zoom(d: Digest, call) -> tuple[dict, int]:
    """A tool_result for one get_frames call, and how many frames it returned."""
    raw = call.input.get("timestamps") if isinstance(call.input, dict) else None
    try:
        ts = sorted({round(float(t), 2) for t in (raw or [])})[:9]
        blocks = d.frames(ts).to_content("anthropic")
    except (TypeError, ValueError) as e:
        return {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                "content": f"get_frames failed: {e}"}, 0
    return {"type": "tool_result", "tool_use_id": call.id, "content": blocks}, len(ts)


def ask(client, model: str, strategy, questions: list[dict], run_id: str,
        max_rounds: int = 3, max_tokens: int = 16000) -> dict:
    """Ask every question about `strategy`'s material. Returns answers and real usage.

    run_id goes first so every run is a fresh, uncached read of the video
    (identical repeat runs would otherwise get ~90% off and skew the comparison).
    Within a run the video material is cached at its last image, and top-level
    automatic caching moves a second breakpoint to the newest turn, so zoom
    turns re-read earlier context cheaply. History is append-only.
    """
    content = [{"type": "text", "text": f"[run {run_id}]"}, *strategy.content]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    qs = "\n".join(f'- {q["id"]}: {q["question"]}' for q in questions)
    content.append({"type": "text", "text": f"Questions:\n{qs}"})
    msgs = [{"role": "user", "content": content}]

    tin = tout = rounds = frames = 0
    billed = 0.0  # input tokens weighted by cache pricing (write 1.25x, read 0.1x)
    first_turn = None
    t0 = time.time()
    text, stop = "", None
    while True:
        kw = {}
        if strategy.zoom:  # once used, the tool must stay declared; "none" forces an answer
            kw["tools"] = [ZOOM_TOOL]
            if rounds >= max_rounds:
                kw["tool_choice"] = {"type": "none"}
        r = client.messages.create(model=model, max_tokens=max_tokens, system=ANSWER_SYSTEM,
                                   messages=msgs, cache_control={"type": "ephemeral"}, **kw)
        u = r.usage
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        processed = u.input_tokens + cw + cr
        first_turn = processed if first_turn is None else first_turn
        tin += processed
        billed += u.input_tokens + 1.25 * cw + 0.1 * cr
        tout += u.output_tokens
        stop = r.stop_reason
        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:
            text = "".join(b.text for b in r.content if b.type == "text")
            break
        rounds += 1
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for c in calls:
            res, n = _zoom(strategy.digest, c)
            results.append(res)
            frames += n
        if rounds >= max_rounds:
            results.append({"type": "text", "text": "That was the last zoom. Answer now."})
        msgs.append({"role": "user", "content": results})

    m = re.search(r"\{.*\}", text, re.S)
    try:
        answers = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        answers = {}
    return {"answers": {k: str(v) for k, v in answers.items()},
            "input_tokens": tin, "first_turn_input_tokens": first_turn,
            "billed_input_tokens": round(billed), "output_tokens": tout,
            "zoom_rounds": rounds, "zoom_frames": frames, "stop_reason": stop,
            "latency_s": round(time.time() - t0, 1)}
