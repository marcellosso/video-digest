"""qa_bench: grading, gate, the ask loop and the CLI, with no model calls.

A fake client stands in for anthropic.Anthropic. The real-model path is
opt-in: VIDEO_DIGEST_LIVE=1 runs the synthetic fixture live (costs money).
"""
import json
import os
import shutil
from types import SimpleNamespace as NS

import pytest

from bench.qa_bench import cli
from bench.qa_bench.ask import ask
from bench.qa_bench.fixtures import FIXTURES_DIR, load_questions, make_synthetic
from bench.qa_bench.gate import FAIL, NA, PASS, check
from bench.qa_bench.grading import evidence_covered, grade, parse_seconds
from bench.qa_bench.run import summarize
from bench.qa_bench.strategies import digest as digest_strategy
from bench.qa_bench.tokens import image_tokens
from conftest import needs_ffmpeg
from video_digest import Digest

SYNTHETIC_QUESTIONS = FIXTURES_DIR / "synthetic" / "questions.json"


# ------------------------------------------------------------------ fakes
def response(text="", tool=None, tokens=1000):
    content = [NS(type="text", text=text)] if text else []
    if tool is not None:
        content.append(NS(type="tool_use", id="tu_1", name="get_frames", input=tool))
    return NS(content=content, stop_reason="tool_use" if tool is not None else "end_turn",
              usage=NS(input_tokens=tokens, output_tokens=50,
                       cache_creation_input_tokens=0, cache_read_input_tokens=0))


class FakeClient:
    """Answers from a fixed dict; optionally zooms first. Judge calls say "correct".

    Reported input tokens grow with the images sent, like the real API.
    """

    def __init__(self, answers, zoom=None, zoom_forever=False):
        self.answers, self.zoom, self.zoom_forever = answers, zoom, zoom_forever
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if "system" not in kw:  # judge
            return response('{"verdict": "correct", "reason": "same"}', tokens=100)
        msgs = kw["messages"]
        images = json.dumps(msgs, default=lambda o: None).count('"type": "image"')
        tokens = 100 + 1000 * images
        if "tools" in kw and self.zoom and kw.get("tool_choice") != {"type": "none"} \
                and (len(msgs) == 1 or self.zoom_forever):
            return response(tool={"timestamps": self.zoom}, tokens=tokens)
        return response(json.dumps(self.answers), tokens=tokens)


# ------------------------------------------------------------------ units
def test_image_tokens():
    assert image_tokens(1920, 1080) == 69 * 39          # ~one token per 28x28 patch
    assert image_tokens(3840, 2160) <= 4784             # downscaled to 2576 px, capped
    assert image_tokens(1920, 1080, "legacy") <= 1600
    assert image_tokens(320, 180) < image_tokens(1280, 720)


def test_grading_checks():
    q = {"id": "a", "question": "?", "check": "contains", "answer": ["blue 42"], "reject": ["red"]}
    assert grade(q, "The code was BLUE-42.")[0]
    assert not grade(q, "blue-42 or red")[0]
    assert grade(q, "unknown") == (False, "no answer")
    num = {"id": "n", "question": "?", "check": "number", "answer": 93, "tol": 1}
    assert grade(num, "about 01:33.4")[0] and not grade(num, "95 seconds")[0]
    assert parse_seconds("50+") == 50
    yn = {"id": "y", "question": "?", "check": "yesno", "answer": "no"}
    assert grade(yn, "No.")[0] and not grade(yn, "Yes, one")[0]
    judged = {"id": "j", "question": "?", "check": "judge", "answer": "ref"}
    assert grade(judged, "anything", FakeClient({}), "judge-model") == (True, "same")


def test_evidence_proxy():
    q = {"evidence": [{"start": 93, "end": 94.99}, {"start": 45, "end": 60, "min_frames": 2}]}
    assert evidence_covered(q, [46, 50, 93.5])
    assert not evidence_covered(q, [46, 93.5])           # motion needs two frames
    assert not evidence_covered(q, [46, 50, 92, 95])     # the flash fell between samples
    assert evidence_covered({"evidence": []}, []) is True
    assert evidence_covered({}, [1, 2]) is None


def test_question_sets_are_valid():
    for name in ("synthetic", "showreel"):
        qs = load_questions(FIXTURES_DIR / name / "questions.json")
        assert len(qs) >= 10
        assert all({"id", "question", "check", "answer"} <= q.keys() for q in qs)
    assert all("evidence" in q for q in load_questions(SYNTHETIC_QUESTIONS))


# ------------------------------------------------------------------ gate
def results_with(base_acc, dig_acc, base_tok, dig_tok, runs=3, zoom_max=0, build_ratio=0.2):
    def strat(acc, tok, zoom=None):
        return {"summary": {"accuracy": acc, "input_tokens": tok, "consistency": 1.0,
                            "consistent": 10, "graded": 10, "zoom_rounds_mean": zoom,
                            "zoom_rounds_max": zoom}}
    return {"fixture": "x", "mode": "live", "runs": runs, "duration": 30.0,
            "reference_mode": False, "build": {"seconds": 30 * build_ratio, "ratio": build_ratio},
            "strategies": {"baseline": strat(base_acc, base_tok),
                           "digest": strat(dig_acc, dig_tok, zoom_max)}}


def statuses(g):
    return {r["check"]: r["status"] for r in g["checks"]}


def test_gate_passes_on_the_prd_story():
    g = check(results_with(0.71, 1.0, 41049, 6688))
    assert g["passed"] and set(statuses(g).values()) == {PASS}


def test_gate_fails_each_target():
    assert "accuracy vs baseline" in check(results_with(0.9, 0.7, 40000, 5000))["failed"]
    assert "input tokens vs baseline" in check(results_with(0.7, 0.9, 40000, 12000))["failed"]
    assert "zoom rounds" in check(results_with(0.7, 0.9, 40000, 5000, zoom_max=4))["failed"]
    assert "digest build time" in check(results_with(0.7, 0.9, 40000, 5000, build_ratio=1.5))["failed"]


def test_gate_marks_unmeasured_checks_na():
    s = statuses(check(results_with(0.7, 0.9, 40000, 5000, runs=1)))
    assert s["consistency across runs"] == NA
    r = results_with(0.7, 0.9, 40000, 5000)
    r["strategies"]["digest"]["summary"]["accuracy"] = None
    g = check(r)
    assert statuses(g)["accuracy vs baseline"] == NA and g["passed"]


def test_gate_against_previous_run():
    prev = results_with(0.71, 1.0, 41049, 6000)
    s = statuses(check(results_with(0.71, 0.9, 41049, 6000), against=prev))
    assert s["accuracy vs previous run"] == FAIL
    s = statuses(check(results_with(0.71, 1.0, 41049, 6500), against=prev))
    assert s["input tokens vs previous run"] == FAIL
    s = statuses(check(results_with(0.71, 1.0, 41049, 5000), against=prev))
    assert s["accuracy vs previous run"] == PASS and s["input tokens vs previous run"] == PASS


def test_summarize_consistency():
    runs = [{"input_tokens": 10, "grades": {"a": [True, ""], "b": [True, ""]}},
            {"input_tokens": 20, "grades": {"a": [True, ""], "b": [False, ""]}},
            {"input_tokens": 30, "grades": {"a": [True, ""], "b": [True, ""]}}]
    m = summarize(runs)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert (m["accuracy_min"], m["accuracy_max"]) == (0.5, 1.0)
    assert (m["consistent"], m["graded"]) == (1, 2)
    assert m["input_tokens"] == 20


# ------------------------------------------------------------------ ask loop
@pytest.fixture(scope="module")
def small_digest(video, tmp_path_factory):
    return Digest.build(video, cache_dir=tmp_path_factory.mktemp("cache"))


@needs_ffmpeg
def test_ask_zooms_then_answers(small_digest):
    s = digest_strategy(small_digest, zoom=True, build_seconds=1.0)
    qs = [{"id": "word", "question": "Which word?"}]
    client = FakeClient({"word": "EVERYTHING"}, zoom=[4.5, 1.0])
    r = ask(client, "m", s, qs, "run1")
    assert r["answers"] == {"word": "EVERYTHING"}
    assert (r["zoom_rounds"], r["zoom_frames"]) == (1, 2)
    n = len(small_digest.sheets)
    assert r["first_turn_input_tokens"] == 100 + 1000 * n
    assert r["input_tokens"] == r["first_turn_input_tokens"] + 100 + 1000 * (n + 1)
    # the zoom result is a labeled sheet from the engine, appended to the history
    second = client.calls[1]["messages"]
    tool_result = second[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert [b["type"] for b in tool_result["content"]] == ["text", "image"]
    first = client.calls[0]["messages"][0]["content"]
    assert first[0]["text"] == "[run run1]"
    assert sum("cache_control" in b for b in first) == 1


@needs_ffmpeg
def test_ask_stops_zooming_after_max_rounds(small_digest):
    s = digest_strategy(small_digest, zoom=True, build_seconds=1.0)
    client = FakeClient({"q": "x"}, zoom=[1.0], zoom_forever=True)
    r = ask(client, "m", s, [{"id": "q", "question": "?"}], "r", max_rounds=2)
    assert r["zoom_rounds"] == 2
    assert client.calls[-1]["tool_choice"] == {"type": "none"}


@needs_ffmpeg
def test_nozoom_has_no_tool_and_no_zoom_line(small_digest):
    s = digest_strategy(small_digest, zoom=False, build_seconds=1.0)
    assert "ask for a full-resolution frame" not in s.content[0]["text"]
    client = FakeClient({"q": "x"}, zoom=[1.0])
    r = ask(client, "m", s, [{"id": "q", "question": "?"}], "r")
    assert r["zoom_rounds"] == 0 and "tools" not in client.calls[0]


# ------------------------------------------------------------------ CLI
@pytest.fixture(scope="session")
def synthetic_small(tmp_path_factory):
    """The synthetic fixture at reduced resolution: same timeline, faster to encode."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    return make_synthetic(tmp_path_factory.mktemp("synthetic") / "synthetic.mp4", 640, 360, fps=10)


@needs_ffmpeg
def test_cli_offline_synthetic_passes_the_gate(synthetic_small, tmp_path, capsys):
    code = cli.main([str(synthetic_small), "--questions", str(SYNTHETIC_QUESTIONS),
                     "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Quality gate (digest) — PASSED" in out
    res = json.loads((tmp_path / "results.json").read_text())
    dig, base = res["strategies"]["digest"]["summary"], res["strategies"]["baseline"]["summary"]
    assert dig["accuracy"] == 1.0 and dig["graded"] == 10
    assert dig["input_tokens"] < 0.25 * base["input_tokens"]
    assert res["build"]["ratio"] < 1.0
    assert (tmp_path / "qa_report.md").read_text().startswith("# QA benchmark")


@needs_ffmpeg
def test_cli_offline_catches_a_missed_flash(synthetic_small, tmp_path, capsys):
    # sampling every 5 s steps over the 2-second code flash at 93-95 s
    cli.main([str(synthetic_small), "--questions", str(SYNTHETIC_QUESTIONS), "--out", str(tmp_path),
              "--baseline-every", "5"])
    res = json.loads((tmp_path / "results.json").read_text())
    grades = res["strategies"]["baseline"]["runs"][0]["grades"]
    assert grades["flash"][0] is False and grades["fact"][0] is True


@needs_ffmpeg
def test_cli_live_path_with_fake_client(video, tmp_path, capsys):
    qs = [{"id": "word", "question": "Which word?", "check": "contains", "answer": ["everything"]},
          {"id": "end", "question": "Last words?", "check": "judge", "answer": "THE END"},
          {"id": "absent", "question": "A phone number?", "check": "yesno", "answer": "no"}]
    qpath = tmp_path / "q.json"
    qpath.write_text(json.dumps(qs))
    client = FakeClient({"word": "EVERYTHING", "end": "The end", "absent": "no"}, zoom=[4.5])
    # a 9 s clip: sample the baseline densely so it is a fair token comparison
    code = cli.main([str(video), "--questions", str(qpath), "--live", "--runs", "3",
                     "--baseline-every", "0.5", "--out", str(tmp_path / "out")], client=client)
    out = capsys.readouterr().out
    assert code == 0, out
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert res["mode"] == "live" and set(res["strategies"]) == {"baseline", "digest_nozoom", "digest"}
    dig = res["strategies"]["digest"]["summary"]
    assert dig["accuracy"] == 1.0 and dig["consistency"] == 1.0 and dig["zoom_rounds_max"] == 1
    assert "## Answers (run 1)" in out


def test_showreel_without_video_explains_how(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_DIGEST_SHOWREEL", str(tmp_path / "missing.mp4"))
    if (FIXTURES_DIR / "showreel" / "benchmarket-showreel.mp4").exists():
        pytest.skip("showreel present locally")
    with pytest.raises(SystemExit, match="VIDEO_DIGEST_SHOWREEL"):
        cli.main(["showreel"])


# ------------------------------------------------------------------ opt-in live
@pytest.mark.skipif(os.environ.get("VIDEO_DIGEST_LIVE") != "1",
                    reason="live model run; set VIDEO_DIGEST_LIVE=1 (costs money)")
def test_live_synthetic_passes_the_gate(tmp_path):
    pytest.importorskip("anthropic")
    runs = os.environ.get("VIDEO_DIGEST_LIVE_RUNS", "1")
    assert cli.main(["synthetic", "--live", "--runs", runs, "--out", str(tmp_path)]) == 0
