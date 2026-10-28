"""Optional local speech-to-text: audio extraction, digest wiring and the CLI flag.

Everything here runs with a fake Whisper model, so no model is downloaded and
the 'transcript' extra isn't needed. The live test at the bottom runs a real
model only when VIDEO_DIGEST_LIVE_STT=1.
"""
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

from conftest import FPS, H, W, needs_ffmpeg
from video_digest import Digest, cli, transcript
from video_digest.digest import timeline_text

pytestmark = needs_ffmpeg


def _no_extra(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "faster_whisper" else real(name, *a))


def test_timeline_renders_segments():
    base = {"duration": 4.0, "width": W, "height": H, "keyframes": [], "sheets": []}
    off = timeline_text({**base, "transcript": None})
    assert "Transcript: not available (speech-to-text was not requested)." in off
    tr = {"model": "base", "language": "en", "has_audio": True,
          "segments": [{"start": 0.0, "end": 1.5, "text": "Hello there."},
                       {"start": 62.0, "end": 63.25, "text": "Bye."}]}
    md = timeline_text({**base, "transcript": tr})
    assert "Transcript (local speech-to-text, whisper base, language en;" in md
    assert "  [00:00.0 – 00:01.5] Hello there.\n" in md
    assert "  [01:02.0 – 01:03.2] Bye.\n" in md
    silent = timeline_text({**base, "transcript": {**tr, "segments": []}})
    assert "Transcript: no speech detected." in silent
    no_audio = timeline_text({**base, "transcript": {**tr, "has_audio": False, "segments": []}})
    assert "Transcript: none (the video has no audio track)." in no_audio


def test_audio_extraction(tone_video, video):
    assert transcript.has_audio(str(tone_video))
    audio = transcript.load_audio(str(tone_video))
    assert audio.dtype.name == "float32"
    assert len(audio) == pytest.approx(4 * transcript.SAMPLE_RATE, rel=0.05)
    assert 0.1 < abs(audio).max() <= 1.0
    assert not transcript.has_audio(str(video))


def test_build_with_transcript(tone_video, tmp_path, fake_whisper):
    d = Digest.build(tone_video, cache_dir=tmp_path, transcript=True)
    assert fake_whisper.loaded == ["base"]
    (audio, kwargs), = fake_whisper.calls
    assert audio.dtype.name == "float32" and kwargs["vad_filter"] is True
    data = json.loads((d.path / "digest.json").read_text())
    assert data["transcript"] == d.transcript == {
        "model": "base", "language": "en", "has_audio": True,
        "segments": [{"start": 0.0, "end": 1.5, "text": "Hello there."},
                     {"start": 2.0, "end": 3.25, "text": "Click save now."}]}
    assert "  [00:00.0 – 00:01.5] Hello there.\n" in d.timeline
    assert "  [00:02.0 – 00:03.2] Click save now.\n" in d.timeline

    # Cached separately from the plain digest, and reused on the next call.
    assert Digest.cached(tone_video, cache_dir=tmp_path, transcript=True).path == d.path
    assert Digest.cached(tone_video, cache_dir=tmp_path) is None
    assert Digest.build(tone_video, cache_dir=tmp_path, transcript=True).path == d.path
    assert len(fake_whisper.calls) == 1
    # Another model is another digest.
    other = Digest.build(tone_video, cache_dir=tmp_path, transcript=True,
                         transcript_model="tiny")
    assert other.path != d.path and other.transcript["model"] == "tiny"


def test_model_from_environment(tone_video, tmp_path, fake_whisper, monkeypatch):
    monkeypatch.setenv("VIDEO_DIGEST_WHISPER_MODEL", "tiny.en")
    d = Digest.build(tone_video, cache_dir=tmp_path, transcript=True)
    assert fake_whisper.loaded == ["tiny.en"] and d.transcript["model"] == "tiny.en"


def test_no_audio_track_loads_no_model(video, tmp_path, fake_whisper):
    d = Digest.build(video, cache_dir=tmp_path, transcript=True)
    assert fake_whisper.loaded == []
    assert d.transcript == {"model": "base", "language": None, "has_audio": False,
                            "segments": []}
    assert "Transcript: none (the video has no audio track)." in d.timeline


def test_flag_off_needs_no_extra_and_loads_no_model(tone_video, tmp_path, monkeypatch):
    _no_extra(monkeypatch)
    monkeypatch.setattr(transcript, "_load_model", lambda name: 1 / 0)
    d = Digest.build(tone_video, cache_dir=tmp_path)
    assert d.transcript is None
    assert "Transcript: not available" in d.timeline


def test_flag_on_without_extra_fails_before_building(tone_video, tmp_path, monkeypatch):
    _no_extra(monkeypatch)
    with pytest.raises(transcript.TranscriptUnavailable, match=r"video-digest\[transcript\]"):
        Digest.build(tone_video, cache_dir=tmp_path, transcript=True)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_cli_transcript(tone_video, tmp_path, fake_whisper, capsys):
    argv = ["build", str(tone_video), "--cache-dir", str(tmp_path), "--transcript"]
    assert cli.main(argv) == 0
    assert "Transcript: 2 segment(s), whisper base" in capsys.readouterr().out
    assert cli.main(argv + ["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cached"] and out["transcript"]["segments"][1]["text"] == "Click save now."
    assert cli.main(argv + ["--timeline"]) == 0
    assert "[00:00.0 – 00:01.5] Hello there." in capsys.readouterr().out


def test_cli_transcript_errors(tone_video, tmp_path, monkeypatch, capsys):
    base = ["build", str(tone_video), "--cache-dir", str(tmp_path)]
    assert cli.main(base + ["--transcript-model", "tiny"]) == cli.EXIT_USAGE
    assert "--transcript-model needs --transcript" in capsys.readouterr().err
    _no_extra(monkeypatch)
    assert cli.main(base + ["--transcript"]) == cli.EXIT_USAGE
    assert "pip install 'video-digest[transcript]'" in capsys.readouterr().err


# ---------------------------------------------------------------- live
def _say(text: str, path) -> bool:
    """Synthesize speech with macOS `say` or espeak-ng; False if neither exists."""
    if shutil.which("say"):
        # An English voice: the system default may speak another language.
        voice = ["-v", "Samantha"] if "Samantha" in subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True).stdout else []
        return subprocess.run(["say", *voice, "-o", str(path), text]).returncode == 0
    if shutil.which("espeak-ng"):
        return subprocess.run(["espeak-ng", "-w", str(path), text]).returncode == 0
    return False


@pytest.mark.live_stt
@pytest.mark.skipif(os.environ.get("VIDEO_DIGEST_LIVE_STT") != "1",
                    reason="set VIDEO_DIGEST_LIVE_STT=1 to run a real Whisper model")
def test_live_transcript_of_synthesized_speech(tmp_path):
    pytest.importorskip("faster_whisper")
    speech = tmp_path / ("speech.aiff" if shutil.which("say") else "speech.wav")
    if not _say("The save button stopped working after the update.", speech):
        pytest.skip("no speech synthesizer (say or espeak-ng)")
    clip = tmp_path / "talk.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=gray:s={W}x{H}:r={FPS}", "-i", str(speech),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip)],
        check=True)
    d = Digest.build(clip, cache_dir=tmp_path / "cache", transcript=True,
                     transcript_model=os.environ.get("VIDEO_DIGEST_WHISPER_MODEL"))
    text = " ".join(s["text"] for s in d.transcript["segments"]).lower()
    assert "save button" in text
    assert d.transcript["segments"][0]["start"] < 1.0
    assert "] The save button" in d.timeline or "] the save button" in d.timeline.lower()
