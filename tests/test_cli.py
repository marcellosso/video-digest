"""The video-digest CLI: argument parsing, errors and exit codes, and both commands end to end."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from conftest import H, W, needs_ffmpeg
from video_digest import __version__, cli


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_DIGEST_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.mark.parametrize("text, seconds", [
    ("12.5", 12.5), ("0", 0.0), ("1:02.5", 62.5), ("01:00:00", 3600.0), ("90", 90.0)])
def test_timestamp_parsing(text, seconds):
    assert cli.timestamp(text) == seconds


@pytest.mark.parametrize("text", ["abc", "", "-1", "1:60", "1:2:3:4", "1::2"])
def test_timestamp_rejects_garbage(text):
    with pytest.raises(Exception):
        cli.timestamp(text)


def test_usage_errors_exit_2(capsys):
    for argv in ([], ["nope"], ["build"], ["frames", "v.mp4"], ["frames", "v.mp4", "x"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2, argv
    assert "invalid timestamp 'x'" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"video-digest {__version__}"


def test_missing_video_exits_2(tmp_path, capsys):
    missing = str(tmp_path / "missing.mp4")
    assert cli.main(["build", missing]) == cli.EXIT_USAGE
    assert cli.main(["frames", missing, "1"]) == cli.EXIT_USAGE
    assert capsys.readouterr().err.count(f"video not found: {missing}") == 2


def test_directory_is_not_a_video(tmp_path, capsys):
    assert cli.main(["build", str(tmp_path)]) == cli.EXIT_USAGE
    assert "not a file" in capsys.readouterr().err


def test_missing_ffmpeg_exits_3(tmp_path, monkeypatch, capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(cli.shutil, "which", lambda tool: None)
    assert cli.main(["build", str(video)]) == cli.EXIT_NO_FFMPEG
    assert cli.main(["frames", str(video), "1"]) == cli.EXIT_NO_FFMPEG
    assert "ffmpeg and ffprobe not found on PATH" in capsys.readouterr().err


@needs_ffmpeg
def test_not_a_video_exits_1(tmp_path, cache, capsys):
    video = tmp_path / "junk.mp4"
    video.write_text("not a video")
    assert cli.main(["build", str(video)]) == cli.EXIT_FAILED
    assert cli.main(["frames", str(video), "1"]) == cli.EXIT_FAILED
    assert "ffprobe failed" in capsys.readouterr().err


@pytest.mark.parametrize("knob, msg", [("bogus=1", "unknown setting 'bogus'"),
                                       ("threshold", "expects KNOB=VALUE"),
                                       ("threshold=high", "not a number")])
def test_bad_settings_exit_2(knob, msg, video, cache, capsys):
    assert cli.main(["build", str(video), "--set", knob]) == cli.EXIT_USAGE
    assert msg in capsys.readouterr().err
    assert not cache.exists()  # rejected before any work


@needs_ffmpeg
def test_build_prints_path_and_timeline_then_reuses_cache(video, cache, capsys):
    assert cli.main(["build", str(video)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "(built in" in out[0]
    folder = cache / out[0].split()[0].rsplit("/", 1)[-1]
    assert (folder / "digest.json").is_file()
    assert out[1].startswith(f"synthetic.mp4: 00:09.0, {W}x{H}, ")
    assert out[2].startswith("  #1   00:00.0 – ")
    assert out[-2].startswith("Timeline: ") and out[-2].endswith("digest.md")
    assert out[-1].startswith("Sheet:    ") and out[-1].endswith("sheet_01.jpg")

    assert cli.main(["build", str(video)]) == 0
    assert "(reused from cache)" in capsys.readouterr().out
    assert cli.main(["build", str(video), "--force"]) == 0
    assert "(built in" in capsys.readouterr().out


@needs_ffmpeg
def test_build_json_and_settings(video, tmp_path, capsys):
    argv = ["build", str(video), "--json", "--cache-dir", str(tmp_path / "c")]
    assert cli.main(argv) == 0
    default = json.loads(capsys.readouterr().out)
    assert default["cached"] is False
    assert (default["width"], default["height"]) == (W, H)
    assert default["keyframes"][0] == 0.0
    assert all(p.endswith(".jpg") for p in default["sheets"])
    assert open(default["timeline"]).read().startswith("VIDEO DIGEST")

    assert cli.main(argv + ["--set", "threshold=5.0", "--set", "settle=false"]) == 0
    other = json.loads(capsys.readouterr().out)
    assert other["path"] != default["path"]
    data = json.loads(open(other["path"] + "/digest.json").read())
    assert data["settings"]["threshold"] == 5.0 and data["settings"]["settle"] is False


@needs_ffmpeg
def test_build_timeline_flag(video, cache, capsys):
    assert cli.main(["build", str(video), "--timeline"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("VIDEO DIGEST — 00:09.0 long") and "Keyframe timeline:" in out


@needs_ffmpeg
def test_frames_single_is_full_resolution(video, tmp_path, capsys):
    assert cli.main(["frames", str(video), "4.5", "-o", str(tmp_path)]) == 0
    label, path = capsys.readouterr().out.splitlines()
    assert label == "Frame at 00:04.5 (4.5s)"
    assert Image.open(path).size == (W, H)


@needs_ffmpeg
def test_frames_sheet_json(video, tmp_path, capsys):
    argv = ["frames", str(video), "0:07", "1", "4.5", "--out", str(tmp_path), "--json"]
    assert cli.main(argv) == 0
    z = json.loads(capsys.readouterr().out)
    assert z["timestamps"] == [1.0, 4.5, 7.0]  # sorted
    assert z["label"].startswith("Frames at 00:01.0 (1s), 00:04.5 (4.5s)")
    assert Image.open(z["path"]).size == (z["width"], z["height"])


@needs_ffmpeg
def test_frames_too_many_exits_2(video, tmp_path, capsys):
    ts = [str(t) for t in range(10)]
    assert cli.main(["frames", str(video), *ts, "-o", str(tmp_path)]) == cli.EXIT_USAGE
    assert "at most 9 timestamps" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@needs_ffmpeg
def test_python_dash_m_entry_point(video, tmp_path):
    proc = subprocess.run([sys.executable, "-m", "video_digest", "frames", str(video), "2",
                           "--json"], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    path = json.loads(proc.stdout)["path"]
    assert Path(path).resolve() == (tmp_path / "at_2.00.jpg").resolve()
