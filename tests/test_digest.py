"""End-to-end: build a digest of a short synthetic video and zoom into it."""
import base64
import json
from io import BytesIO

import pytest
from PIL import Image

from conftest import H, SCRAMBLE_END, W, needs_ffmpeg
from video_digest import Digest, get_frames

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def digest(video, tmp_path_factory):
    return Digest.build(video, cache_dir=tmp_path_factory.mktemp("cache"))


def test_digest_folder_layout(digest):
    assert (digest.path / "digest.json").is_file()
    assert (digest.path / "digest.md").is_file()
    data = json.loads((digest.path / "digest.json").read_text())
    assert data["format_version"] == 1
    assert (data["width"], data["height"]) == (W, H)
    assert data["duration"] == pytest.approx(9.0, abs=0.1)
    assert data["keyframes"] and data["sheets"]
    for rel in data["sheets"]:
        assert (digest.path / rel).is_file()
        assert rel.startswith("sheets/")  # relative, so the folder is portable
    fhash = data["file_sha256"]
    assert digest.path.name.startswith(fhash[:16] + "-")


def test_timeline_is_readable(digest):
    md = digest.timeline
    assert md.startswith("VIDEO DIGEST — 00:09.0 long, 320x180.")
    assert "Keyframe timeline:" in md
    assert "#1   00:00.0" in md
    assert "settled after animation" in md
    assert md.count("\n  #") == len(digest.keyframes)


def test_scrambled_text_is_settled_and_cuts_respected(digest):
    kfs = digest.keyframes
    # one keyframe per scene, each inside its own scene
    assert any(0 <= k["t"] < 3 for k in kfs)
    assert any(6 <= k["t"] < 9 for k in kfs)
    scene2 = [k for k in kfs if 3 <= k["t"] < 6]
    assert scene2, "the EVERYTHING scene must be kept"
    # no keyframe left mid-scramble, and the settled one did not leave its scene
    assert not any(3 <= k["t"] < SCRAMBLE_END for k in kfs)
    assert any(k.get("settled_from") == 3.0 for k in scene2)


def test_cache_reuse_and_settings_key(digest, video):
    again = Digest.build(video, cache_dir=digest.path.parent)
    assert again.path == digest.path
    assert again.data["build_seconds"] == digest.data["build_seconds"]
    other = Digest.build(video, cache_dir=digest.path.parent, threshold=5.0)
    assert other.path != digest.path
    assert other.path.name.split("-")[0] == digest.path.name.split("-")[0]
    assert other.settings.threshold == 5.0


def test_single_frame_is_full_resolution(digest):
    f = digest.frames(4.5)
    assert f.image.size == (W, H)
    assert f.path and f.path.is_file()
    # center of the EVERYTHING scene background is green
    r, g, b = f.image.getpixel((5, 5))
    assert g > r and g > b


def test_multiple_frames_make_one_labeled_sheet(digest):
    f = digest.frames([7.0, 1.0, 4.5])
    assert f.timestamps == [1.0, 4.5, 7.0]
    assert f.image.size == (3 * 512, 288)
    assert "00:01.0" in f.label and "00:07.0" in f.label


def test_frames_limits(digest):
    with pytest.raises(ValueError):
        digest.frames([float(i) for i in range(10)])
    with pytest.raises(ValueError):
        digest.frames([])
    # past the end clamps to the last frame instead of failing
    assert digest.frames(100).image.size == (W, H)


def test_get_frames_without_a_digest(video):
    f = get_frames(video, [0.5, 8.5])
    assert f.image.width == 2 * 512


def test_to_content_anthropic(digest):
    blocks = digest.to_content("anthropic")
    assert blocks[0] == {"type": "text", "text": digest.timeline}
    images = [b for b in blocks if b["type"] == "image"]
    assert len(images) == len(digest.sheets)
    img = Image.open(BytesIO(base64.b64decode(images[0]["source"]["data"])))
    assert img.format == "JPEG"
    zoom = digest.frames([1.0, 4.5]).to_content("anthropic")
    assert [b["type"] for b in zoom] == ["text", "image"]
    with pytest.raises(ValueError):
        digest.to_content("openai")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Digest.build(tmp_path / "nope.mp4", cache_dir=tmp_path)
