"""The MCP server's two tools, called in-process (no MCP client needed)."""
import asyncio
import base64
from io import BytesIO

import pytest
from PIL import Image

pytest.importorskip("mcp.server.mcpserver")

from conftest import H, W  # noqa: E402
from video_digest import digest as digest_mod  # noqa: E402
from video_digest import mcp_server  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402


def call(name, **arguments):
    """Call a tool through the server, as a client would, and return its content blocks."""
    result = asyncio.run(mcp_server.server.call_tool(name, arguments))
    assert not result.is_error, result.content
    return result.content


def decode(block) -> Image.Image:
    assert block.type == "image" and block.mime_type == "image/jpeg"
    return Image.open(BytesIO(base64.b64decode(block.data)))


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "cache")


def test_lists_both_tools():
    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {"digest_video", "get_frames"}


def test_digest_video_returns_timeline_and_sheets(video, cache):
    content = call("digest_video", path=str(video), cache_dir=cache)
    meta, timeline = content[0].text, content[1].text
    assert "built in" in meta and "keyframes in 1 contact sheet(s)" in meta
    assert timeline.startswith("VIDEO DIGEST") and "Keyframe timeline:" in timeline
    images = [b for b in content if b.type == "image"]
    assert len(images) == 1
    assert decode(images[0]).size[0] > 0
    assert "Transcript: requested" not in meta


def test_transcript_flag_is_accepted_but_deferred(video, cache):
    content = call("digest_video", path=str(video), cache_dir=cache, transcript=True)
    assert "not implemented yet" in content[0].text


def test_get_frames_after_digest_does_not_rebuild(video, cache, monkeypatch):
    call("digest_video", path=str(video), cache_dir=cache)

    def no_build(*a, **k):
        raise AssertionError("digest was rebuilt")
    monkeypatch.setattr(digest_mod, "_build_into", no_build)

    content = call("digest_video", path=str(video), cache_dir=cache)
    assert "reused from cache" in content[0].text

    one = call("get_frames", path=str(video), timestamps=[4.5], cache_dir=cache)
    assert one[0].text.startswith("Frame at 00:04.5") and "frames/at_4.50.jpg" in one[0].text
    assert decode(one[1]).size == (W, H)

    many = call("get_frames", path=str(video), timestamps=[1, 4.5, 7], cache_dir=cache)
    assert many[0].text.startswith("Frames at 00:01.0 (1s), 00:04.5 (4.5s), 00:07.0 (7s)")
    assert decode(many[1]).size != (W, H)


def test_get_frames_without_digest_does_not_build(video, tmp_path, monkeypatch):
    monkeypatch.setattr(digest_mod, "_build_into", lambda *a, **k: 1 / 0)
    content = mcp_server.get_frames(str(video), [2.0], cache_dir=str(tmp_path / "empty"))
    assert content[0] == "Frame at 00:02.0 (2s)"
    assert not (tmp_path / "empty").exists()


@pytest.mark.parametrize("timestamps", [[], list(range(10))])
def test_get_frames_rejects_bad_timestamp_counts(video, timestamps):
    with pytest.raises(ToolError, match="timestamps"):
        call("get_frames", path=str(video), timestamps=timestamps)


def test_missing_video_is_a_tool_error(tmp_path):
    with pytest.raises(ToolError, match="video not found"):
        mcp_server.digest_video(str(tmp_path / "missing.mp4"))
    with pytest.raises(ToolError, match="video not found"):
        mcp_server.get_frames(str(tmp_path / "missing.mp4"), [1.0])


def test_not_a_video_is_a_tool_error(tmp_path, cache):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(ToolError):
        mcp_server.digest_video(str(junk), cache_dir=cache)
