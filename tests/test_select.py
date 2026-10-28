"""Frame selection on hand-made thumbnails (no ffmpeg needed)."""
import numpy as np

from video_digest.select import SelectSettings, change_score, select_keyframes

FPS = 4.0


def solid(color):
    return np.full((72, 128, 3), color, np.uint8)


def with_box(base, x0, x1, color=(255, 255, 255)):
    a = base.copy()
    a[24:48, x0:x1] = color
    return a


def noisy(base, seed):
    """Base frame with a scrambled patch, different every call (mid-animation)."""
    a = base.copy()
    a[24:48, 32:96] = np.random.default_rng(seed).integers(0, 256, (24, 64, 3), dtype=np.uint8)
    return a


def samples_of(frames):
    return [(i / FPS, f) for i, f in enumerate(frames)]


def test_change_score_block_and_color():
    a = solid((40, 40, 40))
    assert change_score(a, a) == 0
    # a hue shift at roughly constant brightness still registers
    assert change_score(solid((120, 40, 40)), solid((40, 40, 120))) == 100
    # a small object registers even though the frame average barely moves
    assert 0 < change_score(a, with_box(a, 0, 8)) < 5


def test_static_video_collapses_to_one_keyframe():
    kfs = select_keyframes(samples_of([solid((10, 20, 30))] * 40), duration=10.0)
    assert len(kfs) == 1
    assert kfs[0]["t"] == 0 and kfs[0]["change"] is None and kfs[0]["until"] == 10.0


def test_settles_scrambled_text_to_final_state():
    blue, green = solid((30, 40, 70)), solid((20, 110, 50))
    final = with_box(green, 32, 96)
    frames = [blue] * 12 + [noisy(green, s) for s in range(3)] + [final] * 13
    kfs = select_keyframes(samples_of(frames), duration=7.0)
    second = kfs[1]
    assert second["settled_from"] == 3.0          # caught at the cut, mid-scramble
    assert second["t"] == 3.75                     # moved to the first still frame
    assert not any(3.0 <= k["t"] < 3.75 for k in kfs)


def test_settling_never_crosses_a_scene_cut():
    blue, green, red = solid((30, 40, 70)), solid((20, 110, 50)), solid((160, 30, 30))
    # scene B animates right up to the cut into scene C
    frames = [blue] * 8 + [noisy(green, s) for s in range(4)] + [red] * 12
    kfs = select_keyframes(samples_of(frames), duration=6.0)
    cut = 3.0
    scene_b = [k for k in kfs if 2.0 <= k.get("settled_from", k["t"]) < cut]
    assert scene_b, "scene B must keep a keyframe"
    assert all(k["t"] < cut for k in scene_b)


def test_keeps_scene_final_state():
    blue, green, red = solid((30, 40, 70)), solid((20, 110, 50)), solid((160, 30, 30))
    built = with_box(green, 16, 112)  # the scene's fullest state, reached right before the cut
    frames = [blue] * 8 + [green] * 2 + [built] * 4 + [red] * 12
    kfs = select_keyframes(samples_of(frames), duration=6.5,
                           s=SelectSettings(settle=False))
    ends = [k for k in kfs if k.get("scene_end")]
    assert [k["t"] for k in ends] == [3.25]


def test_until_chains_keyframes():
    frames = [solid((0, 0, 0))] * 10 + [solid((200, 200, 200))] * 10
    kfs = select_keyframes(samples_of(frames), duration=5.0)
    for a, b in zip(kfs, kfs[1:]):
        assert a["until"] == b["t"]
    assert kfs[-1]["until"] == 5.0
