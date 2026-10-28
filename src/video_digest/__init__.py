"""video-digest: turn a video into a compact digest an AI model can read cheaply."""

__version__ = "0.1.0.dev0"

from .digest import Digest, Frames, get_frames  # noqa: E402
from .select import SelectSettings  # noqa: E402

__all__ = ["Digest", "Frames", "SelectSettings", "get_frames", "__version__"]
