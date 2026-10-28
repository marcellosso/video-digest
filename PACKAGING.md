# Packaging and publishing

Nothing is published yet. The first PyPI release needs an explicit go from the maintainer, including the final package name.

## Package name (open)

Checked 2026-10-28 against `https://pypi.org/simple/<name>/`:

- **`video-digest` can't be registered as is.** It's free as a name, but PyPI rejects new names that are "too similar" to an existing project: it compares names after removing `-`, `_` and `.` and mapping `l`/`i` to `1` and `o` to `0` (`ultranormalize_name` in Warehouse). [`videodigest`](https://pypi.org/project/videodigest/) (a 2015 subtitle summarizer, one release, 0.0.1) collides with it.
- Options:
  1. Ask for `videodigest` under [PEP 541](https://peps.python.org/pep-0541/) (abandoned project). Slow and not guaranteed.
  2. Publish under another distribution name and keep the `video-digest` command and `video_digest` import. Users then run `uvx --from <name> video-digest`, unless the name matches a command.
  3. Rename the project (the issue lists this as an open question).
- Free on the same check: `vdigest`, `framedigest`, `keyframe-digest`, `video-digester`, `screendigest`, `digest-video`, `video-digest-ai`. Taken: `videodigest`, `clip-digest` (so `clipdigest` too). Only the PyPI upload can confirm a name, because the similarity check runs against every project, including ones that have no files.

## Version

`src/video_digest/__init__.py` holds the version (`pyproject.toml` reads it from there). It stays `0.1.0.dev0` until the first release. Bump it to `0.1.0` in the release PR. The plugin manifest (`plugin/.claude-plugin/plugin.json`) has its own version.

## Release checklist

1. Decide the name. If it changes, update `name` in `pyproject.toml`, the install lines in `README.md`, and the error message in `src/video_digest/mcp_server.py`.
2. Bump the version, then run `pytest` and `python -m bench.qa_bench synthetic`.
3. `rm -rf dist && uv build`. Check the wheel contains only `video_digest/` and that the README renders (`uvx twine check dist/*`).
4. Smoke-test the built wheel in a fresh venv with no clone on the path:
   ```bash
   uv venv /tmp/vd && uv pip install --python /tmp/vd/bin/python "$(ls dist/*.whl)[mcp]"
   /tmp/vd/bin/video-digest --version && /tmp/vd/bin/video-digest build some.mp4 --cache-dir /tmp/vd-cache
   ```
5. Optional dry run on TestPyPI: `uv publish --publish-url https://test.pypi.org/legacy/`.
6. Publish with `uv publish` (a trusted publisher on GitHub Actions is preferable to an API token), tag `v0.1.0` and create a GitHub release.
7. In `README.md`, switch the install section from the clone and `git+` lines to `uvx <name>` / `pip install <name>`.
