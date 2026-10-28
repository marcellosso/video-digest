## What changed


## Quality gate

Does this change which frames end up in a digest (`select.py`, `sample.py`, `sheets.py`, default settings, timeline text)? If yes, read [bench/README.md](../bench/README.md#quality-gate) and fill this in. If no, delete this section.

- [ ] `python -m bench.qa_bench synthetic --against <main results.json>` passes
- [ ] `python -m bench.qa_bench showreel --live --model claude-sonnet-5-5 --against bench/fixtures/showreel/reference-results.json` passes (paste the Summary + Quality gate tables):

```
```
