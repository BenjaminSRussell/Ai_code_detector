# Fixture "PR" for the aicd gate (#12)

`tests/test_gate.py` commits these files on top of an empty base commit in a
temporary git repo, takes `git diff --name-only base...HEAD` exactly as
`.github/workflows/ai-scan.yml` does, and runs `aicd gate` on the result.

| File | Expected (basic mode, `--threshold 0.4`) |
|---|---|
| `generated_helper.py` | flagged (copy of `examples/sample_ai_code.py`) |
| `hand_written.py` | passes |
| `third_party/vendored.py` | suppressed by this directory's `.aicdignore` |
| `README.md` | skipped (not a source file) |

Reproduce by hand from the repo root:

```bash
aicd gate --root tests/fixtures/gate_pr generated_helper.py hand_written.py third_party/vendored.py README.md -t 0.4
```
