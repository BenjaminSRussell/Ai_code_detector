# AI Code Detector — Agent-Prep Scanner Design

Date: 2026-08-02
Status: Approved, Phase 1 in progress

## Context

`Ai_code_detector` currently ships as a CLI (`src/cli.py` / `src/cli_enhanced.py`) that scores a repo
for AI-generated-code likelihood using 34 stylometric/structural/history heuristics, plus an
optional (currently untrained) ML classifier and templated/LLM explanations.

Goal: turn this into a tool the user actually runs day-to-day, structured as a **pre-agent scan** —
something that runs on a codebase *before* handing it to a coding agent (e.g. Claude Code), producing
findings the agent can read and act on directly.

## Decisions made

- Private target-repo scanning is explicitly out of scope — not needed.
- Local codebases must be scannable (already true: `GitLoader._is_url()` treats any non-URL source
  as a local path).
- The originally-designed GitHub Actions `workflow_dispatch` on-demand scan (see "Phase 4" below)
  is deferred, not dropped — the local dashboard is the priority.
- The "executable" the user wants is a standalone packaged app (PyInstaller-style), not just a
  launcher script — but that's a late phase; it wraps a working local app, it doesn't come first.
- Hotspot ("slowest functions") detection should include both a static heuristic pass and real
  profiling, gated appropriately (see Phase 1 below).
- Findings should be written both as human-readable Markdown and machine-readable JSON.
- Scope is large enough to need phasing. Build order is backend-out: detection engine first, then
  UI, then packaging, then the CI workflow.

## Research summary (informed Phase 1's feature list)

- GitClear's 211M-line study (2020–2024): copy/paste-style duplication overtook refactored code for
  the first time in 2024; duplicated blocks up ~8x. → repo-wide duplication is a real, growing gap in
  the current stylometry features (which only compare *within* a file).
  [gitclear.com](https://www.gitclear.com/ai_assistant_code_quality_2025_research)
- AI coding tools (Claude Code, Copilot, Cursor) commonly leave literal attribution trailers in commit
  messages (`Co-Authored-By: Claude`, "Generated with Claude Code", etc.) — a near-certain signal the
  detector doesn't check for, despite already parsing every commit message.
  [explainx.ai](https://explainx.ai/blog/claude-code-commit-co-author-attribution-disable-guide-2026),
  [coderbuds.com](https://coderbuds.com/blog/open-source-ai-code-detection-yaml-rules)
- Self-admitted technical debt (SATD: `TODO`/`FIXME`/`HACK`) research found AI-era SATD takes ~44x
  longer to get addressed (10 days → 441 days median) — not currently scanned for at all.
  [arxiv.org/2601.06266](https://arxiv.org/html/2601.06266v1)
- A 2026 taxonomy of LLM code inefficiencies found AI-generated code runs 2.6–3.4x slower on average
  (worst case ~68x), concentrated in sub-optimal time complexity (18.5% of issues) and redundant
  steps (5.3%) — directly motivates the "slowest functions" hotspot feature the user asked for.
  [arxiv.org/2503.06327](https://arxiv.org/html/2503.06327v3)
- Detectors generally remain evadable via light refactoring, and current tools show real accuracy
  gaps — reinforces the existing README caveat that this stays "one signal among many," never sole
  evidence.

## Phase breakdown

1. **Detection engine additions + agent-ready findings docs** (this spec's focus) — CLI-callable,
   no UI.
2. **Local dashboard** — folder picker, run-scan button, results view, built on Phase 1's engine.
3. **Standalone packaged executable** (PyInstaller) — wraps Phase 2.
4. **GitHub Actions on-demand workflow** — `workflow_dispatch`, target-repo-URL input, `basic`/
   `enhanced` mode choice, results rendered in the Actions run summary, run always exits green
   (report tool, not a gate). Independent of 1–3; lower priority.

## Phase 1 design

### New detection signals (`src/analysis/`)

**Repo-wide duplication** (new, extends the duplication story beyond `metrics_stylometry.py`'s
existing intra-file-only `code_duplication_score`/`intra_file_similarity`):
- Hash n-gram blocks (line-based, same 3-line-gram approach already used intra-file) across every
  analyzed file in the repo.
- Flag blocks that recur in 2+ distinct files; report the file pairs/locations and a repo-level
  duplication ratio.

**AI-attribution commit trailers** (new feature on `HistoryAnalyzer` /
`metrics_history.py`, reusing commit messages already being read):
- Maintain a pattern list of known AI-tool commit markers (`Co-Authored-By: Claude`, `Generated with
  Claude Code`, Copilot/Cursor equivalents), extensible without code changes (config-driven list).
- When matched, this is treated as **direct evidence**, not folded anonymously into the weighted
  heuristic score — surfaced separately in output with high confidence, since it's near-certain
  rather than probabilistic.

**SATD marker scan** (new, lightweight):
- Scan comments for `TODO`/`FIXME`/`HACK`/`XXX` density per file.
- v1 scope: count/density only. Git-blame-based aging (how long a marker has survived) is a
  reasonable v2 addition — explicitly deferred to control scope.

**Performance hotspots** (new module, `src/analysis/metrics_performance.py`):
- *Static pass, always runs*: AST-based heuristics — nested loops, unmemoized recursion, expensive
  operations (I/O, string concatenation, linear scans) inside loops. Produces a per-function risk
  score; surfaces the riskiest functions repo-wide.
- *Dynamic pass, opt-in only*: when a runnable entry point or test suite is detected, optionally run
  it under `cProfile` for real per-function timings, merged with the static findings.
  - **Safety requirement**: this executes code from the scanned repo. Must never run automatically —
    Phase 1 exposes it as an explicit CLI flag (e.g. `--profile`); Phase 2's dashboard must gate it
    behind an explicit user action with a visible warning, never a default/automatic step.

### Findings output

Written to the **scanned repo's root** by default (the point is for a coding agent operating in that
repo to find it there on a later invocation):

- `AI_SCAN_FINDINGS.md` — prioritized, prose findings: AI-likelihood verdict with reasons, any direct
  AI-attribution-trailer hits called out first (highest confidence), duplication clusters, SATD
  hotspots, top performance hotspots with the specific pattern and line flagged.
- `ai_scan_findings.json` — the same findings as a structured task queue:
  `{type, file, line, function, severity, description, evidence}` per entry, for an agent or script
  to consume programmatically.

### Explicitly out of scope for Phase 1

Dashboard UI, executable packaging, the GitHub Actions workflow, git-blame-based SATD aging,
security-vulnerability pattern detection, cross-repo (as opposed to cross-file-within-one-repo)
duplication, and any changes to the existing (untrained) ML classifier.

### Testing

New analyzers follow the existing test layout (`tests/test_basic.py`-style unit tests per module,
plus fixtures under `examples/` such as `sample_ai_code.py`/`sample_human_code.py`); each new signal
needs at least one positive and one negative case.
