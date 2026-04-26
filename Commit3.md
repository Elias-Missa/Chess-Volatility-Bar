# Commit 3 — Explain panel, calibration scaffolding, move classification

Three new product capabilities, all fully local, no LLM, no extra
runtime dependencies.

## 1. Explain Vol Bar — "Why this volatility?"

Deterministic, template-based explainer that decomposes any
`VolatilityResult` into a one-sentence summary, named contributions,
and machine-readable pattern tags.

- `src/chess_vol/explain.py` — `Explanation` + `Component` dataclasses,
  11 pattern tags (`knife_edge`, `defensive_crisis`, `forgiving`,
  `reply_dominates`, `scale_dampened`, `decided`, `mate_available`,
  …), pattern detector, summary writer that pulls real move SANs from
  `top_lines` so users see moves, not numbers.
- `src/chess_vol/cli_report.py` — `VolatilityJson` now carries
  `explanation: { summary, components, patterns, headline_pattern }`
  on every result. Same schema flows through `/analyze/fen` and the
  per-ply SSE stream.
- `web/index.html`, `web/app.js`, `web/styles.css` — vol bar is now a
  clickable button with an "i" hint; new "Why this volatility?" panel
  in both side panels with pattern-coloured badges, stacked component
  bar, and an actionable "what to do" hint per pattern. Click the bar
  to flash the panel.
- 23 unit tests in `tests/test_explain.py`.

## 2. Move classification — chess.com-style, plus volatility tags

Standard primary labels (best / great / good / inaccuracy / mistake /
blunder / brilliant) **plus** secondary tags that nobody else has:
`practical`, `simplification`, `complication`, `defusal`,
`routine_miss`, `critical_miss` — built on the V × eval-drop axis.

- `src/chess_vol/classify.py` — `Classification` dataclass,
  `classify_move(prev, next)` rule engine, and a per-classification
  summary sentence.
- `src/chess_vol/analyze.py` — `PlyResult` extended with `move_uci`
  and optional `classification`; `analyze_pgn` does a second pass to
  attach classifications once neighbouring plies are known.
- `src/chess_vol/cli_report.py` — `PlyJson` extended with `move_uci`
  and `classification: ClassificationJson | None`.
- `src/chess_vol/server.py` — SSE `done` event now ships the full
  per-ply report so library/import flows can persist without re-
  analysing.
- Frontend — move list shows classification glyphs; new
  "Classifications" stat card per side.
- Unit tests in `tests/test_classify.py`.

## 3. Calibration scaffolding (Phase 2.5)

Tooling to tune the eight coupled volatility constants against a real
corpus, with a hard slow/fast split: Stockfish runs once per corpus,
then the optimiser iterates against cached engine output.

- `src/chess_vol/calibrate.py` — `Constants` (8-vector with
  `as_vector` / `from_vector`), `CachedAnalysis` / `CorpusEntry` /
  `RawScore` data classes, `recompute_v` (engine-free, matches
  `compute_volatility` for shallow mode), three loss functions
  (`expert_loss` MSE, `distributional_loss` smoothed-KL,
  `blended_loss`), `tune_constants` via scipy L-BFGS-B with sane
  bounds, and `build_report` for before/after diffs.
- `scripts/calibrate.py` — `dump` / `tune` / `report` subcommands.
  `tune` prints a paste-ready `config.py` patch.
- `tests/fixtures/calibration_corpus.json` — 20 starter labelled
  positions across `quiet` / `middlegame` / `sharp` / `tactical` /
  `endgame` categories.
- 36 unit tests in `tests/test_calibrate.py` including a synthetic
  recovery test that walks `K_SHALLOW` from default 150 back to a
  hidden target of 100 (validates the optimiser actually optimises).

## 4. Docs

- README §7 fully rewritten with the calibration workflow,
  mode-comparison table, recommended corpus shape, and tuning order.
- README §12 milestone checklist updated: Phase 2.5 and Explain panel
  marked complete.

## Health

- 202 tests pass (was 149 at session start)
- `mypy --strict` clean on `src/chess_vol/`
- `ruff` clean across `src/`, `tests/`, `scripts/`
- No new runtime dependencies — `scipy` is optional, only needed for
  `tune`; `dump` and `report` work without it
- No regressions: `recurse_depth=0` produces identical output to before
