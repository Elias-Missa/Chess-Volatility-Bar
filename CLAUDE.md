# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (choose extras as needed)
pip install -e .          # base library + CLI
pip install -e .[dev]     # + pytest, mypy, ruff, black, rich, fastapi, httpx
pip install -e .[cli]     # + rich (colored terminal output)
pip install -e .[web]     # + fastapi, uvicorn, sse-starlette

# Tests
pytest                                          # all unit tests
pytest tests/test_volatility.py                 # single file
pytest tests/test_volatility.py::test_name      # single test
pytest -m integration                           # integration tests (require Stockfish)

# Linting / formatting / type-checking
ruff check src/ tests/
black src/ tests/
mypy src/chess_vol/      # must pass --strict (configured in pyproject.toml)

# CLI
chess-vol analyze tests/fixtures/sample_game.pgn --max-plies 10
chess-vol analyze game.pgn --deep --output report.json
chess-vol fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
chess-vol serve                       # http://127.0.0.1:8000/
chess-vol serve --host 0.0.0.0 --port 9000 --reload

# Stockfish (required for integration tests and real use)
export STOCKFISH_PATH=/path/to/stockfish.exe   # Windows override
```

Integration tests are skipped automatically when Stockfish is unavailable; they are marked `@pytest.mark.integration` and gated by `requires_stockfish` from `tests/conftest.py`.

## Architecture

### Module map

| Module | Role |
|---|---|
| `config.py` | All algorithm constants (`K`, `DROP_CAP`, `EVAL_SCALE_*`, `MATE_*`, color thresholds). Always change constants here, never inline. |
| `engine.py` | `Engine` context manager wrapping Stockfish. Path resolution: explicit arg → `$STOCKFISH_PATH` → `shutil.which` → known install paths. Raises `EngineNotFoundError` with install instructions. |
| `volatility.py` | Core algorithm. Public: `compute_volatility(board, engine, ...) → VolatilityResult`. Internal: `_compute_raw()` does recursion; normalization to 0-100 happens **once** at the root. |
| `analyze.py` | `analyze_pgn(pgn, engine, ...) → list[PlyResult]`. Iterates PGN moves, calls `compute_volatility` on each pre-move position. |
| `cli.py` | Typer app with `analyze`, `fen`, `serve` commands. `ENGINE_FACTORY` is a module-level callable that tests monkey-patch to inject `FakeEngine`. |
| `cli_report.py` | JSON serialization (`build_analyze_report`, `build_fen_report`, `volatility_to_json`). |
| `server.py` | FastAPI server. `POST /analyze/fen`, `POST /analyze/pgn` (SSE stream), `GET /`. CORS allows localhost + chess.com. |
| `web/` | Vendored frontend (chessboard.js, chess.js, Chart.js). No CDN, no build step. |

### Volatility algorithm (one-ply)

1. Run Stockfish with `MultiPV=6` to get the top-N eval lines.
2. Convert all evals to centipawns from **side-to-move POV** (mates via `mate_to_cp`: M1=1950, each step −50, floor at M20=1000).
3. Sort descending; compute drops from best: `drop_i = e1 - e_i`, capped at `DROP_CAP=2000`.
4. Weighted sum with weights `[1, 0.5, 0.333, 0.25, 0.2]` / their sum (2.283).
5. Eval-aware scale: `scale = 1 / (1 + ((min(|e1|,|e2|,2000) - 200) / 300)²)` — dampens V when both best and backup are already decisive wins.
6. Normalize: `V = 100 * (1 - exp(-scale * weighted_sum / K))`.

**Recursive mode** (`recurse_depth > 0`): after computing `V_local`, recurse into the top-k candidate positions (reusing the same MultiPV result — no extra engine call) and blend: `V_total = V_local + alpha * mean(child_Vs)`. Normalize only at the root. The side-to-move flips at each child; `info_to_cp` always uses `board.turn` at the current recursion level.

**`decided` flag**: set when `|e1|>800 AND sign(e1)==sign(e2) AND |e2|>400`. UI dims the bar.

**Terminal / edge cases**: checkmate → `score=None, reason="checkmate"`; stalemate → `reason="stalemate"`; only legal move → `reason="only_move"`. These propagate up only when they are the *root* result; during recursion they contribute `0` to the child sum.

### Testing patterns

Unit tests inject `FakeEngine` from `conftest.py`:
```python
from tests.conftest import FakeEngine, evals_to_infos

engine = FakeEngine(scripts=[
    evals_to_infos([80, 30, 10, -5, -20, -40]),  # root call
])
result = compute_volatility(board, engine)
assert engine.call_count == 1
```

`evals_to_infos` accepts `int` (cp) or `"M3"` / `"-M3"` (mate) entries, always from side-to-move POV.

For recursive budget tests, assert `engine.call_count == 1 + k + k²` (e.g., 13 for `recurse_depth=2, recurse_k=3`).

### Build phases

The README tracks four phases. Phase 3 (web app) is marked complete. The phases must be built in order and each phase's tests must pass before the next begins. The `recurse_depth=0` default preserves exact Phase 1 behavior — never break this regression.
