"""Chess Volatility Bar — core library.

Public API:
    - :class:`Engine` — context-managed Stockfish wrapper.
    - :func:`compute_volatility` — one-ply (Phase 1) and recursive (Phase 1.5).
    - :class:`VolatilityResult` — return type of ``compute_volatility``.
    - :func:`analyze_pgn` — per-ply volatility analysis of a PGN.
    - :exc:`EngineNotFoundError` — raised when Stockfish cannot be located.
"""

from chess_vol.analyze import PlyResult, analyze_pgn
from chess_vol.engine import Engine, EngineNotFoundError
from chess_vol.volatility import VolatilityResult, compute_volatility, mate_to_cp

__all__ = [
    "Engine",
    "EngineNotFoundError",
    "PlyResult",
    "VolatilityResult",
    "analyze_pgn",
    "compute_volatility",
    "mate_to_cp",
]
