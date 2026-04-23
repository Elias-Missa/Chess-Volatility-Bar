"""Pure helpers that convert volatility results into JSON-serializable dicts.

Kept separate from :mod:`chess_vol.cli` so the schema is directly unit-testable
without instantiating the Typer app or an engine.

See README §7 for the CLI specification; the dict produced here is what
``chess-vol ... --output report.json`` writes to disk.
"""

from __future__ import annotations

from typing import TypedDict

from chess_vol.analyze import PlyResult
from chess_vol.config import color_for
from chess_vol.volatility import TopLine, VolatilityResult


class TopLineJson(TypedDict):
    """JSON shape for a single engine line (one MultiPV entry)."""

    uci: str
    san: str
    pv_san: list[str]
    eval_cp: int


class VolatilityJson(TypedDict):
    """JSON shape for a single :class:`VolatilityResult`."""

    score: float | None
    raw_cp: float | None
    local_raw_cp: float | None
    best_eval_cp: int
    alt_evals_cp: list[int]
    scale: float
    decided: bool
    reason: str | None
    recurse_depth_used: int
    analyses: int
    color: str | None
    top_lines: list[TopLineJson]


class PlyJson(TypedDict):
    """JSON shape for a single :class:`PlyResult`."""

    ply: int
    san: str
    fen_before: str
    fen_after: str
    eval_cp: int
    volatility: VolatilityJson


class ParamsJson(TypedDict, total=False):
    """Parameters echoed back into the report for reproducibility."""

    depth: int
    multipv: int
    recurse_depth: int
    recurse_k: int
    recurse_alpha: float
    child_depth: int
    max_plies: int | None


class AnalyzeReportJson(TypedDict):
    """Top-level JSON for ``chess-vol analyze``."""

    mode: str
    params: ParamsJson
    plies: list[PlyJson]


class FenReportJson(TypedDict):
    """Top-level JSON for ``chess-vol fen``."""

    mode: str
    fen: str
    params: ParamsJson
    volatility: VolatilityJson


def mode_label(recurse_depth: int) -> str:
    """Return ``"shallow"`` for ``recurse_depth == 0`` else ``"deep"``."""
    return "shallow" if recurse_depth == 0 else "deep"


def _top_line_to_json(line: TopLine) -> TopLineJson:
    return TopLineJson(
        uci=line.uci,
        san=line.san,
        pv_san=list(line.pv_san),
        eval_cp=line.eval_cp,
    )


def volatility_to_json(result: VolatilityResult) -> VolatilityJson:
    """Convert a :class:`VolatilityResult` to a JSON-serializable dict."""
    color = color_for(result.score) if result.score is not None else None
    return VolatilityJson(
        score=result.score,
        raw_cp=result.raw_cp,
        local_raw_cp=result.local_raw_cp,
        best_eval_cp=result.best_eval_cp,
        alt_evals_cp=list(result.alt_evals_cp),
        scale=result.scale,
        decided=result.decided,
        reason=result.reason,
        recurse_depth_used=result.recurse_depth_used,
        analyses=result.analyses,
        color=color,
        top_lines=[_top_line_to_json(line) for line in result.top_lines],
    )


def ply_to_json(ply: PlyResult) -> PlyJson:
    """Convert a :class:`PlyResult` to a JSON-serializable dict."""
    return PlyJson(
        ply=ply.ply,
        san=ply.san,
        fen_before=ply.fen_before,
        fen_after=ply.fen_after,
        eval_cp=ply.eval_cp,
        volatility=volatility_to_json(ply.volatility),
    )


def build_params(
    *,
    depth: int,
    multipv: int,
    recurse_depth: int,
    recurse_k: int,
    recurse_alpha: float,
    child_depth: int,
    max_plies: int | None = None,
) -> ParamsJson:
    """Assemble a ``params`` dict for the JSON report."""
    params: ParamsJson = {
        "depth": depth,
        "multipv": multipv,
        "recurse_depth": recurse_depth,
        "recurse_k": recurse_k,
        "recurse_alpha": recurse_alpha,
        "child_depth": child_depth,
    }
    if max_plies is not None:
        params["max_plies"] = max_plies
    return params


def build_analyze_report(
    plies: list[PlyResult],
    *,
    params: ParamsJson,
) -> AnalyzeReportJson:
    """Build the top-level report for ``chess-vol analyze``."""
    return AnalyzeReportJson(
        mode=mode_label(params.get("recurse_depth", 0)),
        params=params,
        plies=[ply_to_json(p) for p in plies],
    )


def build_fen_report(
    fen: str,
    result: VolatilityResult,
    *,
    params: ParamsJson,
) -> FenReportJson:
    """Build the top-level report for ``chess-vol fen``."""
    return FenReportJson(
        mode=mode_label(params.get("recurse_depth", 0)),
        fen=fen,
        params=params,
        volatility=volatility_to_json(result),
    )


__all__: list[str] = [
    "AnalyzeReportJson",
    "FenReportJson",
    "ParamsJson",
    "PlyJson",
    "TopLineJson",
    "VolatilityJson",
    "build_analyze_report",
    "build_fen_report",
    "build_params",
    "mode_label",
    "ply_to_json",
    "volatility_to_json",
]
