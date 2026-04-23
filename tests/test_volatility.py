"""Unit tests for the one-ply volatility algorithm (README §3.1 / §3.7).

All tests here use :class:`FakeEngine` so they run with no Stockfish binary.
"""

from __future__ import annotations

import math

import chess
import pytest

from chess_vol.config import (
    DROP_CAP,
    EVAL_SCALE_GRACE,
    EVAL_SCALE_MAX,
    EVAL_SCALE_WIDTH,
    K_SHALLOW,
)
from chess_vol.volatility import (
    _compute_local,
    _is_decided,
    compute_volatility,
    default_scale_fn,
    default_weights,
)

from .conftest import FakeEngine, evals_to_infos

# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


class TestDefaultWeights:
    def test_n6_matches_readme(self) -> None:
        assert default_weights(6) == pytest.approx([1.0, 0.5, 1 / 3, 0.25, 0.2])

    def test_sum_for_n6(self) -> None:
        assert sum(default_weights(6)) == pytest.approx(2.283, abs=1e-3)

    @pytest.mark.parametrize("n", [0, 1])
    def test_too_small(self, n: int) -> None:
        assert default_weights(n) == []


class TestDefaultScaleFn:
    """Verifies the §3.1 scale lookup table (canonical values)."""

    @pytest.mark.parametrize(
        "scale_eval,expected",
        [
            (0, 1.00),
            (100, 1.00),
            (200, 1.00),  # equal to grace → exactly 1.0
            (300, 0.90),
            (500, 0.50),
            (800, 0.20),
            (1000, 0.12),
            (1500, 0.047),
            (2000, 0.028),
        ],
    )
    def test_lookup_table(self, scale_eval: int, expected: float) -> None:
        value = default_scale_fn(scale_eval, scale_eval)
        assert value == pytest.approx(expected, abs=0.01)

    def test_uses_min_of_e1_e2(self) -> None:
        """README §3.1: `scale_eval = min(|e_1|, |e_2|, EVAL_SCALE_MAX)`.

        Best +400, backup -200 → min is 200 → scale = 1.0 (no dampening)."""
        assert default_scale_fn(400, -200) == pytest.approx(1.0, abs=1e-6)

    def test_caps_at_eval_scale_max(self) -> None:
        """Very large mate-like scores get clamped at EVAL_SCALE_MAX."""
        v = default_scale_fn(9999, 9999)
        expected = 1.0 / (1.0 + ((EVAL_SCALE_MAX - EVAL_SCALE_GRACE) / EVAL_SCALE_WIDTH) ** 2)
        assert v == pytest.approx(expected)

    def test_matches_formula(self) -> None:
        e = 700
        expected = 1.0 / (1.0 + ((e - EVAL_SCALE_GRACE) / EVAL_SCALE_WIDTH) ** 2)
        assert default_scale_fn(e, e) == pytest.approx(expected)


class TestDecided:
    """README §3.4."""

    def test_positive_decided(self) -> None:
        assert _is_decided(1500, 900) is True

    def test_negative_decided(self) -> None:
        assert _is_decided(-1500, -900) is True

    def test_opposite_signs_not_decided(self) -> None:
        """Knife's-edge: winning line but backup loses → not decided."""
        assert _is_decided(1500, -500) is False

    def test_best_below_threshold(self) -> None:
        assert _is_decided(700, 500) is False

    def test_alt_below_threshold(self) -> None:
        """README §3.7 'Mate available' row: e_1=1950, e_2=200 → not decided."""
        assert _is_decided(1950, 200) is False


# --------------------------------------------------------------------------- #
# _compute_local — direct formula verification                                 #
# --------------------------------------------------------------------------- #


def _expected_weighted_sum(evals: list[int]) -> float:
    e1 = evals[0]
    weights = default_weights(len(evals))
    drops = [min(e1 - e_i, DROP_CAP) for e_i in evals[1:]]
    return sum(w * d for w, d in zip(weights, drops, strict=True)) / sum(weights)


class TestComputeLocal:
    """Validate :func:`_compute_local` against hand-calculated values."""

    def test_quiet_opening_readme_row(self) -> None:
        """README §3.7 row 1: [+5, 0, -5, -10, -15, -25] → V_raw ≈ 11."""
        evals = [5, 0, -5, -10, -15, -25]
        v_local, scale, decided = _compute_local(evals, default_weights, default_scale_fn)
        assert scale == pytest.approx(1.0)
        assert decided is False
        assert v_local == pytest.approx(_expected_weighted_sum(evals))
        assert v_local == pytest.approx(11.39, abs=0.5)

    def test_normal_middlegame_readme_row(self) -> None:
        """§3.7 row 2: [+80, +30, +10, -5, -20, -40] → V_raw ≈ 71, V ≈ 38."""
        evals = [80, 30, 10, -5, -20, -40]
        v_local, scale, decided = _compute_local(evals, default_weights, default_scale_fn)
        assert scale == pytest.approx(1.0)
        assert v_local == pytest.approx(71.1, abs=0.5)
        v = 100.0 * (1.0 - math.exp(-v_local / K_SHALLOW))
        assert v == pytest.approx(38.0, abs=1.0)
        assert decided is False

    def test_only_move_crisis_readme_row(self) -> None:
        """§3.7 row 3: [0, -250, -400, -600, -900, -1200] → V_raw ≈ 488, V ≈ 96.

        scale = 1.0 (e_1 = 0)."""
        evals = [0, -250, -400, -600, -900, -1200]
        v_local, scale, decided = _compute_local(evals, default_weights, default_scale_fn)
        assert scale == pytest.approx(1.0)
        assert decided is False
        assert v_local == pytest.approx(_expected_weighted_sum(evals))
        assert v_local == pytest.approx(488.0, abs=2.0)
        v = 100.0 * (1.0 - math.exp(-v_local / K_SHALLOW))
        assert v == pytest.approx(96.0, abs=1.0)

    def test_dead_drawn_endgame(self) -> None:
        """§3.7: tiny drops → V near zero."""
        evals = [0, 0, 0, -5, -5, -10]
        v_local, scale, decided = _compute_local(evals, default_weights, default_scale_fn)
        assert scale == pytest.approx(1.0)
        v = 100.0 * (1.0 - math.exp(-v_local / K_SHALLOW))
        assert v < 3
        assert decided is False

    def test_drop_cap_applied(self) -> None:
        """When drops would exceed DROP_CAP, they're clamped.

        Uses ``min(|e1|, |e2|) <= EVAL_SCALE_GRACE`` so scale = 1.0 and we can
        read off the raw weighted sum directly.
        """
        # e_1 = 0 → scale_fn min(0, huge) = 0 → scale = 1.0
        huge_evals = [0, -5000, -10000, -20000, -30000, -40000]
        v_local, scale, _ = _compute_local(huge_evals, default_weights, default_scale_fn)
        assert scale == pytest.approx(1.0)
        # Every drop would exceed DROP_CAP → weighted mean collapses to DROP_CAP
        assert v_local == pytest.approx(DROP_CAP)


# --------------------------------------------------------------------------- #
# compute_volatility — end-to-end with FakeEngine                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def multi_move_board() -> chess.Board:
    """A position guaranteed to have >= 6 legal moves so multipv=6 stays intact."""
    return chess.Board()


class TestComputeVolatilityOnePly:
    def test_quiet_opening(self, multi_move_board: chess.Board) -> None:
        engine = FakeEngine(scripts=[evals_to_infos([5, 0, -5, -10, -15, -25])])
        result = compute_volatility(multi_move_board, engine)
        assert result.reason is None
        assert result.recurse_depth_used == 0
        assert result.score is not None
        assert result.score < 15
        assert result.best_eval_cp == 5
        assert result.alt_evals_cp == [0, -5, -10, -15, -25]
        assert result.scale == pytest.approx(1.0)
        assert result.decided is False
        assert result.analyses == 1

    def test_middlegame(self, multi_move_board: chess.Board) -> None:
        engine = FakeEngine(scripts=[evals_to_infos([80, 30, 10, -5, -20, -40])])
        result = compute_volatility(multi_move_board, engine)
        assert result.score == pytest.approx(38.0, abs=1.5)
        assert result.local_raw_cp == pytest.approx(71.1, abs=1.0)
        assert result.raw_cp == result.local_raw_cp

    def test_only_move_crisis_position(self, multi_move_board: chess.Board) -> None:
        """Many legal moves, but only one is remotely good → bar screams red."""
        engine = FakeEngine(scripts=[evals_to_infos([0, -250, -400, -600, -900, -1200])])
        result = compute_volatility(multi_move_board, engine)
        assert result.score is not None
        assert result.score > 90
        assert result.decided is False

    def test_v_endpoint_near_zero(self, multi_move_board: chess.Board) -> None:
        """All moves essentially equal → V ≈ 0."""
        engine = FakeEngine(scripts=[evals_to_infos([0, 0, 0, 0, 0, 0])])
        result = compute_volatility(multi_move_board, engine)
        assert result.score == pytest.approx(0.0, abs=1e-6)
        assert result.raw_cp == pytest.approx(0.0, abs=1e-6)

    def test_v_endpoint_near_100(self, multi_move_board: chess.Board) -> None:
        """Cataclysmic drops → V saturates near 100."""
        engine = FakeEngine(scripts=[evals_to_infos([0, -3000, -3000, -3000, -3000, -3000])])
        result = compute_volatility(multi_move_board, engine)
        assert result.score is not None
        assert result.score > 99

    def test_decided_dampens_score(self, multi_move_board: chess.Board) -> None:
        """[+1500, +1200, +1000, +800, +600, +400] — should flag decided."""
        engine = FakeEngine(scripts=[evals_to_infos([1500, 1200, 1000, 800, 600, 400])])
        result = compute_volatility(multi_move_board, engine)
        assert result.decided is True
        assert result.scale < 0.2

    def test_black_to_move_pov(self) -> None:
        """Engine scores are reported to the side-to-move; result reflects that POV."""
        board = chess.Board()
        board.push_san("e4")  # now Black to move
        assert board.turn == chess.BLACK
        engine = FakeEngine(scripts=[evals_to_infos([40, 20, 0, -10, -20, -30], turn=chess.BLACK)])
        result = compute_volatility(board, engine)
        assert result.best_eval_cp == 40


# --------------------------------------------------------------------------- #
# Edge cases (README §3.5)                                                     #
# --------------------------------------------------------------------------- #


class TestCheckmateAndStalemate:
    def test_checkmate_root_returns_none(self) -> None:
        """Scholar's mate final position: Black is checkmated."""
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        assert board.is_checkmate()

        def no_call(*a: object, **kw: object) -> list[dict[str, object]]:
            raise AssertionError("Engine should not be called on terminal positions")

        engine = FakeEngine(producer=no_call)
        result = compute_volatility(board, engine)
        assert result.score is None
        assert result.raw_cp is None
        assert result.reason == "checkmate"
        assert engine.call_count == 0

    def test_stalemate_root_returns_none(self) -> None:
        """Classic stalemate: Kh8 boxed by Kf7 + Qg6, not in check, no legal moves."""
        board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
        assert board.is_stalemate()
        engine = FakeEngine(scripts=[])
        result = compute_volatility(board, engine)
        assert result.score is None
        assert result.reason == "stalemate"


class TestOnlyMove:
    """README §3.5: K_legal == 1 → root returns None."""

    def test_only_one_legal_move_root_returns_none(self) -> None:
        board = chess.Board("7k/5K2/6R1/8/8/8/8/8 b - - 0 1")
        # Sanity check: only Kh7 is legal.
        assert len(list(board.legal_moves)) == 1
        engine = FakeEngine(scripts=[evals_to_infos([-500], turn=board.turn)])
        result = compute_volatility(board, engine)
        assert result.score is None
        assert result.reason == "only_move"
        assert engine.call_count == 1  # engine was still called to get the eval


class TestFewerLegalMovesThanMultiPV:
    def test_multipv_clamped_to_legal_count(self) -> None:
        """A position with 2 legal moves analyzed with multipv=6 must not crash."""
        board = chess.Board("8/8/8/8/8/8/7p/k6K w - - 0 1")
        legal = list(board.legal_moves)
        assert len(legal) == 2
        engine = FakeEngine(
            producer=lambda b, d, mpv: evals_to_infos(list(range(0, -mpv * 50, -50))[:mpv])
        )
        result = compute_volatility(board, engine)
        # Engine should have been asked for multipv = 2, not 6
        assert engine.calls[0][2] == 2
        assert result.score is not None
        assert len(result.alt_evals_cp) == 1
