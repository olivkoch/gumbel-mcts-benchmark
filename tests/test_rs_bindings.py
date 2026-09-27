"""Smoke tests for the optional Rust extension (gumbel_mcts_rs).

Build it with:  cd rust-python && maturin develop --release
Tests are skipped if the extension is not installed.
"""

import numpy as np
import pytest
import torch

gmrs = pytest.importorskip("gumbel_mcts_rs")

from game_logic.gomoku import GomokuLogic
from game_logic.tictactoe import TicTacToeLogic


class UniformModel:
    """Uniform-legal policy, zero value. No torch ops needed."""

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float().numpy()
        B, BL = boards.shape
        policies = np.zeros((B, BL), dtype=np.float32)
        for i in range(B):
            mask = (boards[i] == 0).astype(np.float32)
            s = mask.sum()
            policies[i] = mask / s if s > 0 else mask
        return {
            "policy": torch.from_numpy(policies),
            "value": torch.zeros(B),
        }


@pytest.fixture
def gomoku():
    return GomokuLogic()


def _roots(logic, n=2):
    boards = np.zeros((n,) + tuple(logic.BOARD_SHAPE), dtype=np.int8)
    players = np.ones(n, dtype=np.int64)
    return boards, players, list(range(n))


def test_puct_visits_sum_to_sims(gomoku):
    boards, players, active = _roots(gomoku)
    tree = gmrs.PUCT(2, 5000, gomoku, device="cpu")
    tree.initialize_roots(active, boards, players)
    tree.run_simulation_batch(UniformModel(), active, num_simulations=200)
    visits, root_q = tree.get_all_root_data(2)
    assert visits.shape == (2, 225)
    np.testing.assert_allclose(visits.sum(axis=1), 200.0)
    assert np.isfinite(root_q).all()
    assert tree.get_max_depth() >= 1


def test_gumbel_sparse_returns_legal_move(gomoku):
    boards, players, active = _roots(gomoku)
    tree = gmrs.GumbelSparse(2, 5000, gomoku, device="cpu")
    tree.initialize_roots(active, boards, players)
    moves = tree.run_simulation_batch(UniformModel(), active, num_simulations=50)
    assert moves.shape == (2,)
    for g in range(2):
        mv = int(moves[g])
        assert 0 <= mv < 225
        assert boards[g].ravel()[mv] == 0
    pi = tree.get_improved_policy(2)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-5)


def test_gumbel_dense_returns_legal_move(gomoku):
    boards, players, active = _roots(gomoku)
    tree = gmrs.GumbelDense(2, 5000, gomoku, device="cpu")
    tree.initialize_roots(active, boards, players)
    moves = tree.run_simulation_batch(UniformModel(), active, num_simulations=50)
    for g in range(2):
        mv = int(moves[g])
        assert 0 <= mv < 225
        assert boards[g].ravel()[mv] == 0


def test_tictactoe_puct():
    logic = TicTacToeLogic()
    boards = np.zeros((1, 3, 3), dtype=np.int8)
    players = np.array([1])
    tree = gmrs.PUCT(1, 500, logic, device="cpu")
    tree.initialize_roots([0], boards, players)
    tree.run_simulation_batch(UniformModel(), [0], num_simulations=50)
    visits, _ = tree.get_all_root_data(1)
    assert visits.shape == (1, 9)
    assert visits.sum() == 50.0


def test_string_logic_accepted():
    """`logic` may be a plain game name string instead of a logic object."""
    boards = np.zeros((1, 3, 3), dtype=np.int8)
    players = np.array([1])
    tree = gmrs.PUCT(1, 500, "tictactoe")
    tree.initialize_roots([0], boards, players)
    tree.run_simulation_batch(UniformModel(), [0], num_simulations=20)


def test_bad_game_name_rejected():
    with pytest.raises(Exception):
        gmrs.PUCT(1, 500, "chess")
