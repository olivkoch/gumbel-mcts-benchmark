"""Benchmark of throughput for PUCT vs GumbelDense vs GumbelSparse on Tic-Tac-Toe and Gomoku.

If the Rust extension (`rust-python/`, built with maturin) is installed, each
variant is also benchmarked through it (rows suffixed "-Rs").
"""

import time
import numpy as np
import torch
import torch.nn as nn
from game_logic.tictactoe import TicTacToeLogic
from game_logic.gomoku import GomokuLogic
from gumbel_mcts import PUCT, GumbelDense, GumbelSparse

try:
    import gumbel_mcts_rs as gmrs
    HAS_RS = True
except ImportError:
    HAS_RS = False


def bench(name, create_and_run, n_warmup=3, n_iter=50):
    """Warm up, then time n_iter calls. Returns list of per-call times."""
    for _ in range(n_warmup):
        create_and_run()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        create_and_run()
        times.append(time.perf_counter() - t0)

    arr = np.array(times) * 1000  # ms
    print(f"  {name:20s}  mean={arr.mean():7.2f} ms  std={arr.std():6.2f} ms  "
          f"min={arr.min():7.2f} ms  max={arr.max():7.2f} ms")
    return arr


class TicTacToeModel(nn.Module):
    """Minimal NN for Tic-Tac-Toe (3x3 = 9 actions)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.policy_head = nn.Linear(64, 9)
        self.value_head = nn.Linear(64, 1)
        self.logic = TicTacToeLogic()

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float()
        h = self.net(boards)
        policy = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h))
        return {"policy": policy, "value": value}


class GomokuModel(nn.Module):
    """Minimal NN for Gomoku (15x15 = 225 actions)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(225, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.policy_head = nn.Linear(128, 225)
        self.value_head = nn.Linear(128, 1)
        self.logic = GomokuLogic()

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float()
        h = self.net(boards)
        policy = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h))
        return {"policy": policy, "value": value}


def bench_tictactoe():
    logic = TicTacToeLogic()
    model = TicTacToeModel()
    model.eval()

    board = np.zeros((3, 3), dtype=np.int8)
    board_flat = board.ravel()
    player = np.array([1])
    num_sims = 50

    def run_puct():
        tree = PUCT(n_games=1, max_nodes=500, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], player)
        tree.run_simulation_batch(model, [0], num_simulations=num_sims)

    def run_dense():
        tree = GumbelDense(n_games=1, max_nodes=500, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], player)
        tree.run_simulation_batch(model, [0], num_simulations=num_sims)

    def run_sparse():
        tree = GumbelSparse(n_games=1, max_nodes=500, logic=logic, device="cpu")
        tree.initialize_roots([0], board_flat[None], player)
        tree.run_simulation_batch(model, [0], num_simulations=num_sims)

    def run_puct_rs():
        tree = gmrs.PUCT(1, 500, logic, device="cpu")
        tree.initialize_roots([0], board[None], player)
        tree.run_simulation_batch(model, [0], num_simulations=num_sims)

    def run_dense_rs():
        tree = gmrs.GumbelDense(1, 500, logic, device="cpu")
        tree.initialize_roots([0], board[None], player)
        tree.run_simulation_batch(model, [0], num_simulations=num_sims)

    def run_sparse_rs():
        tree = gmrs.GumbelSparse(1, 500, logic, device="cpu")
        tree.initialize_roots([0], board_flat[None], player)
        tree.run_simulation_batch(model, [0], num_simulations=num_sims)

    print(f"Tic-Tac-Toe (9 actions)  |  {num_sims} sims  |  50 iterations\n")
    bench("PUCT", run_puct)
    bench("GumbelDense", run_dense)
    bench("GumbelSparse", run_sparse)
    if HAS_RS:
        bench("PUCT-Rs", run_puct_rs)
        bench("GumbelDense-Rs", run_dense_rs)
        bench("GumbelSparse-Rs", run_sparse_rs)


def bench_gomoku():
    logic = GomokuLogic()
    model = GomokuModel()
    model.eval()

    board = np.zeros((15, 15), dtype=np.int8)
    board_flat = board.ravel()
    player = np.array([1])

    for num_sims, max_nodes in [(50, 500), (200, 5000), (800, 50000)]:
        print(f"\nGomoku (225 actions)  |  {num_sims} sims  |  max_nodes={max_nodes}  |  20 iterations\n")

        def run_puct(ns=num_sims, mn=max_nodes):
            tree = PUCT(n_games=1, max_nodes=mn, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], player)
            tree.run_simulation_batch(model, [0], num_simulations=ns)

        def run_dense(ns=num_sims, mn=max_nodes):
            tree = GumbelDense(n_games=1, max_nodes=mn, logic=logic, device="cpu")
            tree.initialize_roots([0], board[None], player)
            tree.run_simulation_batch(model, [0], num_simulations=ns)

        def run_sparse(ns=num_sims, mn=max_nodes):
            tree = GumbelSparse(n_games=1, max_nodes=mn, logic=logic, device="cpu")
            tree.initialize_roots([0], board_flat[None], player)
            tree.run_simulation_batch(model, [0], num_simulations=ns)

        def run_puct_rs(ns=num_sims, mn=max_nodes):
            tree = gmrs.PUCT(1, mn, logic, device="cpu")
            tree.initialize_roots([0], board[None], player)
            tree.run_simulation_batch(model, [0], num_simulations=ns)

        def run_dense_rs(ns=num_sims, mn=max_nodes):
            tree = gmrs.GumbelDense(1, mn, logic, device="cpu")
            tree.initialize_roots([0], board[None], player)
            tree.run_simulation_batch(model, [0], num_simulations=ns)

        def run_sparse_rs(ns=num_sims, mn=max_nodes):
            tree = gmrs.GumbelSparse(1, mn, logic, device="cpu")
            tree.initialize_roots([0], board_flat[None], player)
            tree.run_simulation_batch(model, [0], num_simulations=ns)

        bench("PUCT", run_puct, n_warmup=2, n_iter=20)
        bench("GumbelDense", run_dense, n_warmup=2, n_iter=20)
        bench("GumbelSparse", run_sparse, n_warmup=2, n_iter=20)
        if HAS_RS:
            bench("PUCT-Rs", run_puct_rs, n_warmup=2, n_iter=20)
            bench("GumbelDense-Rs", run_dense_rs, n_warmup=2, n_iter=20)
            bench("GumbelSparse-Rs", run_sparse_rs, n_warmup=2, n_iter=20)


if __name__ == "__main__":
    bench_tictactoe()
    bench_gomoku()
