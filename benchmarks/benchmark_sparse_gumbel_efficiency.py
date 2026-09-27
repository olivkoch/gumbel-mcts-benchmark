"""Asymmetry benchmark: how many PUCT sims to match Gumbel at fixed budget?

Fixes Gumbel at a low sim count and ramps up PUCT sims until win rate equalizes.
Uses the heuristic (simulated trained) model only.
"""

import sys
import numpy as np
import torch
from game_logic.gomoku import GomokuLogic
from gumbel_mcts import PUCT, GumbelSparse

try:
    import gumbel_mcts_rs as gmrs
    HAS_RS = True
except ImportError:
    HAS_RS = False

# Pass --rust to run the search through the Rust extension (rust-python/).
USE_RUST = "--rust" in sys.argv
if USE_RUST and not HAS_RS:
    sys.exit("--rust requested but gumbel_mcts_rs is not installed "
             "(cd rust-python && maturin develop --release)")

BOARD_SIZE = 15
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE


class HeuristicGomokuModel:
    """Heuristic model simulating a trained network.
    
    noise_scale controls policy quality:
      0.0 = perfect heuristic
      higher = noisier policy (simulates partially trained network)
    """

    def __init__(self, noise_scale=0.0):
        self.logic = GomokuLogic()
        self.noise_scale = noise_scale

    def _score_board(self, board_2d, player):
        opponent = 3 - player
        scores = np.zeros(NUM_ACTIONS, dtype=np.float32)

        occupied = (board_2d != 0)
        if occupied.any():
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if board_2d[r, c] != 0:
                        continue
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                                if board_2d[rr, cc] != 0:
                                    dist = max(abs(dr), abs(dc))
                                    scores[r * BOARD_SIZE + c] += 2.0 / dist
        else:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    dist_center = abs(r - 7) + abs(c - 7)
                    scores[r * BOARD_SIZE + c] = max(0, 7 - dist_center)

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        my_threat_score = 0.0
        opp_threat_score = 0.0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_2d[r, c] != 0:
                    continue
                action = r * BOARD_SIZE + c
                for dr, dc in directions:
                    for p, multiplier in [(player, 3.0), (opponent, 2.5)]:
                        count = 0
                        for sign in [1, -1]:
                            rr, cc = r + sign * dr, c + sign * dc
                            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board_2d[rr, cc] == p:
                                count += 1
                                rr += sign * dr
                                cc += sign * dc
                        if count >= 4:
                            scores[action] += multiplier * 50
                        elif count >= 3:
                            scores[action] += multiplier * 10
                        elif count >= 2:
                            scores[action] += multiplier * 3
                        elif count >= 1:
                            scores[action] += multiplier * 1

                        if p == player:
                            my_threat_score += count
                        else:
                            opp_threat_score += count

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_2d[r, c] != 0:
                    scores[r * BOARD_SIZE + c] = -1e9

        value = np.tanh(0.05 * (my_threat_score - opp_threat_score))
        return scores, value

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float().numpy()
        players = batch["current_player"].numpy()
        B = boards.shape[0]
        all_policies = np.zeros((B, NUM_ACTIONS), dtype=np.float32)
        all_values = np.zeros(B, dtype=np.float32)
        for i in range(B):
            board_2d = boards[i].reshape(BOARD_SIZE, BOARD_SIZE).astype(np.int8)
            player = int(players[i])
            logits, val = self._score_board(board_2d, player)

            # Add noise to degrade policy quality
            if self.noise_scale > 0:
                legal = logits > -1e8
                logits[legal] += np.random.randn(legal.sum()) * self.noise_scale

            logits_max = logits.max()
            exp_logits = np.exp(logits - logits_max)
            all_policies[i] = exp_logits / exp_logits.sum()
            all_values[i] = val
        return {
            "policy": torch.from_numpy(all_policies),
            "value": torch.from_numpy(all_values),
        }


def pick_move_puct(logic, model, board, player, num_sims, max_nodes):
    cls = gmrs.PUCT if USE_RUST else PUCT
    tree = cls(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
    tree.initialize_roots([0], board[None], np.array([player]))
    tree.run_simulation_batch(model, [0], num_simulations=num_sims)
    visits, _ = tree.get_all_root_data(n_active=1)
    return int(np.argmax(visits[0]))


def pick_move_gumbel(logic, model, board, player, num_sims, max_nodes):
    cls = gmrs.GumbelSparse if USE_RUST else GumbelSparse
    tree = cls(n_games=1, max_nodes=max_nodes, logic=logic, device="cpu")
    tree.initialize_roots([0], board.ravel()[None], np.array([player]))
    moves = tree.run_simulation_batch(model, [0], num_simulations=num_sims)
    return int(moves[0])


def play_game(logic, model, puct_sims, gumbel_sims, max_nodes, puct_is_player1):
    """Play one game with asymmetric sim budgets. Returns winner (1, 2, or 0)."""
    board = np.zeros(logic.BOARD_SHAPE, dtype=np.int8)
    player = 1
    puct_color = 1 if puct_is_player1 else 2

    for _ in range(logic.MAX_MOVES):
        if player == puct_color:
            action = pick_move_puct(logic, model, board, player, puct_sims, max_nodes)
        else:
            action = pick_move_gumbel(logic, model, board, player, gumbel_sims, max_nodes)

        _, winner, done, board = logic.fast_step(board, action, player)
        if done:
            return winner
        player = 3 - player
    return 0


def run_asymmetric_match(n_games, puct_sims, gumbel_sims, max_nodes, model):
    logic = GomokuLogic()
    results = {"puct": 0, "gumbel": 0, "draw": 0}

    for game_idx in range(n_games):
        puct_is_p1 = (game_idx % 2 == 0)
        winner = play_game(logic, model, puct_sims, gumbel_sims, max_nodes, puct_is_p1)

        puct_color = 1 if puct_is_p1 else 2
        gumbel_color = 3 - puct_color

        if winner == puct_color:
            results["puct"] += 1
        elif winner == gumbel_color:
            results["gumbel"] += 1
        else:
            results["draw"] += 1

    return results


def main():
    gumbel_sims = 8
    puct_sims_list = [8, 16, 32, 64, 128, 256, 512]
    n_games = 40
    max_nodes = 5000
    noise_scale = 15.0  # partially trained network

    model = HeuristicGomokuModel(noise_scale=noise_scale)

    if USE_RUST:
        print("Engine: gumbel_mcts_rs (Rust extension)")
    print(f"Asymmetry benchmark: PUCT vs Gumbel (heuristic model, noise={noise_scale})")
    print(f"Gumbel fixed at {gumbel_sims} sims  |  Gomoku 15x15  |  "
          f"{n_games} games per setting (alternating colors)\n")
    print(f"{'PUCT sims':>10s}  {'Gumbel sims':>12s}  {'PUCT wins':>10s}  "
          f"{'Gumbel wins':>12s}  {'Draws':>6s}  {'PUCT win%':>9s}")
    print("-" * 70)

    for puct_sims in puct_sims_list:
        mn = max(max_nodes, puct_sims * 10)
        results = run_asymmetric_match(n_games, puct_sims, gumbel_sims, mn, model)
        puct_pct = 100 * results["puct"] / n_games
        print(f"{puct_sims:>10d}  {gumbel_sims:>12d}  {results['puct']:>10d}  "
              f"{results['gumbel']:>12d}  {results['draw']:>6d}  {puct_pct:>8.1f}%")

    print()
    print("When PUCT win% reaches ~50%, the sim counts are equivalent in strength.")


if __name__ == "__main__":
    main()
