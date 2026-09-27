"""Win-rate benchmark: PUCT vs GumbelSparse on Gomoku at fixed simulation budgets.

This benchmark demonstrates that Sparse Gumbel MCTS achieves higher win rate than PUCT at constant simulation budget.

"""

import sys
import numpy as np
import torch
import torch.nn as nn
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
    """A fake 'trained' model that uses Gomoku heuristics instead of a neural network.

    Policy: scores each empty cell by proximity to existing stones and line patterns.
    Value:  simple material/threat count normalized to [-1, 1].

    This gives Gumbel MCTS informative logits to exploit, simulating a trained network.
    """

    def __init__(self):
        self.logic = GomokuLogic()

    def _score_board(self, board_2d, player):
        """Return (policy_logits, value) for a single position."""
        opponent = 3 - player
        scores = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # 1. Proximity bonus: moves near existing stones are more interesting
        occupied = (board_2d != 0)
        if occupied.any():
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if board_2d[r, c] != 0:
                        continue
                    # Count nearby stones (Manhattan distance <= 2)
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                                if board_2d[rr, cc] != 0:
                                    dist = max(abs(dr), abs(dc))
                                    scores[r * BOARD_SIZE + c] += 2.0 / dist
        else:
            # Empty board — prefer center
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    dist_center = abs(r - 7) + abs(c - 7)
                    scores[r * BOARD_SIZE + c] = max(0, 7 - dist_center)

        # 2. Line-threat scoring: reward extending own lines, blocking opponent lines
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
                        # Count consecutive stones of p in both directions from (r,c)
                        count = 0
                        for sign in [1, -1]:
                            rr, cc = r + sign * dr, c + sign * dc
                            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board_2d[rr, cc] == p:
                                count += 1
                                rr += sign * dr
                                cc += sign * dc
                        if count >= 4:
                            scores[action] += multiplier * 50   # win/block-win
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

        # Mask illegal moves
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_2d[r, c] != 0:
                    scores[r * BOARD_SIZE + c] = -1e9

        # Value: simple threat advantage, clamped to [-1, 1]
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

            # Softmax
            logits_max = logits.max()
            exp_logits = np.exp(logits - logits_max)
            all_policies[i] = exp_logits / exp_logits.sum()
            all_values[i] = val

        return {
            "policy": torch.from_numpy(all_policies),
            "value": torch.from_numpy(all_values),
        }


class RandomGomokuModel:
    """Untrained baseline: uniform policy, zero value."""

    def __init__(self):
        self.logic = GomokuLogic()

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float().numpy()
        B = boards.shape[0]
        policies = np.zeros((B, NUM_ACTIONS), dtype=np.float32)
        for i in range(B):
            board_flat = boards[i].astype(np.int8)
            mask = (board_flat == 0).astype(np.float32)
            s = mask.sum()
            if s > 0:
                policies[i] = mask / s
        return {
            "policy": torch.from_numpy(policies),
            "value": torch.zeros(B),
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


def play_game(logic, model, searchers, num_sims, max_nodes):
    """Play one game. searchers = {1: pick_fn, 2: pick_fn}. Returns winner (1, 2, or 0 for draw)."""
    board = np.zeros(logic.BOARD_SHAPE, dtype=np.int8)
    player = 1
    for _ in range(logic.MAX_MOVES):
        action = searchers[player](logic, model, board, player, num_sims, max_nodes)
        _, winner, done, board = logic.fast_step(board, action, player)
        if done:
            return winner
        player = 3 - player
    return 0  # draw


def run_match(n_games, num_sims, max_nodes, model):
    logic = GomokuLogic()

    results = {"puct": 0, "gumbel": 0, "draw": 0}

    for game_idx in range(n_games):
        # Alternate colors each game
        if game_idx % 2 == 0:
            searchers = {1: pick_move_puct, 2: pick_move_gumbel}
            puct_color, gumbel_color = 1, 2
        else:
            searchers = {1: pick_move_gumbel, 2: pick_move_puct}
            puct_color, gumbel_color = 2, 1

        winner = play_game(logic, model, searchers, num_sims, max_nodes)

        if winner == puct_color:
            results["puct"] += 1
            tag = "PUCT"
        elif winner == gumbel_color:
            results["gumbel"] += 1
            tag = "Gumbel"
        else:
            results["draw"] += 1
            tag = "Draw"

        print(f"  Game {game_idx + 1:3d}/{n_games}: {tag:>6s}  "
              f"(PUCT={results['puct']}  Gumbel={results['gumbel']}  Draw={results['draw']})")

    return results


def print_results(results, n_games):
    print(f"\n{'='*50}")
    print(f"PUCT wins:    {results['puct']:3d}  ({100*results['puct']/n_games:.1f}%)")
    print(f"Gumbel wins:  {results['gumbel']:3d}  ({100*results['gumbel']/n_games:.1f}%)")
    print(f"Draws:        {results['draw']:3d}  ({100*results['draw']/n_games:.1f}%)")
    print(f"{'='*50}")


def main():
    n_games = 30
    num_sims = 50
    max_nodes = 2000

    if USE_RUST:
        print("Engine: gumbel_mcts_rs (Rust extension)\n")

    # --- Round 1: Random model (uniform policy, zero value) ---
    print(f"Round 1: RANDOM MODEL (uniform policy)")
    print(f"PUCT vs GumbelSparse  |  Gomoku 15x15  |  {num_sims} sims/move  |  "
          f"{n_games} games\n")

    random_model = RandomGomokuModel()
    results1 = run_match(n_games, num_sims, max_nodes, random_model)
    print_results(results1, n_games)

    # --- Round 2: Heuristic model (informative policy + value) ---
    print(f"\n\nRound 2: HEURISTIC MODEL (simulates trained network)")
    print(f"PUCT vs GumbelSparse  |  Gomoku 15x15  |  {num_sims} sims/move  |  "
          f"{n_games} games\n")

    heuristic_model = HeuristicGomokuModel()
    results2 = run_match(n_games, num_sims, max_nodes, heuristic_model)
    print_results(results2, n_games)


if __name__ == "__main__":
    main()
