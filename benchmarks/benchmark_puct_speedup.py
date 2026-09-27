""" This benchmark demonstrates the speedup of the puct.py implementation over the reference MCTS (reference.py) 
"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from gumbel_mcts.reference import parallel_uct_search
from gumbel_mcts.puct import PUCT
from utils.gomoku_env import GomokuEnv
from game_logic.gomoku import GomokuLogic
from tests.test_utils import create_eval_func

try:
    import gumbel_mcts_rs as gmrs
    HAS_RS = True
except ImportError:
    HAS_RS = False

import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


# =============================================================================
# 1. MODEL
# =============================================================================

class GomokuModel(nn.Module):
    """Minimal NN for Gomoku (15x15 = 225 actions)."""
    def __init__(self):
        super().__init__()
        self.logic = GomokuLogic()
        self.net = nn.Sequential(nn.Linear(225, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.policy_head = nn.Linear(128, 225)
        self.value_head = nn.Linear(128, 1)

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float()
        h = self.net(boards)
        policy = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return {"policy": policy, "value": value}


# =============================================================================
# 2. BENCHMARK SUITE
# =============================================================================

def create_test_environments(n_envs: int, seed: int = 42) -> list:
    """Create test environments with random opening positions."""
    envs = []
    np.random.seed(seed)
    for _ in range(n_envs):
        env = GomokuEnv()
        env.reset()
        # Random opening moves (0-7 moves)
        for _ in range(np.random.randint(0, 8)): 
            valid = np.where(env.legal_actions)[0]
            if len(valid) > 0:
                env.step(np.random.choice(valid))
        envs.append(env)
    return envs


def warmup(model_v3, eval_func_v2, logic, device: str, n_warmup: int = 3):
    """Warmup to trigger JIT compilation and GPU initialization."""
    print("Warming up...", end=" ", flush=True)
    
    env = GomokuEnv()
    env.reset()
    
    for _ in range(n_warmup):
        # Warmup V2
        parallel_uct_search(
            env=env, eval_func=eval_func_v2, root_node=None,
            c_puct_base=19652, c_puct_init=1.25,
            num_simulations=10, num_parallel=4,
            deterministic=True
        )
        
        # Warmup V3
        num_simulations=10
        max_nodes = int((1 + num_simulations) * 2.0)
        tree = PUCT(n_games=4, max_nodes=max_nodes, logic=logic, device=device)
        tree.reset()
        boards = np.array([env.board] * 4)
        players = np.array([env.to_play] * 4)
        tree.initialize_roots(list(range(4)), boards, players)
        tree.run_simulation_batch(
            model_v3, active_games=list(range(4)), num_simulations=10,
            c_puct_base=19652, c_puct_init=1.25
        )
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print("Done.")


def run_benchmark(model_v3, model_name: str, device: str, eval_func_v2,
                  logic, envs: list, configs: list, n_repeats: int = 3):
    """
    Benchmark MCTS V2 (parallel simulations) vs V3 (batched games).
    
    Args:
        model_v3: Model with forward_for_mcts interface
        model_name: Display name
        device: 'cpu', 'cuda', or 'mps'
        eval_func_v2: Evaluation function for V2
        logic: Game logic instance (e.g. GomokuLogic)
        envs: Pre-created test environments
        configs: List of benchmark configurations
        n_repeats: Number of repetitions for timing
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    if HAS_RS:
        print(f"{'Config':<40} | {'V2 (s)':<10} | {'V3 (s)':<10} | {'V3 sims/s':<12} | {'V3rs (s)':<10} | {'V3rs sims/s':<12} | {'Speedup':<10}")
    else:
        print(f"{'Config':<40} | {'V2 (s)':<10} | {'V3 (s)':<10} | {'V3 sims/s':<12} | {'Speedup':<10}")
    print("-" * 80)

    results = []

    for cfg in configs:
        n_games = cfg["n_games"]
        sims = cfg["sims"]
        n_parallel = cfg["parallel"]
        
        current_envs = envs[:n_games]
        cfg_str = f"{n_games} games × {sims} sims × {n_parallel} parallel"
        
        # --- Benchmark V2 (loop over games, parallel simulations) ---
        v2_times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            for env in current_envs:
                parallel_uct_search(
                    env=env, eval_func=eval_func_v2, root_node=None,
                    c_puct_base=19652, c_puct_init=1.25,
                    num_simulations=sims, num_parallel=n_parallel,
                    deterministic=True
                )
            if device == 'cuda':
                torch.cuda.synchronize()
            v2_times.append(time.perf_counter() - start)
        v2_time = np.median(v2_times)
        
        # --- Benchmark V3 (batched games) ---
        boards = np.array([e.board for e in current_envs])
        players = np.array([e.to_play for e in current_envs])
        active = list(range(n_games))
        
        v3_times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            max_nodes = int((1 + sims) * n_games * 2.0)  # Estimate max nodes needed
            tree = PUCT(n_games=n_games, max_nodes=max_nodes, logic=logic, device=device)
            tree.reset()
            tree.initialize_roots(active, boards, players)
            tree.run_simulation_batch(
                model_v3, active_games=active, num_simulations=sims,
                c_puct_base=19652, c_puct_init=1.25
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            v3_times.append(time.perf_counter() - start)
        v3_time = np.median(v3_times)

        # --- Benchmark V3-Rs (Rust extension, same model + eval callback) ---
        v3rs_time = None
        v3rs_sps = None
        if HAS_RS:
            v3rs_times = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                max_nodes = int((1 + sims) * n_games * 2.0)
                tree = gmrs.PUCT(n_games, max_nodes, logic, device=device)
                tree.initialize_roots(active, boards, players)
                tree.run_simulation_batch(
                    model_v3, active_games=active, num_simulations=sims,
                    c_puct_base=19652, c_puct_init=1.25
                )
                if device == 'cuda':
                    torch.cuda.synchronize()
                v3rs_times.append(time.perf_counter() - start)
            v3rs_time = np.median(v3rs_times)
            v3rs_sps = (n_games * sims) / v3rs_time

        # Calculate metrics
        speedup = v2_time / v3_time
        v3_sps = (n_games * sims) / v3_time

        if HAS_RS:
            speedup_rs = v2_time / v3rs_time
            print(f"{cfg_str:<40} | {v2_time:<10.3f} | {v3_time:<10.3f} | {v3_sps:<12.0f} | "
                  f"{v3rs_time:<10.3f} | {v3rs_sps:<12.0f} | {speedup_rs:<10.2f}x")
        else:
            print(f"{cfg_str:<40} | {v2_time:<10.3f} | {v3_time:<10.3f} | {v3_sps:<12.0f} | {speedup:<10.2f}x")

        results.append({
            "config": cfg,
            "v2_time": v2_time,
            "v3_time": v3_time,
            "v3rs_time": v3rs_time,
            "v3rs_sims_per_sec": v3rs_sps,
            "speedup": speedup,
            "v3_sims_per_sec": v3_sps
        })
        
    return results


def print_summary(all_results: dict):
    """Print final comparison across all models."""
    print(f"\n{'='*80}")
    print("SUMMARY: V3 Throughput Comparison")
    print(f"{'='*80}")
    
    model_names = list(all_results.keys())
    configs = [r["config"] for r in all_results[model_names[0]]]
    
    # Header
    header = f"{'Config':<40}"
    for name in model_names:
        header += f" | {name[:12]:<12}"
    print(header)
    print("-" * 80)
    
    # Rows
    for i, cfg in enumerate(configs):
        cfg_str = f"{cfg['n_games']} × {cfg['sims']} x {cfg['parallel']} parallel"
        row = f"{cfg_str:<40}"
        for name in model_names:
            sps = all_results[name][i]["v3_sims_per_sec"]
            row += f" | {sps:<12.0f}"
        print(row)
    
    print("-" * 80)


# =============================================================================
# 3. MAIN
# =============================================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: CUDA ({torch.cuda.get_device_name()})")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Device: MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Device: CPU")
    
    # --- Benchmark Configs ---
    configs = [
        {"n_games": 8,   "sims": 50,  "parallel": 8},
        {"n_games": 8,   "sims": 50,  "parallel": 16},
        {"n_games": 8,   "sims": 50,  "parallel": 32},
        {"n_games": 32,  "sims": 50,  "parallel": 8},
        {"n_games": 32,  "sims": 50,  "parallel": 32},
        {"n_games": 64,  "sims": 100, "parallel": 8},
        {"n_games": 64,  "sims": 100, "parallel": 16},
        {"n_games": 64,  "sims": 100, "parallel": 64},
        {"n_games": 128, "sims": 200, "parallel": 8},
        {"n_games": 128, "sims": 200, "parallel": 16},
        {"n_games": 128, "sims": 200, "parallel": 128},
        {"n_games": 256, "sims": 200, "parallel": 8}, 
        {"n_games": 256, "sims": 200, "parallel": 32}, 
        {"n_games": 256, "sims": 200, "parallel": 128}, 
        {"n_games": 1024, "sims": 800, "parallel":8}, 
        {"n_games": 1024, "sims": 800, "parallel": 64}, 
        {"n_games": 1024, "sims": 800, "parallel": 512},
        {"n_games": 1024, "sims": 800, "parallel": 1024},
        ]
    
    max_games = max(cfg["n_games"] for cfg in configs)

    # --- Create Test Environments ---
    print("Creating test environments...")
    envs = create_test_environments(n_envs=max_games, seed=42)
    
    # --- Setup Model ---
    logic = GomokuLogic()
    model = GomokuModel().eval().to(device)
    eval_func_v2 = create_eval_func(model)
        
    # --- Warmup ---
    warmup(model, eval_func_v2, logic, device)
    
    # --- Run Benchmarks ---
    all_results = {}
    
    all_results["GomokuModel"] = run_benchmark(
        model, "GomokuModel", device, eval_func_v2, logic, envs, configs
    )
        
    # --- Print Summary ---
    print_summary(all_results)