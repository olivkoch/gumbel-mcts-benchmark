//! Rust port of benchmarks/benchmark_puct_speedup.py:
//! V2 reference (`reference.rs`) vs V3 batched PUCT (`puct.rs`).

use burn::backend::NdArray;
use gumbel_mcts_bench::bench_util::median;
use gumbel_mcts_bench::env::GomokuEnv;
use gumbel_mcts_bench::model::Mlp;
use gumbel_mcts_bench::puct::Puct;
use gumbel_mcts_bench::reference::parallel_uct_search;
use gumbel_mcts_bench::game::Gomoku;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

type B = NdArray;

const C_PUCT_BASE: f64 = 19652.0;
const C_PUCT_INIT: f64 = 1.25;

fn create_test_environments(n_envs: usize, seed: u64) -> Vec<GomokuEnv> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_envs)
        .map(|_| {
            let mut env = GomokuEnv::new();
            // Random opening moves (0-7).
            for _ in 0..rng.gen_range(0..8) {
                let legal = env.legal_actions();
                let valid: Vec<usize> =
                    (0..225).filter(|&a| legal[a] >= 1.0).collect();
                if !valid.is_empty() {
                    let a = valid[rng.gen_range(0..valid.len())];
                    env.step(a);
                }
            }
            env
        })
        .collect()
}

fn warmup<M>(model: &M, n_warmup: usize)
where
    M: gumbel_mcts_bench::model::EvalModel,
{
    print!("Warming up... ");
    use std::io::Write;
    std::io::stdout().flush().unwrap();

    let env = GomokuEnv::new();
    for _ in 0..n_warmup {
        parallel_uct_search(&env, model, C_PUCT_BASE, C_PUCT_INIT, 10, 4, true);

        let max_nodes = ((1 + 10) * 2) as usize;
        let mut tree = Puct::<Gomoku>::new(4, max_nodes);
        tree.reset();
        let boards: Vec<i8> = env.board.iter().cloned().cycle().take(4 * 225).collect();
        let players = [env.to_play; 4];
        tree.initialize_roots(&[0, 1, 2, 3], &boards, &players);
        tree.run_simulation_batch(model, &[0, 1, 2, 3], 10, C_PUCT_BASE, C_PUCT_INIT);
    }
    println!("Done.");
}

fn main() {
    let device = burn::tensor::Device::<B>::default();
    let model = Mlp::<B>::new(225, 128, 225, &device);

    let configs = [
        (8usize, 50usize, 8usize),
        (8, 50, 16),
        (8, 50, 32),
        (32, 50, 8),
        (32, 50, 32),
        (64, 100, 8),
        (64, 100, 16),
        (64, 100, 64),
        (128, 200, 8),
        (128, 200, 16),
        (128, 200, 128),
        (256, 200, 8),
        (256, 200, 32),
        (256, 200, 128),
        (1024, 800, 8),
        (1024, 800, 64),
        (1024, 800, 512),
        (1024, 800, 1024),
    ];

    let max_games = configs.iter().map(|c| c.0).max().unwrap();
    println!("Creating test environments...");
    let envs = create_test_environments(max_games, 42);

    warmup(&model, 3);

    println!();
    println!("{:<40} | {:<10} | {:<10} | {:<12} | {:<10}", "Config", "V2 (s)", "V3 (s)", "V3 sims/s", "Speedup");
    println!("{}", "-".repeat(80));

    let n_repeats = 3;
    for &(n_games, sims, n_parallel) in &configs {
        let cfg_str = format!("{n_games} games × {sims} sims × {n_parallel} parallel");
        let current_envs = &envs[..n_games];

        let mut v2_times = Vec::new();
        for _ in 0..n_repeats {
            let start = Instant::now();
            for env in current_envs {
                parallel_uct_search(
                    env, &model, C_PUCT_BASE, C_PUCT_INIT, sims, n_parallel, true,
                );
            }
            v2_times.push(start.elapsed().as_secs_f64());
        }
        let v2_time = median(v2_times);

        let mut boards = vec![0i8; n_games * 225];
        let mut players = vec![0i8; n_games];
        for (i, e) in current_envs.iter().enumerate() {
            boards[i * 225..(i + 1) * 225].copy_from_slice(&e.board);
            players[i] = e.to_play;
        }
        let active: Vec<usize> = (0..n_games).collect();

        let mut v3_times = Vec::new();
        for _ in 0..n_repeats {
            let start = Instant::now();
            let max_nodes = ((1 + sims) * n_games) as f64 * 2.0;
            let mut tree = Puct::<Gomoku>::new(n_games, max_nodes as usize);
            tree.reset();
            tree.initialize_roots(&active, &boards, &players);
            tree.run_simulation_batch(&model, &active, sims, C_PUCT_BASE, C_PUCT_INIT);
            v3_times.push(start.elapsed().as_secs_f64());
        }
        let v3_time = median(v3_times);

        let speedup = v2_time / v3_time;
        let v3_sps = (n_games * sims) as f64 / v3_time;
        println!(
            "{:<40} | {:<10.3} | {:<10.3} | {:<12.0} | {:.2}x",
            cfg_str, v2_time, v3_time, v3_sps, speedup
        );
    }
}
