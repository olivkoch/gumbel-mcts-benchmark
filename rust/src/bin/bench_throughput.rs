//! Rust port of benchmarks/benchmark_throughput.py:
//! PUCT vs GumbelDense vs GumbelSparse on Tic-Tac-Toe and Gomoku.

use burn::backend::NdArray;
use gumbel_mcts_bench::bench_util::bench;
use gumbel_mcts_bench::game::{Gomoku, TicTacToe};
use gumbel_mcts_bench::gumbel_dense::GumbelDense;
use gumbel_mcts_bench::gumbel_sparse::GumbelSparse;
use gumbel_mcts_bench::model::Mlp;
use gumbel_mcts_bench::puct::Puct;
use rand::rngs::StdRng;
use rand::SeedableRng;

type B = NdArray;

fn bench_tictactoe() {
    let device = burn::tensor::Device::<B>::default();
    let model = Mlp::<B>::new(9, 64, 9, &device);
    let board = [0i8; 9];
    let player = [1i8];
    let num_sims = 50;
    let mut rng = StdRng::seed_from_u64(0);

    println!("Tic-Tac-Toe (9 actions)  |  {num_sims} sims  |  50 iterations\n");

    bench(
        "PUCT",
        || {
            let mut tree = Puct::<TicTacToe>::new(1, 500);
            tree.initialize_roots(&[0], &board, &player);
            tree.run_simulation_batch(&model, &[0], num_sims, 19652.0, 1.25);
        },
        3,
        50,
    );
    bench(
        "GumbelDense",
        || {
            let mut tree = GumbelDense::<TicTacToe>::new(1, 500, 50.0, 1.0);
            tree.initialize_roots(&[0], &board, &player, &mut rng);
            tree.run_simulation_batch(&model, &[0], num_sims, &mut rng);
        },
        3,
        50,
    );
    bench(
        "GumbelSparse",
        || {
            let mut tree = GumbelSparse::<TicTacToe>::new(1, 500, 50.0, 1.0, 35, 256);
            tree.initialize_roots(&[0], &board, &player);
            tree.run_simulation_batch(&model, &[0], num_sims, &mut rng, None);
        },
        3,
        50,
    );
}

fn bench_gomoku() {
    let device = burn::tensor::Device::<B>::default();
    let model = Mlp::<B>::new(225, 128, 225, &device);
    let board = [0i8; 225];
    let player = [1i8];
    let mut rng = StdRng::seed_from_u64(0);

    for (num_sims, max_nodes) in [(50usize, 500usize), (200, 5000), (800, 50000)] {
        println!(
            "\nGomoku (225 actions)  |  {num_sims} sims  |  max_nodes={max_nodes}  |  20 iterations\n"
        );
        bench(
            "PUCT",
            || {
                let mut tree = Puct::<Gomoku>::new(1, max_nodes);
                tree.initialize_roots(&[0], &board, &player);
                tree.run_simulation_batch(&model, &[0], num_sims, 19652.0, 1.25);
            },
            2,
            20,
        );
        bench(
            "GumbelDense",
            || {
                let mut tree = GumbelDense::<Gomoku>::new(1, max_nodes, 50.0, 1.0);
                tree.initialize_roots(&[0], &board, &player, &mut rng);
                tree.run_simulation_batch(&model, &[0], num_sims, &mut rng);
            },
            2,
            20,
        );
        bench(
            "GumbelSparse",
            || {
                let mut tree =
                    GumbelSparse::<Gomoku>::new(1, max_nodes, 50.0, 1.0, 35, 225);
                tree.initialize_roots(&[0], &board, &player);
                tree.run_simulation_batch(&model, &[0], num_sims, &mut rng, None);
            },
            2,
            20,
        );
    }
}

fn main() {
    bench_tictactoe();
    bench_gomoku();
}
