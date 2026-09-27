//! Rust port of benchmarks/benchmark_sparse_gumbel_efficiency.py:
//! Gumbel fixed at 8 sims — how many PUCT sims to match?

use gumbel_mcts_bench::game::{GameLogic, Gomoku};
use gumbel_mcts_bench::gumbel_sparse::GumbelSparse;
use gumbel_mcts_bench::model::{EvalModel, HeuristicGomoku};
use gumbel_mcts_bench::puct::Puct;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn pick_move_puct<M: EvalModel>(
    model: &M,
    board: &[i8; 225],
    player: i8,
    num_sims: usize,
    max_nodes: usize,
) -> usize {
    let mut tree = Puct::<Gomoku>::new(1, max_nodes);
    tree.initialize_roots(&[0], board, &[player]);
    tree.run_simulation_batch(model, &[0], num_sims, 19652.0, 1.25);
    let (visits, _) = tree.get_all_root_data(1);
    let mut best = 0usize;
    for a in 1..Gomoku::NUM_ACTIONS {
        if visits[a] > visits[best] {
            best = a;
        }
    }
    best
}

fn pick_move_gumbel<M: EvalModel>(
    model: &M,
    board: &[i8; 225],
    player: i8,
    num_sims: usize,
    max_nodes: usize,
    rng: &mut StdRng,
) -> usize {
    let mut tree = GumbelSparse::<Gomoku>::new(1, max_nodes, 50.0, 1.0, 35, 256);
    tree.initialize_roots(&[0], board, &[player]);
    let moves = tree.run_simulation_batch(model, &[0], num_sims, rng, None);
    moves[0] as usize
}

fn play_game<M: EvalModel>(
    model: &M,
    puct_sims: usize,
    gumbel_sims: usize,
    max_nodes: usize,
    puct_is_p1: bool,
    rng: &mut StdRng,
) -> i8 {
    let mut board = [0i8; 225];
    let mut player = 1i8;
    let puct_color = if puct_is_p1 { 1 } else { 2 };
    for _ in 0..Gomoku::MAX_MOVES {
        let action = if player == puct_color {
            pick_move_puct(model, &board, player, puct_sims, max_nodes)
        } else {
            pick_move_gumbel(model, &board, player, gumbel_sims, max_nodes, rng)
        };
        let res = Gomoku::fast_step(&mut board, action, player);
        if res.done {
            return res.winner;
        }
        player = 3 - player;
    }
    0
}

fn main() {
    let gumbel_sims = 8;
    let puct_sims_list = [8usize, 16, 32, 64, 128, 256, 512];
    let n_games = 40;
    let max_nodes = 5000usize;
    let noise_scale = 15.0;

    let model = HeuristicGomoku::new(noise_scale, StdRng::seed_from_u64(7));
    let mut rng = StdRng::seed_from_u64(0);

    println!("Asymmetry benchmark: PUCT vs Gumbel (heuristic model, noise={noise_scale})");
    println!("Gumbel fixed at {gumbel_sims} sims  |  Gomoku 15x15  |  {n_games} games per setting (alternating colors)\n");
    println!(
        "{:>10}  {:>12}  {:>10}  {:>12}  {:>6}  {:>9}",
        "PUCT sims", "Gumbel sims", "PUCT wins", "Gumbel wins", "Draws", "PUCT win%"
    );
    println!("{}", "-".repeat(70));

    for &puct_sims in &puct_sims_list {
        let mn = max_nodes.max(puct_sims * 10);
        let (mut puct, mut gumbel, mut draw) = (0usize, 0usize, 0usize);
        for game_idx in 0..n_games {
            let puct_is_p1 = game_idx % 2 == 0;
            let winner = play_game(&model, puct_sims, gumbel_sims, mn, puct_is_p1, &mut rng);
            let puct_color = if puct_is_p1 { 1 } else { 2 };
            if winner == puct_color {
                puct += 1;
            } else if winner == 3 - puct_color {
                gumbel += 1;
            } else {
                draw += 1;
            }
        }
        let puct_pct = 100.0 * puct as f64 / n_games as f64;
        println!(
            "{:>10}  {:>12}  {:>10}  {:>12}  {:>6}  {:>8.1}%",
            puct_sims, gumbel_sims, puct, gumbel, draw, puct_pct
        );
    }
    println!();
    println!("When PUCT win% reaches ~50%, the sim counts are equivalent in strength.");
}
