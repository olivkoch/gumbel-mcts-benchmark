//! Rust port of benchmarks/benchmark_winrate.py:
//! PUCT vs GumbelSparse on Gomoku at fixed simulation budgets.

use gumbel_mcts_bench::game::{GameLogic, Gomoku};
use gumbel_mcts_bench::gumbel_sparse::GumbelSparse;
use gumbel_mcts_bench::model::{EvalModel, HeuristicGomoku, RandomGomoku};
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
    puct_is_p1: bool,
    num_sims: usize,
    max_nodes: usize,
    rng: &mut StdRng,
) -> i8 {
    let mut board = [0i8; 225];
    let mut player = 1i8;
    for _ in 0..Gomoku::MAX_MOVES {
        let is_puct = (player == 1) == puct_is_p1;
        let action = if is_puct {
            pick_move_puct(model, &board, player, num_sims, max_nodes)
        } else {
            pick_move_gumbel(model, &board, player, num_sims, max_nodes, rng)
        };
        let res = Gomoku::fast_step(&mut board, action, player);
        if res.done {
            return res.winner;
        }
        player = 3 - player;
    }
    0
}

fn run_match<M: EvalModel>(
    n_games: usize,
    num_sims: usize,
    max_nodes: usize,
    model: &M,
    rng: &mut StdRng,
) -> (usize, usize, usize) {
    let (mut puct, mut gumbel, mut draw) = (0, 0, 0);
    for game_idx in 0..n_games {
        let puct_is_p1 = game_idx % 2 == 0;
        let winner = play_game(model, puct_is_p1, num_sims, max_nodes, rng);
        let puct_color = if puct_is_p1 { 1 } else { 2 };
        let gumbel_color = 3 - puct_color;
        let tag = if winner == puct_color {
            puct += 1;
            "PUCT"
        } else if winner == gumbel_color {
            gumbel += 1;
            "Gumbel"
        } else {
            draw += 1;
            "Draw"
        };
        println!(
            "  Game {:>3}/{}: {:>6}  (PUCT={}  Gumbel={}  Draw={})",
            game_idx + 1,
            n_games,
            tag,
            puct,
            gumbel,
            draw
        );
    }
    (puct, gumbel, draw)
}

fn print_results(results: (usize, usize, usize), n_games: usize) {
    let (p, g, d) = results;
    println!("\n{}", "=".repeat(50));
    println!("PUCT wins:    {:>3}  ({:.1}%)", p, 100.0 * p as f64 / n_games as f64);
    println!("Gumbel wins:  {:>3}  ({:.1}%)", g, 100.0 * g as f64 / n_games as f64);
    println!("Draws:        {:>3}  ({:.1}%)", d, 100.0 * d as f64 / n_games as f64);
    println!("{}", "=".repeat(50));
}

fn main() {
    let n_games = 30;
    let num_sims = 50;
    let max_nodes = 2000;
    let mut rng = StdRng::seed_from_u64(0);

    println!("Round 1: RANDOM MODEL (uniform policy)");
    println!("PUCT vs GumbelSparse  |  Gomoku 15x15  |  {num_sims} sims/move  |  {n_games} games\n");
    let random_model = RandomGomoku;
    let r1 = run_match(n_games, num_sims, max_nodes, &random_model, &mut rng);
    print_results(r1, n_games);

    println!("\n\nRound 2: HEURISTIC MODEL (simulates trained network)");
    println!("PUCT vs GumbelSparse  |  Gomoku 15x15  |  {num_sims} sims/move  |  {n_games} games\n");
    let heuristic = HeuristicGomoku::new(0.0, StdRng::seed_from_u64(1));
    let r2 = run_match(n_games, num_sims, max_nodes, &heuristic, &mut rng);
    print_results(r2, n_games);
}
