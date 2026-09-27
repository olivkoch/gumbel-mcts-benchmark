//! Smoke tests for the Rust port.

use gumbel_mcts_bench::game::{GameLogic, Gomoku, TicTacToe};
use gumbel_mcts_bench::gumbel_dense::GumbelDense;
use gumbel_mcts_bench::gumbel_sparse::GumbelSparse;
use gumbel_mcts_bench::model::{EvalModel, HeuristicGomoku, RandomGomoku};
use gumbel_mcts_bench::puct::Puct;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn gomoku_win_detection() {
    let mut board = [0i8; 225];
    for i in 0..4 {
        board[7 * 15 + i] = 1;
    }
    let res = Gomoku::fast_step(&mut board, 7 * 15 + 4, 1);
    assert!(res.done && res.winner == 1 && res.reward == 1.0);
}

#[test]
fn gomoku_diagonal_win() {
    let mut board = [0i8; 225];
    for i in 0..4 {
        board[i * 15 + i] = 2;
    }
    let res = Gomoku::fast_step(&mut board, 4 * 15 + 4, 2);
    assert!(res.done && res.winner == 2);
}

#[test]
fn ttt_win_detection() {
    let mut board = [0i8; 9];
    board[0] = 1;
    board[1] = 1;
    let res = TicTacToe::fast_step(&mut board, 2, 1);
    assert!(res.done && res.winner == 1);
}

/// PUCT with the heuristic model should block an open four.
#[test]
fn puct_blocks_open_four() {
    let model = HeuristicGomoku::new(0.0, StdRng::seed_from_u64(0));
    // Opponent (player 2) has four in a row at row 7, cols 3-6.
    let mut board = [0i8; 225];
    for c in 3..7 {
        board[7 * 15 + c] = 2;
    }
    board[6 * 15 + 7] = 1;

    let mut tree = Puct::<Gomoku>::new(1, 500);
    tree.initialize_roots(&[0], &board, &[1]);
    tree.run_simulation_batch(&model, &[0], 50, 19652.0, 1.25);
    let (visits, _) = tree.get_all_root_data(1);
    let best = (0..225).max_by(|&a, &b| visits[a].total_cmp(&visits[b])).unwrap();
    assert!(
        best == 7 * 15 + 2 || best == 7 * 15 + 7,
        "expected blocking move at (7,2) or (7,7), got action {best}"
    );
    // Root should have received sims+1 visits.
    let root = tree.root_indices[0] as usize;
    assert!(tree.visit_counts[root] >= 51);
}

/// GumbelSparse with the heuristic model should also block.
#[test]
fn gumbel_sparse_blocks_open_four() {
    let model = HeuristicGomoku::new(0.0, StdRng::seed_from_u64(0));
    let mut rng = StdRng::seed_from_u64(42);
    let mut board = [0i8; 225];
    for c in 3..7 {
        board[7 * 15 + c] = 2;
    }
    board[6 * 15 + 7] = 1;

    let mut tree = GumbelSparse::<Gomoku>::new(1, 500, 50.0, 1.0, 35, 256);
    tree.initialize_roots(&[0], &board, &[1]);
    let moves = tree.run_simulation_batch(&model, &[0], 50, &mut rng, None);
    let mv = moves[0] as usize;
    assert!(
        mv == 7 * 15 + 2 || mv == 7 * 15 + 7,
        "expected blocking move, got action {mv}"
    );
}

#[test]
fn gumbel_dense_returns_legal_move() {
    let model = RandomGomoku;
    let mut rng = StdRng::seed_from_u64(3);
    let board = [0i8; 225];
    let mut tree = GumbelDense::<Gomoku>::new(1, 500, 50.0, 1.0);
    tree.initialize_roots(&[0], &board, &[1], &mut rng);
    let moves = tree.run_simulation_batch(&model, &[0], 50, &mut rng);
    assert!(moves[0] >= 0 && moves[0] < 225);
}
