//! Game logic for Gomoku (15x15) and Tic-Tac-Toe (3x3).
//! Direct port of `src/game_logic/gomoku.py` and `tictactoe.py`.

pub struct StepResult {
    /// Reward from the perspective of the player who just moved.
    pub reward: f64,
    /// Winning player id (0 = draw/none).
    pub winner: i8,
    pub done: bool,
}

pub trait GameLogic: Copy + Send + Sync + 'static {
    const NUM_ACTIONS: usize;
    const BOARD_LEN: usize;
    const MAX_MOVES: usize;
    const MAX_LEGAL: usize;
    const NAME: &'static str;
    const PLAYER_1: i8 = 1;
    const PLAYER_2: i8 = 2;

    /// Apply `action` for `player`, mutating `board` in place.
    fn fast_step(board: &mut [i8], action: usize, player: i8) -> StepResult;

    /// Fill `mask` (length NUM_ACTIONS) with legal-move flags.
    fn valid_mask(board: &[i8], mask: &mut [bool]);

    fn other_player(player: i8) -> i8 {
        if player == Self::PLAYER_1 {
            Self::PLAYER_2
        } else {
            Self::PLAYER_1
        }
    }
}

// =============================================================================
// Gomoku
// =============================================================================

#[derive(Clone, Copy)]
pub struct Gomoku;

const GOMOKU_SIZE: usize = 15;

fn count_direction(board: &[i8], r: isize, c: isize, dr: isize, dc: isize, player: i8) -> usize {
    let mut count = 0;
    let mut rr = r + dr;
    let mut cc = c + dc;
    while rr >= 0 && rr < GOMOKU_SIZE as isize && cc >= 0 && cc < GOMOKU_SIZE as isize {
        if board[rr as usize * GOMOKU_SIZE + cc as usize] != player {
            break;
        }
        count += 1;
        rr += dr;
        cc += dc;
    }
    count
}

fn check_win(board: &[i8], r: usize, c: usize, player: i8) -> bool {
    for (dr, dc) in [(0isize, 1isize), (1, 0), (1, 1), (1, -1)] {
        let total = 1
            + count_direction(board, r as isize, c as isize, dr, dc, player)
            + count_direction(board, r as isize, c as isize, -dr, -dc, player);
        if total >= 5 {
            return true;
        }
    }
    false
}

impl GameLogic for Gomoku {
    const NUM_ACTIONS: usize = 225;
    const BOARD_LEN: usize = 225;
    const MAX_MOVES: usize = 225;
    const MAX_LEGAL: usize = 225;
    const NAME: &'static str = "gomoku";

    fn fast_step(board: &mut [i8], action: usize, player: i8) -> StepResult {
        let r = action / GOMOKU_SIZE;
        let c = action % GOMOKU_SIZE;
        board[r * GOMOKU_SIZE + c] = player;

        if check_win(board, r, c, player) {
            return StepResult { reward: 1.0, winner: player, done: true };
        }
        if board.iter().all(|&v| v != 0) {
            return StepResult { reward: 0.0, winner: 0, done: true };
        }
        StepResult { reward: 0.0, winner: 0, done: false }
    }

    fn valid_mask(board: &[i8], mask: &mut [bool]) {
        for i in 0..Self::NUM_ACTIONS {
            mask[i] = board[i] == 0;
        }
    }
}

// =============================================================================
// Tic-Tac-Toe
// =============================================================================

#[derive(Clone, Copy)]
pub struct TicTacToe;

fn line_match(board: &[i8], a: usize, b: usize, c: usize, p: i8) -> bool {
    board[a] == p && board[b] == p && board[c] == p
}

impl GameLogic for TicTacToe {
    const NUM_ACTIONS: usize = 9;
    const BOARD_LEN: usize = 9;
    const MAX_MOVES: usize = 9;
    const MAX_LEGAL: usize = 9;
    const NAME: &'static str = "tictactoe";

    fn fast_step(board: &mut [i8], action: usize, player: i8) -> StepResult {
        board[action] = player;
        // Same check order as the Python kernel: for each player, row i +
        // column i per i, then both diagonals.
        for p in [Self::PLAYER_1, Self::PLAYER_2] {
            for i in 0..3 {
                if line_match(board, i * 3, i * 3 + 1, i * 3 + 2, p)
                    || line_match(board, i, i + 3, i + 6, p)
                {
                    return StepResult {
                        reward: if p == player { 1.0 } else { -1.0 },
                        winner: p,
                        done: true,
                    };
                }
            }
            if line_match(board, 0, 4, 8, p) || line_match(board, 2, 4, 6, p) {
                return StepResult {
                    reward: if p == player { 1.0 } else { -1.0 },
                    winner: p,
                    done: true,
                };
            }
        }
        if board.iter().all(|&v| v != 0) {
            return StepResult { reward: 0.0, winner: 0, done: true };
        }
        StepResult { reward: 0.0, winner: 0, done: false }
    }

    fn valid_mask(board: &[i8], mask: &mut [bool]) {
        for i in 0..Self::NUM_ACTIONS {
            mask[i] = board[i] == 0;
        }
    }
}
