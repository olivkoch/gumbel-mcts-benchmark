//! Gomoku environment compatible with the V2 reference search.
//! Port of `src/utils/gomoku_env.py`.

use crate::game::{GameLogic, Gomoku};

#[derive(Clone)]
pub struct GomokuEnv {
    pub board: [i8; 225],
    pub to_play: i8,
    pub steps: usize,
    pub winner: Option<i8>,
    pub done: bool,
    pub last_player: i8,
}

impl GomokuEnv {
    pub fn new() -> Self {
        Self {
            board: [0; 225],
            to_play: Gomoku::PLAYER_1,
            steps: 0,
            winner: None,
            done: false,
            last_player: 0,
        }
    }

    pub fn action_dim(&self) -> usize {
        Gomoku::NUM_ACTIONS
    }

    pub fn opponent_player(&self) -> i8 {
        3 - self.to_play
    }

    /// Legal-action mask as f32 (1.0 = legal), matching the Python env.
    pub fn legal_actions(&self) -> [f32; 225] {
        let mut mask = [false; 225];
        Gomoku::valid_mask(&self.board, &mut mask);
        let mut out = [0.0f32; 225];
        for i in 0..225 {
            out[i] = if mask[i] { 1.0 } else { 0.0 };
        }
        out
    }

    pub fn step(&mut self, action: usize) -> (f64, bool) {
        assert!(!self.done, "Game is over, call reset.");
        let legal = self.legal_actions();
        assert!(legal[action] >= 1.0, "Illegal action {action}");

        self.last_player = self.to_play;
        let res = Gomoku::fast_step(&mut self.board, action, self.to_play);
        self.to_play = 3 - self.to_play;
        self.steps += 1;
        self.done = res.done;

        if res.done {
            self.winner = Some(res.winner);
            let r = if res.winner == self.last_player {
                1.0
            } else if res.winner == 0 {
                0.0
            } else {
                -1.0
            };
            (r, true)
        } else {
            (0.0, false)
        }
    }

    /// Flat observation: [board (225), current_player (1)].
    pub fn observation(&self) -> [f32; 226] {
        let mut obs = [0.0f32; 226];
        for i in 0..225 {
            obs[i] = self.board[i] as f32;
        }
        obs[225] = self.to_play as f32;
        obs
    }

    pub fn is_game_over(&self) -> bool {
        self.done
    }
}
