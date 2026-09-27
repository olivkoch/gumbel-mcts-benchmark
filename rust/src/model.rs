//! Evaluation models for MCTS. Port of the models used by the Python
//! benchmarks: a Burn MLP (matching the `nn.Sequential` nets), plus the
//! heuristic/random Gomoku models from the win-rate benchmarks.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use rand::Rng;

/// Batched policy/value evaluator.
///
/// `boards` is `B × BOARD_LEN` row-major i8; `players` is `B` i8.
/// Returns `(policy, value)` where policy is `B × NUM_ACTIONS` row-major f32
/// (softmax probabilities) and value is `B` f32 from the side-to-move's
/// perspective.
pub trait EvalModel {
    fn forward(&self, boards: &[i8], players: &[i8]) -> (Vec<f32>, Vec<f32>);
}

// =============================================================================
// Burn MLP — mirrors benchmark_throughput.py's TicTacToeModel / GomokuModel
// =============================================================================

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
    #[module(skip)]
    pub board_len: usize,
    #[module(skip)]
    pub num_actions: usize,
    #[module(skip)]
    device: B::Device,
}

impl<B: Backend> Mlp<B> {
    pub fn new(board_len: usize, hidden: usize, num_actions: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(board_len, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            policy_head: LinearConfig::new(hidden, num_actions).init(device),
            value_head: LinearConfig::new(hidden, 1).init(device),
            board_len,
            num_actions,
            device: device.clone(),
        }
    }

    fn infer(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = relu(self.fc2.forward(relu(self.fc1.forward(x))));
        let policy = softmax(self.policy_head.forward(h.clone()), 1);
        let value = self.value_head.forward(h).tanh();
        (policy, value)
    }
}

impl<B: Backend> EvalModel for Mlp<B> {
    fn forward(&self, boards: &[i8], _players: &[i8]) -> (Vec<f32>, Vec<f32>) {
        let b = boards.len() / self.board_len;
        let data: Vec<f32> = boards.iter().map(|&v| v as f32).collect();
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(data, [b, self.board_len]),
            &self.device,
        );
        let (p, v) = self.infer(x);
        let policy: Vec<f32> = p.into_data().to_vec().expect("policy f32");
        let value: Vec<f32> = v.reshape([b]).into_data().to_vec().expect("value f32");
        (policy, value)
    }
}

// =============================================================================
// Random model — uniform policy over legal moves, zero value
// =============================================================================

pub struct RandomGomoku;

impl EvalModel for RandomGomoku {
    fn forward(&self, boards: &[i8], _players: &[i8]) -> (Vec<f32>, Vec<f32>) {
        let b = boards.len() / 225;
        let mut policy = vec![0.0f32; b * 225];
        let value = vec![0.0f32; b];
        for i in 0..b {
            let board = &boards[i * 225..(i + 1) * 225];
            let n_legal = board.iter().filter(|&&v| v == 0).count();
            if n_legal > 0 {
                let p = 1.0 / n_legal as f32;
                for a in 0..225 {
                    if board[a] == 0 {
                        policy[i * 225 + a] = p;
                    }
                }
            }
        }
        (policy, value)
    }
}

// =============================================================================
// Heuristic model — port of HeuristicGomokuModel from benchmark_winrate.py /
// benchmark_sparse_gumbel_efficiency.py
// =============================================================================

pub struct HeuristicGomoku<R: Rng> {
    pub noise_scale: f32,
    pub rng: std::cell::RefCell<R>,
}

impl<R: Rng> HeuristicGomoku<R> {
    pub fn new(noise_scale: f32, rng: R) -> Self {
        Self { noise_scale, rng: std::cell::RefCell::new(rng) }
    }

    fn score_board(&self, board: &[i8], player: i8) -> ([f32; 225], f64) {
        const BS: usize = 15;
        let opponent = 3 - player;
        let mut scores = [0.0f32; 225];

        let occupied = board.iter().any(|&v| v != 0);
        if occupied {
            for r in 0..BS {
                for c in 0..BS {
                    if board[r * BS + c] != 0 {
                        continue;
                    }
                    for dr in -2i32..=2 {
                        for dc in -2i32..=2 {
                            let rr = r as i32 + dr;
                            let cc = c as i32 + dc;
                            if rr >= 0 && rr < BS as i32 && cc >= 0 && cc < BS as i32
                                && board[rr as usize * BS + cc as usize] != 0
                            {
                                let dist = dr.abs().max(dc.abs()) as f32;
                                scores[r * BS + c] += 2.0 / dist;
                            }
                        }
                    }
                }
            }
        } else {
            for r in 0..BS {
                for c in 0..BS {
                    let dist_center = (r as i32 - 7).abs() + (c as i32 - 7).abs();
                    scores[r * BS + c] = (7 - dist_center).max(0) as f32;
                }
            }
        }

        let directions = [(0i32, 1i32), (1, 0), (1, 1), (1, -1)];
        let mut my_threat = 0.0f64;
        let mut opp_threat = 0.0f64;

        for r in 0..BS {
            for c in 0..BS {
                if board[r * BS + c] != 0 {
                    continue;
                }
                let action = r * BS + c;
                for (dr, dc) in directions {
                    for (p, multiplier) in [(player, 3.0f32), (opponent, 2.5f32)] {
                        let mut count = 0;
                        for sign in [1i32, -1] {
                            let mut rr = r as i32 + sign * dr;
                            let mut cc = c as i32 + sign * dc;
                            while rr >= 0
                                && rr < BS as i32
                                && cc >= 0
                                && cc < BS as i32
                                && board[rr as usize * BS + cc as usize] == p
                            {
                                count += 1;
                                rr += sign * dr;
                                cc += sign * dc;
                            }
                        }
                        if count >= 4 {
                            scores[action] += multiplier * 50.0;
                        } else if count >= 3 {
                            scores[action] += multiplier * 10.0;
                        } else if count >= 2 {
                            scores[action] += multiplier * 3.0;
                        } else if count >= 1 {
                            scores[action] += multiplier;
                        }
                        if p == player {
                            my_threat += count as f64;
                        } else {
                            opp_threat += count as f64;
                        }
                    }
                }
            }
        }

        for i in 0..225 {
            if board[i] != 0 {
                scores[i] = -1e9;
            }
        }

        let value = (0.05 * (my_threat - opp_threat)).tanh();
        (scores, value)
    }
}

impl<R: Rng> EvalModel for HeuristicGomoku<R> {
    fn forward(&self, boards: &[i8], players: &[i8]) -> (Vec<f32>, Vec<f32>) {
        let b = boards.len() / 225;
        let mut policy = vec![0.0f32; b * 225];
        let mut value = vec![0.0f32; b];
        let mut rng = self.rng.borrow_mut();
        for i in 0..b {
            let board = &boards[i * 225..(i + 1) * 225];
            let (mut logits, val) = self.score_board(board, players[i]);

            if self.noise_scale > 0.0 {
                for a in 0..225 {
                    if logits[a] > -1e8 {
                        // Standard normal via Box-Muller.
                        let u1: f32 = rng.gen::<f32>().max(1e-12);
                        let u2: f32 = rng.gen();
                        logits[a] += (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                            * self.noise_scale;
                    }
                }
            }

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for a in 0..225 {
                logits[a] = (logits[a] - max_l).exp();
                sum += logits[a];
            }
            for a in 0..225 {
                policy[i * 225 + a] = logits[a] / sum;
            }
            value[i] = val as f32;
        }
        (policy, value)
    }
}
