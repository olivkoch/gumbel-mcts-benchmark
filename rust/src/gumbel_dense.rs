//! Gumbel MCTS with dense storage. Port of `src/gumbel_mcts/gumbel_dense.py`
//! and `src/kernels/gumbel_dense_kernels.py`.

use crate::game::GameLogic;
use crate::model::EvalModel;
use crate::puct::Puct;
use rand::Rng;

pub struct GumbelDense<L: GameLogic> {
    pub tree: Puct<L>,
    /// (n_games, NUM_ACTIONS) Gumbel noise sampled at root init.
    pub gumbel_noises: Vec<f32>,
    /// (n_games, NUM_ACTIONS) log-priors at the root.
    pub root_logits: Vec<f32>,
    /// (n_games, NUM_ACTIONS) legal moves at the root.
    pub root_legal_masks: Vec<bool>,
    /// (n_games,) NN value estimate at the root (for v_mix).
    pub root_nn_values: Vec<f32>,
    pub c_visit: f64,
    pub c_scale: f64,
}

impl<L: GameLogic> GumbelDense<L> {
    pub fn new(n_games: usize, max_nodes: usize, c_visit: f64, c_scale: f64) -> Self {
        let a = L::NUM_ACTIONS;
        Self {
            tree: Puct::<L>::new(n_games, max_nodes),
            gumbel_noises: vec![0.0; n_games * a],
            root_logits: vec![0.0; n_games * a],
            root_legal_masks: vec![false; n_games * a],
            root_nn_values: vec![0.0; n_games],
            c_visit,
            c_scale,
        }
    }

    pub fn reset(&mut self) {
        self.tree.reset();
    }

    pub fn initialize_roots<R: Rng>(
        &mut self,
        active_games: &[usize],
        boards: &[i8],
        players: &[i8],
        rng: &mut R,
    ) {
        self.tree.initialize_roots(active_games, boards, players);
        // Fresh Gumbel noise per game: G = -log(-log(U)).
        let a = L::NUM_ACTIONS;
        for i in 0..active_games.len() {
            for m in 0..a {
                let u: f32 = rng
                    .gen::<f32>()
                    .clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
                self.gumbel_noises[i * a + m] = -(-u.ln()).ln();
            }
        }
    }

    pub fn run_simulation_batch<M: EvalModel, R: Rng>(
        &mut self,
        model: &M,
        active_games: &[usize],
        num_simulations: usize,
        _rng: &mut R,
    ) -> Vec<i32> {
        let a = L::NUM_ACTIONS;
        let n_active = active_games.len();
        let game_indices: Vec<i32> = active_games.iter().map(|&g| g as i32).collect();

        // 1. Initial root expansion (fetch logits).
        self.expand_roots(model, active_games);

        // 2. Sequential halving setup.
        let max_k = a.min(16);
        let mut num_phases = ((max_k as f64).log2() as usize).max(1);
        let first_phase_budget = num_simulations / num_phases;
        let k_initial = max_k.min(first_phase_budget / 2).max(2);
        num_phases = ((k_initial as f64).log2() as usize).max(1);

        let mut candidate_mask =
            self.initial_gumbel_candidates(&game_indices, k_initial, n_active);

        // 3. Halving loop.
        let mut remaining = num_simulations;
        for phase in 0..num_phases {
            let k_phase = (k_initial / (1 << phase)).max(1);
            let phases_left = num_phases - phase;
            let budget_this_phase = if phase == num_phases - 1 {
                remaining
            } else {
                remaining / phases_left
            };
            let sims_per_action = (budget_this_phase / k_phase).max(1);

            for candidate_rank in 0..k_phase {
                let root_moves =
                    forced_root_moves(&candidate_mask, candidate_rank, n_active, a);
                for _ in 0..sims_per_action {
                    let leaf_indices =
                        self.descend_tree(&game_indices, &root_moves);
                    self.evaluate_and_backprop(model, &leaf_indices);
                }
            }

            remaining = remaining.saturating_sub(k_phase * sims_per_action);

            if phase < num_phases - 1 {
                candidate_mask = self.halve_candidates(&game_indices, &candidate_mask);
            }
        }

        self.final_survivors(&game_indices, &candidate_mask, n_active)
    }

    /// Port of `_expand_roots_v4`.
    fn expand_roots<M: EvalModel>(&mut self, model: &M, active_games: &[usize]) {
        let a = L::NUM_ACTIONS;
        let bl = L::BOARD_LEN;
        let n_active = active_games.len();
        let root_indices: Vec<usize> = active_games
            .iter()
            .map(|&g| self.tree.root_indices[g] as usize)
            .collect();

        let mut boards = vec![0i8; n_active * bl];
        let mut players = vec![0i8; n_active];
        for (i, &r) in root_indices.iter().enumerate() {
            boards[i * bl..(i + 1) * bl].copy_from_slice(&self.tree.boards[r * bl..(r + 1) * bl]);
            players[i] = self.tree.players[r];
        }

        let (probs, vals) = model.forward(&boards, &players);

        let mut mask = [false; 256];
        for i in 0..n_active {
            let r = root_indices[i];
            for m in 0..a {
                self.tree.prior_probs[r * a + m] = probs[i * a + m] as f64;
                self.root_logits[i * a + m] = (probs[i * a + m] + 1e-10).ln();
            }
            self.root_nn_values[i] = vals[i];
            L::valid_mask(&self.tree.boards[r * bl..(r + 1) * bl], &mut mask[..a]);
            self.root_legal_masks[i * a..(i + 1) * a].copy_from_slice(&mask[..a]);
            self.tree.is_expanded[r] = true;
        }

        let ri32: Vec<i32> = root_indices.iter().map(|&r| r as i32).collect();
        let vals64: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        self.tree.backpropagate_batch(&ri32, &vals64);
    }

    /// Port of `_evaluate_and_backprop_v3`.
    fn evaluate_and_backprop<M: EvalModel>(&mut self, model: &M, leaf_indices: &[i32]) {
        self.tree.evaluate_leaves(model, leaf_indices);
        let leaf_values: Vec<f64> = leaf_indices
            .iter()
            .map(|&l| self.tree.last_eval(l as usize))
            .collect();
        self.tree.backpropagate_batch(leaf_indices, &leaf_values);
    }

    /// Port of `descend_tree_kernel` (Gumbel variant).
    fn descend_tree(&mut self, game_indices: &[i32], root_moves: &[i32]) -> Vec<i32> {
        let a = L::NUM_ACTIONS;
        let bl = L::BOARD_LEN;
        let n_active = game_indices.len();
        let mut leaf_indices = vec![0i32; n_active];

        let mut mask = [false; 256];
        let mut new_board = [0i8; 256];
        let mut node_q = [0.0f64; 256];
        let mut combined = [0.0f64; 256];
        let mut pi_prime = [0.0f64; 256];

        for i in 0..n_active {
            let mut node_idx = self.tree.root_indices[game_indices[i] as usize] as usize;
            let mut search_depth = 0usize;
            let mut move_to_take = root_moves[i];

            loop {
                if self.tree.is_terminal[node_idx]
                    || !self.tree.is_expanded[node_idx]
                    || search_depth >= L::MAX_MOVES
                {
                    break;
                }
                search_depth += 1;

                if search_depth > 1 {
                    // Interior node: deterministic π'-based selection (Eq. 14).
                    L::valid_mask(
                        &self.tree.boards[node_idx * bl..(node_idx + 1) * bl],
                        &mut mask[..a],
                    );
                    let parent_n = self.tree.visit_counts[node_idx];
                    let v_node = if parent_n > 0 {
                        self.tree.values[node_idx] / parent_n as f64
                    } else {
                        0.0
                    };

                    let mut sum_child_n = 0i64;
                    let mut max_child_n = 0i32;
                    for act in 0..a {
                        let c_idx = self.tree.children[node_idx * a + act];
                        if c_idx != -1 {
                            let n_c = self.tree.visit_counts[c_idx as usize];
                            sum_child_n += n_c as i64;
                            if n_c > 0 {
                                node_q[act] = -self.tree.values[c_idx as usize] / n_c as f64;
                                if n_c > max_child_n {
                                    max_child_n = n_c;
                                }
                            } else {
                                node_q[act] = v_node;
                            }
                        } else {
                            node_q[act] = v_node;
                        }
                    }

                    let sigma_scale = (self.c_visit + max_child_n as f64) * self.c_scale;

                    let mut q_min = 1e10f64;
                    let mut q_max = -1e10f64;
                    for act in 0..a {
                        if mask[act] {
                            q_min = q_min.min(node_q[act]);
                            q_max = q_max.max(node_q[act]);
                        }
                    }
                    let q_range = if q_max - q_min < 1e-6 { 1.0 } else { q_max - q_min };

                    let mut max_combined = -1e10f64;
                    for act in 0..a {
                        if mask[act] {
                            let log_prior =
                                (self.tree.prior_probs[node_idx * a + act] + 1e-10).ln();
                            let q_norm = (node_q[act] - q_min) / q_range;
                            combined[act] = log_prior + sigma_scale * q_norm;
                            if combined[act] > max_combined {
                                max_combined = combined[act];
                            }
                        } else {
                            combined[act] = -1e10;
                        }
                    }

                    let mut exp_sum = 0.0f64;
                    for act in 0..a {
                        if mask[act] {
                            pi_prime[act] = (combined[act] - max_combined).exp();
                            exp_sum += pi_prime[act];
                        } else {
                            pi_prime[act] = 0.0;
                        }
                    }
                    if exp_sum > 0.0 {
                        for act in 0..a {
                            pi_prime[act] /= exp_sum;
                        }
                    }

                    let denom = 1.0 + sum_child_n as f64;
                    let mut best_score = -1e10f64;
                    move_to_take = -1;
                    for act in 0..a {
                        if !mask[act] {
                            continue;
                        }
                        let c_idx = self.tree.children[node_idx * a + act];
                        let n_a = if c_idx != -1 {
                            self.tree.visit_counts[c_idx as usize] as f64
                        } else {
                            0.0
                        };
                        let score = pi_prime[act] - n_a / denom;
                        if score > best_score {
                            best_score = score;
                            move_to_take = act as i32;
                        }
                    }
                }

                // Transition to child.
                if move_to_take < 0 {
                    leaf_indices[i] = node_idx as i32;
                    break;
                }
                let child_idx = self.tree.children[node_idx * a + move_to_take as usize];
                if child_idx == -1 {
                    let new_idx = self.tree.next_free_idx;
                    if new_idx >= self.tree.max_nodes {
                        leaf_indices[i] = node_idx as i32;
                        break;
                    }
                    self.tree.next_free_idx += 1;

                    new_board[..bl].copy_from_slice(
                        &self.tree.boards[node_idx * bl..(node_idx + 1) * bl],
                    );
                    let cur_player = self.tree.players[node_idx];
                    let res =
                        L::fast_step(&mut new_board[..bl], move_to_take as usize, cur_player);
                    let t_val = if res.done {
                        if res.reward == 1.0 {
                            -1.0
                        } else if res.reward == -1.0 {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let next_p = L::other_player(cur_player);

                    self.tree.init_node_pub(
                        new_idx,
                        node_idx as i32,
                        move_to_take as i16,
                        &new_board[..bl],
                        next_p,
                        t_val,
                        res.done,
                    );
                    self.tree.children[node_idx * a + move_to_take as usize] = new_idx as i32;
                    node_idx = new_idx;
                    break;
                } else {
                    node_idx = child_idx as usize;
                }
            }
            leaf_indices[i] = node_idx as i32;
        }
        leaf_indices
    }

    /// Port of `get_gumbel_score_kernel`.
    fn gumbel_scores(
        &self,
        game_indices: &[i32],
        candidate_mask: &[bool],
    ) -> Vec<f32> {
        let a = L::NUM_ACTIONS;
        let n_active = game_indices.len();
        let mut scores = vec![-1e10f32; n_active * a];

        for i in 0..n_active {
            let g_idx = game_indices[i] as usize;
            let r_idx = self.tree.root_indices[g_idx] as usize;

            let mut max_n = 0i32;
            let mut q_min = 1e10f64;
            let mut q_max = -1e10f64;

            // v_mix (Eq. 33)
            let v_hat = self.root_nn_values[i] as f64;
            let sum_n = (self.tree.visit_counts[r_idx] - 1) as f64;
            let mut sum_weighted_q = 0.0f64;
            let mut sum_pi_visited = 0.0f64;

            for m in 0..a {
                let c_idx = self.tree.children[r_idx * a + m];
                if c_idx != -1 {
                    let n_c = self.tree.visit_counts[c_idx as usize];
                    if n_c > max_n {
                        max_n = n_c;
                    }
                    if n_c > 0 {
                        let q_c = -self.tree.values[c_idx as usize] / n_c as f64;
                        q_min = q_min.min(q_c);
                        q_max = q_max.max(q_c);
                        let pi_a = self.tree.prior_probs[r_idx * a + m];
                        sum_weighted_q += pi_a * q_c;
                        sum_pi_visited += pi_a;
                    }
                }
            }

            let v_mix = if sum_pi_visited > 1e-10 && sum_n > 0.0 {
                (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
            } else {
                v_hat
            };

            q_min = q_min.min(v_mix);
            q_max = q_max.max(v_mix);
            let q_range = if q_max - q_min < 1e-6 { 1.0 } else { q_max - q_min };
            let dynamic_q_scale = (self.c_visit + max_n as f64) * self.c_scale;

            for m in 0..a {
                if !candidate_mask[i * a + m] {
                    continue;
                }
                let c_idx = self.tree.children[r_idx * a + m];
                let n_v = if c_idx != -1 {
                    self.tree.visit_counts[c_idx as usize]
                } else {
                    0
                };
                let q_v = if n_v > 0 {
                    -self.tree.values[c_idx as usize] / n_v as f64
                } else {
                    v_mix
                };
                let q_normalized = (q_v - q_min) / q_range;
                scores[i * a + m] = (self.root_logits[g_idx * a + m] as f64
                    + self.gumbel_noises[g_idx * a + m] as f64
                    + dynamic_q_scale * q_normalized) as f32;
            }
        }
        scores
    }

    fn halve_candidates(&self, game_indices: &[i32], candidate_mask: &[bool]) -> Vec<bool> {
        let a = L::NUM_ACTIONS;
        let n_active = game_indices.len();
        let scores = self.gumbel_scores(game_indices, candidate_mask);
        let mut new_mask = vec![false; n_active * a];

        for i in 0..n_active {
            let active_moves: Vec<usize> = (0..a).filter(|&m| candidate_mask[i * a + m]).collect();
            if active_moves.len() <= 1 {
                for m in 0..a {
                    new_mask[i * a + m] = candidate_mask[i * a + m];
                }
                continue;
            }
            let num_to_keep = (active_moves.len() / 2).max(1);
            // Top-num_to_keep by score.
            let mut scored: Vec<(usize, f32)> = active_moves
                .iter()
                .map(|&m| (m, scores[i * a + m]))
                .collect();
            scored.sort_by(|x, y| x.1.total_cmp(&y.1));
            for (m, _) in scored.iter().skip(active_moves.len() - num_to_keep) {
                new_mask[i * a + m] = true;
            }
        }
        new_mask
    }

    fn final_survivors(
        &self,
        game_indices: &[i32],
        candidate_mask: &[bool],
        n_active: usize,
    ) -> Vec<i32> {
        let a = L::NUM_ACTIONS;
        let scores = self.gumbel_scores(game_indices, candidate_mask);
        let mut survivors = vec![0i32; n_active];
        for i in 0..n_active {
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for m in 0..a {
                if scores[i * a + m] > best_s {
                    best_s = scores[i * a + m];
                    best = m;
                }
            }
            survivors[i] = best as i32;
        }
        survivors
    }

    /// Port of `_get_initial_gumbel_candidates`.
    fn initial_gumbel_candidates(
        &self,
        _game_indices: &[i32],
        k: usize,
        n_active: usize,
    ) -> Vec<bool> {
        let a = L::NUM_ACTIONS;
        let mut mask = vec![false; n_active * a];
        for i in 0..n_active {
            let mut legal_scored: Vec<(usize, f32)> = Vec::new();
            for m in 0..a {
                if self.root_legal_masks[i * a + m] {
                    legal_scored
                        .push((m, self.root_logits[i * a + m] + self.gumbel_noises[i * a + m]));
                }
            }
            if legal_scored.is_empty() {
                continue;
            }
            let k_actual = k.min(legal_scored.len());
            legal_scored.sort_by(|x, y| x.1.total_cmp(&y.1));
            for (m, _) in legal_scored.iter().skip(legal_scored.len() - k_actual) {
                mask[i * a + m] = true;
            }
        }
        mask
    }

    /// Port of `compute_gumbel_policy_kernel` — improved policy π' for training.
    pub fn get_improved_policy(&self, n_active: usize) -> Vec<f32> {
        let a = L::NUM_ACTIONS;
        let mut out = vec![0.0f32; n_active * a];

        for i in 0..n_active {
            let r_idx = self.tree.root_indices[i] as usize;
            let v_hat = self.root_nn_values[i] as f64;

            let sum_n = (self.tree.visit_counts[r_idx] - 1) as f64;
            let mut sum_weighted_q = 0.0f64;
            let mut sum_pi_visited = 0.0f64;
            let mut max_n = 0i32;
            let mut q_values = vec![0.0f64; a];

            for m in 0..a {
                let c_idx = self.tree.children[r_idx * a + m];
                if c_idx != -1 {
                    let n_v = self.tree.visit_counts[c_idx as usize];
                    if n_v > 0 {
                        let q_v = -self.tree.values[c_idx as usize] / n_v as f64;
                        q_values[m] = q_v;
                        let pi_a = self.tree.prior_probs[r_idx * a + m];
                        sum_weighted_q += pi_a * q_v;
                        sum_pi_visited += pi_a;
                        if n_v > max_n {
                            max_n = n_v;
                        }
                    }
                }
            }

            let v_mix = if sum_pi_visited > 1e-10 && sum_n > 0.0 {
                (1.0 / (1.0 + sum_n)) * (v_hat + (sum_n / sum_pi_visited) * sum_weighted_q)
            } else {
                v_hat
            };

            let sigma_scale = (self.c_visit + max_n as f64) * self.c_scale;

            let mut q_min = 1e10f64;
            let mut q_max = -1e10f64;
            for m in 0..a {
                if !self.root_legal_masks[i * a + m] {
                    continue;
                }
                let c_idx = self.tree.children[r_idx * a + m];
                let q_completed = if c_idx != -1 && self.tree.visit_counts[c_idx as usize] > 0 {
                    q_values[m]
                } else {
                    v_mix
                };
                q_values[m] = q_completed;
                q_min = q_min.min(q_completed);
                q_max = q_max.max(q_completed);
            }

            let q_range = if q_max - q_min < 1e-6 { 1.0 } else { q_max - q_min };
            let mut max_combined = -1e10f64;
            for m in 0..a {
                if !self.root_legal_masks[i * a + m] {
                    continue;
                }
                let q_norm = (q_values[m] - q_min) / q_range;
                let combined_v = self.root_logits[i * a + m] as f64 + sigma_scale * q_norm;
                out[i * a + m] = combined_v as f32;
                if combined_v > max_combined {
                    max_combined = combined_v;
                }
            }

            let mut exp_sum = 0.0f32;
            for m in 0..a {
                if self.root_legal_masks[i * a + m] {
                    out[i * a + m] = (out[i * a + m] as f64 - max_combined).exp() as f32;
                    exp_sum += out[i * a + m];
                } else {
                    out[i * a + m] = 0.0;
                }
            }
            if exp_sum > 0.0 {
                for m in 0..a {
                    out[i * a + m] /= exp_sum;
                }
            }
        }
        out
    }

    /// Root Q value for the chosen (or best-visited) move per game.
    pub fn get_gumbel_root_value(&self, n_active: usize, chosen_moves: Option<&[i32]>) -> Vec<f32> {
        let a = L::NUM_ACTIONS;
        let mut out = vec![0.0f32; n_active];
        for i in 0..n_active {
            let r_idx = self.tree.root_indices[i] as usize;
            if let Some(chosen) = chosen_moves {
                let c_idx = self.tree.children[r_idx * a + chosen[i] as usize];
                if c_idx != -1 {
                    let n = self.tree.visit_counts[c_idx as usize];
                    if n > 0 {
                        out[i] = (-self.tree.values[c_idx as usize] / n as f64) as f32;
                    }
                }
            } else {
                let mut best_n = 0;
                for m in 0..a {
                    let c_idx = self.tree.children[r_idx * a + m];
                    if c_idx != -1 {
                        let n = self.tree.visit_counts[c_idx as usize];
                        if n > best_n {
                            best_n = n;
                            out[i] = (-self.tree.values[c_idx as usize] / n as f64) as f32;
                        }
                    }
                }
            }
        }
        out
    }

    /// (n_active, NUM_ACTIONS) visit counts + per-root best child Q.
    pub fn get_all_root_data(&self, n_active: usize) -> (Vec<f32>, Vec<f32>) {
        let a = L::NUM_ACTIONS;
        let mut visits = vec![0.0f32; n_active * a];
        let mut root_q = vec![0.0f32; n_active];
        for i in 0..n_active {
            let r_idx = self.tree.root_indices[i] as usize;
            let mut best_n = 0i32;
            for m in 0..a {
                let c_idx = self.tree.children[r_idx * a + m];
                if c_idx != -1 {
                    let n_c = self.tree.visit_counts[c_idx as usize];
                    visits[i * a + m] = n_c as f32;
                    if n_c > best_n {
                        best_n = n_c;
                        root_q[i] = (-self.tree.values[c_idx as usize] / n_c as f64) as f32;
                    }
                }
            }
        }
        (visits, root_q)
    }
}

/// Port of `get_forced_root_moves_kernel`.
fn forced_root_moves(candidate_mask: &[bool], forced_rank: usize, n_active: usize, a: usize) -> Vec<i32> {
    let mut moves = vec![0i32; n_active];
    for i in 0..n_active {
        let active: Vec<usize> = (0..a).filter(|&m| candidate_mask[i * a + m]).collect();
        if active.is_empty() {
            moves[i] = 0;
            continue;
        }
        moves[i] = active[forced_rank % active.len()] as i32;
    }
    moves
}
