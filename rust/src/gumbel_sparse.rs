//! Gumbel MCTS with sparse edge storage. Port of
//! `src/gumbel_mcts/gumbel_sparse.py` and
//! `src/kernels/gumbel_sparse_kernels.py`.
//!
//! Edges live in flat arrays with per-node offset/count, avoiding
//! (max_nodes, NUM_ACTIONS) storage for large action spaces.

use crate::game::GameLogic;
use crate::model::EvalModel;
use rand::Rng;
use std::marker::PhantomData;

pub struct GumbelSparse<L: GameLogic> {
    pub n_games: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_legal: usize,
    pub c_visit: f64,
    pub c_scale: f64,

    // Per-node arrays.
    pub parents: Vec<i32>,
    pub visit_counts: Vec<i32>,
    pub values: Vec<f64>,
    pub is_expanded: Vec<bool>,
    pub is_terminal: Vec<bool>,
    pub terminal_values: Vec<f64>,
    pub boards: Vec<i8>, // (max_nodes, BOARD_LEN)
    pub players: Vec<i8>,
    pub depths: Vec<i32>,
    pub edge_from_parent: Vec<i16>,
    pub root_indices: Vec<i32>,
    /// NN value stored at expansion time (needed for v_mix).
    pub node_nn_value: Vec<f32>,

    // Per-node edge metadata.
    pub node_edge_offset: Vec<i32>,
    pub node_num_edges: Vec<i16>,

    // Flat edge arrays.
    pub edge_action: Vec<i16>,
    pub edge_child: Vec<i32>,
    pub edge_prior: Vec<f64>,

    pub next_free_node: usize,
    pub next_free_edge: usize,

    // Root-level Gumbel data, padded to max_legal.
    pub root_logits: Vec<f32>,
    pub root_gumbel_noise: Vec<f32>,
    pub root_actions: Vec<i16>,
    pub root_num_legal: Vec<i16>,
    pub candidate_mask: Vec<bool>,

    _logic: PhantomData<L>,
}

impl<L: GameLogic> GumbelSparse<L> {
    pub fn new(
        n_games: usize,
        max_nodes: usize,
        c_visit: f64,
        c_scale: f64,
        avg_branching: usize,
        max_legal_moves: usize,
    ) -> Self {
        let max_edges = max_nodes * avg_branching;
        let bl = L::BOARD_LEN;
        Self {
            n_games,
            max_nodes,
            max_edges,
            max_legal: max_legal_moves,
            c_visit,
            c_scale,
            parents: vec![-1; max_nodes],
            visit_counts: vec![0; max_nodes],
            values: vec![0.0; max_nodes],
            is_expanded: vec![false; max_nodes],
            is_terminal: vec![false; max_nodes],
            terminal_values: vec![0.0; max_nodes],
            boards: vec![0; max_nodes * bl],
            players: vec![0; max_nodes],
            depths: vec![0; max_nodes],
            edge_from_parent: vec![0; max_nodes],
            root_indices: vec![0; n_games],
            node_nn_value: vec![0.0; max_nodes],
            node_edge_offset: vec![0; max_nodes],
            node_num_edges: vec![0; max_nodes],
            edge_action: vec![0; max_edges],
            edge_child: vec![-1; max_edges],
            edge_prior: vec![0.0; max_edges],
            next_free_node: 1,
            next_free_edge: 0,
            root_logits: vec![0.0; n_games * max_legal_moves],
            root_gumbel_noise: vec![0.0; n_games * max_legal_moves],
            root_actions: vec![-1; n_games * max_legal_moves],
            root_num_legal: vec![0; n_games],
            candidate_mask: vec![false; n_games * max_legal_moves],
            _logic: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        let num_used = self.next_free_node;
        self.next_free_node = 1;
        self.next_free_edge = 0;
        self.depths[..num_used + 1].fill(0);
        self.is_expanded[..num_used + 1].fill(false);
        self.is_terminal[..num_used + 1].fill(false);
    }

    pub fn initialize_roots(&mut self, active_games: &[usize], boards: &[i8], players: &[i8]) {
        let bl = L::BOARD_LEN;
        let current_alloc = self.next_free_node;
        for (i, &game_idx) in active_games.iter().enumerate() {
            let root_idx = current_alloc + i;
            self.root_indices[game_idx] = root_idx as i32;
            self.boards[root_idx * bl..(root_idx + 1) * bl]
                .copy_from_slice(&boards[i * bl..(i + 1) * bl]);
            self.players[root_idx] = players[i];
            self.parents[root_idx] = -1;
            self.visit_counts[root_idx] = 0;
            self.values[root_idx] = 0.0;
            self.is_expanded[root_idx] = false;
            self.is_terminal[root_idx] = false;
            self.depths[root_idx] = 0;
        }
        self.next_free_node += active_games.len();
    }

    // =========================================================================
    // Root expansion
    // =========================================================================

    fn expand_roots<M: EvalModel, R: Rng>(
        &mut self,
        model: &M,
        active_games: &[usize],
        rng: &mut R,
    ) {
        let bl = L::BOARD_LEN;
        let n_active = active_games.len();
        let root_indices: Vec<usize> = active_games
            .iter()
            .map(|&g| self.root_indices[g] as usize)
            .collect();

        let mut boards = vec![0i8; n_active * bl];
        let mut players = vec![0i8; n_active];
        for (i, &r) in root_indices.iter().enumerate() {
            boards[i * bl..(i + 1) * bl].copy_from_slice(&self.boards[r * bl..(r + 1) * bl]);
            players[i] = self.players[r];
        }

        let (probs, vals) = model.forward(&boards, &players);

        let mut mask = [false; 256];
        let a = L::NUM_ACTIONS;
        for i in 0..n_active {
            let r_idx = root_indices[i];
            L::valid_mask(&self.boards[r_idx * bl..(r_idx + 1) * bl], &mut mask[..a]);
            let legal_moves: Vec<i16> = (0..a)
                .filter(|&m| mask[m])
                .map(|m| m as i16)
                .collect();
            let n_legal = legal_moves.len();

            let dense: Vec<f64> = (0..a).map(|m| probs[i * a + m] as f64).collect();
            self.allocate_edges(r_idx, &legal_moves, &dense);

            self.node_nn_value[r_idx] = vals[i];

            self.root_num_legal[i] = n_legal as i16;
            for e in 0..n_legal {
                self.root_actions[i * self.max_legal + e] = legal_moves[e];
            }
            for e in n_legal..self.max_legal {
                self.root_actions[i * self.max_legal + e] = -1;
            }
            for e in 0..n_legal {
                let p = (probs[i * a + legal_moves[e] as usize]).max(1e-10);
                self.root_logits[i * self.max_legal + e] = p.ln();
            }
            for e in n_legal..self.max_legal {
                self.root_logits[i * self.max_legal + e] = -1e18;
            }
            for e in 0..n_legal {
                let u: f32 = rng
                    .gen::<f32>()
                    .clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
                self.root_gumbel_noise[i * self.max_legal + e] = -(-u.ln()).ln();
            }
            for e in n_legal..self.max_legal {
                self.root_gumbel_noise[i * self.max_legal + e] = -1e18;
            }
        }

        for &r in &root_indices {
            self.is_expanded[r] = true;
        }
        let ri32: Vec<i32> = root_indices.iter().map(|&r| r as i32).collect();
        let vals64: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        backpropagate(&ri32, &vals64, &mut self.parents, &mut self.visit_counts, &mut self.values);
    }

    /// Allocate a contiguous edge block for `node_idx`.
    fn allocate_edges(&mut self, node_idx: usize, legal_moves: &[i16], dense_probs: &[f64]) {
        let n_legal = legal_moves.len();
        let start = self.next_free_edge;
        assert!(
            start + n_legal <= self.max_edges,
            "Edge pool exhausted: need {}, have {}. Increase avg_branching or max_nodes.",
            start + n_legal,
            self.max_edges
        );

        self.node_edge_offset[node_idx] = start as i32;
        self.node_num_edges[node_idx] = n_legal as i16;

        for e in 0..n_legal {
            self.edge_action[start + e] = legal_moves[e];
            self.edge_child[start + e] = -1;
        }

        let mut legal_priors: Vec<f64> = legal_moves
            .iter()
            .map(|&m| dense_probs[m as usize])
            .collect();
        let prior_sum: f64 = legal_priors.iter().sum();
        if prior_sum > 1e-10 {
            for p in legal_priors.iter_mut() {
                *p /= prior_sum;
            }
        } else {
            for p in legal_priors.iter_mut() {
                *p = 1.0 / n_legal.max(1) as f64;
            }
        }
        for e in 0..n_legal {
            self.edge_prior[start + e] = legal_priors[e];
        }

        self.next_free_edge = start + n_legal;
    }

    // =========================================================================
    // Leaf evaluation & expansion
    // =========================================================================

    fn evaluate_and_expand<M: EvalModel>(&mut self, model: &M, leaf_indices: &[i32]) {
        let bl = L::BOARD_LEN;
        let a = L::NUM_ACTIONS;
        let mut leaf_values = vec![0.0f64; leaf_indices.len()];

        let mut nn_indices: Vec<usize> = Vec::new();
        let mut eval_pos = vec![usize::MAX; leaf_indices.len()];
        for (i, &l) in leaf_indices.iter().enumerate() {
            let li = l as usize;
            if self.is_terminal[li] {
                leaf_values[i] = self.terminal_values[li];
            } else if !self.is_expanded[li] {
                eval_pos[i] = nn_indices.len();
                nn_indices.push(li);
            }
        }

        if !nn_indices.is_empty() {
            let b = nn_indices.len();
            let mut boards = vec![0i8; b * bl];
            let mut players = vec![0i8; b];
            let mut masks = vec![false; b * a];
            let mut mask = [false; 256];
            for (j, &n) in nn_indices.iter().enumerate() {
                boards[j * bl..(j + 1) * bl]
                    .copy_from_slice(&self.boards[n * bl..(n + 1) * bl]);
                players[j] = self.players[n];
                L::valid_mask(&self.boards[n * bl..(n + 1) * bl], &mut mask[..a]);
                masks[j * a..(j + 1) * a].copy_from_slice(&mask[..a]);
            }

            let (probs, vals) = model.forward(&boards, &players);

            for (j, &node_idx) in nn_indices.iter().enumerate() {
                let legal_moves: Vec<i16> = (0..a)
                    .filter(|&m| masks[j * a + m])
                    .map(|m| m as i16)
                    .collect();
                let dense: Vec<f64> = (0..a).map(|m| probs[j * a + m] as f64).collect();
                self.allocate_edges(node_idx, &legal_moves, &dense);
                self.node_nn_value[node_idx] = vals[j];
                self.is_expanded[node_idx] = true;
            }
            for (i, &pos) in eval_pos.iter().enumerate() {
                if pos != usize::MAX {
                    leaf_values[i] = vals[pos] as f64;
                }
            }
        }

        backpropagate(
            leaf_indices,
            &leaf_values,
            &mut self.parents,
            &mut self.visit_counts,
            &mut self.values,
        );
    }

    // =========================================================================
    // Descent kernel — port of `descend_batch`
    // =========================================================================

    fn descend_batch(
        &mut self,
        game_indices: &[i32],
        forced_edge_locals: &[i32],
        c_scale: &[f32],
    ) -> Vec<i32> {
        let bl = L::BOARD_LEN;
        let n_active = game_indices.len();
        let mut leaf_indices = vec![0i32; n_active];
        let mut new_board = [0i8; 256];

        for i in 0..n_active {
            let mut node_idx = self.root_indices[game_indices[i] as usize] as usize;
            let forced = forced_edge_locals[i];
            let mut first_move = true;

            loop {
                if self.is_terminal[node_idx] || !self.is_expanded[node_idx] {
                    leaf_indices[i] = node_idx as i32;
                    break;
                }

                let e_start = self.node_edge_offset[node_idx] as usize;
                let n_edges = self.node_num_edges[node_idx] as usize;
                if n_edges == 0 {
                    leaf_indices[i] = node_idx as i32;
                    break;
                }

                let chosen_local = if first_move && forced >= 0 {
                    first_move = false;
                    forced as usize
                } else {
                    self.select_edge(node_idx, e_start, n_edges, c_scale[i] as f64)
                };

                let eidx = e_start + chosen_local;
                let action = self.edge_action[eidx];
                let child_idx = self.edge_child[eidx];

                if child_idx == -1 {
                    let new_idx = self.next_free_node;
                    if new_idx >= self.max_nodes {
                        leaf_indices[i] = node_idx as i32;
                        break;
                    }
                    self.next_free_node += 1;
                    self.edge_child[eidx] = new_idx as i32;

                    new_board[..bl]
                        .copy_from_slice(&self.boards[node_idx * bl..(node_idx + 1) * bl]);
                    let cur_player = self.players[node_idx];
                    let res = L::fast_step(&mut new_board[..bl], action as usize, cur_player);
                    let new_player = L::other_player(cur_player);

                    self.boards[new_idx * bl..(new_idx + 1) * bl]
                        .copy_from_slice(&new_board[..bl]);
                    self.players[new_idx] = new_player;
                    self.parents[new_idx] = node_idx as i32;
                    self.edge_from_parent[new_idx] = action;
                    self.depths[new_idx] = self.depths[node_idx] + 1;
                    self.visit_counts[new_idx] = 0;
                    self.values[new_idx] = 0.0;
                    self.is_expanded[new_idx] = false;
                    if res.done {
                        self.is_terminal[new_idx] = true;
                        self.terminal_values[new_idx] = -res.reward;
                    } else {
                        self.is_terminal[new_idx] = false;
                        self.terminal_values[new_idx] = 0.0;
                    }

                    leaf_indices[i] = new_idx as i32;
                    break;
                }

                node_idx = child_idx as usize;
            }
        }
        leaf_indices
    }

    /// Port of `_compute_v_mix` (Numba kernel).
    fn compute_v_mix(&self, node_idx: usize, e_start: usize, n_edges: usize) -> f64 {
        let v_hat = self.node_nn_value[node_idx] as f64;
        let mut sum_n = self.visit_counts[node_idx] - 1;
        if sum_n < 0 {
            sum_n = 0;
        }
        let mut sum_weighted_q = 0.0f64;
        let mut sum_pi_visited = 0.0f64;

        for e in 0..n_edges {
            let eidx = e_start + e;
            let c_idx = self.edge_child[eidx];
            if c_idx != -1 {
                let n_c = self.visit_counts[c_idx as usize];
                if n_c > 0 {
                    let q_c = -self.values[c_idx as usize] / n_c as f64;
                    let pi_a = self.edge_prior[eidx];
                    sum_weighted_q += pi_a * q_c;
                    sum_pi_visited += pi_a;
                }
            }
        }
        if sum_pi_visited > 1e-10 && sum_n > 0 {
            (1.0 / (1.0 + sum_n as f64)) * (v_hat + (sum_n as f64 / sum_pi_visited) * sum_weighted_q)
        } else {
            v_hat
        }
    }

    /// Port of `_select_edge` — interior-node sigma-scaled completed-Q selection.
    fn select_edge(
        &self,
        node_idx: usize,
        e_start: usize,
        n_edges: usize,
        c_scale: f64,
    ) -> usize {
        let v_mix = self.compute_v_mix(node_idx, e_start, n_edges);

        let mut q_values = vec![0.0f64; n_edges];
        let mut q_min = 1e10f64;
        let mut q_max = -1e10f64;
        let mut max_child_n = 0i32;

        for e in 0..n_edges {
            let eidx = e_start + e;
            let c_idx = self.edge_child[eidx];
            let mut n_c = 0i32;
            let q = if c_idx != -1 && self.visit_counts[c_idx as usize] > 0 {
                n_c = self.visit_counts[c_idx as usize];
                -self.values[c_idx as usize] / n_c as f64
            } else {
                v_mix
            };
            if n_c > max_child_n {
                max_child_n = n_c;
            }
            q_values[e] = q;
            q_min = q_min.min(q);
            q_max = q_max.max(q);
        }

        let q_range = if q_max - q_min < 1e-6 { 1.0 } else { q_max - q_min };
        let sigma = c_scale * (self.c_visit + max_child_n as f64);

        let mut combined = vec![0.0f64; n_edges];
        let mut max_combined = -1e18f64;
        for e in 0..n_edges {
            let p = self.edge_prior[e_start + e].max(1e-10);
            combined[e] = p.ln() + sigma * (q_values[e] - q_min) / q_range;
            if combined[e] > max_combined {
                max_combined = combined[e];
            }
        }

        let mut pi_prime = vec![0.0f64; n_edges];
        let mut exp_sum = 0.0f64;
        for e in 0..n_edges {
            pi_prime[e] = (combined[e] - max_combined).exp();
            exp_sum += pi_prime[e];
        }
        if exp_sum > 0.0 {
            for e in 0..n_edges {
                pi_prime[e] /= exp_sum;
            }
        }

        let mut sum_child_n = 0i64;
        let mut n_per_edge = vec![0i64; n_edges];
        for e in 0..n_edges {
            let c_idx = self.edge_child[e_start + e];
            let n_a = if c_idx != -1 {
                self.visit_counts[c_idx as usize] as i64
            } else {
                0
            };
            n_per_edge[e] = n_a;
            sum_child_n += n_a;
        }

        let denom = 1.0 + sum_child_n as f64;
        let mut best_local = 0usize;
        let mut best_score = -1e18f64;
        for e in 0..n_edges {
            let score = pi_prime[e] - n_per_edge[e] as f64 / denom;
            if score > best_score {
                best_score = score;
                best_local = e;
            }
        }
        best_local
    }

    // =========================================================================
    // Sequential halving — port of run_simulation_batch
    // =========================================================================

    pub fn run_simulation_batch<M: EvalModel, R: Rng>(
        &mut self,
        model: &M,
        active_games: &[usize],
        num_simulations: usize,
        rng: &mut R,
        c_scale_overrides: Option<&[f32]>,
    ) -> Vec<i32> {
        let n_active = active_games.len();
        let game_indices: Vec<i32> = active_games.iter().map(|&g| g as i32).collect();

        let current_c_scales: Vec<f32> = match c_scale_overrides {
            Some(v) => v.to_vec(),
            None => vec![self.c_scale as f32; n_active],
        };

        self.expand_roots(model, active_games, rng);

        let max_k = self.max_legal.min(16);
        let mut num_phases = ((max_k as f64).log2() as usize).max(1);
        let first_budget = num_simulations / num_phases;
        let k_initial = max_k.min(first_budget / 2).max(2);
        num_phases = ((k_initial as f64).log2() as usize).max(1);

        self.init_candidates(n_active, k_initial);

        let mut remaining = num_simulations;
        for phase in 0..num_phases {
            let k_phase = (k_initial / (1 << phase)).max(1);
            let phases_left = num_phases - phase;
            let budget = if phase == num_phases - 1 {
                remaining
            } else {
                remaining / phases_left
            };
            let sims_per_action = (budget / k_phase).max(1);

            for candidate_rank in 0..k_phase {
                let forced_edges =
                    forced_edge_local(&self.candidate_mask[..n_active * self.max_legal], candidate_rank, self.max_legal);
                for _ in 0..sims_per_action {
                    let leaf_indices =
                        self.descend_batch(&game_indices, &forced_edges, &current_c_scales);
                    self.evaluate_and_expand(model, &leaf_indices);
                }
            }

            remaining = remaining.saturating_sub(k_phase * sims_per_action);

            if phase < num_phases - 1 {
                self.halve_candidates(n_active, &current_c_scales);
            }
        }

        self.final_moves(n_active, &current_c_scales)
    }

    fn init_candidates(&mut self, n_active: usize, k: usize) {
        for i in 0..n_active {
            for e in 0..self.max_legal {
                self.candidate_mask[i * self.max_legal + e] = false;
            }
            let n_legal = self.root_num_legal[i] as usize;
            if n_legal == 0 {
                continue;
            }
            let mut scored: Vec<(usize, f32)> = (0..n_legal)
                .map(|e| {
                    (
                        e,
                        self.root_logits[i * self.max_legal + e]
                            + self.root_gumbel_noise[i * self.max_legal + e],
                    )
                })
                .collect();
            scored.sort_by(|x, y| x.1.total_cmp(&y.1));
            let k_actual = k.min(n_legal);
            for (e, _) in scored.iter().skip(n_legal - k_actual) {
                self.candidate_mask[i * self.max_legal + e] = true;
            }
        }
    }

    /// Port of `compute_gumbel_scores`.
    fn gumbel_scores(&self, n_active: usize, c_scale: &[f32]) -> Vec<f32> {
        let ml = self.max_legal;
        let mut scores = vec![-1e18f32; n_active * ml];

        for i in 0..n_active {
            let r_idx = self.root_indices[i] as usize;
            let e_start = self.node_edge_offset[r_idx] as usize;
            let n_edges = self.node_num_edges[r_idx] as usize;

            let v_mix = self.compute_v_mix(r_idx, e_start, n_edges);
            let n_legal = (self.root_num_legal[i] as usize).min(n_edges);
            if n_legal == 0 {
                continue;
            }

            let mut q_values = vec![0.0f64; n_legal];
            let mut q_min = 1e10f64;
            let mut q_max = -1e10f64;
            let mut max_child_n = 0i32;

            for e in 0..n_legal {
                let eidx = e_start + e;
                let c_idx = self.edge_child[eidx];
                let mut n_c = 0i32;
                let q = if c_idx != -1 && self.visit_counts[c_idx as usize] > 0 {
                    n_c = self.visit_counts[c_idx as usize];
                    -self.values[c_idx as usize] / n_c as f64
                } else {
                    v_mix
                };
                if n_c > max_child_n {
                    max_child_n = n_c;
                }
                q_values[e] = q;
                q_min = q_min.min(q);
                q_max = q_max.max(q);
            }

            let q_range = if q_max - q_min < 1e-6 { 1.0 } else { q_max - q_min };
            let sigma = c_scale[i] as f64 * (self.c_visit + max_child_n as f64);

            for e in 0..n_legal {
                if !self.candidate_mask[i * ml + e] {
                    continue;
                }
                scores[i * ml + e] = (sigma * ((q_values[e] - q_min) / q_range)
                    + self.root_logits[i * ml + e] as f64
                    + self.root_gumbel_noise[i * ml + e] as f64)
                    as f32;
            }
        }
        scores
    }

    fn halve_candidates(&mut self, n_active: usize, c_scale: &[f32]) {
        let ml = self.max_legal;
        let scores = self.gumbel_scores(n_active, c_scale);
        let mut new_mask = vec![false; n_active * ml];

        for i in 0..n_active {
            let active: Vec<usize> = (0..ml).filter(|&e| self.candidate_mask[i * ml + e]).collect();
            if active.len() <= 1 {
                for e in 0..ml {
                    new_mask[i * ml + e] = self.candidate_mask[i * ml + e];
                }
                continue;
            }
            let num_keep = (active.len() / 2).max(1);
            let mut scored: Vec<(usize, f32)> =
                active.iter().map(|&e| (e, scores[i * ml + e])).collect();
            scored.sort_by(|x, y| x.1.total_cmp(&y.1));
            for (e, _) in scored.iter().skip(active.len() - num_keep) {
                new_mask[i * ml + e] = true;
            }
        }
        self.candidate_mask[..n_active * ml].copy_from_slice(&new_mask);
    }

    fn final_moves(&self, n_active: usize, c_scale: &[f32]) -> Vec<i32> {
        let ml = self.max_legal;
        let scores = self.gumbel_scores(n_active, c_scale);
        let mut moves = vec![0i32; n_active];
        for i in 0..n_active {
            let mut best_local = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for e in 0..ml {
                if scores[i * ml + e] > best_s {
                    best_s = scores[i * ml + e];
                    best_local = e;
                }
            }
            let r_idx = self.root_indices[i] as usize;
            let eidx = self.node_edge_offset[r_idx] as usize + best_local;
            moves[i] = self.edge_action[eidx] as i32;
        }
        moves
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Improved policy π' = softmax(logit + sigma * q_completed), dense output.
    pub fn get_improved_policy(&self, n_active: usize) -> Vec<f32> {
        let a = L::NUM_ACTIONS;
        let ml = self.max_legal;
        let c_scale = vec![self.c_scale as f32; n_active];
        let mut dense = vec![0.0f32; n_active * a];

        for i in 0..n_active {
            let r_idx = self.root_indices[i] as usize;
            let e_start = self.node_edge_offset[r_idx] as usize;
            let n_edges = self.node_num_edges[r_idx] as usize;

            let v_mix = self.compute_v_mix(r_idx, e_start, n_edges);
            let n_legal = (self.root_num_legal[i] as usize).min(n_edges);
            if n_legal == 0 {
                continue;
            }

            let mut local_actions = vec![0i32; n_legal];
            let mut local_q = vec![0.0f64; n_legal];
            let mut q_min = 1e10f64;
            let mut q_max = -1e10f64;
            let mut max_child_n = 0i32;

            for e in 0..n_legal {
                let eidx = e_start + e;
                let c_idx = self.edge_child[eidx];
                let mut n_c = 0i32;
                let q = if c_idx != -1 && self.visit_counts[c_idx as usize] > 0 {
                    n_c = self.visit_counts[c_idx as usize];
                    -self.values[c_idx as usize] / n_c as f64
                } else {
                    v_mix
                };
                if n_c > max_child_n {
                    max_child_n = n_c;
                }
                local_actions[e] = self.edge_action[eidx] as i32;
                local_q[e] = q;
                q_min = q_min.min(q);
                q_max = q_max.max(q);
            }

            let q_range = if q_max - q_min < 1e-6 { 1.0 } else { q_max - q_min };
            let sigma = c_scale[i] as f64 * (self.c_visit + max_child_n as f64);

            let mut local_scores = vec![0.0f64; n_legal];
            for e in 0..n_legal {
                local_scores[e] =
                    self.root_logits[i * ml + e] as f64 + sigma * (local_q[e] - q_min) / q_range;
            }
            let mx = local_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exp_sum = 0.0f64;
            for e in 0..n_legal {
                local_scores[e] = (local_scores[e] - mx).exp();
                exp_sum += local_scores[e];
            }
            for e in 0..n_legal {
                dense[i * a + local_actions[e] as usize] =
                    (local_scores[e] / exp_sum.max(1e-10)) as f32;
            }
        }
        dense
    }

    /// Root Q value for the chosen (or best-visited) move per game.
    pub fn get_gumbel_root_value(&self, n_active: usize, chosen_moves: Option<&[i32]>) -> Vec<f32> {
        let mut out = vec![0.0f32; n_active];
        for i in 0..n_active {
            let r_idx = self.root_indices[i] as usize;
            let e_start = self.node_edge_offset[r_idx] as usize;
            let n_edges = self.node_num_edges[r_idx] as usize;

            if let Some(chosen) = chosen_moves {
                let target = chosen[i] as i16;
                for e in 0..n_edges {
                    if self.edge_action[e_start + e] == target {
                        let c_idx = self.edge_child[e_start + e];
                        if c_idx != -1 && self.visit_counts[c_idx as usize] > 0 {
                            out[i] = (-self.values[c_idx as usize]
                                / self.visit_counts[c_idx as usize] as f64)
                                as f32;
                        }
                        break;
                    }
                }
            } else {
                let mut best_n = 0;
                for e in 0..n_edges {
                    let c_idx = self.edge_child[e_start + e];
                    if c_idx != -1 {
                        let n_c = self.visit_counts[c_idx as usize];
                        if n_c > best_n {
                            best_n = n_c;
                            out[i] =
                                (-self.values[c_idx as usize] / n_c as f64) as f32;
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
            let r_idx = self.root_indices[i] as usize;
            let e_start = self.node_edge_offset[r_idx] as usize;
            let n_edges = self.node_num_edges[r_idx] as usize;
            let mut best_n = 0i32;
            for e in 0..n_edges {
                let c_idx = self.edge_child[e_start + e];
                if c_idx != -1 {
                    let n_c = self.visit_counts[c_idx as usize];
                    let action = self.edge_action[e_start + e] as usize;
                    visits[i * a + action] = n_c as f32;
                    if n_c > best_n {
                        best_n = n_c;
                        root_q[i] = (-self.values[c_idx as usize] / n_c as f64) as f32;
                    }
                }
            }
        }
        (visits, root_q)
    }

    pub fn get_max_depth(&self) -> i32 {
        let upper = self.next_free_node;
        if upper <= 1 {
            return 0;
        }
        *self.depths[..upper].iter().max().unwrap_or(&0)
    }
}

/// Port of `get_forced_edge_local`: local edge index of the
/// `candidate_rank`-th active candidate per game.
fn forced_edge_local(candidate_mask: &[bool], candidate_rank: usize, ml: usize) -> Vec<i32> {
    let n_active = candidate_mask.len() / ml;
    let mut edges = vec![-1i32; n_active];
    for i in 0..n_active {
        let mut count = 0;
        for j in 0..ml {
            if candidate_mask[i * ml + j] {
                if count == candidate_rank {
                    edges[i] = j as i32;
                    break;
                }
                count += 1;
            }
        }
    }
    edges
}

/// Port of `backpropagate_batch`.
fn backpropagate(
    leaf_indices: &[i32],
    nn_values: &[f64],
    parents: &mut [i32],
    visit_counts: &mut [i32],
    values: &mut [f64],
) {
    for (i, &leaf) in leaf_indices.iter().enumerate() {
        let mut node_idx = leaf;
        let mut value = nn_values[i];
        while node_idx != -1 {
            let n = node_idx as usize;
            visit_counts[n] += 1;
            values[n] += value;
            value = -value;
            node_idx = parents[n];
        }
    }
}
