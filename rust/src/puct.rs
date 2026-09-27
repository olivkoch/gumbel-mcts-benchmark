//! Dense PUCT MCTS. Port of `src/gumbel_mcts/puct.py` and
//! `src/kernels/puct_kernels.py`.

use crate::game::GameLogic;
use crate::model::EvalModel;
use std::marker::PhantomData;

pub struct Puct<L: GameLogic> {
    pub n_games: usize,
    pub max_nodes: usize,

    // Flat struct-of-arrays node storage (ported 1:1 from PUCTStorage).
    pub children: Vec<i32>,       // (max_nodes, NUM_ACTIONS)
    pub parents: Vec<i32>,        // (max_nodes,)
    pub edge_from_parent: Vec<i16>,
    pub visit_counts: Vec<i32>,
    pub values: Vec<f64>,
    pub prior_probs: Vec<f64>,    // (max_nodes, NUM_ACTIONS)
    pub terminal_values: Vec<f64>,
    pub is_expanded: Vec<bool>,
    pub is_terminal: Vec<bool>,
    pub boards: Vec<i8>,          // (max_nodes, BOARD_LEN)
    pub players: Vec<i8>,
    pub root_indices: Vec<i32>,   // (n_games,)
    pub depths: Vec<i32>,

    pub next_free_idx: usize,
    /// Scratch: per-node value produced by the most recent `evaluate_leaves`
    /// (terminal value for terminals, NN value otherwise). Indexed by node.
    last_vals: Vec<f64>,
    _logic: PhantomData<L>,
}

impl<L: GameLogic> Puct<L> {
    pub fn new(n_games: usize, max_nodes: usize) -> Self {
        let a = L::NUM_ACTIONS;
        let bl = L::BOARD_LEN;
        Self {
            n_games,
            max_nodes,
            children: vec![-1; max_nodes * a],
            parents: vec![-1; max_nodes],
            edge_from_parent: vec![0; max_nodes],
            visit_counts: vec![0; max_nodes],
            values: vec![0.0; max_nodes],
            prior_probs: vec![0.0; max_nodes * a],
            terminal_values: vec![0.0; max_nodes],
            is_expanded: vec![false; max_nodes],
            is_terminal: vec![false; max_nodes],
            boards: vec![0; max_nodes * bl],
            players: vec![0; max_nodes],
            root_indices: vec![0; n_games],
            depths: vec![0; max_nodes],
            next_free_idx: 1,
            last_vals: Vec::new(),
            _logic: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        let num_used = self.next_free_idx;
        self.next_free_idx = 1;
        self.depths[..num_used + 1].fill(0);
        self.is_expanded[..num_used + 1].fill(false);
        self.is_terminal[..num_used + 1].fill(false);
    }

    #[allow(clippy::too_many_arguments)]
    fn init_node(
        &mut self,
        idx: usize,
        parent: i32,
        edge: i16,
        board: &[i8],
        player: i8,
        terminal_value: f64,
        done: bool,
    ) {
        let a = L::NUM_ACTIONS;
        self.children[idx * a..(idx + 1) * a].fill(-1);
        self.parents[idx] = parent;
        self.edge_from_parent[idx] = edge;
        self.visit_counts[idx] = 0;
        self.values[idx] = 0.0;
        self.is_expanded[idx] = false;
        self.is_terminal[idx] = done;
        self.terminal_values[idx] = terminal_value;
        self.boards[idx * L::BOARD_LEN..(idx + 1) * L::BOARD_LEN].copy_from_slice(board);
        self.players[idx] = player;
        self.depths[idx] = if parent == -1 {
            0
        } else {
            self.depths[parent as usize] + 1
        };
    }

    /// `boards` is `active_games.len() × BOARD_LEN`, `players` is per game.
    pub fn initialize_roots(&mut self, active_games: &[usize], boards: &[i8], players: &[i8]) {
        let current_alloc = self.next_free_idx;
        for (i, &game_idx) in active_games.iter().enumerate() {
            let root_idx = current_alloc + i;
            self.root_indices[game_idx] = root_idx as i32;
            let board = &boards[i * L::BOARD_LEN..(i + 1) * L::BOARD_LEN];
            self.init_node(root_idx, -1, -1, board, players[i], 0.0, false);
        }
        self.next_free_idx += active_games.len();
    }

    /// Selection phase — port of `select_leaves_batch`.
    pub fn select_leaves_batch(
        &mut self,
        game_indices: &[i32],
        c_puct_base: f64,
        c_puct_init: f64,
    ) -> Vec<i32> {
        let a = L::NUM_ACTIONS;
        let bl = L::BOARD_LEN;
        let n_active = game_indices.len();
        let mut leaf_indices = vec![0i32; n_active];
        let mut mask = [false; 256]; // covers up to 225 actions
        let mut new_board = [0i8; 256];

        for i in 0..n_active {
            let game_idx = game_indices[i] as usize;
            let mut node_idx = self.root_indices[game_idx] as usize;
            let mut search_depth = 0usize;

            loop {
                if self.is_terminal[node_idx]
                    || !self.is_expanded[node_idx]
                    || search_depth >= L::MAX_MOVES
                {
                    break;
                }
                search_depth += 1;

                L::valid_mask(&self.boards[node_idx * bl..(node_idx + 1) * bl], &mut mask[..a]);

                let mut best_score = -1e9f64;
                let mut best_move = -1i32;
                let parent_n = self.visit_counts[node_idx] as f64;
                let sqrt_parent_n = parent_n.sqrt();
                let pb_c = ((1.0 + parent_n + c_puct_base) / c_puct_base).ln() + c_puct_init;
                let mut has_valid_move = false;

                for m in 0..a {
                    if !mask[m] {
                        continue;
                    }
                    has_valid_move = true;
                    let child_idx = self.children[node_idx * a + m];
                    let mut child_q = 0.0f64;
                    let mut child_n = 0f64;
                    if child_idx != -1 {
                        let ci = child_idx as usize;
                        let cn = self.visit_counts[ci];
                        if cn > 0 {
                            child_q = -self.values[ci] / cn as f64;
                            child_n = cn as f64;
                        }
                    }
                    let prior = self.prior_probs[node_idx * a + m];
                    let u_score = pb_c * prior * sqrt_parent_n / (1.0 + child_n);
                    let score = child_q + u_score;
                    if score > best_score {
                        best_score = score;
                        best_move = m as i32;
                    }
                }

                if !has_valid_move {
                    break;
                }

                let child_idx = self.children[node_idx * a + best_move as usize];
                if child_idx == -1 {
                    // Allocate new child.
                    let new_idx = self.next_free_idx;
                    if new_idx >= self.max_nodes {
                        leaf_indices[i] = node_idx as i32;
                        break;
                    }
                    self.next_free_idx += 1;

                    new_board[..bl]
                        .copy_from_slice(&self.boards[node_idx * bl..(node_idx + 1) * bl]);
                    let cur_player = self.players[node_idx];
                    let res = L::fast_step(&mut new_board[..bl], best_move as usize, cur_player);

                    let term_val = if res.done {
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
                    let next_player = L::other_player(cur_player);

                    self.init_node(
                        new_idx,
                        node_idx as i32,
                        best_move as i16,
                        &new_board[..bl],
                        next_player,
                        term_val,
                        res.done,
                    );
                    self.children[node_idx * a + best_move as usize] = new_idx as i32;
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

    /// Port of `backpropagate_batch`: values negate each level up.
    pub fn backpropagate_batch(&mut self, leaf_indices: &[i32], nn_values: &[f64]) {
        for (i, &leaf) in leaf_indices.iter().enumerate() {
            let mut node_idx = leaf;
            let mut value = nn_values[i];
            while node_idx != -1 {
                let n = node_idx as usize;
                self.visit_counts[n] += 1;
                self.values[n] += value;
                value = -value;
                node_idx = self.parents[n];
            }
        }
    }

    /// Port of `PUCT.run_simulation_batch`.
    pub fn run_simulation_batch<M: EvalModel>(
        &mut self,
        model: &M,
        active_games: &[usize],
        num_simulations: usize,
        c_puct_base: f64,
        c_puct_init: f64,
    ) {
        let game_indices: Vec<i32> = active_games.iter().map(|&g| g as i32).collect();

        // Pre-expand unexpanded roots.
        let unexpanded: Vec<i32> = active_games
            .iter()
            .map(|&g| self.root_indices[g])
            .filter(|&r| !self.is_expanded[r as usize])
            .collect();

        if !unexpanded.is_empty() {
            self.evaluate_leaves_impl(model, &unexpanded, false);
            let vals: Vec<f64> = unexpanded
                .iter()
                .map(|&r| self.last_vals[r as usize])
                .collect();
            self.backpropagate_batch(&unexpanded, &vals);
        }

        loop {
            let min_visits = active_games
                .iter()
                .map(|&g| self.visit_counts[self.root_indices[g] as usize])
                .min()
                .unwrap_or(0);
            if min_visits as usize >= num_simulations + 1 {
                break;
            }

            let leaf_indices =
                self.select_leaves_batch(&game_indices, c_puct_base, c_puct_init);
            self.evaluate_leaves(model, &leaf_indices);
            let leaf_values: Vec<f64> = leaf_indices
                .iter()
                .map(|&l| self.last_vals[l as usize])
                .collect();
            self.backpropagate_batch(&leaf_indices, &leaf_values);
        }
    }

    /// Evaluate a batch of leaf nodes: terminals use ground truth, others get
    /// an NN eval which also marks them expanded and stores their priors.
    pub fn evaluate_leaves<M: EvalModel>(&mut self, model: &M, leaf_indices: &[i32]) {
        self.evaluate_leaves_impl(model, leaf_indices, true)
    }

    fn evaluate_leaves_impl<M: EvalModel>(
        &mut self,
        model: &M,
        leaf_indices: &[i32],
        respect_terminal: bool,
    ) {
        let a = L::NUM_ACTIONS;
        let bl = L::BOARD_LEN;

        if self.last_vals.len() < self.max_nodes {
            self.last_vals.resize(self.max_nodes, 0.0);
        }

        let mut nn_indices: Vec<i32> = Vec::with_capacity(leaf_indices.len());
        for &l in leaf_indices {
            let li = l as usize;
            if respect_terminal && self.is_terminal[li] {
                self.last_vals[li] = self.terminal_values[li];
            } else {
                nn_indices.push(l);
            }
        }

        if !nn_indices.is_empty() {
            let b = nn_indices.len();
            let mut boards = vec![0i8; b * bl];
            let mut players = vec![0i8; b];
            for (j, &n) in nn_indices.iter().enumerate() {
                let ni = n as usize;
                boards[j * bl..(j + 1) * bl]
                    .copy_from_slice(&self.boards[ni * bl..(ni + 1) * bl]);
                players[j] = self.players[ni];
            }
            let (priors, vals) = model.forward(&boards, &players);
            for (j, &n) in nn_indices.iter().enumerate() {
                let ni = n as usize;
                self.is_expanded[ni] = true;
                for m in 0..a {
                    self.prior_probs[ni * a + m] = priors[j * a + m] as f64;
                }
                self.last_vals[ni] = vals[j] as f64;
            }
        }
    }

    pub fn get_root_data(&self, game_idx: usize) -> (Vec<f32>, f64) {
        let a = L::NUM_ACTIONS;
        let root_idx = self.root_indices[game_idx] as usize;
        let mut child_visits = vec![0.0f32; a];
        for m in 0..a {
            let c = self.children[root_idx * a + m];
            if c != -1 {
                child_visits[m] = self.visit_counts[c as usize] as f32;
            }
        }
        let root_n = self.visit_counts[root_idx];
        let root_w = self.values[root_idx];
        let root_q = if root_n > 0 { root_w / root_n as f64 } else { 0.0 };
        (child_visits, root_q)
    }

    /// Vectorized extraction for games 0..n_active.
    pub fn get_all_root_data(&self, n_active: usize) -> (Vec<f32>, Vec<f32>) {
        let a = L::NUM_ACTIONS;
        let mut child_visits = vec![0.0f32; n_active * a];
        let mut root_q = vec![0.0f32; n_active];
        for i in 0..n_active {
            let r = self.root_indices[i] as usize;
            for m in 0..a {
                let c = self.children[r * a + m];
                if c != -1 {
                    child_visits[i * a + m] = self.visit_counts[c as usize] as f32;
                }
            }
            let n = self.visit_counts[r];
            root_q[i] = if n > 0 { (self.values[r] / n as f64) as f32 } else { 0.0 };
        }
        (child_visits, root_q)
    }

    /// Value assigned to `node` by the last `evaluate_leaves` call
    /// (terminal ground truth or NN value).
    pub fn last_eval(&self, node: usize) -> f64 {
        self.last_vals[node]
    }

    /// Same as the private `init_node` but callable by sibling modules
    /// (GumbelDense creates children inside its descent kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn init_node_pub(
        &mut self,
        idx: usize,
        parent: i32,
        edge: i16,
        board: &[i8],
        player: i8,
        terminal_value: f64,
        done: bool,
    ) {
        self.init_node(idx, parent, edge, board, player, terminal_value, done);
    }

    pub fn get_max_depth(&self) -> i32 {
        let upper = self.next_free_idx;
        if upper <= 1 {
            return 0;
        }
        *self.depths[..upper].iter().max().unwrap_or(&0)
    }
}
