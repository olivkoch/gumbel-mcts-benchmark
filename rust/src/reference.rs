//! Port of the golden-standard reference MCTS (`src/gumbel_mcts/reference.py`,
//! originally from michaelnny/alpha_zero `mcts_v2.py`). This is the "V2"
//! baseline in the PUCT speedup benchmark: object-graph nodes, per-node dict
//! children, deepcopy'd env per simulation, virtual-loss leaf batching.

use crate::env::GomokuEnv;
use crate::model::EvalModel;
use std::collections::HashMap;

const NUM_ACTIONS: usize = 225;
/// Arena index of the placeholder "DummyNode" every root hangs off of.
const DUMMY: usize = 0;

struct V2Node {
    #[allow(dead_code)]
    to_play: i8,
    mov: i32,
    parent: usize,
    is_expanded: bool,
    child_w: Vec<f64>,
    child_n: Vec<f64>,
    child_p: Vec<f64>,
    children: HashMap<i32, usize>,
    losses_applied: i32,
}

impl V2Node {
    fn new(to_play: i8, mov: i32, parent: usize) -> Self {
        Self {
            to_play,
            mov,
            parent,
            is_expanded: false,
            child_w: vec![0.0; NUM_ACTIONS],
            child_n: vec![0.0; NUM_ACTIONS],
            child_p: vec![0.0; NUM_ACTIONS],
            children: HashMap::new(),
            losses_applied: 0,
        }
    }
}

pub struct V2Tree {
    arena: Vec<V2Node>,
}

impl V2Tree {
    fn n(&self, node: usize) -> f64 {
        let nd = &self.arena[node];
        self.arena[nd.parent].child_n[nd.mov.max(0) as usize]
    }



    /// `best_child`: max UCB child, created on demand. Returns node index.
    fn best_child(
        &mut self,
        node: usize,
        legal_actions: &[f32; NUM_ACTIONS],
        c_puct_base: f64,
        c_puct_init: f64,
        child_to_play: i8,
    ) -> usize {
        let n = self.n(node);
        let pb_c = ((1.0 + n + c_puct_base) / c_puct_base).ln() + c_puct_init;
        let sqrt_n = n.sqrt();

        let mut best_move = -1i32;
        let mut best_score = f64::NEG_INFINITY;
        for m in 0..NUM_ACTIONS {
            if legal_actions[m] != 1.0 {
                continue;
            }
            let nd = &self.arena[node];
            let child_n = nd.child_n[m].max(0.0);
            let child_q = nd.child_w[m] / if nd.child_n[m] > 0.0 { nd.child_n[m] } else { 1.0 };
            let u = pb_c * nd.child_p[m] * (sqrt_n / (1.0 + child_n));
            let score = -child_q + u;
            if score > best_score {
                best_score = score;
                best_move = m as i32;
            }
        }
        assert!(best_move >= 0 && legal_actions[best_move as usize] == 1.0);

        let mov = best_move;
        if let Some(&c) = self.arena[node].children.get(&mov) {
            c
        } else {
            let idx = self.arena.len();
            self.arena.push(V2Node::new(child_to_play, mov, node));
            self.arena[node].children.insert(mov, idx);
            idx
        }
    }

    fn backup(&mut self, mut node: usize, mut value: f64) {
        while node != DUMMY {
            let nd = &self.arena[node];
            let parent = nd.parent;
            let mov = nd.mov.max(0) as usize;
            self.arena[parent].child_n[mov] += 1.0;
            self.arena[parent].child_w[mov] += value;
            node = parent;
            value = -value;
        }
    }

    fn add_virtual_loss(&mut self, mut node: usize) {
        while node != DUMMY {
            let nd = &self.arena[node];
            let parent = nd.parent;
            let mov = nd.mov.max(0) as usize;
            self.arena[node].losses_applied += 1;
            self.arena[parent].child_w[mov] += 1.0;
            node = parent;
        }
    }

    fn revert_virtual_loss(&mut self, mut node: usize) {
        while node != DUMMY {
            let nd = &self.arena[node];
            let parent = nd.parent;
            let mov = nd.mov.max(0) as usize;
            if self.arena[node].losses_applied > 0 {
                self.arena[node].losses_applied -= 1;
                self.arena[parent].child_w[mov] -= 1.0;
            }
            node = parent;
        }
    }
}

/// Port of `parallel_uct_search`. Returns the chosen move.
#[allow(clippy::too_many_arguments)]
pub fn parallel_uct_search<M: EvalModel>(
    env: &GomokuEnv,
    model: &M,
    c_puct_base: f64,
    c_puct_init: f64,
    num_simulations: usize,
    num_parallel: usize,
    deterministic: bool,
) -> i32 {
    assert!(num_simulations >= 1);
    assert!(!env.is_game_over());

    let mut tree = V2Tree {
        arena: vec![V2Node::new(0, 0, DUMMY)],
    };

    // Create root node (evaluate + expand + backup, as in the reference).
    let obs = env.observation();
    let boards_i8: Vec<i8> = obs[..225].iter().map(|&v| v as i8).collect();
    let players_i8 = [env.to_play];
    let (priors, values) = model.forward(&boards_i8, &players_i8);
    let root = tree.arena.len();
    tree.arena.push(V2Node::new(env.to_play, 0, DUMMY));
    // Root's N/W live in the dummy's child arrays at slot `mov` (=0).
    tree.arena[root].child_p.copy_from_slice(
        &priors.iter().map(|&p| p as f64).collect::<Vec<f64>>()[..],
    );
    tree.arena[root].is_expanded = true;
    tree.backup(root, values[0] as f64);

    let root_legal = env.legal_actions();

    while tree.n(root) < (num_simulations + num_parallel) as f64 {
        let mut leaves: Vec<(usize, GomokuEnv)> = Vec::new();
        let mut failsafe = 0;

        while leaves.len() < num_parallel && failsafe < num_parallel * 2 {
            failsafe += 1;
            let mut node = root;
            let mut sim_env = env.clone();
            let mut done = sim_env.is_game_over();

            // Phase 1 — select.
            let mut reward = 0.0;
            while tree.arena[node].is_expanded {
                node = tree.best_child(
                    node,
                    &sim_env.legal_actions(),
                    c_puct_base,
                    c_puct_init,
                    sim_env.opponent_player(),
                );
                let mov = tree.arena[node].mov;
                let (r, d) = sim_env.step(mov as usize);
                reward = r;
                done = d;
                if done {
                    break;
                }
            }

            if done {
                // The reward is for the last player who made the move.
                tree.backup(node, -reward);
                continue;
            } else {
                tree.add_virtual_loss(node);
                leaves.push((node, sim_env));
            }
        }

        if !leaves.is_empty() {
            let b = leaves.len();
            let mut boards = vec![0i8; b * 225];
            let mut players = vec![0i8; b];
            for (j, (_, e)) in leaves.iter().enumerate() {
                boards[j * 225..(j + 1) * 225].copy_from_slice(&e.board);
                players[j] = e.to_play;
            }
            let (priors, values) = model.forward(&boards, &players);

            for (j, (leaf, _)) in leaves.iter().enumerate() {
                tree.revert_virtual_loss(*leaf);
                if tree.arena[*leaf].is_expanded {
                    continue;
                }
                for m in 0..NUM_ACTIONS {
                    tree.arena[*leaf].child_p[m] = priors[j * NUM_ACTIONS + m] as f64;
                }
                tree.arena[*leaf].is_expanded = true;
                tree.backup(*leaf, values[j] as f64);
            }
        }
    }

    // Play — the benchmark only ever uses deterministic=true (most visits).
    let _ = deterministic;
    let mut move_idx = 0usize;
    let mut best_n = f64::NEG_INFINITY;
    for m in 0..NUM_ACTIONS {
        if tree.arena[root].child_n[m] > best_n {
            best_n = tree.arena[root].child_n[m];
            move_idx = m;
        }
    }

    debug_assert!(root_legal[move_idx] == 1.0);
    move_idx as i32
}
