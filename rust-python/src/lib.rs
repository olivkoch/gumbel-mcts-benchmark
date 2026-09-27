//! Python bindings for the Rust gumbel-mcts engines.
//!
//! Exposes `PUCT`, `GumbelDense`, `GumbelSparse` with the same API as their
//! Python counterparts, so the benchmark scripts can use them as drop-in
//! replacements. The NN evaluation is a callback into Python:
//! `model.forward_for_mcts({"boards": torch.Tensor[B, BL],
//!                          "current_player": torch.Tensor[B]})`
//! returning `{"policy": ..., "value": ...}`.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::SeedableRng;

use gumbel_mcts_bench::game::{Gomoku, TicTacToe};
use gumbel_mcts_bench::gumbel_dense::GumbelDense as DenseRs;
use gumbel_mcts_bench::gumbel_sparse::GumbelSparse as SparseRs;
use gumbel_mcts_bench::model::EvalModel;
use gumbel_mcts_bench::puct::Puct as PuctRs;

// =============================================================================
// Model bridge
// =============================================================================

/// Convert a torch tensor (or numpy array) to a numpy float32 array object.
fn to_numpy_f32<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.hasattr("detach")? {
        obj.call_method0("detach")?
            .call_method0("float")?
            .call_method0("cpu")?
            .call_method0("numpy")
    } else if obj.hasattr("astype")? {
        obj.call_method1("astype", ("float32",))
    } else {
        Err(PyValueError::new_err(
            "model output must be a torch tensor or numpy array",
        ))
    }
}

struct PyEvalModel {
    model: Py<PyAny>,
    device: String,
}

impl EvalModel for PyEvalModel {
    fn forward(&self, boards: &[i8], players: &[i8]) -> (Vec<f32>, Vec<f32>) {
        let b = players.len();
        let bl = boards.len() / b.max(1);
        Python::attach(|py| -> PyResult<(Vec<f32>, Vec<f32>)> {
            let torch = pyo3::types::PyModule::import(py, "torch")?;

            let boards_np = PyArray1::from_slice(py, boards).reshape([b, bl])?;
            let boards_t = torch
                .getattr("from_numpy")?
                .call1((boards_np,))?
                .call_method1("to", (&self.device,))?;

            let players64: Vec<i64> = players.iter().map(|&p| p as i64).collect();
            let players_np = PyArray1::from_slice(py, &players64);
            let players_t = torch
                .getattr("from_numpy")?
                .call1((players_np,))?
                .call_method1("to", (&self.device,))?;

            let batch = PyDict::new(py);
            batch.set_item("boards", boards_t)?;
            batch.set_item("current_player", players_t)?;

            let out = self.model.call_method1(py, "forward_for_mcts", (batch,))?;
            let out = out.bind(py);

            let pol_obj = to_numpy_f32(&out.get_item("policy")?)?;
            let pol = pol_obj.cast::<PyArray2<f32>>()?.clone();
            let policy = pol.readonly().as_slice()?.to_vec();

            let val_flat = to_numpy_f32(&out.get_item("value")?)?.call_method0("ravel")?;
            let val = val_flat.cast::<PyArray1<f32>>()?.clone();
            let value = val.readonly().as_slice()?.to_vec();

            Ok((policy, value))
        })
        .expect("python eval callback failed")
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn logic_name(logic: &Bound<PyAny>) -> PyResult<String> {
    if let Ok(s) = logic.extract::<String>() {
        return Ok(s);
    }
    logic.getattr("GAME_NAME")?.extract::<String>()
}

fn read_boards(boards: &Bound<PyAny>) -> PyResult<Vec<i8>> {
    let arr: PyReadonlyArrayDyn<i8> = boards.extract()?;
    let view = arr.as_array();
    Ok(view.iter().copied().collect())
}

fn read_players(players: &Bound<PyAny>) -> PyResult<Vec<i8>> {
    let arr: PyReadonlyArrayDyn<i64> = players.extract()?;
    Ok(arr.as_array().iter().map(|&v| v as i8).collect())
}

// =============================================================================
// Engine enums (dispatch on game)
// =============================================================================

enum PuctEngine {
    Gomoku(PuctRs<Gomoku>),
    Ttt(PuctRs<TicTacToe>),
}

enum DenseEngine {
    Gomoku(DenseRs<Gomoku>),
    Ttt(DenseRs<TicTacToe>),
}

enum SparseEngine {
    Gomoku(SparseRs<Gomoku>),
    Ttt(SparseRs<TicTacToe>),
}

// =============================================================================
// PyO3 classes
// =============================================================================

#[pyclass(name = "PUCT")]
struct PyPuct {
    engine: PuctEngine,
    device: String,
}

#[pymethods]
impl PyPuct {
    #[new]
    #[pyo3(signature = (n_games, max_nodes, logic, device="cpu"))]
    fn new(n_games: usize, max_nodes: usize, logic: &Bound<PyAny>, device: &str) -> PyResult<Self> {
        let name = logic_name(logic)?;
        let engine = match name.as_str() {
            "gomoku" => PuctEngine::Gomoku(PuctRs::<Gomoku>::new(n_games, max_nodes)),
            "tictactoe" => PuctEngine::Ttt(PuctRs::<TicTacToe>::new(n_games, max_nodes)),
            other => return Err(PyValueError::new_err(format!("unknown game: {other}"))),
        };
        Ok(Self { engine, device: device.to_string() })
    }

    fn reset(&mut self) {
        match &mut self.engine {
            PuctEngine::Gomoku(t) => t.reset(),
            PuctEngine::Ttt(t) => t.reset(),
        }
    }

    fn initialize_roots(
        &mut self,
        active_games: Vec<usize>,
        boards: &Bound<PyAny>,
        players: &Bound<PyAny>,
    ) -> PyResult<()> {
        let boards = read_boards(boards)?;
        let players = read_players(players)?;
        match &mut self.engine {
            PuctEngine::Gomoku(t) => t.initialize_roots(&active_games, &boards, &players),
            PuctEngine::Ttt(t) => t.initialize_roots(&active_games, &boards, &players),
        }
        Ok(())
    }

    #[pyo3(signature = (model, active_games, num_simulations=50, c_puct_base=19652.0, c_puct_init=1.25))]
    fn run_simulation_batch(
        &mut self,
        model: Py<PyAny>,
        active_games: Vec<usize>,
        num_simulations: usize,
        c_puct_base: f64,
        c_puct_init: f64,
    ) {
        let eval = PyEvalModel { model, device: self.device.clone() };
        match &mut self.engine {
            PuctEngine::Gomoku(t) => t.run_simulation_batch(
                &eval, &active_games, num_simulations, c_puct_base, c_puct_init,
            ),
            PuctEngine::Ttt(t) => t.run_simulation_batch(
                &eval, &active_games, num_simulations, c_puct_base, c_puct_init,
            ),
        }
    }

    fn get_all_root_data<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let (visits, root_q) = match &self.engine {
            PuctEngine::Gomoku(t) => t.get_all_root_data(n_active),
            PuctEngine::Ttt(t) => t.get_all_root_data(n_active),
        };
        let a = visits.len() / n_active.max(1);
        let visits = PyArray1::from_vec(py, visits)
            .reshape([n_active, a])
            .expect("reshape visits");
        (visits, PyArray1::from_vec(py, root_q))
    }

    fn get_max_depth(&self) -> i32 {
        match &self.engine {
            PuctEngine::Gomoku(t) => t.get_max_depth(),
            PuctEngine::Ttt(t) => t.get_max_depth(),
        }
    }
}

// -----------------------------------------------------------------------------

#[pyclass(name = "GumbelDense")]
struct PyGumbelDense {
    engine: DenseEngine,
    device: String,
    rng: StdRng,
}

#[pymethods]
impl PyGumbelDense {
    #[new]
    #[pyo3(signature = (n_games, max_nodes, logic, device="cpu", c_visit=50.0, c_scale=1.0))]
    fn new(
        n_games: usize,
        max_nodes: usize,
        logic: &Bound<PyAny>,
        device: &str,
        c_visit: f64,
        c_scale: f64,
    ) -> PyResult<Self> {
        let name = logic_name(logic)?;
        let engine = match name.as_str() {
            "gomoku" => DenseEngine::Gomoku(DenseRs::<Gomoku>::new(n_games, max_nodes, c_visit, c_scale)),
            "tictactoe" => DenseEngine::Ttt(DenseRs::<TicTacToe>::new(n_games, max_nodes, c_visit, c_scale)),
            other => return Err(PyValueError::new_err(format!("unknown game: {other}"))),
        };
        Ok(Self { engine, device: device.to_string(), rng: StdRng::from_entropy() })
    }

    fn initialize_roots(
        &mut self,
        active_games: Vec<usize>,
        boards: &Bound<PyAny>,
        players: &Bound<PyAny>,
    ) -> PyResult<()> {
        let boards = read_boards(boards)?;
        let players = read_players(players)?;
        match &mut self.engine {
            DenseEngine::Gomoku(t) => t.initialize_roots(&active_games, &boards, &players, &mut self.rng),
            DenseEngine::Ttt(t) => t.initialize_roots(&active_games, &boards, &players, &mut self.rng),
        }
        Ok(())
    }

    #[pyo3(signature = (model, active_games, num_simulations=50))]
    fn run_simulation_batch<'py>(
        &mut self,
        py: Python<'py>,
        model: Py<PyAny>,
        active_games: Vec<usize>,
        num_simulations: usize,
    ) -> Bound<'py, PyArray1<i32>> {
        let eval = PyEvalModel { model, device: self.device.clone() };
        let moves = match &mut self.engine {
            DenseEngine::Gomoku(t) => {
                t.run_simulation_batch(&eval, &active_games, num_simulations, &mut self.rng)
            }
            DenseEngine::Ttt(t) => {
                t.run_simulation_batch(&eval, &active_games, num_simulations, &mut self.rng)
            }
        };
        PyArray1::from_vec(py, moves)
    }

    fn get_improved_policy<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        let pi = match &self.engine {
            DenseEngine::Gomoku(t) => t.get_improved_policy(n_active),
            DenseEngine::Ttt(t) => t.get_improved_policy(n_active),
        };
        let a = pi.len() / n_active.max(1);
        PyArray1::from_vec(py, pi).reshape([n_active, a]).expect("reshape pi")
    }

    #[pyo3(signature = (n_active, chosen_moves=None))]
    fn get_gumbel_root_value<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
        chosen_moves: Option<Vec<i32>>,
    ) -> Bound<'py, PyArray1<f32>> {
        let v = match &self.engine {
            DenseEngine::Gomoku(t) => t.get_gumbel_root_value(n_active, chosen_moves.as_deref()),
            DenseEngine::Ttt(t) => t.get_gumbel_root_value(n_active, chosen_moves.as_deref()),
        };
        PyArray1::from_vec(py, v)
    }

    fn get_all_root_data<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let (visits, root_q) = match &self.engine {
            DenseEngine::Gomoku(t) => t.get_all_root_data(n_active),
            DenseEngine::Ttt(t) => t.get_all_root_data(n_active),
        };
        let a = visits.len() / n_active.max(1);
        let visits = PyArray1::from_vec(py, visits)
            .reshape([n_active, a])
            .expect("reshape visits");
        (visits, PyArray1::from_vec(py, root_q))
    }

    fn get_max_depth(&self) -> i32 {
        match &self.engine {
            DenseEngine::Gomoku(t) => t.tree.get_max_depth(),
            DenseEngine::Ttt(t) => t.tree.get_max_depth(),
        }
    }
}

// -----------------------------------------------------------------------------

#[pyclass(name = "GumbelSparse")]
struct PyGumbelSparse {
    engine: SparseEngine,
    device: String,
    rng: StdRng,
}

#[pymethods]
impl PyGumbelSparse {
    #[new]
    #[pyo3(signature = (n_games, max_nodes, logic, device="cpu", c_visit=50.0, c_scale=1.0, avg_branching=35, max_legal_moves=256))]
    fn new(
        n_games: usize,
        max_nodes: usize,
        logic: &Bound<PyAny>,
        device: &str,
        c_visit: f64,
        c_scale: f64,
        avg_branching: usize,
        max_legal_moves: usize,
    ) -> PyResult<Self> {
        let name = logic_name(logic)?;
        let engine = match name.as_str() {
            "gomoku" => SparseEngine::Gomoku(SparseRs::<Gomoku>::new(
                n_games, max_nodes, c_visit, c_scale, avg_branching, max_legal_moves,
            )),
            "tictactoe" => SparseEngine::Ttt(SparseRs::<TicTacToe>::new(
                n_games, max_nodes, c_visit, c_scale, avg_branching, max_legal_moves,
            )),
            other => return Err(PyValueError::new_err(format!("unknown game: {other}"))),
        };
        Ok(Self { engine, device: device.to_string(), rng: StdRng::from_entropy() })
    }

    fn initialize_roots(
        &mut self,
        active_games: Vec<usize>,
        boards: &Bound<PyAny>,
        players: &Bound<PyAny>,
    ) -> PyResult<()> {
        let boards = read_boards(boards)?;
        let players = read_players(players)?;
        match &mut self.engine {
            SparseEngine::Gomoku(t) => t.initialize_roots(&active_games, &boards, &players),
            SparseEngine::Ttt(t) => t.initialize_roots(&active_games, &boards, &players),
        }
        Ok(())
    }

    #[pyo3(signature = (model, active_games, num_simulations=50))]
    fn run_simulation_batch<'py>(
        &mut self,
        py: Python<'py>,
        model: Py<PyAny>,
        active_games: Vec<usize>,
        num_simulations: usize,
    ) -> Bound<'py, PyArray1<i32>> {
        let eval = PyEvalModel { model, device: self.device.clone() };
        let moves = match &mut self.engine {
            SparseEngine::Gomoku(t) => {
                t.run_simulation_batch(&eval, &active_games, num_simulations, &mut self.rng, None)
            }
            SparseEngine::Ttt(t) => {
                t.run_simulation_batch(&eval, &active_games, num_simulations, &mut self.rng, None)
            }
        };
        PyArray1::from_vec(py, moves)
    }

    fn get_improved_policy<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        let pi = match &self.engine {
            SparseEngine::Gomoku(t) => t.get_improved_policy(n_active),
            SparseEngine::Ttt(t) => t.get_improved_policy(n_active),
        };
        let a = pi.len() / n_active.max(1);
        PyArray1::from_vec(py, pi).reshape([n_active, a]).expect("reshape pi")
    }

    #[pyo3(signature = (n_active, chosen_moves=None))]
    fn get_gumbel_root_value<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
        chosen_moves: Option<Vec<i32>>,
    ) -> Bound<'py, PyArray1<f32>> {
        let v = match &self.engine {
            SparseEngine::Gomoku(t) => t.get_gumbel_root_value(n_active, chosen_moves.as_deref()),
            SparseEngine::Ttt(t) => t.get_gumbel_root_value(n_active, chosen_moves.as_deref()),
        };
        PyArray1::from_vec(py, v)
    }

    fn get_all_root_data<'py>(
        &self,
        py: Python<'py>,
        n_active: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let (visits, root_q) = match &self.engine {
            SparseEngine::Gomoku(t) => t.get_all_root_data(n_active),
            SparseEngine::Ttt(t) => t.get_all_root_data(n_active),
        };
        let a = visits.len() / n_active.max(1);
        let visits = PyArray1::from_vec(py, visits)
            .reshape([n_active, a])
            .expect("reshape visits");
        (visits, PyArray1::from_vec(py, root_q))
    }

    fn get_max_depth(&self) -> i32 {
        match &self.engine {
            SparseEngine::Gomoku(t) => t.get_max_depth(),
            SparseEngine::Ttt(t) => t.get_max_depth(),
        }
    }
}

#[pymodule]
fn gumbel_mcts_rs(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPuct>()?;
    m.add_class::<PyGumbelDense>()?;
    m.add_class::<PyGumbelSparse>()?;
    Ok(())
}
