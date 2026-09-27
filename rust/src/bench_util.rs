//! Timing helpers shared by the benchmark binaries.

use std::time::Instant;

pub struct Stats {
    pub mean_ms: f64,
    pub std_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

/// Warm up `n_warmup` times, then time `n_iter` calls. Prints like the Python
/// `bench()` helper and returns per-call stats.
pub fn bench<F: FnMut()>(name: &str, mut f: F, n_warmup: usize, n_iter: usize) -> Stats {
    for _ in 0..n_warmup {
        f();
    }
    let mut times = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let var =
        times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let stats = Stats {
        mean_ms: mean,
        std_ms: var.sqrt(),
        min_ms: times.iter().cloned().fold(f64::INFINITY, f64::min),
        max_ms: times.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    };
    println!(
        "  {:<20}  mean={:7.2} ms  std={:6.2} ms  min={:7.2} ms  max={:7.2} ms",
        name, stats.mean_ms, stats.std_ms, stats.min_ms, stats.max_ms
    );
    stats
}

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
