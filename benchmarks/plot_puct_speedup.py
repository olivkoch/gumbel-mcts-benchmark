"""Plot the speedup of PUCT v3 over v2 (reference) from benchmark results."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Data from README ---

configs = [
    "8×50",
    "32×50",
    "64×100",
    "128×200",
    "256×200",
    "1024×800",
]

total_sims = [
    8 * 50,
    32 * 50,
    64 * 100,
    128 * 200,
    256 * 200,
    1024 * 800,
]

speedup_m3 = [1.24, 4.37, 6.58, 10.9, 13.7, 12.1]
speedup_a100 = [1.33, 3.61, 6.92, 8.15, 5.30, 14.76]

# Rust engines, also vs the Python V2 baseline, same machine (M3 Max).
# V2 time is the most favorable `n_parallel` per config, matching the
# convention above.
# Rust ext = gumbel_mcts_rs (PyO3, model called back into Python).
# Rust bin = pure-Rust binaries (Burn model in-process).
speedup_rs_ext = [0.81, 6.39, 2.91, 8.49, 17.63, 17.93]
speedup_rs_bin = [157.5, 229.1, 147.5, 39.7, 24.9, 23.2]

# --- Plot ---

x_labels = [
    "8 games\n× 50 sims",
    "32 games\n× 50 sims",
    "64 games\n× 100 sims",
    "128 games\n× 200 sims",
    "256 games\n× 200 sims",
    "1024 games\n× 800 sims",
]
x_pos = np.arange(len(configs))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x_pos, speedup_m3, "o-", color="#2563eb", linewidth=2.2,
        markersize=8, label="Mac M3 Pro", zorder=3)
ax.plot(x_pos, speedup_a100, "s-", color="#dc2626", linewidth=2.2,
        markersize=8, label="NVIDIA A100", zorder=3)
ax.plot(x_pos, speedup_rs_ext, "^-", color="#9333ea", linewidth=2.2,
        markersize=8, label="Rust ext (PyO3), M3 Max", zorder=3)
ax.plot(x_pos, speedup_rs_bin, "D-", color="#ea580c", linewidth=2.2,
        markersize=8, label="Rust bin (Burn), M3 Max", zorder=3)

ax.axhline(1, color="black", linewidth=2.5, linestyle="--", label="Baseline (1×)", zorder=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_yscale("log")
ax.set_ylabel("Speedup (×)", fontsize=12)
ax.set_title("Speedup of our PUCT implementation against baseline (V2 reference)", fontsize=14, pad=12)

y_ticks = [1, 2, 5, 10, 15, 50, 100, 250]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{v}×" for v in y_ticks], fontsize=12)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, which="major", linestyle=":", alpha=0.5)
ax.set_ylim(bottom=0.8)

fig.tight_layout()
fig.savefig("benchmarks/puct_speedup.png", dpi=180)
print("Saved benchmarks/puct_speedup.png")
plt.show()
