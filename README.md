Blazing fast MCTS on CPU: 100,000s sims per second! :rocket: 

<p align="center">
  <img src="benchmarks/puct_speedup.png.png" width="100%" alt="" />
</p>

This repository provides the benchmark for [gumbel-mcts](https://github.com/olivkoch/gumbel-mcts). 

We use a separate repository to keep the original repo minimal and due to a dependency on an external reference.

This benchmark demonstrates the following:

1. Our PUCT implementation strictly matches the output/policy of a golden standard [mcts_v2.py](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py)
2. Our PUCT implementation is **2-15X faster** than this golden standard, both on Mac and NVIDIA GPUs
3. Gumbel is **much more simulation-efficient** than PUCT at similar throughput

All benchmarks use Gomoku (15×15, 225 actions). Run them from the `benchmarks/` directory using `uv` (e.g. `uv run python benchmarks/benchmark_puct_speedup.py`)

### PUCT Efficiency (`benchmark_puct_speedup.py`)

Measures the speedup of our PUCT implementation (V3 = `puct.py`) compared to V2 = `reference.py` taken from [mcts_v2.py](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py)

The `n_parallel` parameter is used by the reference only. We pick the value that is most favorable to the reference for each games x sims configuration.

On a Mac M3 Pro:

| Config                                   | V2 (s)     | V3 (s)     | V3 sims/s    | Speedup     |
| -----------------------------------------|------------|------------|--------------|------------ |
| 8 games × 50 sims × 32 parallel          | 0.048      | 0.039      | 10254        | 1.24 x |
| 32 games × 50 sims × 32 parallel         | 0.209      | 0.048      | 33397        | 4.37 x |
| 64 games × 100 sims × 64 parallel        | 0.511      | 0.078      | 82449        | 6.58 x |
| 128 games × 200 sims × 16 parallel       | 2.210      | 0.203      | 126362       | 10.9 x |
| 256 games × 200 sims × 32 parallel       | 3.742      | 0.272      | 188123       | 13.7 x |
| 1024 games × 800 sims × 64 parallel      | 47.262     | 3.907      | 209676       | 12.1 x |

On an NVIDIA A100:

| Config                                   | V2 (s)     | V3 (s)     | V3 sims/s    | Speedup     |
| -----------------------------------------|------------|------------|--------------|------------ |
| 8 games × 50 sims × 16 parallel          | 0.079      | 0.059      | 6741         | 1.33 x |
| 32 games × 50 sims × 8 parallel          | 0.541      | 0.150      | 10667        | 3.61 x |
| 64 games × 100 sims × 64 parallel        | 0.974      | 0.141      | 45488        | 6.92 x |
| 128 games × 200 sims × 16 parallel       | 3.478      | 0.427      | 59945        | 8.15 x |
| 256 games × 200 sims × 32 parallel       | 6.967      | 1.314      | 38961        | 5.30 x |
| 1024 games × 800 sims × 64 parallel      | 94.432     | 6.400      | 128002       | 14.76 x |


### Throughput (`benchmark_throughput.py`)

Measures wall-clock time per MCTS search call at increasing scale.

| Variant | 50 sims / 500 nodes | 200 sims / 5K nodes | 800 sims / 50K nodes |
|---|---|---|---|
| **PUCT** | 2.70 ms | 11.0 ms | 52.5 ms |
| **GumbelDense** | 2.91 ms | 13.9 ms | 72.8 ms |
| **GumbelSparse** | 3.40 ms | 16.4 ms | 89.0 ms |

PUCT is fastest because it skips sequential halving. GumbelSparse is slower than GumbelDense due to indirection overhead, but the relationship reverses for larger action spaces (e.g. chess, 4672 actions) where sparse storage avoids iterating over thousands of illegal moves.

### Win rate (`benchmark_winrate.py`)

PUCT vs GumbelSparse head-to-head, 30 games, 50 sims/move, alternating colors.

| Model | PUCT wins | Gumbel wins |
|---|---|---|
| **Random** (uniform policy) | **100%** | 0% |
| **Heuristic** (simulates trained network) | 0% | **100%** |

With an uninformative policy, PUCT's broader UCB exploration wins. With an informative policy, Gumbel's sequential halving exploits the logits and dominates.

### Simulation efficiency (`benchmark_sparse_gumbel_efficiency.py`)

Gumbel fixed at 8 sims — how many PUCT sims to match? (we proxy a trained model with a noisy heuristic one)

| PUCT sims | PUCT win% |
|---|---|
| 8 | 45% |
| 16 | 35% |
| 32 | 38% |
| 64 | 43% |
| 128 | 43% |
| 256 | **55%** |

PUCT needs roughly **200× the simulation budget** to match Gumbel when the policy prior is informative.

### Validation of PUCT against reference

To compare the output of `puct.py` against the golden reference: `uv run python tests/test_puct.py gomoku`