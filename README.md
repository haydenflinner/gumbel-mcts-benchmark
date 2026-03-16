This repository provides the benchmark for [gumbel-mcts](https://github.com/olivkoch/gumbel-mcts). 

We use a separate repository to keep the original repo minimal and due to a dependency on an external reference.

All benchmarks use Gomoku (15×15, 225 actions). Run them from the `benchmarks/` directory using `uv` (e.g. `uv run python benchmarks/benchmark_puct_speedup.py`)

### PUCT Efficiency (`benchmark_puct_speedup.py`)

Measures the speedup of the PUCT implementation (`puct.py`) compared to `reference.py` taken from [mcts_v2.py](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py)

On a Mac M3 Pro:

| Config                    | V2 (s)     | V3 (s)     | V3 sims/s   | Speedup    |
| --------------------------|------------|------------|-------------|------------|
| 8 games × 50 sims         | 0.076      | 0.038      | 10647       | 2.02x      | 
| 32 games × 50 sims        | 0.276      | 0.039      | 41526       | 7.18x      |
| 64 games × 100 sims       | 0.713      | 0.078      | 81939       | 9.12x      |
| 128 games × 200 sims      | 2.240      | 0.190      | 134855      | 11.80x     |
| 256 games × 200 sims      | 3.724      | 0.273      | 187861      | 13.66x     | 
| 1024 games × 800 sims     | 47.146     | 3.357      | 244013      | 14.04x     |

On an NVIDIA A100:

| Config                    | V2 (s)     | V3 (s)     | V3 sims/s   | Speedup    | 
| --------------------------|------------|------------|-------------|------------|
| 8 games × 50 sims         | 0.073      | 0.035      | 11361       | 2.07x      |
| 32 games × 50 sims        | 0.275      | 0.056      | 28639       | 4.92x      |
| 64 games × 100 sims       | 0.802      | 0.086      | 74596       | 9.34x      |
| 128 games × 200 sims      | 2.797      | 0.206      | 124129      | 13.56x     |
| 256 games × 200 sims      | 5.385      | 2.327      | 22003       | 2.31x      |
| 1024 games × 800 sims     | 89.579     | 4.676      | 175196      | 19.16x     |

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