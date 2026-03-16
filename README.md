# gumbel-mcts

A lightweight and modular Gumbel MCTS implementation

## Motivation

Most open-source MCTS implementations provide only standard PUCT (AlphaZero-style) or dense Gumbel MCTS (MuZero, EfficientZero). Sparse Gumbel MCTS — which makes Gumbel planning practical for large action spaces like chess (4672 actions) or Go (362) — is essentially absent.

This project provides three MCTS variants that cover the full spectrum:

### 1. PUCT (`puct.py`) — Standard AlphaZero UCB

Classic PUCT selects actions by maximizing $Q(s,a) + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$, then samples moves proportionally to visit counts. It's the **fastest** variant and explores broadly, making it robust when the policy prior is weak or untrained. The downside: it needs many simulations to concentrate on the best move, and doesn't produce a theoretically improved policy.

This version has been extensively tested against [mcts_v2](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/mcts_v2.py) which is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python." by Michael Hu.


### 2. GumbelDense (`gumbel_dense.py`) — Gumbel MCTS with dense storage

Implements *Sequential Halving with Gumbel* from [Danihelka et al. (2022)](https://openreview.net/forum?id=bERaNdoegnO). At the root, Gumbel noise converts action selection into a sample-without-replacement problem. The simulation budget is split across halving phases that progressively prune weaker candidates. After search, the output is a theoretically grounded improved policy $\pi' = \text{softmax}(\log \pi + \sigma \cdot \bar{Q}_{\text{completed}})$ — directly usable as a training target without temperature tuning.

This makes Gumbel MCTS **dramatically more sample-efficient** than PUCT when the policy prior is informative (benchmarks show PUCT needs ~32× the simulation budget to match). The trade-off: it stores edges as dense `(max_nodes, NUM_ACTIONS)` arrays, so memory scales with the full action space regardless of how many moves are actually legal.

### 3. GumbelSparse (`gumbel_sparse.py`) — Gumbel MCTS with sparse storage

Runs the **exact same** Gumbel sequential halving algorithm as GumbelDense, but stores edges in a flat pool where each node only allocates slots for its legal moves. The difference is purely in memory layout:

| | Dense | Sparse |
|---|---|---|
| Storage | `(max_nodes, NUM_ACTIONS)` per node | `O(legal_moves)` per node |
| 200K nodes, chess | ~10.5 GB | ~98 MB |

This makes GumbelSparse the only variant that can realistically run games with large action spaces. It's also ~19% faster than GumbelDense at scale on Gomoku, and the gap widens with larger action spaces.

### Which one should I use?

| Scenario | Variant |
|---|---|
| Weak / random policy, need broad exploration | **PUCT** |
| Small action space + trained policy | **GumbelDense** |
| Large action space (chess, Go) or memory-constrained | **GumbelSparse** |
| Need improved policy $\pi'$ as training target | **GumbelDense** or **GumbelSparse** |
| Not sure | **GumbelSparse** (drop-in replacement for GumbelDense, works everywhere) |



## Usage

```
def play_game():
    logic = TicTacToeLogic()
    model = TinyModel()
    model.eval()

    board = np.zeros((3, 3), dtype=np.int8)
    player = 1
    symbols = {0: ".", 1: "X", 2: "O"}

    while True:
        tree = GumbelSparse(n_games=1, max_nodes=500, device="cpu", logic=logic)
        tree.initialize_roots([0], board.ravel()[None], np.array([player]))
        move = tree.run_simulation_batch(model, [0], num_simulations=50)
        action = move[0]

        _, winner, done, board = logic.fast_step(board, action, player)
```

## Benchmarks

All benchmarks use Gomoku (15×15, 225 actions) on CPU. Run them from the `benchmarks/` directory.

### Speed (`benchmark_throughput.py`)

Measures wall-clock time per MCTS search call at increasing scale.

| Variant | 50 sims / 500 nodes | 200 sims / 5K nodes | 800 sims / 50K nodes |
|---|---|---|---|
| **PUCT** | 2.70 ms | 11.0 ms | 52.5 ms |
| **GumbelDense** | 2.91 ms | 13.9 ms | 72.8 ms |
| **GumbelSparse** | 3.40 ms | 16.4 ms | 89.0 ms |

PUCT is fastest because it skips sequential halving. GumbelSparse is slower than GumbelDense on Gomoku (225 actions) due to indirection overhead, but the relationship reverses for larger action spaces (e.g. chess, 4672 actions) where sparse storage avoids iterating over thousands of illegal moves.

### Win rate (`benchmark_winrate.py`)

PUCT vs GumbelSparse head-to-head, 30 games, 50 sims/move, alternating colors.

| Model | PUCT wins | Gumbel wins |
|---|---|---|
| **Random** (uniform policy) | **100%** | 0% |
| **Heuristic** (simulates trained network) | 0% | **100%** |

With an uninformative policy, PUCT's broader UCB exploration wins. With an informative policy, Gumbel's sequential halving exploits the logits and dominates — matching the paper's thesis.

### Simulation efficiency (`benchmark_asymmetry.py`)

Gumbel fixed at 8 sims — how many PUCT sims to match? (Heuristic model, noise=15.0, 40 games per tier.)

| PUCT sims | PUCT win% |
|---|---|
| 8 | 45% |
| 16 | 35% |
| 32 | 38% |
| 64 | 43% |
| 128 | 43% |
| 256 | **55%** |

PUCT needs roughly **32× the simulation budget** to match Gumbel when the policy prior is informative.

