# Training for all ten minnows

The September 6 changes target the nine-survivor local optimum. They have CPU
regression coverage, but have not been trained or shown to converge at slow speed
0.2. No GPU computation or optimizer updates were run during this change.

The existing `perfect_priority_v1/training_history.csv` records 98.14% perfect
games at slow speed 0.4 (update 240). Its final training row at speed 0.3 records
8.992 mean survivors and zero perfect games. These are historical results under
that run's settings, not evaluations of the revised rewards. The actor in
`checkpoints/perfect_priority_v1/best_speed_0.400.pt` is shape-compatible with the
current model; compatibility alone does not establish current performance.

| Factor | Previous | Revised |
| --- | --- | --- |
| Slow death | -1 at episode end | -1 immediately, no duplicate terminal charge |
| First team failure | No explicit penalty | -10 once, on first death or unresolved timeout |
| Perfect completion | +10 team bonus | Retained |
| Actor advantages | 80% team / 20% individual, separately normalized | 100% team |
| GAE lambda | 0.95 | 0.99 |
| Rollout length | 128 decisions | 256 decisions |
| Fast arrival heuristic | Penalty for unresolved slow teammates | Disabled by default |
| Trailing progress | Changing multiplier times progress difference | Discount-correct state potential difference |
| Best checkpoint within a speed | Mean saved first | Perfect rate first; mean saved breaks ties |
| Curriculum gate | 8.8 average saved; 10% validation perfect | 9.8 average saved; 90% recent training AND validation perfect |

Episodes continue after a loss, so recovering the other minnows remains useful.
Timeout without a collision also loses the first-failure reward; hiding in the
safe starting zone cannot avoid failure. The first-failure penalty is shared,
so actions by the fast rescuer and the rest of the team receive that signal.

For an undiscounted accounting example with progress/wall shaping excluded:
saving all ten pays 10 crossings + 10 terminal survival + 10 perfect = 30.
Saving nine after losing a slow minnow pays 9 + 9 - 1 death - 10 failure = 7.
Actual discounted returns also depend on event timing and the other shaping
terms. This remains a weighted reward objective, not a mathematical guarantee
of maximizing perfect-game probability or finding a feasible perfect strategy.

The trailing potential includes both trailing progress and the safe-fraction
multiplier. Its reward is `gamma * Phi(next_state) - Phi(state)`, with terminal
potential zero. This prevents that particular shaping term from creating extra
discounted return through backtracking as the multiplier changes. Other reward
terms are not claimed to be policy invariant. See
[Ng, Harada and Russell on potential shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

At gamma 0.999, the direct GAE weight on a residual 100 decisions away changes
from `(0.999 * 0.95)^100`, about 0.54%, to `(0.999 * 0.99)^100`, about 33%.
This extends credit assignment but increases variance; the existing KL rollback
and gradient clipping remain. Bootstrapped values also carry future information,
so these numbers are not the entire influence of distant outcomes. See the
[GAE paper](https://arxiv.org/abs/1506.02438). Longer rollouts also increase memory
and work per update.

## Starting a later experiment

Old checkpoints cannot be resumed under changed reward/PPO settings. Use a new
output directory and `--initialize-actor-from` (Make variable
`INITIALIZE_ACTOR_FROM`) to reuse compatible encoders, attention, actor and
exploration scale with fresh critics, optimizer, history and curriculum state.
Omit it for a fresh policy. Start the curriculum at the checkpoint's trained
speed, such as 0.4, rather than assuming a 0.4 policy already works at 0.2.

The command below only prints a proposed CPU command; it does not train:

```sh
make -n train-curriculum RUN=all_ten_v3 DEVICE=cpu \
  INITIALIZE_ACTOR_FROM=checkpoints/perfect_priority_v1/best_speed_0.400.pt
```

For a later learning experiment, inspect both stochastic training and deterministic
validation perfect-game rates at each speed. After tuning, evaluate at the actual
0.2 target on fresh seeds; the repeated validation seed is a tuning set. A stage
that fails the gate should remain there. If it cannot improve, that is evidence
to investigate exploration or task feasibility, not a reason to lower the gate
and report convergence. Keep per-speed checkpoints: `best.pt` still prioritizes
the hardest speed reached before comparing performance at that speed.

## CPU verification

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/shark-mpl \
  .venv/bin/python -m unittest -v test_all_ten
```

The tests cover immediate and one-time loss accounting, multiple deaths, timeout,
reset, 10/10 versus 9/10 reward accounting, absorbing terminal rewards, potential
telescoping across changing safe fractions, all-fast teams, perfect-first
selection, actor-only initialization, and finite CPU rollout targets. They guard
against CUDA initialization and do not call the optimizer.
