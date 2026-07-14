# GDT-NEAT Iteration Log

Each round documents: (1) the change made, (2) the rationale, (3) results, (4) what was learned.

---

## Round 1: Baseline GDT-NEAT

**Change:** Initial implementation of Gradient-Directed Topogenesis NEAT.

**Core principle:** Topology grows in the direction of policy gradient. For
each non-existent edge, compute the policy-gradient signal it would receive
if it existed with weight 0 (via one backward pass through a
candidate-extended network). Add the top-K edges by |gradient|. Existing
weights refined by REINFORCE. Behavioral clustering (state-action stats)
replaces genetic speciation.

**Config:**
- pop_size=30, n_episodes=2, max_steps=500
- lr_weights=1e-2, pg_steps=3 (3 policy-gradient steps per individual per gen)
- n_candidate_edges=16, edges_to_add=1, p_split_edge=0.1
- diversity_penalty=0.0 (off)

**Results (2 seeds × 15 gens):**
| Benchmark       | Best mean | Best std | Individual |
|-----------------|-----------|----------|------------|
| CartPole-v1     | 500.00    | 0.00     | 500, 500   |
| MountainCar-v0  | -155.25   | 41.25    | -110.5, -200 |
| Acrobot-v1      | -65.75    | 1.75     | -67.5, -64 |

**Baseline comparison (from Phase 0):**
| Benchmark      | NEAT       | RandomSearch | FixedGA    | CMAES      | DQN    |
|----------------|------------|--------------|------------|------------|--------|
| CartPole-v1    | 500        | 500          | 500        | 500        | ~200   |
| MountainCar-v0 | -110       | -200         | -200       | -200       | ?      |
| Acrobot-v1     | -69        | ?            | ?          | ?          | ?      |

**What was learned:**
1. GDT-NEAT solves CartPole (matching NEAT/baselines) and Acrobot (-65 vs
   NEAT's -69, slightly better). On MountainCar it's mixed: one seed solves
   (-110.5), one fails (-200). MountainCar is an exploration problem — the
   reward is -1 per step until the car reaches the flag, so most random
   policies get -200 (timeout).
2. The behavioral diversity penalty is currently off (diversity_penalty=0).
   Without it, the population collapses to similar behaviors, hurting
   exploration on sparse-reward tasks like MountainCar.
3. Topology grows slowly: avg_size 10-12 (started at 8 for CartPole), avg_hidden
   ~1. The algorithm is conservative about adding structure, which is good.
4. The algorithm is slow: ~50s per individual per generation due to 4 rollouts
   per individual (eval, policy-grad, topology-grad, +1 in topology-grad). Need
   to reuse rollouts.

**Next round:** Round 2 — turn on diversity penalty + reuse evaluation rollout
for policy gradient (cut rollouts from 4 to 2 per individual per generation).
