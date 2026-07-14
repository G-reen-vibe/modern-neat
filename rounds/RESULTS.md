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

**Config:** pop=30, episodes=2, lr=1e-2, pg_steps=3, candidates=16, edges_to_add=1

**Results (2 seeds × 15 gens):**
- CartPole-v1: 500, 500 (solves)
- MountainCar-v0: -110.5, -200 (mixed)
- Acrobot-v1: -67.5, -64 (solves; threshold -100)

**Learned:** Algorithm works. Topology grows slowly (8→12 edges). MountainCar
is the exploration challenge.

---

## Round 2: Gradient-based pruning

**Change:** Unify all structural decisions under gradient. Pruning now uses
both weight magnitude AND gradient magnitude — an edge with tiny gradient
isn't contributing and should be pruned.

**Results (2 seeds × 15 gens):**
- CartPole-v1: 500, 500 (same)
- MountainCar-v0: -117, -200 (similar)
- Acrobot-v1: -64.5, -65.5 (similar)

**Learned:** Gradient-based pruning is more aggressive (avg_size 13-23 vs
8-12). Performance similar. Algorithm is more principled now.

---

## Round 3: Diversity penalty

**Change:** Turn on behavioral diversity penalty (diversity_penalty=20).

**Results (2 seeds × 15 gens):**
- CartPole-v1: 499.33 (slight regression)
- MountainCar-v0: -200, -191 (worse!)

**Learned:** Diversity penalty is too blunt. Penalizes good individuals in
dense clusters. Made things worse. Reverted.

---

## Round 4: Separate deterministic eval from stochastic training

**Change:** Use argmax for fitness evaluation (deterministic), but sample
from softmax for policy-gradient training (stochastic, on-policy).

**Results (2 seeds × 15 gens, pop=20):**
- CartPole-v1: 500, 500 (solves; recovered from R3 regression)
- MountainCar-v0: -200, -200 (still fails)
- Acrobot-v1: -63, -64 (solves, slightly better than R1-R2)

**Learned:** Separating eval from training is essential. Argmax eval gives
clean fitness; stochastic training enables exploration. But MountainCar
still has zero reward signal → zero gradient → no learning.

---

## Round 5: Novelty-augmented reward (small bonus)

**Change:** Add novelty bonus to reward during stochastic rollouts.
Novelty = avg distance to k-nearest neighbors in a state-visitation archive.

**Config:** novelty_bonus=0.1, novelty_k=5, archive_size=200

**Results (2 seeds × 20 gens, pop=20):**
- CartPole-v1: 500, 500 (solves)
- MountainCar-v0: -200, -200 (still fails)

**Learned:** Small novelty bonus doesn't help MountainCar. The bonus is too
weak relative to the -1/step env reward.

---

## Round 6: Novelty-augmented reward (large bonus)

**Change:** Increase novelty_bonus to 1.0, archive_size to 500.

**Results (2 seeds × 20 gens, pop=20):**
- CartPole-v1: 500, 500 (solves)
- MountainCar-v0: -200, -200 (still fails)

**Learned:** Even large novelty bonus doesn't help. The issue is that
novelty bonus is added at each step, but the policy gradient through the
softmax is still too noisy. MountainCar is fundamentally hard for
policy-gradient methods. (Note: DQN baseline also failed on MountainCar in
Phase 0.)

---

## Summary so far (Rounds 1-6)

**What works:**
- Gradient-directed topogenesis: topology grows meaningfully (8→15 edges)
- Policy gradient for weights: faster convergence than NEAT's Gaussian perturbation
- Behavioral clustering: principled replacement for genetic speciation
- Deterministic eval + stochastic training: clean fitness + exploration

**What doesn't work:**
- Diversity penalty (too blunt)
- Novelty bonus on MountainCar (insufficient signal)

**Open questions:**
- Can we make the algorithm faster? (Currently ~20s/seed vs NEAT's ~1s)
- Can we solve MountainCar? (Fundamental exploration challenge)
- Can we improve sample efficiency further?
