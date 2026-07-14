"""Smoke test: run GDT-NEAT on CartPole for 5 generations."""
import sys
sys.path.insert(0, "/home/z/my-project/modern-neat")
import time
from src.gdt_neat import GDTNEAT, GDTConfig

cfg = GDTConfig(
    pop_size=10,
    n_inputs=4,
    n_outputs=2,
    n_episodes=2,
    max_steps=500,
    seed=0,
    pg_steps=2,
    n_candidate_edges=8,
    edges_to_add=1,
    p_split_edge=0.1,
    diversity_penalty=0.0,  # off for first test
)
algo = GDTNEAT(cfg, "CartPole-v1")
t0 = time.time()
algo.run(5, verbose=True)
print(f"elapsed: {time.time()-t0:.1f}s, best={algo.best_fitness:.2f}")
