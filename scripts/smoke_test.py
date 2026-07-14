"""Quick smoke test: run NEAT on CartPole for 5 generations."""
import sys
sys.path.insert(0, "/home/z/my-project/modern-neat")
from src.neat import NEAT, NEATConfig
import time

cfg = NEATConfig(
    pop_size=30,
    n_inputs=4,
    n_outputs=2,
    n_episodes=2,
    max_steps=500,
    seed=0,
    p_add_node=0.05,
    p_add_edge=0.1,
    compatibility_threshold=3.0,
)
neat = NEAT(cfg, "CartPole-v1")
t0 = time.time()
neat.run(5, verbose=True)
print(f"elapsed: {time.time() - t0:.1f}s")
