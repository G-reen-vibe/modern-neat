"""Run all baselines on CartPole-v1 for a quick sanity check."""
import sys
sys.path.insert(0, "/home/z/my-project/modern-neat")
from src.neat import NEAT, NEATConfig
from src.baselines import RandomSearch, RandomSearchConfig, FixedGA, FixedGAConfig, CMAES
from src.dqn import DQN, DQNConfig
import time

print("=== Vanilla NEAT ===")
t0 = time.time()
neat = NEAT(NEATConfig(pop_size=50, n_episodes=3, max_steps=500, seed=0), "CartPole-v1")
neat.run(10, verbose=True)
print(f"elapsed: {time.time()-t0:.1f}s, best={neat.best_fitness:.1f}")

print("\n=== Random Search ===")
t0 = time.time()
rs = RandomSearch(RandomSearchConfig(pop_size=50, n_episodes=3, max_steps=500, seed=0, n_hidden=8), "CartPole-v1")
rs.run(10, verbose=True)
print(f"elapsed: {time.time()-t0:.1f}s, best={rs.best_fitness:.1f}")

print("\n=== Fixed GA ===")
t0 = time.time()
ga = FixedGA(FixedGAConfig(pop_size=50, n_episodes=3, max_steps=500, seed=0, n_hidden=8), "CartPole-v1")
ga.run(10, verbose=True)
print(f"elapsed: {time.time()-t0:.1f}s, best={ga.best_fitness:.1f}")

print("\n=== CMA-ES ===")
t0 = time.time()
cma = CMAES("CartPole-v1", n_inputs=4, n_outputs=2, n_hidden=8, pop_size=30, n_episodes=3, max_steps=500, seed=0)
cma.run(15, verbose=True)
print(f"elapsed: {time.time()-t0:.1f}s, best={cma.best_fitness:.1f}")

print("\n=== DQN ===")
t0 = time.time()
dqn = DQN(DQNConfig(n_inputs=4, n_outputs=2, total_steps=8000, eval_interval=1000, seed=0), "CartPole-v1")
dqn.run(verbose=True)
dqn.close()
print(f"elapsed: {time.time()-t0:.1f}s, best={dqn.best_fitness:.1f}")
