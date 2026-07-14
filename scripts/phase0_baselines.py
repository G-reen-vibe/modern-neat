"""Phase 0 baseline experiments: establish reference numbers for each algorithm
on each benchmark across multiple seeds.

Benchmarks:
  - CartPole-v1       (4 obs, 2 actions, max 500 steps)
  - MountainCar-v0    (2 obs, 3 actions, max 200 steps)
  - Acrobot-v1        (6 obs, 3 actions, max 500 steps)

Algorithms:
  - NEAT (pop 50, 30 generations)
  - RandomSearch (pop 50, 30 generations, 8-hidden MLP)
  - FixedGA (pop 50, 30 generations, 8-hidden MLP)
  - CMAES (pop 30, 30 generations, 8-hidden MLP)
  - DQN (10k steps on CartPole, 15k on MountainCar/Acrobot)

Seeds: 0, 1, 2 (3 seeds for error bars)

Outputs: results/phase0_baselines/{csv,json,summary.json}
"""
import sys
sys.path.insert(0, "/home/z/my-project/modern-neat")
import os
import json
import time
from pathlib import Path
from experiments.runner import (
    RUNNERS, run_neat, run_random_search, run_fixed_ga, run_cmaes, run_dqn,
    save_run_csv, save_run_json, aggregate_runs, RunResult,
)

OUT_DIR = "/home/z/my-project/modern-neat/results/phase0_baselines"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# (env_name, neat_generations, max_steps, dqn_total_steps, n_episodes)
BENCHMARKS = [
    ("CartPole-v1",     30, 500, 10000, 3),
    ("MountainCar-v0",  30, 200, 15000, 3),
    ("Acrobot-v1",      30, 500, 15000, 3),
]
SEEDS = [0, 1, 2]

all_runs: list[RunResult] = []

for env_name, n_gen, max_steps, dqn_steps, n_eps in BENCHMARKS:
    print(f"\n===== {env_name} (gens={n_gen}, max_steps={max_steps}) =====")
    for seed in SEEDS:
        print(f"  seed {seed}...")
        # NEAT
        r = run_neat(env_name, seed, n_gen, pop_size=50, n_episodes=n_eps, max_steps=max_steps)
        print(f"    NEAT:          best={r.best_fitness:7.2f}  ({r.elapsed_seconds:.1f}s)")
        save_run_csv(r, os.path.join(OUT_DIR, "csv"))
        save_run_json(r, os.path.join(OUT_DIR, "json"))
        all_runs.append(r)

        # Random search
        r = run_random_search(env_name, seed, n_gen, pop_size=50, n_episodes=n_eps, max_steps=max_steps, n_hidden=8)
        print(f"    RandomSearch:  best={r.best_fitness:7.2f}  ({r.elapsed_seconds:.1f}s)")
        save_run_csv(r, os.path.join(OUT_DIR, "csv"))
        save_run_json(r, os.path.join(OUT_DIR, "json"))
        all_runs.append(r)

        # Fixed GA
        r = run_fixed_ga(env_name, seed, n_gen, pop_size=50, n_episodes=n_eps, max_steps=max_steps, n_hidden=8)
        print(f"    FixedGA:       best={r.best_fitness:7.2f}  ({r.elapsed_seconds:.1f}s)")
        save_run_csv(r, os.path.join(OUT_DIR, "csv"))
        save_run_json(r, os.path.join(OUT_DIR, "json"))
        all_runs.append(r)

        # CMA-ES
        r = run_cmaes(env_name, seed, n_gen, pop_size=30, n_episodes=n_eps, max_steps=max_steps, n_hidden=8)
        print(f"    CMAES:         best={r.best_fitness:7.2f}  ({r.elapsed_seconds:.1f}s)")
        save_run_csv(r, os.path.join(OUT_DIR, "csv"))
        save_run_json(r, os.path.join(OUT_DIR, "json"))
        all_runs.append(r)

        # DQN
        r = run_dqn(env_name, seed, total_steps=dqn_steps, eval_interval=500, max_steps=max_steps, eval_episodes=n_eps)
        print(f"    DQN:           best={r.best_fitness:7.2f}  ({r.elapsed_seconds:.1f}s)")
        save_run_csv(r, os.path.join(OUT_DIR, "csv"))
        save_run_json(r, os.path.join(OUT_DIR, "json"))
        all_runs.append(r)

# aggregate
summary = aggregate_runs(all_runs)
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# print summary table
print("\n\n===== SUMMARY (mean ± std across seeds) =====")
print(f"{'Algorithm':<14} {'Benchmark':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 70)
for key, stats in sorted(summary.items()):
    algo, bench = key.split("|", 1)
    print(f"{algo:<14} {bench:<18} {stats['mean']:>8.2f} {stats['std']:>8.2f} {stats['min']:>8.2f} {stats['max']:>8.2f}")

print(f"\nResults saved to {OUT_DIR}")
