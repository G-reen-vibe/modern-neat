"""Experiment runner: standardize comparisons across algorithms/benchmarks/seeds.

Produces CSV files of per-generation (or per-eval-step) best/mean/std fitness
for each (algorithm, benchmark, seed), plus summary tables with error bars.
"""
from __future__ import annotations

import os
import csv
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

# project root on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neat import NEAT, NEATConfig
from src.baselines import RandomSearch, RandomSearchConfig, FixedGA, FixedGAConfig, CMAES
from src.dqn import DQN, DQNConfig


@dataclass
class RunResult:
    algorithm: str
    benchmark: str
    seed: int
    history: List[Dict[str, float]]  # per-step records
    best_fitness: float
    elapsed_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


def run_neat(env_name: str, seed: int, n_generations: int, pop_size: int = 50,
             n_episodes: int = 3, max_steps: int = 500, **kwargs) -> RunResult:
    cfg = NEATConfig(
        pop_size=pop_size, n_episodes=n_episodes, max_steps=max_steps, seed=seed,
        **kwargs,
    )
    # determine n_inputs / n_outputs from env
    import gymnasium as gym
    env = gym.make(env_name); env.reset(seed=seed)
    n_in = int(np.prod(env.observation_space.shape))
    n_out = env.action_space.n
    env.close()
    cfg.n_inputs = n_in
    cfg.n_outputs = n_out
    algo = NEAT(cfg, env_name)
    t0 = time.time()
    algo.run(n_generations, verbose=False)
    elapsed = time.time() - t0
    hist = [
        {
            "gen": h.generation, "best": h.best_fitness, "mean": h.mean_fitness,
            "std": h.std_fitness, "n_species": h.n_species,
            "genome_size": h.best_genome_size, "n_hidden": h.best_n_hidden,
        }
        for h in algo.history
    ]
    return RunResult("NEAT", env_name, seed, hist, algo.best_fitness, elapsed,
                     extra={"n_evals": n_generations * pop_size * n_episodes})


def run_random_search(env_name: str, seed: int, n_generations: int, pop_size: int = 50,
                      n_episodes: int = 3, max_steps: int = 500, n_hidden: int = 8,
                      **kwargs) -> RunResult:
    import gymnasium as gym
    env = gym.make(env_name); env.reset(seed=seed)
    n_in = int(np.prod(env.observation_space.shape))
    n_out = env.action_space.n
    env.close()
    cfg = RandomSearchConfig(
        n_inputs=n_in, n_outputs=n_out, n_hidden=n_hidden,
        pop_size=pop_size, n_episodes=n_episodes, max_steps=max_steps, seed=seed, **kwargs,
    )
    algo = RandomSearch(cfg, env_name)
    t0 = time.time()
    algo.run(n_generations, verbose=False)
    elapsed = time.time() - t0
    hist = [{"gen": g, "best": b, "mean": m} for (g, b, m) in algo.history]
    return RunResult("RandomSearch", env_name, seed, hist, algo.best_fitness, elapsed,
                     extra={"n_evals": n_generations * pop_size * n_episodes})


def run_fixed_ga(env_name: str, seed: int, n_generations: int, pop_size: int = 50,
                 n_episodes: int = 3, max_steps: int = 500, n_hidden: int = 8,
                 **kwargs) -> RunResult:
    import gymnasium as gym
    env = gym.make(env_name); env.reset(seed=seed)
    n_in = int(np.prod(env.observation_space.shape))
    n_out = env.action_space.n
    env.close()
    cfg = FixedGAConfig(
        n_inputs=n_in, n_outputs=n_out, n_hidden=n_hidden,
        pop_size=pop_size, n_episodes=n_episodes, max_steps=max_steps, seed=seed, **kwargs,
    )
    algo = FixedGA(cfg, env_name)
    t0 = time.time()
    algo.run(n_generations, verbose=False)
    elapsed = time.time() - t0
    hist = [{"gen": g, "best": b, "mean": m} for (g, b, m) in algo.history]
    return RunResult("FixedGA", env_name, seed, hist, algo.best_fitness, elapsed,
                     extra={"n_evals": n_generations * pop_size * n_episodes})


def run_cmaes(env_name: str, seed: int, n_generations: int, pop_size: int = 30,
              n_episodes: int = 3, max_steps: int = 500, n_hidden: int = 8,
              **kwargs) -> RunResult:
    import gymnasium as gym
    env = gym.make(env_name); env.reset(seed=seed)
    n_in = int(np.prod(env.observation_space.shape))
    n_out = env.action_space.n
    env.close()
    algo = CMAES(env_name, n_inputs=n_in, n_outputs=n_out, n_hidden=n_hidden,
                 pop_size=pop_size, n_episodes=n_episodes, max_steps=max_steps, seed=seed, **kwargs)
    t0 = time.time()
    algo.run(n_generations, verbose=False)
    elapsed = time.time() - t0
    hist = [{"gen": g, "best": b, "mean": m, "sigma": float(algo.history[g][1])} for (g, b, m) in algo.history]
    # actually rebuild properly
    hist = [{"gen": g, "best": b, "mean": m} for (g, b, m) in algo.history]
    return RunResult("CMAES", env_name, seed, hist, algo.best_fitness, elapsed,
                     extra={"n_evals": n_generations * pop_size * n_episodes})


def run_dqn(env_name: str, seed: int, total_steps: int = 30000, eval_interval: int = 1000,
            max_steps: int = 500, eval_episodes: int = 5, **kwargs) -> RunResult:
    import gymnasium as gym
    env = gym.make(env_name); env.reset(seed=seed)
    n_in = int(np.prod(env.observation_space.shape))
    n_out = env.action_space.n
    env.close()
    cfg = DQNConfig(
        n_inputs=n_in, n_outputs=n_out, total_steps=total_steps,
        eval_interval=eval_interval, max_steps=max_steps, eval_episodes=eval_episodes,
        seed=seed, **kwargs,
    )
    algo = DQN(cfg, env_name)
    t0 = time.time()
    algo.run(verbose=False)
    elapsed = time.time() - t0
    algo.close()
    hist = [{"step": s, "best": m, "std": sd} for (s, m, sd) in algo.history]
    # convert to "generation-like" history indexed by eval number
    return RunResult("DQN", env_name, seed, hist, algo.best_fitness, elapsed,
                     extra={"total_steps": total_steps, "n_evals": len(hist)})


# Map of algorithm name -> runner
RUNNERS = {
    "NEAT": run_neat,
    "RandomSearch": run_random_search,
    "FixedGA": run_fixed_ga,
    "CMAES": run_cmaes,
    "DQN": run_dqn,
}


def save_run_csv(result: RunResult, out_dir: str) -> str:
    """Save a single run's history to a CSV file. Returns the path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{result.algorithm}_{result.benchmark}_seed{result.seed}.csv"
    path = os.path.join(out_dir, fname)
    if not result.history:
        return path
    keys = list(result.history[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in result.history:
            w.writerow([row[k] for k in keys])
    return path


def save_run_json(result: RunResult, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{result.algorithm}_{result.benchmark}_seed{result.seed}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump({
            "algorithm": result.algorithm,
            "benchmark": result.benchmark,
            "seed": result.seed,
            "best_fitness": result.best_fitness,
            "elapsed_seconds": result.elapsed_seconds,
            "extra": result.extra,
            "history": result.history,
        }, f, indent=2)
    return path


def aggregate_runs(runs: List[RunResult]) -> Dict[str, Any]:
    """Aggregate best-fitness across seeds for the same (algorithm, benchmark)."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r.algorithm, r.benchmark)].append(r.best_fitness)
    summary = {}
    for (algo, bench), vals in by_key.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[f"{algo}|{bench}"] = {
            "n_seeds": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "ci95_half": float(1.96 * arr.std(ddof=1) / np.sqrt(max(1, len(arr)))) if len(arr) > 1 else 0.0,
        }
    return summary
