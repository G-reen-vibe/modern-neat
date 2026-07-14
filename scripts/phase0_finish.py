"""Finish Phase 0 baselines: add missing seeds and other benchmarks.
Strategically reduced to fit time budget:
- CartPole-v1: seeds 0,1,2 (canonical NEAT benchmark)
- MountainCar-v0: seeds 0,1 (sparse reward, harder)
- Acrobot-v1: seeds 0,1 (medium difficulty)
"""
import sys, os, json, time
sys.path.insert(0, "/home/z/my-project/modern-neat")
from pathlib import Path
from experiments.runner import (
    run_neat, run_random_search, run_fixed_ga, run_cmaes, run_dqn,
    save_run_csv, save_run_json, aggregate_runs, RunResult,
)

OUT_DIR = "/home/z/my-project/modern-neat/results/phase0_baselines"
Path(os.path.join(OUT_DIR, "csv")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(OUT_DIR, "json")).mkdir(parents=True, exist_ok=True)

# Load existing results
all_runs = []
json_dir = os.path.join(OUT_DIR, "json")
if os.path.isdir(json_dir):
    for fname in os.listdir(json_dir):
        with open(os.path.join(json_dir, fname)) as f:
            d = json.load(f)
        r = RunResult(d["algorithm"], d["benchmark"], d["seed"],
                      d["history"], d["best_fitness"], d["elapsed_seconds"],
                      d.get("extra", {}))
        all_runs.append(r)

# Helper: only run if (algo, env, seed) not already done
def have(algo, env, seed):
    return any(r.algorithm == algo and r.benchmark == env and r.seed == seed for r in all_runs)

def maybe_run(algo, env, seed, fn, *args, **kwargs):
    if have(algo, env, seed):
        return None
    print(f"  running {algo} {env} seed={seed}...", flush=True)
    t0 = time.time()
    r = fn(env, seed, *args, **kwargs)
    print(f"    done: best={r.best_fitness:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    save_run_csv(r, os.path.join(OUT_DIR, "csv"))
    save_run_json(r, os.path.join(OUT_DIR, "json"))
    all_runs.append(r)
    return r

# ---------- CartPole-v1: 3 seeds, 30 gens ----------
print("=== CartPole-v1 ===", flush=True)
for seed in [0, 1, 2]:
    maybe_run("NEAT", "CartPole-v1", seed, run_neat,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=500)
    maybe_run("RandomSearch", "CartPole-v1", seed, run_random_search,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)
    maybe_run("FixedGA", "CartPole-v1", seed, run_fixed_ga,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)
    maybe_run("CMAES", "CartPole-v1", seed, run_cmaes,
              n_generations=30, pop_size=30, n_episodes=3, max_steps=500, n_hidden=8)
    maybe_run("DQN", "CartPole-v1", seed, run_dqn,
              total_steps=10000, eval_interval=500, max_steps=500, eval_episodes=3)

# ---------- MountainCar-v0: 2 seeds, 30 gens ----------
# max_steps=200 default; need to allow longer episodes to learn
print("\n=== MountainCar-v0 ===", flush=True)
for seed in [0, 1]:
    maybe_run("NEAT", "MountainCar-v0", seed, run_neat,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=200)
    maybe_run("RandomSearch", "MountainCar-v0", seed, run_random_search,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=200, n_hidden=8)
    maybe_run("FixedGA", "MountainCar-v0", seed, run_fixed_ga,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=200, n_hidden=8)
    maybe_run("CMAES", "MountainCar-v0", seed, run_cmaes,
              n_generations=30, pop_size=30, n_episodes=3, max_steps=200, n_hidden=8)
    maybe_run("DQN", "MountainCar-v0", seed, run_dqn,
              total_steps=15000, eval_interval=500, max_steps=200, eval_episodes=3)

# ---------- Acrobot-v1: 2 seeds, 30 gens ----------
print("\n=== Acrobot-v1 ===", flush=True)
for seed in [0, 1]:
    maybe_run("NEAT", "Acrobot-v1", seed, run_neat,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=500)
    maybe_run("RandomSearch", "Acrobot-v1", seed, run_random_search,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)
    maybe_run("FixedGA", "Acrobot-v1", seed, run_fixed_ga,
              n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)
    maybe_run("CMAES", "Acrobot-v1", seed, run_cmaes,
              n_generations=30, pop_size=30, n_episodes=3, max_steps=500, n_hidden=8)
    maybe_run("DQN", "Acrobot-v1", seed, run_dqn,
              total_steps=15000, eval_interval=500, max_steps=500, eval_episodes=3)

# Aggregate & save
summary = aggregate_runs(all_runs)
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n===== SUMMARY (mean ± std across seeds) =====", flush=True)
print(f"{'Algorithm':<14} {'Benchmark':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 70)
for key, stats in sorted(summary.items()):
    algo, bench = key.split("|", 1)
    print(f"{algo:<14} {bench:<18} {stats['mean']:>8.2f} {stats['std']:>8.2f} {stats['min']:>8.2f} {stats['max']:>8.2f}")
print(f"\nTotal runs: {len(all_runs)}")
