"""Quick phase 0: skip DQN, run only the cheaper baselines to get tables fast.
We'll add DQN results from the existing seed0 runs only."""
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

# Load existing
all_runs = []
json_dir = os.path.join(OUT_DIR, "json")
for fname in os.listdir(json_dir):
    with open(os.path.join(json_dir, fname)) as f:
        d = json.load(f)
    r = RunResult(d["algorithm"], d["benchmark"], d["seed"],
                  d["history"], d["best_fitness"], d["elapsed_seconds"],
                  d.get("extra", {}))
    all_runs.append(r)

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

# CartPole seed 2 (only the cheap algos)
print("=== CartPole-v1 seed 2 ===", flush=True)
maybe_run("NEAT", "CartPole-v1", 2, run_neat,
          n_generations=30, pop_size=50, n_episodes=3, max_steps=500)
maybe_run("RandomSearch", "CartPole-v1", 2, run_random_search,
          n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)
maybe_run("FixedGA", "CartPole-v1", 2, run_fixed_ga,
          n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)
maybe_run("CMAES", "CartPole-v1", 2, run_cmaes,
          n_generations=30, pop_size=30, n_episodes=3, max_steps=500, n_hidden=8)

# Save summary
summary = aggregate_runs(all_runs)
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nDone. Total runs:", len(all_runs), flush=True)
