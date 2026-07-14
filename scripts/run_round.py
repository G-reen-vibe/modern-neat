"""Round-by-round experiment runner.

Each round:
- Loads a specific GDT-NEAT config (defined inline)
- Runs on CartPole-v1, MountainCar-v0, Acrobot-v1 with N seeds
- Saves results to results/gdt/round_<N>/
- Appends a summary to rounds/RESULTS.md

Usage:
    python3 scripts/run_round.py <N>
"""
import sys, os, json, time, argparse
sys.path.insert(0, "/home/z/my-project/modern-neat")
import numpy as np
import gymnasium as gym
from src.gdt_neat import GDTNEAT, GDTConfig

OUT_BASE = "/home/z/my-project/modern-neat/results/gdt"


def get_round_config(round_n: int) -> dict:
    """Return config kwargs for each round. Round 1 = baseline GDT-NEAT."""
    configs = {
        1: dict(  # baseline GDT-NEAT
            pop_size=30, n_episodes=2, max_steps=500, seed=0,
            lr_weights=1e-2, pg_steps=3,
            n_candidate_edges=16, edges_to_add=1, p_split_edge=0.1,
            diversity_penalty=0.0, n_behavior_clusters=5,
        ),
    }
    return configs.get(round_n, configs[1])


def run_one(env_name: str, seed: int, n_gens: int, cfg_kwargs: dict):
    env = gym.make(env_name); env.reset(seed=seed)
    n_in = int(np.prod(env.observation_space.shape))
    n_out = env.action_space.n
    env.close()
    max_steps = {"CartPole-v1": 500, "MountainCar-v0": 200, "Acrobot-v1": 500}.get(env_name, 500)
    # build cfg, letting caller override anything except n_inputs/n_outputs
    kw = dict(cfg_kwargs)
    kw.pop("seed", None)  # caller's seed is the actual seed
    kw.pop("max_steps", None)
    cfg = GDTConfig(n_inputs=n_in, n_outputs=n_out, max_steps=max_steps, seed=seed, **kw)
    algo = GDTNEAT(cfg, env_name)
    t0 = time.time()
    algo.run(n_gens, verbose=False)
    elapsed = time.time() - t0
    return {
        "history": algo.history,
        "best_fitness": algo.best_fitness,
        "elapsed": elapsed,
        "final_avg_size": algo.history[-1].get("avg_size", 0) if algo.history else 0,
        "final_avg_hidden": algo.history[-1].get("avg_hidden", 0) if algo.history else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round_n", type=int)
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--envs", type=str, default="CartPole-v1,MountainCar-v0,Acrobot-v1")
    args = parser.parse_args()

    cfg_kwargs = get_round_config(args.round_n)
    out_dir = os.path.join(OUT_BASE, f"round_{args.round_n:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # Load existing results to merge (so we can run envs separately)
    existing = {}
    results_path = os.path.join(out_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)

    envs = args.envs.split(",")
    results = existing.get("results", {})
    for env_name in envs:
        env_results = []
        for seed in range(args.seeds):
            print(f"  Round {args.round_n} | {env_name} | seed {seed}...", flush=True)
            t0 = time.time()
            r = run_one(env_name, seed, args.gens, cfg_kwargs)
            print(f"    best={r['best_fitness']:.2f}  ({r['elapsed']:.1f}s)  size={r['final_avg_size']:.1f}  hidden={r['final_avg_hidden']:.1f}", flush=True)
            env_results.append(r)
        results[env_name] = env_results

    # Save
    with open(results_path, "w") as f:
        json.dump({
            "round": args.round_n,
            "config": cfg_kwargs,
            "gens": args.gens,
            "seeds": args.seeds,
            "results": results,
        }, f, indent=2)

    # Print summary
    print(f"\n=== Round {args.round_n} summary ===")
    for env_name, runs in results.items():
        bests = [r["best_fitness"] for r in runs]
        print(f"  {env_name:<20} best mean={np.mean(bests):7.2f}  std={np.std(bests):6.2f}  individual={bests}")


if __name__ == "__main__":
    main()
