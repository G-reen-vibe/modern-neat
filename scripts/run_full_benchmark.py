"""Run a full benchmark of the current GDT-NEAT config on all 3 envs × 3 seeds."""
import sys, os, json, time
sys.path.insert(0, '/home/z/my-project/modern-neat')
import numpy as np
import gymnasium as gym
from src.gdt_neat import GDTNEAT, GDTConfig

def run(round_n, gens, seeds, envs, tag=""):
    cfg_kw = dict(pop_size=20, n_episodes=1, max_steps=500, seed=0,
                  lr_weights=1e-2, pg_steps=2, entropy_coef=0.05,
                  use_adam=True,
                  n_candidate_edges=12, edges_to_add=1, adaptive_edges=True,
                  adaptive_edge_threshold=0.01,
                  p_split_edge=0.2, prune_threshold=1e-3, prune_grad_threshold=1e-4, prune_patience=3,
                  diversity_penalty=0.0, n_behavior_clusters=5,
                  restart_on_convergence=True, restart_threshold=0.1, restart_fraction=0.5,
                  novelty_bonus=0.0)
    OUT_DIR = f'/home/z/my-project/modern-neat/results/gdt/round_{round_n:02d}'
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    rp = os.path.join(OUT_DIR, 'results.json')
    if os.path.exists(rp):
        with open(rp) as f: results = json.load(f)['results']
    for env_name in envs:
        env_results = []
        for seed in range(seeds):
            env = gym.make(env_name); env.reset(seed=seed)
            n_in = int(np.prod(env.observation_space.shape)); n_out = env.action_space.n; env.close()
            max_steps = {'CartPole-v1': 500, 'MountainCar-v0': 200, 'Acrobot-v1': 500}.get(env_name, 500)
            kw = dict(cfg_kw); kw.pop('seed', None); kw.pop('max_steps', None)
            cfg = GDTConfig(n_inputs=n_in, n_outputs=n_out, max_steps=max_steps, seed=seed, **kw)
            algo = GDTNEAT(cfg, env_name)
            t0 = time.time(); algo.run(gens, verbose=False); elapsed = time.time() - t0
            algo._env.close()
            r = {'history': algo.history, 'best_fitness': algo.best_fitness, 'elapsed': elapsed,
                 'final_avg_size': algo.history[-1].get('avg_size', 0) if algo.history else 0,
                 'final_avg_hidden': algo.history[-1].get('avg_hidden', 0) if algo.history else 0}
            print(f'  R{round_n}{tag} | {env_name} | seed {seed}: best={r["best_fitness"]:.2f}  ({r["elapsed"]:.1f}s)', flush=True)
            env_results.append(r)
        results[env_name] = env_results
    with open(rp, 'w') as f:
        json.dump({'round': round_n, 'config': cfg_kw, 'gens': gens, 'seeds': seeds, 'results': results}, f, indent=2)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('round_n', type=int)
    p.add_argument('--gens', type=int, default=25)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--envs', type=str, default='CartPole-v1,MountainCar-v0,Acrobot-v1')
    a = p.parse_args()
    run(a.round_n, a.gens, a.seeds, a.envs.split(','))
