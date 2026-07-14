import sys, os, json, time
sys.path.insert(0, '/home/z/my-project/modern-neat')
from experiments.runner import run_random_search, run_fixed_ga, run_cmaes, save_run_csv, save_run_json
OUT_DIR = 'results/phase0_baselines'
print('Acrobot-v1 seed 0 remaining baselines', flush=True)
for name, fn, kw in [
    ('RandomSearch', run_random_search, dict(n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)),
    ('FixedGA', run_fixed_ga, dict(n_generations=30, pop_size=50, n_episodes=3, max_steps=500, n_hidden=8)),
    ('CMAES', run_cmaes, dict(n_generations=30, pop_size=30, n_episodes=3, max_steps=500, n_hidden=8)),
]:
    fname = f'{name}_Acrobot-v1_seed0.json'
    if os.path.exists(os.path.join(OUT_DIR, 'json', fname)):
        print(f'  {name} already done', flush=True); continue
    print(f'  running {name}...', flush=True); t0 = time.time()
    r = fn('Acrobot-v1', 0, **kw)
    print(f'    done: best={r.best_fitness:.2f}  ({time.time()-t0:.1f}s)', flush=True)
    save_run_csv(r, os.path.join(OUT_DIR, 'csv')); save_run_json(r, os.path.join(OUT_DIR, 'json'))
print('all done', flush=True)
