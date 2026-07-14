"""Generate comparison charts for the report."""
import sys, os, json
sys.path.insert(0, '/home/z/my-project/modern-neat')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/home/z/my-project/modern-neat/reports/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Load all results
def load_json(path):
    with open(path) as f:
        return json.load(f)

# Phase 0 baselines
baseline_dir = '/home/z/my-project/modern-neat/results/phase0_baselines'
baselines = {}
if os.path.isdir(os.path.join(baseline_dir, 'json')):
    for fname in os.listdir(os.path.join(baseline_dir, 'json')):
        d = load_json(os.path.join(baseline_dir, 'json', fname))
        key = f"{d['algorithm']}_{d['benchmark']}"
        if key not in baselines:
            baselines[key] = []
        baselines[key].append(d['best_fitness'])

# PT-NEAT results (from round_29)
pt_dir = '/home/z/my-project/modern-neat/results/gdt/round_29'
pt_results = {}
if os.path.exists(os.path.join(pt_dir, 'results.json')):
    d = load_json(os.path.join(pt_dir, 'results.json'))
    for env, runs in d['results'].items():
        pt_results[env] = [r['best_det'] for r in runs]

# GDT-NEAT results (from round_14)
gdt_dir = '/home/z/my-project/modern-neat/results/gdt/round_14'
gdt_results = {}
if os.path.exists(os.path.join(gdt_dir, 'results.json')):
    d = load_json(os.path.join(gdt_dir, 'results.json'))
    for env, runs in d['results'].items():
        gdt_results[env] = [r['best_fitness'] for r in runs]

# Print summary
print("=== Baseline Results (Phase 0) ===")
for key in sorted(baselines.keys()):
    vals = baselines[key]
    print(f"  {key:<40} mean={np.mean(vals):7.2f}  std={np.std(vals):6.2f}  n={len(vals)}")

print("\n=== GDT-NEAT Results (Round 14) ===")
for env, vals in gdt_results.items():
    print(f"  {env:<20} {vals}")

print("\n=== PT-NEAT Results (Round 29) ===")
for env, vals in pt_results.items():
    print(f"  {env:<20} {vals}")

# Create comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
algos = ['NEAT', 'RandomSearch', 'FixedGA', 'CMAES', 'DQN', 'GDT-NEAT', 'PT-NEAT']
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#607D8B', '#E91E63']

for ax, env in zip(axes, envs):
    means = []
    stds = []
    labels = []
    for algo in algos:
        if algo in ['GDT-NEAT', 'PT-NEAT']:
            if algo == 'GDT-NEAT':
                vals = gdt_results.get(env, [])
            else:
                vals = pt_results.get(env, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
                labels.append(algo)
        else:
            key = f"{algo}_{env}"
            vals = baselines.get(key, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
                labels.append(algo)
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(labels)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(env, fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Fitness (mean ± std)')
    ax.grid(axis='y', alpha=0.3)
    # Add threshold lines
    if env == 'CartPole-v1':
        ax.axhline(y=475, color='green', linestyle='--', alpha=0.5, label='Solved (475)')
    elif env == 'MountainCar-v0':
        ax.axhline(y=-110, color='green', linestyle='--', alpha=0.5, label='Solved (-110)')
    elif env == 'Acrobot-v1':
        ax.axhline(y=-100, color='green', linestyle='--', alpha=0.5, label='Solved (-100)')
    ax.legend(fontsize=8)

fig.suptitle('Algorithm Comparison on RL Benchmarks', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(OUT_DIR, 'comparison_bar.png'), dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR}/comparison_bar.png")

# Create learning curves for PT-NEAT
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for ax, env in zip(axes, envs):
    if env in pt_results:
        # load history
        d = load_json(os.path.join(pt_dir, 'results.json'))
        for i, run in enumerate(d['results'][env]):
            history = run['history']
            gens = [h['gen'] for h in history]
            best = [h.get('best_det', h.get('best', 0)) for h in history]
            ax.plot(gens, best, alpha=0.7, label=f'Seed {i}')
    ax.set_title(env, fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Deterministic Reward')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if env == 'CartPole-v1':
        ax.axhline(y=475, color='green', linestyle='--', alpha=0.5)
    elif env == 'MountainCar-v0':
        ax.axhline(y=-110, color='green', linestyle='--', alpha=0.5)
    elif env == 'Acrobot-v1':
        ax.axhline(y=-100, color='green', linestyle='--', alpha=0.5)

fig.suptitle('PT-NEAT Learning Curves', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(OUT_DIR, 'pt_neat_curves.png'), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/pt_neat_curves.png")
