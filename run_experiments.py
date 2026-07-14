#!/usr/bin/env python3
"""
Main experiment runner for Atlas-NEAT research.
Runs benchmarks comparing standard NEAT vs Atlas-NEAT.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.evaluator import run_benchmark, get_env_info


# Benchmark environments
BENCHMARKS = {
    'cartpole': {
        'env': 'CartPole-v1',
        'threshold': 475,
        'generations': 100,
        'episodes': 3,
        'max_steps': 500,
    },
    'acrobot': {
        'env': 'Acrobot-v1',
        'threshold': -100,
        'generations': 100,
        'episodes': 3,
        'max_steps': 500,
    },
    'mountaincar': {
        'env': 'MountainCar-v0',
        'threshold': -110,
        'generations': 100,
        'episodes': 3,
        'max_steps': 200,
    },
    'lunarlander': {
        'env': 'LunarLander-v3',
        'threshold': 200,
        'generations': 150,
        'episodes': 3,
        'max_steps': 1000,
    },
}


def plot_comparison(neat_results, atlas_results, env_name, output_path):
    """Plot comparison between NEAT and Atlas-NEAT."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{env_name}: NEAT vs Atlas-NEAT', fontsize=14, fontweight='bold')
    
    gens = range(len(neat_results['best_fitness_mean']))
    
    # Best fitness
    ax = axes[0, 0]
    ax.plot(gens, neat_results['best_fitness_mean'], 'b-', label='NEAT', linewidth=2)
    ax.fill_between(gens, 
        np.array(neat_results['best_fitness_mean']) - np.array(neat_results['best_fitness_std']),
        np.array(neat_results['best_fitness_mean']) + np.array(neat_results['best_fitness_std']),
        alpha=0.2, color='b')
    
    atlas_gens = range(len(atlas_results['best_fitness_mean']))
    ax.plot(atlas_gens, atlas_results['best_fitness_mean'], 'r-', label='Atlas-NEAT', linewidth=2)
    ax.fill_between(atlas_gens,
        np.array(atlas_results['best_fitness_mean']) - np.array(atlas_results['best_fitness_std']),
        np.array(atlas_results['best_fitness_mean']) + np.array(atlas_results['best_fitness_std']),
        alpha=0.2, color='r')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average fitness
    ax = axes[0, 1]
    ax.plot(gens, neat_results['avg_fitness_mean'], 'b-', label='NEAT', linewidth=2)
    ax.plot(atlas_gens, atlas_results['avg_fitness_mean'], 'r-', label='Atlas-NEAT', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Diversity (species/cells)
    ax = axes[1, 0]
    ax.plot(gens, neat_results['n_species_mean'], 'b-', label='NEAT (species)', linewidth=2)
    ax.plot(atlas_gens, atlas_results['n_species_mean'], 'r-', label='Atlas-NEAT (cells)', linewidth=2)
    
    if 'coverage_mean' in atlas_results:
        ax2 = ax.twinx()
        ax2.plot(atlas_gens, atlas_results['coverage_mean'], 'g--', label='Coverage', linewidth=1.5)
        ax2.set_ylabel('Archive Coverage', color='g')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Count')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Time per generation
    ax = axes[1, 1]
    ax.plot(gens, neat_results['time_mean'], 'b-', label='NEAT', linewidth=2)
    ax.plot(atlas_gens, atlas_results['time_mean'], 'r-', label='Atlas-NEAT', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Time per Generation (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def run_single_experiment(benchmark_name, algorithm, n_trials, pop_size, output_dir, seed):
    """Run a single experiment configuration."""
    bench = BENCHMARKS[benchmark_name]
    env_name = bench['env']
    
    print(f"\n{'='*70}")
    print(f"Running {algorithm.upper()} on {benchmark_name} ({env_name})")
    print(f"  Trials: {n_trials}, Pop size: {pop_size}, Generations: {bench['generations']}")
    print(f"{'='*70}\n")
    
    results = run_benchmark(
        env_name=env_name,
        algorithm=algorithm,
        n_trials=n_trials,
        generations=bench['generations'],
        pop_size=pop_size,
        episodes=bench['episodes'],
        max_steps=bench['max_steps'],
        fitness_threshold=bench['threshold'],
        seed_base=seed
    )
    
    # Save results
    output_file = os.path.join(output_dir, f'{algorithm}_{benchmark_name}.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults for {algorithm} on {benchmark_name}:")
    print(f"  Best fitness: {results['best_fitness_mean'][-1]:.2f} ± {results['best_fitness_std'][-1]:.2f}")
    print(f"  Avg fitness: {results['avg_fitness_mean'][-1]:.2f} ± {results['avg_fitness_std'][-1]:.2f}")
    print(f"  Total time: {results['total_time_mean']:.1f} ± {results['total_time_std']:.1f}s")
    print(f"  Saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run NEAT benchmarks')
    parser.add_argument('--benchmarks', nargs='+', default=['cartpole'],
                       choices=list(BENCHMARKS.keys()),
                       help='Benchmarks to run')
    parser.add_argument('--algorithms', nargs='+', default=['neat', 'atlas'],
                       choices=['neat', 'atlas'],
                       help='Algorithms to compare')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of independent trials')
    parser.add_argument('--pop-size', type=int, default=150,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=None,
                       help='Override max generations')
    parser.add_argument('--seed', type=int, default=0,
                       help='Base random seed')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    all_results = {}
    
    for benchmark_name in args.benchmarks:
        all_results[benchmark_name] = {}
        
        for algorithm in args.algorithms:
            results = run_single_experiment(
                benchmark_name=benchmark_name,
                algorithm=algorithm,
                n_trials=args.trials,
                pop_size=args.pop_size,
                output_dir=args.output_dir,
                seed=args.seed
            )
            all_results[benchmark_name][algorithm] = results
        
        # Plot comparison if both algorithms ran
        if args.plot and len(args.algorithms) == 2:
            plot_path = os.path.join(args.output_dir, f'comparison_{benchmark_name}.png')
            plot_comparison(
                all_results[benchmark_name]['neat'],
                all_results[benchmark_name]['atlas'],
                benchmark_name,
                plot_path
            )
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for benchmark_name in args.benchmarks:
        print(f"\n{benchmark_name}:")
        for algorithm in args.algorithms:
            r = all_results[benchmark_name][algorithm]
            print(f"  {algorithm:10s}: Best={r['best_fitness_mean'][-1]:8.2f} ± {r['best_fitness_std'][-1]:6.2f}  "
                  f"Time={r['total_time_mean']:7.1f}s")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {summary_file}")


if __name__ == '__main__':
    main()
