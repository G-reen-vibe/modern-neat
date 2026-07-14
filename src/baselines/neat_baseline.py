"""
Standard NEAT baseline implementation for benchmarking.
Uses the neat-python library with configurable environments.
"""

import os
import sys
import json
import time
import pickle
import neat
import gymnasium as gym
import numpy as np
from pathlib import Path


def make_env(env_name, render_mode=None):
    """Create a gymnasium environment."""
    return gym.make(env_name, render_mode=render_mode)


def eval_genome(genome, config, env_name, episodes=3, max_steps=500, render=False):
    """Evaluate a single genome."""
    env = make_env(env_name, render_mode="human" if render else None)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_reward = 0.0
    for _ in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_flat = np.array(obs).flatten()
            action = net.activate(obs_flat)
            
            # Handle discrete vs continuous action spaces
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(np.argmax(action))
            else:
                action = np.array(action).clip(env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_reward += episode_reward
    
    env.close()
    return total_reward / episodes


def eval_genomes(genomes, config, env_name, episodes=3, max_steps=500):
    """Evaluate all genomes in the population."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, env_name, episodes, max_steps)


def run_neat(config_path, env_name, generations=100, episodes=3, max_steps=500, 
             checkpoint_dir=None, seed=0):
    """
    Run standard NEAT on a given environment.
    
    Returns:
        dict with results: best_fitnesses, avg_fitnesses, species_counts, times, best_genome
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Adjust genome config for environment
    env = make_env(env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    
    if isinstance(obs_space, gym.spaces.Box):
        config.genome_config.num_inputs = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, gym.spaces.Discrete):
        config.genome_config.num_inputs = obs_space.n
    
    if isinstance(act_space, gym.spaces.Discrete):
        config.genome_config.num_outputs = act_space.n
    elif isinstance(act_space, gym.spaces.Box):
        config.genome_config.num_outputs = int(np.prod(act_space.shape))
    
    env.close()
    
    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        p.add_reporter(neat.Checkpointer(generations // 10, 
                                        filename_prefix=os.path.join(checkpoint_dir, 'neat_checkpoint_')))
    
    # Run evolution
    results = {
        'best_fitnesses': [],
        'avg_fitnesses': [],
        'species_counts': [],
        'times': [],
        'generation': [],
        'best_genome': None,
        'config': str(config_path),
        'env': env_name,
    }
    
    start_time = time.time()
    
    for gen in range(generations):
        gen_start = time.time()
        
        # Evaluate population
        genomes = list(p.population.items())
        eval_genomes(genomes, config, env_name, episodes, max_steps)
        
        # Post-evaluate
        p.population = dict(genomes)
        p.reporters.start_generation(p.generation, p.config)
        
        # Get fitnesses
        fitnesses = [g.fitness for _, g in genomes if g.fitness is not None]
        if fitnesses:
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
        else:
            best_fitness = 0
            avg_fitness = 0
        
        # Create next generation
        p.population = p.reproduction.reproduce(p.config, p.species, p.config.pop_size, p.generation)
        p.species.speciate(p.config, p.population, p.generation)
        
        # Check stagnation
        p.reporters.post_evaluate(p.config, p.population, p.species, best_fitness)
        
        # Collect results
        species_count = len(p.species.species)
        gen_time = time.time() - gen_start
        
        results['best_fitnesses'].append(best_fitness)
        results['avg_fitnesses'].append(avg_fitness)
        results['species_counts'].append(species_count)
        results['times'].append(gen_time)
        results['generation'].append(gen)
        
        print(f"Gen {gen}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, "
              f"Species={species_count}, Time={gen_time:.2f}s")
        
        # Check termination
        if best_fitness >= config.fitness_threshold:
            print(f"Fitness threshold reached at generation {gen}!")
            break
        
        p.generation += 1
    
    results['total_time'] = time.time() - start_time
    results['best_genome'] = stats.best_genome()
    
    return results


def run_trials(config_path, env_name, n_trials=5, generations=100, 
               episodes=3, max_steps=500, seed_base=0):
    """Run multiple trials and aggregate results."""
    all_results = []
    
    for trial in range(n_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{n_trials} - Seed {seed_base + trial}")
        print(f"{'='*60}")
        
        results = run_neat(config_path, env_name, generations, episodes, max_steps, 
                          seed=seed_base + trial)
        all_results.append(results)
    
    # Aggregate
    max_gens = max(len(r['best_fitnesses']) for r in all_results)
    
    agg = {
        'best_fitness_mean': [],
        'best_fitness_std': [],
        'avg_fitness_mean': [],
        'avg_fitness_std': [],
        'species_mean': [],
        'species_std': [],
        'time_mean': [],
        'success_rate': 0,
        'trials': n_trials,
    }
    
    for gen in range(max_gens):
        bf = [r['best_fitnesses'][gen] if gen < len(r['best_fitnesses']) else r['best_fitnesses'][-1] 
              for r in all_results]
        af = [r['avg_fitnesses'][gen] if gen < len(r['avg_fitnesses']) else r['avg_fitnesses'][-1] 
              for r in all_results]
        sc = [r['species_counts'][gen] if gen < len(r['species_counts']) else r['species_counts'][-1] 
              for r in all_results]
        tm = [r['times'][gen] if gen < len(r['times']) else 0 for r in all_results]
        
        agg['best_fitness_mean'].append(float(np.mean(bf)))
        agg['best_fitness_std'].append(float(np.std(bf)))
        agg['avg_fitness_mean'].append(float(np.mean(af)))
        agg['avg_fitness_std'].append(float(np.std(af)))
        agg['species_mean'].append(float(np.mean(sc)))
        agg['species_std'].append(float(np.std(sc)))
        agg['time_mean'].append(float(np.mean(tm)))
    
    # Count successes (fitness threshold reached)
    threshold = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path).fitness_threshold
    successes = sum(1 for r in all_results if max(r['best_fitnesses']) >= threshold)
    agg['success_rate'] = successes / n_trials
    
    return agg, all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--config', default='src/baselines/neat_config.ini')
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='results/neat_baseline.json')
    args = parser.parse_args()
    
    agg, all_results = run_trials(args.config, args.env, args.trials, args.generations, seed_base=args.seed)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(agg, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Success rate: {agg['success_rate']:.1%}")
