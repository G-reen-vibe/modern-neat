"""
Evaluation utilities for benchmarking neuroevolution algorithms.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional

from atlas_neat.genome import Genome, FeedForwardNetwork


def get_env_info(env_name: str) -> Tuple[int, int, str]:
    """
    Get input/output dimensions and action type for an environment.
    Returns (num_inputs, num_outputs, action_type)
    """
    env = gym.make(env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    
    if isinstance(obs_space, gym.spaces.Box):
        num_inputs = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, gym.spaces.Discrete):
        num_inputs = obs_space.n
    else:
        raise ValueError(f"Unsupported observation space: {type(obs_space)}")
    
    if isinstance(act_space, gym.spaces.Discrete):
        num_outputs = act_space.n
        action_type = 'discrete'
    elif isinstance(act_space, gym.spaces.Box):
        num_outputs = int(np.prod(act_space.shape))
        action_type = 'continuous'
    else:
        raise ValueError(f"Unsupported action space: {type(act_space)}")
    
    env.close()
    return num_inputs, num_outputs, action_type


def create_evaluator(env_name: str, episodes: int = 3, max_steps: int = 500,
                     seed: int = 0):
    """
    Create an evaluation function for a given environment.
    Returns a function that takes a Genome and returns fitness.
    """
    env = gym.make(env_name)
    action_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'
    act_low = env.action_space.low if hasattr(env.action_space, 'low') else None
    act_high = env.action_space.high if hasattr(env.action_space, 'high') else None
    env.close()
    
    eval_counter = [0]  # Use list for mutable closure
    
    def evaluate(genome: Genome) -> float:
        """Evaluate a genome on the environment."""
        network = genome.create_network()
        env = gym.make(env_name)
        
        total_reward = 0.0
        for ep in range(episodes):
            ep_seed = seed + eval_counter[0] * episodes + ep
            obs, info = env.reset(seed=ep_seed)
            episode_reward = 0.0
            
            for step in range(max_steps):
                obs_flat = np.array(obs).flatten()
                outputs = network.activate(obs_flat)
                
                if action_type == 'discrete':
                    action = int(np.argmax(outputs))
                else:
                    action = np.array(outputs).clip(act_low, act_high)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
        
        env.close()
        eval_counter[0] += 1
        fitness = total_reward / episodes
        genome.fitness = fitness
        return fitness
    
    return evaluate


def run_benchmark(env_name: str, algorithm: str, n_trials: int = 5,
                  generations: int = 100, pop_size: int = 150,
                  episodes: int = 3, max_steps: int = 500,
                  seed_base: int = 0, **kwargs):
    """
    Run a benchmark comparison.
    
    Args:
        env_name: Gymnasium environment name
        algorithm: 'neat' or 'atlas'
        n_trials: Number of independent trials
        generations: Max generations per trial
        pop_size: Population size
        episodes: Evaluation episodes per genome
        max_steps: Max steps per episode
        seed_base: Base random seed
        **kwargs: Additional algorithm-specific parameters
    
    Returns:
        dict with aggregated results
    """
    num_inputs, num_outputs, action_type = get_env_info(env_name)
    
    all_results = []
    
    for trial in range(n_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{n_trials} - {algorithm} on {env_name}")
        print(f"{'='*60}")
        
        seed = seed_base + trial
        np.random.seed(seed)
        random_state = np.random.get_state()
        
        if algorithm == 'neat':
            result = run_neat_trial(env_name, num_inputs, num_outputs,
                                   generations, pop_size, episodes, 
                                   max_steps, seed, **kwargs)
        elif algorithm == 'atlas':
            result = run_atlas_trial(env_name, num_inputs, num_outputs,
                                    generations, pop_size, episodes,
                                    max_steps, seed, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        all_results.append(result)
    
    # Aggregate results
    return aggregate_results(all_results)


def run_neat_trial(env_name, num_inputs, num_outputs, generations, 
                   pop_size, episodes, max_steps, seed, **kwargs):
    """Run a single NEAT trial."""
    import neat
    
    # Create config
    config_path = 'src/baselines/neat_config.ini'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    config.genome_config.num_inputs = num_inputs
    config.genome_config.num_outputs = num_outputs
    config.genome_config.num_hidden = 0
    config.genome_config.initial_connection = 'full'
    config.pop_size = pop_size
    
    # Create evaluator
    evaluator = create_evaluator(env_name, episodes, max_steps, seed)
    
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            # Convert neat genome to our Genome for evaluation
            # (Actually just use neat's network directly)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            env = gym.make(env_name)
            
            total_reward = 0.0
            for ep in range(episodes):
                obs, info = env.reset(seed=seed + ep)
                episode_reward = 0.0
                for step in range(max_steps):
                    obs_flat = np.array(obs).flatten()
                    action = net.activate(obs_flat)
                    
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action = int(np.argmax(action))
                    else:
                        action = np.array(action).clip(env.action_space.low, 
                                                        env.action_space.high)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                total_reward += episode_reward
            
            env.close()
            genome.fitness = total_reward / episodes
    
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'n_species': [],
        'times': [],
    }
    
    start_time = time.time()
    
    for gen in range(generations):
        gen_start = time.time()
        
        genomes = list(p.population.items())
        eval_genomes(genomes, config)
        p.population = dict(genomes)
        
        fitnesses = [g.fitness for _, g in genomes if g.fitness is not None]
        best = max(fitnesses) if fitnesses else 0
        avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        
        p.reporters.start_generation(p.generation, p.config)
        p.population = p.reproduction.reproduce(p.config, p.species, p.config.pop_size, p.generation)
        p.species.speciate(p.config, p.population, p.generation)
        p.reporters.post_evaluate(p.config, p.population, p.species, best)
        
        n_species = len(p.species.species)
        gen_time = time.time() - gen_start
        
        history['best_fitness'].append(best)
        history['avg_fitness'].append(avg)
        history['n_species'].append(n_species)
        history['times'].append(gen_time)
        
        p.generation += 1
    
    history['total_time'] = time.time() - start_time
    return history


def run_atlas_trial(env_name, num_inputs, num_outputs, generations,
                    pop_size, episodes, max_steps, seed, **kwargs):
    """Run a single Atlas-NEAT trial."""
    import sys
    sys.path.insert(0, 'src')
    
    from atlas_neat.atlas_neat import AtlasNEAT, AtlasConfig
    
    # Override fitness threshold based on environment
    config = AtlasConfig(
        pop_size=pop_size,
        max_generations=generations,
        fitness_threshold=kwargs.get('fitness_threshold', 500.0),
        **{k: v for k, v in kwargs.items() if k != 'fitness_threshold'}
    )
    
    atlas = AtlasNEAT(config, num_inputs, num_outputs)
    atlas.create_population()
    
    evaluator = create_evaluator(env_name, episodes, max_steps, seed)
    
    result = atlas.run(evaluator, generations)
    history = result['history']
    history['total_time'] = sum(history['times'])
    
    return history


def aggregate_results(all_results):
    """Aggregate results from multiple trials."""
    max_gens = max(len(r['best_fitness']) for r in all_results)
    
    agg = {
        'best_fitness_mean': [],
        'best_fitness_std': [],
        'avg_fitness_mean': [],
        'avg_fitness_std': [],
        'n_species_mean': [],
        'n_species_std': [],
        'time_mean': [],
        'time_std': [],
    }
    
    # Check if coverage exists (Atlas-NEAT only)
    has_coverage = 'coverage' in all_results[0]
    if has_coverage:
        agg['coverage_mean'] = []
        agg['coverage_std'] = []
        agg['qd_score_mean'] = []
        agg['qd_score_std'] = []
    
    for gen in range(max_gens):
        bf = [r['best_fitness'][gen] if gen < len(r['best_fitness']) else r['best_fitness'][-1]
              for r in all_results]
        af = [r['avg_fitness'][gen] if gen < len(r['avg_fitness']) else r['avg_fitness'][-1]
              for r in all_results]
        ns = [r['n_species'][gen] if gen < len(r['n_species']) else r['n_species'][-1]
              for r in all_results]
        tm = [r['times'][gen] if gen < len(r['times']) else 0 for r in all_results]
        
        agg['best_fitness_mean'].append(float(np.mean(bf)))
        agg['best_fitness_std'].append(float(np.std(bf)))
        agg['avg_fitness_mean'].append(float(np.mean(af)))
        agg['avg_fitness_std'].append(float(np.std(af)))
        agg['n_species_mean'].append(float(np.mean(ns)))
        agg['n_species_std'].append(float(np.std(ns)))
        agg['time_mean'].append(float(np.mean(tm)))
        agg['time_std'].append(float(np.std(tm)))
        
        if has_coverage:
            cv = [r['coverage'][gen] if gen < len(r['coverage']) else r['coverage'][-1]
                  for r in all_results]
            qd = [r['qd_score'][gen] if gen < len(r['qd_score']) else r['qd_score'][-1]
                  for r in all_results]
            agg['coverage_mean'].append(float(np.mean(cv)))
            agg['coverage_std'].append(float(np.std(cv)))
            agg['qd_score_mean'].append(float(np.mean(qd)))
            agg['qd_score_std'].append(float(np.std(qd)))
    
    # Total time
    total_times = [r.get('total_time', sum(r['times'])) for r in all_results]
    agg['total_time_mean'] = float(np.mean(total_times))
    agg['total_time_std'] = float(np.std(total_times))
    
    # Success rate (if we can determine threshold)
    agg['n_trials'] = len(all_results)
    
    return agg
