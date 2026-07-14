"""
Atlas-NEAT: Modern Quality-Diversity NeuroEvolution.

Core innovation: Replaces NEAT's speciation mechanism with an adaptive
Quality-Diversity archive. Networks are mapped to archive cells based on
their topological and behavioral characteristics, maintaining diversity
explicitly through the archive structure.
"""

import random
import time
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from atlas_neat.genome import Genome, create_random_genome
from atlas_neat.density_archive import DensityArchive


@dataclass
class AtlasConfig:
    """Configuration for Atlas-NEAT."""
    # Population
    pop_size: int = 150
    
    # Archive
    archive_dims: int = 5
    archive_resolution: int = 2
    archive_max_resolution: int = 8
    archive_adapt_interval: int = 10
    
    # Mutation rates
    weight_mutate_prob: float = 0.9
    weight_mutate_power: float = 0.5
    weight_replace_prob: float = 0.1
    bias_mutate_prob: float = 0.7
    bias_mutate_power: float = 0.5
    activation_mutate_rate: float = 0.1
    conn_add_prob: float = 0.2
    conn_remove_prob: float = 0.1
    node_add_prob: float = 0.2
    node_remove_prob: float = 0.05
    toggle_prob: float = 0.01
    
    # Selection
    selection_method: str = 'tournament'
    tournament_size: int = 5
    elitism: int = 2
    
    # Multi-objective: fitness + complexity penalty (prefer simpler solutions)
    complexity_penalty: float = 0.01  # Penalty per connection+node
    use_multi_objective: bool = True
    
    # Novelty-guided exploration
    novelty_weight: float = 0.3  # Weight for novelty in parent selection
    explore_prob: float = 0.2  # Probability of exploration-oriented mutation
    
    # Adaptive rates
    adaptive_mutation: bool = True
    mutation_boost: float = 2.0  # Multiply rates when diversity is low
    
    # Diversity injection
    immigrant_rate: float = 0.05  # Fraction of population replaced by random immigrants
    
    # Local search: when >N clusters stable, do weight-only refinement
    local_search_trigger: int = 5  # Min clusters to trigger local search
    local_search_prob: float = 0.3  # Prob of weight-only mutation when in local search
    
    # Termination
    max_generations: int = 100
    fitness_threshold: float = 500.0
    patience: int = 20  # Early stopping patience (generations without improvement)


class AtlasNEAT:
    """
    Atlas-NEAT: Quality-Diversity NeuroEvolution.
    
    Main differences from standard NEAT:
    1. No speciation - diversity maintained by archive
    2. No compatibility distance
    3. No stagnation detection
    4. Novelty-guided exploration
    5. Adaptive mutation rates
    """
    
    def __init__(self, config: AtlasConfig, num_inputs: int, num_outputs: int):
        self.config = config
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # Density-based archive for quality-diversity
        self.archive = DensityArchive(
            eps=0.25,
            min_samples=2,
            decay_rate=0.95
        )
        
        # Population
        self.population: Dict[int, Genome] = {}
        self.generation = 0
        self.next_genome_key = 0
        
        # Memory: hall of fame
        self.hall_of_fame: List[Genome] = []
        self.hof_size: int = 10
        
        # Statistics
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'coverage': [],
            'qd_score': [],
            'n_species': [],  # Actually n_occupied cells
            'mutation_rates': [],
            'times': [],
        }
    
    def create_population(self, pop_size: Optional[int] = None):
        """Create initial population of minimal genomes."""
        if pop_size is None:
            pop_size = self.config.pop_size
        
        Genome.reset_innovation()
        self.population = {}
        
        for i in range(pop_size):
            genome = create_random_genome(i, self.num_inputs, self.num_outputs)
            self.population[i] = genome
            self.next_genome_key = i + 1
    
    def _compute_adjusted_fitness(self, genome: Genome) -> float:
        """Compute multi-objective adjusted fitness (fitness - complexity_penalty)."""
        if genome.fitness is None:
            return -9999.0
        if not self.config.use_multi_objective:
            return genome.fitness
        
        # Penalize complexity to prefer simpler solutions
        complexity = len(genome.nodes) + sum(1 for c in genome.connections.values() if c.enabled)
        penalty = self.config.complexity_penalty * complexity
        
        # Direction-aware: for maximization, subtract penalty; for minimization, add penalty
        # Assume maximization by default (standard for RL)
        return genome.fitness - penalty
    
    def evaluate_population(self, eval_func: Callable[[Genome], float]):
        """Evaluate all genomes in the population."""
        for genome in self.population.values():
            if genome.fitness is None:
                raw_fitness = eval_func(genome)
                genome.raw_fitness = raw_fitness
                # Store adjusted fitness for selection
                genome.adjusted_fitness = self._compute_adjusted_fitness(genome)
                # Use adjusted fitness as the main fitness for selection
                genome.fitness = genome.adjusted_fitness
    
    def update_archive(self):
        """Add all population members to the archive."""
        for genome in self.population.values():
            self.archive.add(genome)
    
    def get_mutation_config(self) -> dict:
        """Get mutation configuration, possibly adapted based on progress."""
        config = {
            'weight_mutate_prob': self.config.weight_mutate_prob,
            'weight_mutate_power': self.config.weight_mutate_power,
            'weight_replace_prob': self.config.weight_replace_prob,
            'bias_mutate_prob': self.config.bias_mutate_prob,
            'bias_mutate_power': self.config.bias_mutate_power,
            'activation_mutate_rate': self.config.activation_mutate_rate,
            'conn_add_prob': self.config.conn_add_prob,
            'conn_remove_prob': self.config.conn_remove_prob,
            'node_add_prob': self.config.node_add_prob,
            'node_remove_prob': self.config.node_remove_prob,
            'toggle_prob': self.config.toggle_prob,
        }
        
        if self.config.adaptive_mutation:
            # Check if progress has stalled
            recent_best = self.history['best_fitness'][-5:] if len(self.history['best_fitness']) >= 5 else self.history['best_fitness']
            if len(recent_best) >= 2 and recent_best[-1] <= recent_best[0] * 1.01:
                # Progress stalled - boost structural mutations
                stall_factor = min(3.0, 1.0 + (len(recent_best) - 1) * 0.5)
                for key in ['conn_add_prob', 'node_add_prob', 'activation_mutate_rate',
                           'conn_remove_prob', 'node_remove_prob']:
                    config[key] = min(1.0, config[key] * stall_factor)
                
                # Also increase weight mutation power for more exploration
                config['weight_mutate_power'] *= stall_factor
            
            # Boost when diversity is low
            n_clusters = self.archive.get_n_clusters()
            if n_clusters < 3:
                boost = self.config.mutation_boost
                for key in ['conn_add_prob', 'node_add_prob', 'activation_mutate_rate']:
                    config[key] = min(1.0, config[key] * boost)
        
        return config
    
    def _update_hof(self):
        """Update hall of fame with current best genomes."""
        current_best = sorted(
            self.population.values(),
            key=lambda g: getattr(g, 'raw_fitness', g.fitness) or -9999,
            reverse=True
        )[:3]
        
        for genome in current_best:
            # Check if already in HOF
            if not any(g.key == genome.key for g in self.hall_of_fame):
                self.hall_of_fame.append(copy.deepcopy(genome))
        
        # Keep only top N
        self.hall_of_fame.sort(
            key=lambda g: getattr(g, 'raw_fitness', g.fitness) or -9999,
            reverse=True
        )
        self.hall_of_fame = self.hall_of_fame[:self.hof_size]
    
    def select_parent(self) -> Genome:
        """
        Select a parent using hybrid strategy.
        When archive is sparse, use population-level tournament selection.
        When archive is well-populated, use archive-based selection.
        """
        coverage = self.archive.get_coverage()
        
        if coverage < 0.25 or len(self.archive) < 3:
            # Archive too sparse - use population tournament selection
            candidates = list(self.population.values())
            if len(candidates) < self.config.tournament_size:
                return random.choice(candidates)
            
            contestants = random.sample(candidates, self.config.tournament_size)
            return max(contestants, key=lambda g: getattr(g, 'adjusted_fitness', g.fitness) or -9999)
        else:
            # Archive well-populated - use archive selection
            parent = self.archive.sample_parent(
                self.config.selection_method,
                self.config.tournament_size,
                self.config.novelty_weight
            )
            if parent is not None:
                return parent
            
            # Fallback
            candidates = list(self.population.values())
            return random.choice(candidates)
    
    def create_offspring(self) -> Genome:
        """Create a single offspring through selection and mutation."""
        # Decide: mutation-only or crossover
        if random.random() < 0.75 or len(self.archive) < 2:
            # Asexual reproduction with mutation
            parent = self.select_parent()
            
            child = copy.deepcopy(parent)
            child.key = self.next_genome_key
            self.next_genome_key += 1
            child.fitness = None
            
            # Check if we should do exploration-oriented mutation
            if random.random() < self.config.explore_prob:
                # Boost structural mutations
                mut_config = self.get_mutation_config()
                mut_config['conn_add_prob'] = min(1.0, mut_config['conn_add_prob'] * 2)
                mut_config['node_add_prob'] = min(1.0, mut_config['node_add_prob'] * 2)
                mut_config['activation_mutate_rate'] = min(1.0, mut_config['activation_mutate_rate'] * 2)
                child.mutate(mut_config)
            elif (len(self.archive.clusters) >= self.config.local_search_trigger and 
                  random.random() < self.config.local_search_prob):
                # Local search: weight-only mutation for fine-tuning
                mut_config = self.get_mutation_config()
                mut_config['conn_add_prob'] = 0
                mut_config['node_add_prob'] = 0
                mut_config['conn_remove_prob'] = 0
                mut_config['node_remove_prob'] = 0
                mut_config['activation_mutate_rate'] = 0
                mut_config['skip_add_prob'] = 0
                mut_config['weight_mutate_power'] *= 0.5  # Smaller steps
                child.mutate(mut_config)
            else:
                child.mutate(self.get_mutation_config())
        else:
            # Crossover
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child = parent1.crossover(parent2)
            child.key = self.next_genome_key
            self.next_genome_key += 1
            child.fitness = None
            child.mutate(self.get_mutation_config())
        
        return child
    
    def create_next_generation(self):
        """Create the next generation with cluster-competitive allocation."""
        new_population = {}
        
        # Elitism: keep best from each cluster + global best
        elites_added = 0
        # Global best
        archive_best = self.archive.get_best()
        if archive_best is not None:
            elite = copy.deepcopy(archive_best)
            elite.key = self.next_genome_key
            self.next_genome_key += 1
            new_population[elite.key] = elite
            elites_added += 1
        
        # Per-cluster elites (top genome from each cluster)
        for cluster in self.archive.clusters.values():
            if elites_added >= self.config.elitism:
                break
            if cluster.best_genome and cluster.best_genome.key not in new_population:
                elite = copy.deepcopy(cluster.best_genome)
                elite.key = self.next_genome_key
                self.next_genome_key += 1
                new_population[elite.key] = elite
                elites_added += 1
        
        # Cluster-competitive offspring allocation
        # Better clusters get more offspring slots
        if len(self.archive.clusters) >= 2:
            # Compute cluster fitness shares
            cluster_scores = {}
            total_score = 0
            for cid, cluster in self.archive.clusters.items():
                # Score = best_fitness + bonus for young/improving clusters
                young_bonus = max(0, 1.0 - cluster.age * 0.2) * 0.5
                size_bonus = min(0.3, len(cluster.members) * 0.05)
                score = max(cluster.best_fitness, 0.01) * (1.0 + young_bonus + size_bonus)
                cluster_scores[cid] = score
                total_score += score
            
            # Allocate offspring slots proportionally
            slots = self.config.pop_size - len(new_population)
            cluster_slots = {}
            for cid, score in cluster_scores.items():
                cluster_slots[cid] = max(1, int(slots * score / total_score))
            
            # Create offspring for each cluster
            for cid, n_slots in cluster_slots.items():
                cluster = self.archive.clusters[cid]
                for _ in range(n_slots):
                    if len(new_population) >= self.config.pop_size:
                        break
                    # Create offspring from this cluster's representative
                    if cluster.best_genome:
                        child = copy.deepcopy(cluster.best_genome)
                        child.key = self.next_genome_key
                        self.next_genome_key += 1
                        child.fitness = None
                        child.mutate(self.get_mutation_config())
                        new_population[child.key] = child
        
        # Diversity injection: add random immigrants and HOF reintroductions
        n_immigrants = int(self.config.pop_size * self.config.immigrant_rate)
        for i in range(n_immigrants):
            if len(new_population) >= self.config.pop_size:
                break
            # 50% chance of HOF reintroduction vs random immigrant
            if self.hall_of_fame and random.random() < 0.5:
                hof_genome = random.choice(self.hall_of_fame)
                immigrant = copy.deepcopy(hof_genome)
                immigrant.key = self.next_genome_key
                immigrant.fitness = None
                self.next_genome_key += 1
            else:
                immigrant = create_random_genome(self.next_genome_key, 
                                                self.num_inputs, self.num_outputs)
                self.next_genome_key += 1
            new_population[immigrant.key] = immigrant
        
        # Fill any remaining slots
        while len(new_population) < self.config.pop_size:
            child = self.create_offspring()
            new_population[child.key] = child
        
        self.population = new_population
        self.generation += 1
        self.archive.generation = self.generation
    
    def run_generation(self, eval_func: Callable[[Genome], float]) -> dict:
        """Run a single generation. Returns statistics."""
        gen_start = time.time()
        
        # Evaluate
        self.evaluate_population(eval_func)
        
        # Update archive
        self.update_archive()
        
        # Update hall of fame
        self._update_hof()
        
        # Update density clusters
        self.archive.update()
        
        # Create next generation
        self.create_next_generation()
        
        # Collect stats
        fitnesses = [g.fitness for g in self.population.values() if g.fitness is not None]
        best_fitness = max(fitnesses) if fitnesses else -9999
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        
        diversity = self.archive.get_diversity_stats()
        
        gen_time = time.time() - gen_start
        
        self.history['best_fitness'].append(float(best_fitness))
        self.history['avg_fitness'].append(float(avg_fitness))
        self.history['coverage'].append(diversity['coverage'])
        self.history['qd_score'].append(diversity['qd_score'])
        self.history['n_species'].append(diversity['n_occupied'])
        self.history['times'].append(gen_time)
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'coverage': diversity['coverage'],
            'qd_score': diversity['qd_score'],
            'n_occupied': diversity['n_occupied'],
            'n_clusters': self.archive.get_n_clusters(),
            'time': gen_time,
        }
    
    def run(self, eval_func: Callable[[Genome], float], 
            generations: Optional[int] = None,
            callback: Optional[Callable] = None) -> dict:
        """
        Run Atlas-NEAT evolution.
        
        Args:
            eval_func: Function that takes a Genome and returns fitness
            generations: Max generations (default from config)
            callback: Optional callback(stats) called each generation
        
        Returns:
            Final statistics and history
        """
        if generations is None:
            generations = self.config.max_generations
        
        print(f"Atlas-NEAT: Starting evolution")
        print(f"  Population: {self.config.pop_size}")
        print(f"  Archive: Density-based (eps={self.archive.eps})")
        
        best_ever = -9999.0
        patience_counter = 0
        
        for gen in range(generations):
            stats = self.run_generation(eval_func)
            
            print(f"Gen {stats['generation']}: Best={stats['best_fitness']:.2f}, "
                  f"Avg={stats['avg_fitness']:.2f}, Coverage={stats['coverage']:.2%}, "
                  f"QD={stats['qd_score']:.1f}, Cells={stats['n_occupied']}, "
                  f"Clusters={stats['n_clusters']}, Time={stats['time']:.2f}s")
            
            if callback:
                callback(stats)
            
            # Check improvement for early stopping
            if stats['best_fitness'] > best_ever:
                best_ever = stats['best_fitness']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check termination
            if stats['best_fitness'] >= self.config.fitness_threshold:
                print(f"Fitness threshold reached at generation {gen}!")
                break
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping at generation {gen} (no improvement for {patience_counter} gens)")
                break
        
        return {
            'history': self.history,
            'best_genome': self.archive.get_best(),
            'best_fitness': self.archive.get_best_fitness(),
            'archive': self.archive,
            'final_stats': stats,
        }


def create_atlas_neat(num_inputs: int, num_outputs: int, 
                      pop_size: int = 150, **kwargs) -> AtlasNEAT:
    """Convenience function to create an Atlas-NEAT instance."""
    config = AtlasConfig(pop_size=pop_size, **kwargs)
    atlas = AtlasNEAT(config, num_inputs, num_outputs)
    atlas.create_population()
    return atlas
