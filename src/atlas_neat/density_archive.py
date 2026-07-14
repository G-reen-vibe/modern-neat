"""
Density-based adaptive archive for Atlas-NEAT.

Replaces the rigid grid archive with DBSCAN-style clustering.
Genomes cluster naturally in descriptor space, forming adaptive "species".

Key advantages over grid archive:
1. No fixed resolution - adapts to data density
2. Works with any number of varying dimensions
3. Can find arbitrarily-shaped clusters
4. More robust to sparse initial populations
5. Natural cluster count reflects true diversity
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from atlas_neat.genome import Genome


class Cluster:
    """A cluster of similar genomes (like a species in NEAT)."""
    
    def __init__(self, cluster_id: int):
        self.id = cluster_id
        self.members: List[Genome] = []
        self.centroid: Optional[np.ndarray] = None
        self.best_fitness: float = -9999.0
        self.best_genome: Optional[Genome] = None
        self.age: int = 0
    
    def add(self, genome: Genome, descriptor: np.ndarray):
        self.members.append(genome)
        if genome.fitness and genome.fitness > self.best_fitness:
            self.best_fitness = genome.fitness
            self.best_genome = genome
    
    def update_centroid(self):
        if not self.members:
            return
        descriptors = []
        for g in self.members:
            descriptors.append(np.array(g.get_characterization()))
        self.centroid = np.mean(descriptors, axis=0)
    
    def clear_members(self):
        self.members = []


class DensityArchive:
    """
    Density-based archive using adaptive clustering.
    
    Instead of a fixed grid, genomes are clustered in descriptor space
    using a density-based approach inspired by DBSCAN.
    
    Clusters form naturally, and their number adapts to population diversity.
    """
    
    def __init__(self,
                 eps: float = 0.3,  # Neighborhood radius
                 min_samples: int = 2,  # Min members to form cluster
                 decay_rate: float = 0.95):  # Cluster score decay
        self.eps = eps
        self.min_samples = min_samples
        self.decay_rate = decay_rate
        
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0
        self.noise_points: List[Tuple[Genome, np.ndarray]] = []
        
        # Statistics
        self.generation = 0
    
    def _distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Euclidean distance between descriptors."""
        return np.linalg.norm(d1 - d2)
    
    def _find_nearest_cluster(self, descriptor: np.ndarray) -> Optional[int]:
        """Find the nearest cluster within eps radius."""
        nearest_id = None
        nearest_dist = float('inf')
        
        for cid, cluster in self.clusters.items():
            if cluster.centroid is not None:
                dist = self._distance(descriptor, cluster.centroid)
                if dist < self.eps and dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = cid
        
        return nearest_id
    
    def add(self, genome: Genome):
        """Add a genome to the archive (assign to cluster or noise)."""
        descriptor = np.array(genome.get_characterization())
        
        # Try to assign to existing cluster
        nearest = self._find_nearest_cluster(descriptor)
        
        if nearest is not None:
            self.clusters[nearest].add(genome, descriptor)
        else:
            # Add to noise (may form new cluster later)
            self.noise_points.append((genome, descriptor))
    
    def _form_clusters_from_noise(self):
        """Try to form new clusters from noise points."""
        if len(self.noise_points) < self.min_samples:
            return
        
        # Group noise points that are close to each other
        unassigned = list(self.noise_points)
        new_clusters = []
        
        while len(unassigned) >= self.min_samples:
            # Pick a random seed point
            seed_idx = random.randrange(len(unassigned))
            seed_genome, seed_desc = unassigned[seed_idx]
            
            # Find all points within eps
            neighbors = []
            neighbor_indices = []
            for i, (g, d) in enumerate(unassigned):
                if self._distance(seed_desc, d) < self.eps:
                    neighbors.append((g, d))
                    neighbor_indices.append(i)
            
            if len(neighbors) >= self.min_samples:
                # Form new cluster
                cluster = Cluster(self.next_cluster_id)
                self.next_cluster_id += 1
                
                for g, d in neighbors:
                    cluster.add(g, d)
                
                cluster.update_centroid()
                new_clusters.append(cluster)
                
                # Remove assigned points
                for i in reversed(neighbor_indices):
                    unassigned.pop(i)
            else:
                # Can't form cluster from this seed
                unassigned.pop(seed_idx)
        
        # Add new clusters
        for cluster in new_clusters:
            self.clusters[cluster.id] = cluster
        
        # Remaining unassigned stays as noise
        self.noise_points = unassigned
    
    def update(self):
        """
        Update archive: form clusters from noise, update centroids, decay.
        Call this at the end of each generation.
        """
        # Try to form new clusters from accumulated noise
        self._form_clusters_from_noise()
        
        # Update centroids
        for cluster in self.clusters.values():
            cluster.update_centroid()
            cluster.age += 1
        
        # Remove empty clusters
        self.clusters = {cid: c for cid, c in self.clusters.items() if c.members}
        
        self.generation += 1
    
    def get_best(self) -> Optional[Genome]:
        """Get the best genome across all clusters."""
        best = None
        best_fitness = -9999.0
        for cluster in self.clusters.values():
            if cluster.best_genome and cluster.best_fitness > best_fitness:
                best = cluster.best_genome
                best_fitness = cluster.best_fitness
        return best
    
    def get_best_fitness(self) -> float:
        """Get best fitness across all clusters."""
        return max((c.best_fitness for c in self.clusters.values()), default=-9999.0)
    
    def get_coverage(self) -> float:
        """Fraction of 'active' cluster space (proxy for diversity)."""
        if not self.clusters:
            return 0.0
        # Coverage = ratio of clusters with at least min_samples members
        good_clusters = sum(1 for c in self.clusters.values() 
                           if len(c.members) >= self.min_samples)
        return good_clusters / max(len(self.clusters), 1)
    
    def get_qd_score(self) -> float:
        """Sum of best fitness in each cluster."""
        return sum(c.best_fitness for c in self.clusters.values() 
                   if c.best_fitness > -9998)
    
    def get_n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.clusters)
    
    def sample_parent(self, selection_method: str = 'tournament',
                     tournament_size: int = 5) -> Optional[Genome]:
        """Sample a parent from clusters."""
        if not self.clusters:
            return None
        
        # Collect cluster bests
        candidates = []
        for cluster in self.clusters.values():
            if cluster.best_genome:
                candidates.append(cluster.best_genome)
        
        if not candidates:
            return None
        
        if selection_method == 'tournament':
            contestants = random.sample(candidates, min(tournament_size, len(candidates)))
            return max(contestants, key=lambda g: g.fitness or -9999)
        elif selection_method == 'roulette':
            fitnesses = [max(g.fitness or 0, 0.01) for g in candidates]
            total = sum(fitnesses)
            probs = [f / total for f in fitnesses]
            return candidates[np.random.choice(len(candidates), p=probs)]
        else:
            return random.choice(candidates)
    
    def get_diversity_stats(self) -> dict:
        """Get diversity statistics."""
        n_clusters = len(self.clusters)
        total_members = sum(len(c.members) for c in self.clusters.values())
        
        return {
            'coverage': self.get_coverage(),
            'qd_score': self.get_qd_score(),
            'n_occupied': n_clusters,
            'descriptor_spread': 0.0,  # Would need to compute
            'best_fitness': self.get_best_fitness(),
            'avg_fitness': 0.0,  # Would need to compute
            'total_members': total_members,
            'noise_points': len(self.noise_points),
        }
    
    def __len__(self):
        return len(self.clusters)
