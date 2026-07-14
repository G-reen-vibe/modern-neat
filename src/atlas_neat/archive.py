"""
Adaptive Quality-Diversity Archive for Atlas-NEAT.
Replaces NEAT's speciation with an illumination-style archive.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from atlas_neat.genome import Genome


@dataclass
class ArchiveCell:
    """A single cell in the archive."""
    indices: Tuple[int, ...]  # Grid indices
    descriptor_center: Tuple[float, ...]  # Center of this cell in descriptor space
    resident: Optional[Genome] = None  # Best genome in this cell
    fitness: float = -9999.0
    visit_count: int = 0


class AdaptiveArchive:
    """
    Quality-Diversity archive with adaptive resolution.
    
    The archive maps genomes into a grid based on their behavioral-topological
    characterization. Each cell keeps the best genome for that region.
    
    Unlike fixed archives, the resolution adapts based on population density.
    """
    
    def __init__(self, 
                 n_dimensions: int = 3,
                 initial_resolution: int = 4,
                 max_resolution: int = 16,
                 adapt_interval: int = 10):
        self.n_dimensions = n_dimensions
        self.resolution = initial_resolution
        self.max_resolution = max_resolution
        self.adapt_interval = adapt_interval
        
        # The archive grid: maps indices -> cell
        self.cells: Dict[Tuple[int, ...], ArchiveCell] = {}
        
        # Bounds for each dimension (auto-adjusted)
        self.bounds = [(0.0, 1.0) for _ in range(n_dimensions)]
        
        # Statistics
        self.generation = 0
        self.insertion_count = 0
        
        self._init_cells()
    
    def _init_cells(self):
        """Initialize archive cells for current resolution."""
        self.cells = {}
        
        def create_indices(dim, current):
            if dim == self.n_dimensions:
                indices = tuple(current)
                center = tuple(
                    (i + 0.5) / self.resolution 
                    for i in indices
                )
                self.cells[indices] = ArchiveCell(indices, center)
                return
            for i in range(self.resolution):
                create_indices(dim + 1, current + [i])
        
        create_indices(0, [])
    
    def descriptor_to_indices(self, descriptor: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert a descriptor to grid indices."""
        indices = []
        for i, val in enumerate(descriptor):
            lo, hi = self.bounds[i]
            # Normalize to [0, 1]
            norm = (val - lo) / (hi - lo) if hi > lo else 0.5
            norm = max(0.0, min(0.999, norm))  # Clamp
            idx = int(norm * self.resolution)
            idx = min(idx, self.resolution - 1)
            indices.append(idx)
        return tuple(indices)
    
    def add(self, genome: Genome):
        """Add a genome to the archive. Returns True if it improved its cell."""
        descriptor = genome.get_characterization()
        indices = self.descriptor_to_indices(descriptor)
        
        if indices not in self.cells:
            return False
        
        cell = self.cells[indices]
        cell.visit_count += 1
        self.insertion_count += 1
        
        fitness = genome.fitness or -9999.0
        
        if fitness > cell.fitness:
            cell.resident = genome
            cell.fitness = fitness
            return True
        return False
    
    def get_coverage(self) -> float:
        """Fraction of cells occupied."""
        occupied = sum(1 for c in self.cells.values() if c.resident is not None)
        return occupied / len(self.cells)
    
    def get_qd_score(self) -> float:
        """Quality-Diversity score: sum of all occupied cell fitnesses."""
        return sum(max(c.fitness, 0) for c in self.cells.values() if c.resident is not None)
    
    def get_best(self) -> Optional[Genome]:
        """Get the best genome in the archive."""
        best = None
        best_fitness = -9999.0
        for cell in self.cells.values():
            if cell.resident and cell.fitness > best_fitness:
                best = cell.resident
                best_fitness = cell.fitness
        return best
    
    def get_best_fitness(self) -> float:
        """Get the best fitness in the archive."""
        return max((c.fitness for c in self.cells.values() if c.resident is not None), default=-9999.0)
    
    def get_avg_fitness(self) -> float:
        """Get average fitness of occupied cells."""
        fitnesses = [c.fitness for c in self.cells.values() if c.resident is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    
    def sample_parent(self, selection_method: str = 'tournament',
                     tournament_size: int = 5) -> Optional[Genome]:
        """Sample a parent from the archive."""
        occupied = [c for c in self.cells.values() if c.resident is not None]
        if not occupied:
            return None
        
        if selection_method == 'tournament':
            contestants = random.sample(occupied, min(tournament_size, len(occupied)))
            return max(contestants, key=lambda c: c.fitness).resident
        elif selection_method == 'roulette':
            fitnesses = [max(c.fitness, 0.01) for c in occupied]
            total = sum(fitnesses)
            probs = [f / total for f in fitnesses]
            chosen = occupied[np.random.choice(len(occupied), p=probs)]
            return chosen.resident
        elif selection_method == 'uniform':
            return random.choice(occupied).resident
        else:
            return random.choice(occupied).resident
    
    def get_novelty_gradient(self, descriptor: Tuple[float, ...]) -> np.ndarray:
        """
        Compute the novelty gradient for a descriptor.
        Points toward underexplored regions of the archive.
        """
        indices = self.descriptor_to_indices(descriptor)
        
        # Count neighbors
        gradient = np.zeros(self.n_dimensions)
        for dim in range(self.n_dimensions):
            for delta in [-1, 1]:
                neighbor = list(indices)
                neighbor[dim] += delta
                neighbor = tuple(max(0, min(self.resolution - 1, n)) for n in neighbor)
                
                cell = self.cells.get(neighbor)
                if cell:
                    gradient[dim] += (1.0 if cell.resident is None else -cell.visit_count * 0.1)
        
        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient = gradient / norm
        
        return gradient
    
    def get_empty_cells(self) -> List[ArchiveCell]:
        """Get all empty cells."""
        return [c for c in self.cells.values() if c.resident is None]
    
    def adapt_resolution(self):
        """
        Adapt archive resolution based on coverage.
        If coverage is high, increase resolution to allow finer discrimination.
        """
        if self.generation % self.adapt_interval != 0:
            return
        
        coverage = self.get_coverage()
        
        if coverage > 0.8 and self.resolution < self.max_resolution:
            # Double resolution
            old_res = self.resolution
            self.resolution = min(self.resolution * 2, self.max_resolution)
            
            # Migrate existing residents
            old_cells = list(self.cells.values())
            self._init_cells()
            
            for cell in old_cells:
                if cell.resident:
                    self.add(cell.resident)
        
        self.generation += 1
    
    def get_diversity_stats(self) -> dict:
        """Get diversity statistics about the archive."""
        occupied = [c for c in self.cells.values() if c.resident is not None]
        if not occupied:
            return {'coverage': 0, 'qd_score': 0, 'n_occupied': 0, 'descriptor_spread': 0}
        
        descriptors = np.array([c.descriptor_center for c in occupied])
        spread = np.std(descriptors, axis=0).mean() if len(occupied) > 1 else 0
        
        return {
            'coverage': self.get_coverage(),
            'qd_score': self.get_qd_score(),
            'n_occupied': len(occupied),
            'descriptor_spread': float(spread),
            'best_fitness': self.get_best_fitness(),
            'avg_fitness': self.get_avg_fitness(),
        }
    
    def __len__(self):
        return sum(1 for c in self.cells.values() if c.resident is not None)
