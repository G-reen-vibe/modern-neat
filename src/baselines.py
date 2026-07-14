"""Baselines for comparison.

  - RandomSearch: sample random feed-forward networks of fixed size.
  - FixedGA:      evolve weights of a fixed-topology MLP using a simple GA.
  - CMAES:        CMA-ES on the weights of a fixed-topology MLP.
  - SimpleDQN:    a minimal DQN (PyTorch) for sanity-check comparison.

All baselines share the same evaluation harness so the comparison is fair.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

from .genome import Genome, Node, Gene, InnovationCounter, INPUT, HIDDEN, OUTPUT, Network
from .evaluation import evaluate


# ---------------------------------------------------------------------------
# Shared fixed-topology helper
# ---------------------------------------------------------------------------


def make_fixed_topology(
    n_inputs: int,
    n_outputs: int,
    n_hidden: int,
    innov: InnovationCounter,
    hidden_activation: str = "tanh",
    output_activation: str = "tanh",
) -> Tuple[Genome, List[Tuple[int, int]]]:
    """Create a fixed fully-connected MLP topology.

    Returns (genome, weight_index_to_gene_keys) so callers can flatten/unflatten
    weights easily.
    """
    g = Genome()
    input_ids = [innov.new_node_id() for _ in range(n_inputs)]
    hidden_ids = [innov.new_node_id() for _ in range(n_hidden)]
    output_ids = [innov.new_node_id() for _ in range(n_outputs)]
    for nid in input_ids:
        g.add_node(Node(nid, INPUT, bias=0.0, activation="identity"))
    for nid in hidden_ids:
        g.add_node(Node(nid, HIDDEN, bias=0.0, activation=hidden_activation))
    for nid in output_ids:
        g.add_node(Node(nid, OUTPUT, bias=0.0, activation=output_activation))

    # weights: input->hidden, hidden->output, hidden biases, output biases
    gene_keys: List[Tuple[int, int]] = []  # (innovation, bias_node_id or -1)
    for in_id in input_ids:
        for h_id in hidden_ids:
            inv = innov.get_edge_innov(in_id, h_id)
            g.add_gene(Gene(inv, in_id, h_id, 0.0, enabled=True))
            gene_keys.append((inv, -1))
    for h_id in hidden_ids:
        for o_id in output_ids:
            inv = innov.get_edge_innov(h_id, o_id)
            g.add_gene(Gene(inv, h_id, o_id, 0.0, enabled=True))
            gene_keys.append((inv, -1))
    for h_id in hidden_ids:
        gene_keys.append((-1, h_id))
    for o_id in output_ids:
        gene_keys.append((-1, o_id))

    return g, gene_keys


def genome_to_vector(g: Genome, gene_keys: List[Tuple[int, int]]) -> np.ndarray:
    """Flatten genome weights+bias into a vector in the order of gene_keys."""
    v = np.zeros(len(gene_keys), dtype=np.float64)
    for i, (inv, bid) in enumerate(gene_keys):
        if inv >= 0:
            v[i] = g.genes[inv].weight
        else:
            v[i] = g.nodes[bid].bias
    return v


def vector_to_genome(g: Genome, v: np.ndarray, gene_keys: List[Tuple[int, int]]) -> Genome:
    """Set weights+bias of a genome from a vector."""
    g2 = g.copy()
    for i, (inv, bid) in enumerate(gene_keys):
        if inv >= 0:
            g2.genes[inv].weight = float(v[i])
        else:
            g2.nodes[bid].bias = float(v[i])
    return g2


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------


@dataclass
class RandomSearchConfig:
    n_inputs: int = 4
    n_outputs: int = 2
    n_hidden: int = 8
    pop_size: int = 100
    n_episodes: int = 3
    max_steps: int = 500
    weight_scale: float = 1.0
    seed: int = 0


class RandomSearch:
    """Each generation: sample `pop_size` random weight vectors, keep best."""

    def __init__(self, cfg: RandomSearchConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        self.innov = InnovationCounter()
        self.template, self.gene_keys = make_fixed_topology(
            cfg.n_inputs, cfg.n_outputs, cfg.n_hidden, self.innov
        )
        self.history: List[Tuple[int, float, float]] = []  # gen, best, mean
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -1e9

    def step(self) -> Tuple[float, float]:
        cfg = self.cfg
        rewards = []
        best_g = None
        best_r = -1e9
        for _ in range(cfg.pop_size):
            v = np.random.randn(len(self.gene_keys)) * cfg.weight_scale
            g = vector_to_genome(self.template, v, self.gene_keys)
            m, _, _, _ = evaluate(
                g, self.env_name, n_episodes=cfg.n_episodes,
                max_steps=cfg.max_steps, seed=cfg.seed
            )
            rewards.append(m)
            if m > best_r:
                best_r = m
                best_g = g
        if best_r > self.best_fitness:
            self.best_fitness = best_r
            self.best_genome = best_g
        gen = len(self.history)
        self.history.append((gen, float(np.max(rewards)), float(np.mean(rewards))))
        return float(np.max(rewards)), float(np.mean(rewards))

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            best, mean = self.step()
            if verbose:
                print(f"RS gen {len(self.history)-1:>3} | best {best:7.2f} | mean {mean:7.2f}")


# ---------------------------------------------------------------------------
# Fixed-topology GA (weight evolution only)
# ---------------------------------------------------------------------------


@dataclass
class FixedGAConfig:
    n_inputs: int = 4
    n_outputs: int = 2
    n_hidden: int = 8
    pop_size: int = 100
    n_episodes: int = 3
    max_steps: int = 500
    weight_perturb_std: float = 0.5
    p_replace: float = 0.1
    elite: int = 2
    seed: int = 0


class FixedGA:
    """Simple (mu+lambda)-style GA on a fixed MLP topology."""

    def __init__(self, cfg: FixedGAConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        self.innov = InnovationCounter()
        self.template, self.gene_keys = make_fixed_topology(
            cfg.n_inputs, cfg.n_outputs, cfg.n_hidden, self.innov
        )
        n = len(self.gene_keys)
        self.population: List[np.ndarray] = [
            np.random.randn(n) * 0.5 for _ in range(cfg.pop_size)
        ]
        self.fitness: List[float] = [0.0] * cfg.pop_size
        self.history: List[Tuple[int, float, float]] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -1e9

    def _eval(self, v: np.ndarray) -> float:
        g = vector_to_genome(self.template, v, self.gene_keys)
        m, _, _, _ = evaluate(
            g, self.env_name, n_episodes=self.cfg.n_episodes,
            max_steps=self.cfg.max_steps, seed=self.cfg.seed
        )
        return m

    def step(self) -> Tuple[float, float]:
        cfg = self.cfg
        self.fitness = [self._eval(v) for v in self.population]
        order = np.argsort(self.fitness)[::-1]
        # elitism
        new_pop: List[np.ndarray] = [self.population[i].copy() for i in order[: cfg.elite]]
        # offspring from top half
        parents = [self.population[i] for i in order[: max(2, cfg.pop_size // 2)]]
        while len(new_pop) < cfg.pop_size:
            p = parents[np.random.randint(len(parents))].copy()
            for i in range(len(p)):
                if np.random.random() < cfg.p_replace:
                    p[i] = np.random.randn()
                else:
                    p[i] += np.random.randn() * cfg.weight_perturb_std
            new_pop.append(p)
        self.population = new_pop[: cfg.pop_size]
        best = float(np.max(self.fitness))
        mean = float(np.mean(self.fitness))
        # track best
        bi = int(np.argmax(self.fitness))
        if self.fitness[bi] > self.best_fitness:
            self.best_fitness = self.fitness[bi]
            self.best_genome = vector_to_genome(self.template, self.population[bi], self.gene_keys)
        gen = len(self.history)
        self.history.append((gen, best, mean))
        return best, mean

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            best, mean = self.step()
            if verbose:
                print(f"GA gen {len(self.history)-1:>3} | best {best:7.2f} | mean {mean:7.2f}")


# ---------------------------------------------------------------------------
# CMA-ES on weights (lightweight pure-numpy implementation)
# ---------------------------------------------------------------------------


class CMAES:
    """Minimal CMA-ES on the weights of a fixed MLP.

    Implements the core rank-mu update with diagonal covariance for speed.
    Suitable for ~50-100 dim problems.
    """

    def __init__(
        self,
        env_name: str,
        n_inputs: int = 4,
        n_outputs: int = 2,
        n_hidden: int = 8,
        pop_size: int = 50,
        n_episodes: int = 3,
        max_steps: int = 500,
        seed: int = 0,
        sigma0: float = 0.5,
    ):
        self.env_name = env_name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.pop_size = pop_size
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.seed = seed
        self.innov = InnovationCounter()
        self.template, self.gene_keys = make_fixed_topology(
            n_inputs, n_outputs, n_hidden, self.innov
        )
        self.dim = len(self.gene_keys)
        self.mean = np.zeros(self.dim)
        self.sigma = float(sigma0)
        # diagonal covariance (we use a vector of variances)
        self.C = np.ones(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.gen = 0
        self.history: List[Tuple[int, float, float]] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -1e9
        # strategy params (simplified)
        self.cc = 0.4
        self.cs = 0.3
        self.c1 = 0.1
        self.cmu = 0.2
        self.damps = 0.6
        self.chiN = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2))
        self.rng = np.random.RandomState(seed)

    def _eval(self, v: np.ndarray) -> float:
        g = vector_to_genome(self.template, v, self.gene_keys)
        m, _, _, _ = evaluate(
            g, self.env_name, n_episodes=self.n_episodes,
            max_steps=self.max_steps, seed=self.seed
        )
        return m

    def step(self) -> Tuple[float, float]:
        N = self.dim
        lam = self.pop_size
        # sample
        sqrtC = np.sqrt(self.C)
        zs = self.rng.randn(lam, N)
        vs = self.mean + self.sigma * zs * sqrtC  # element-wise since diagonal
        # evaluate
        fits = np.array([self._eval(v) for v in vs])
        # maximize: sort descending
        order = np.argsort(fits)[::-1]
        vs_sorted = vs[order]
        zs_sorted = zs[order]
        fits_sorted = fits[order]
        # weights (truncation selection, top half)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        cm = 1.0
        # recombine
        yw = (zs_sorted[:mu] * weights[:, None]).sum(axis=0)
        self.mean = self.mean + cm * self.sigma * yw * sqrtC
        # update evolution paths (simplified, diagonal version)
        # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrtC * yw
        # mueff = 1 / sum(weights^2)
        mueff = 1.0 / np.sum(weights**2)
        ps_norm = yw / sqrtC  # since C is diagonal, invsqrtC = 1/sqrtC
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * mueff) * ps_norm
        hsig = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2.0 / (N + 1)) * self.chiN)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * mueff) * yw
        # update C (diagonal)
        # C = (1-c1-cmu)*C + c1*(pc*pc + (1-hsig)*cc*(2-cc)*C) + cmu * sum(wi * yi*yi)
        yw_mat = zs_sorted[:mu] * sqrtC  # actual y vectors
        rank_mu = (yw_mat * weights[:, None]).T @ yw_mat  # weighted sum of outer products, diagonal
        # actually we just want diagonal of that, which is element-wise:
        rank_mu_diag = (yw_mat * weights[:, None]).sum(axis=0) * sqrtC  # wrong, redo
        # diagonal of weighted sum of outer products = sum(w_i * y_i^2)
        rank_mu_diag = (yw_mat**2 * weights[:, None]).sum(axis=0)
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu_diag
        # update sigma
        self.sigma = self.sigma * np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * (self.cs / self.damps))
        self.sigma = float(np.clip(self.sigma, 1e-8, 1e3))
        # track best
        best_idx = int(order[0])
        if fits[best_idx] > self.best_fitness:
            self.best_fitness = float(fits[best_idx])
            self.best_genome = vector_to_genome(self.template, vs[best_idx], self.gene_keys)
        best = float(fits_sorted[0])
        mean = float(np.mean(fits))
        self.gen += 1
        self.history.append((self.gen - 1, best, mean))
        return best, mean

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            best, mean = self.step()
            if verbose:
                print(f"CMA gen {self.gen-1:>3} | best {best:7.2f} | mean {mean:7.2f} | sigma {self.sigma:.3f}")
