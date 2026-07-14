"""Vanilla NEAT (Stanley & Miikkulainen 2002) implementation.

This is the baseline NEAT algorithm we'll compare against. It's a faithful
implementation of the core ideas:

  - Complexification: start from a minimal topology (perceptron-like) and
    add nodes/edges over generations via structural mutations.
  - Historical markings: every structural mutation gets a global innovation
    number so that genomes can be aligned linearly for crossover.
  - Speciation: protect innovation by clustering genomes by genetic distance
    and only competing within a species.
  - Explicit fitness sharing: each species' offspring budget is proportional
    to its mean adjusted fitness.

Mutation operators:
  - add_node: split an existing edge in two, disable the old edge, insert a
    new hidden node with bias 0, set weights so the new subnetwork initially
    computes the same function as the old edge.
  - add_edge: connect two previously-unconnected nodes.
  - mutate_weights: perturb each enabled gene's weight by Gaussian noise
    (with small probability of full replacement).
  - mutate_bias: perturb each node's bias.
  - toggle: enable/disable a random gene.

Crossover: align two parents by innovation number. For matching genes,
randomly pick one parent's gene (with the fitter parent preferred for
weights). For disjoint/excess genes, inherit from the fitter parent
(unless equal fitness, then inherit all from both).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

from .genome import Genome, Node, Gene, InnovationCounter, INPUT, HIDDEN, OUTPUT
from .evaluation import evaluate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NEATConfig:
    # population
    pop_size: int = 100

    # initial topology
    n_inputs: int = 4
    n_outputs: int = 2
    output_activation: str = "tanh"

    # mutation rates
    p_add_node: float = 0.03
    p_add_edge: float = 0.05
    p_mutate_weight: float = 0.8
    p_replace_weight: float = 0.1
    p_mutate_bias: float = 0.5
    p_toggle: float = 0.01
    weight_perturb_std: float = 0.5
    weight_replace_scale: float = 1.0
    bias_perturb_std: float = 0.5

    # speciation
    c1: float = 1.0  # excess gene coefficient
    c2: float = 1.0  # disjoint gene coefficient
    c3: float = 0.4  # weight difference coefficient
    compatibility_threshold: float = 3.0
    species_elitism: int = 1  # keep top-K of each species unchanged

    # selection / crossover
    survival_threshold: float = 0.2  # fraction of each species allowed to reproduce
    interspecies_mate: float = 0.001

    # evaluation
    n_episodes: int = 3
    max_steps: int = 500
    action_mode: str = "argmax"

    # misc
    seed: int = 0


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------


def make_initial_genome(cfg: NEATConfig, innov: InnovationCounter) -> Genome:
    """Minimal initial topology: inputs -> outputs (fully connected).

    Input and output node IDs are shared across all initial genomes (they are
    created once on the first call and cached on the InnovationCounter), so
    that crossover between two initial genomes aligns them perfectly. Only the
    connection weights differ.
    """
    g = Genome()
    # Use cached input/output node ids so all initial genomes align.
    if not hasattr(innov, "_init_inputs"):
        innov._init_inputs = [innov.new_node_id() for _ in range(cfg.n_inputs)]
        innov._init_outputs = [innov.new_node_id() for _ in range(cfg.n_outputs)]
    input_ids = innov._init_inputs
    output_ids = innov._init_outputs
    for nid in input_ids:
        g.add_node(Node(nid, INPUT, bias=0.0, activation="identity"))
    for nid in output_ids:
        g.add_node(Node(nid, OUTPUT, bias=0.0, activation=cfg.output_activation))
    for in_id in input_ids:
        for out_id in output_ids:
            inv = innov.get_edge_innov(in_id, out_id)
            w = float(np.random.uniform(-1.0, 1.0))
            g.add_gene(Gene(inv, in_id, out_id, w, enabled=True))
    return g


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


def mutate_add_node(g: Genome, innov: InnovationCounter) -> None:
    """Split a random enabled edge."""
    enabled = [ge for ge in g.genes.values() if ge.enabled]
    if not enabled:
        return
    ge = enabled[np.random.randint(len(enabled))]
    ge.enabled = False
    # new hidden node
    new_id = innov.new_node_id()
    g.add_node(Node(new_id, HIDDEN, bias=0.0, activation="tanh"))
    # edge in -> new with weight 1
    inv1 = innov.get_edge_innov(ge.in_node, new_id)
    g.add_gene(Gene(inv1, ge.in_node, new_id, 1.0, enabled=True))
    # edge new -> out with weight = old weight
    inv2 = innov.get_edge_innov(new_id, ge.out_node)
    g.add_gene(Gene(inv2, new_id, ge.out_node, ge.weight, enabled=True))


def mutate_add_edge(g: Genome, innov: InnovationCounter) -> None:
    """Add a random edge between two previously-unconnected nodes.

    Only allow edges that respect feed-forward order (no cycles). We do this
    by maintaining a topological order: an edge (i -> j) is allowed only if
    topo(i) < topo(j).
    """
    nodes = list(g.nodes.values())
    if len(nodes) < 2:
        return
    # compute a topo order: {node_id: level}
    order = _topo_order(g)
    n = len(nodes)
    # build set of existing edges
    existing = {(ge.in_node, ge.out_node) for ge in g.genes.values()}
    # try a few random pairs
    for _ in range(20):
        i_idx = np.random.randint(n)
        j_idx = np.random.randint(n)
        if i_idx == j_idx:
            continue
        in_id = nodes[i_idx].node_id
        out_id = nodes[j_idx].node_id
        if order[in_id] >= order[out_id]:
            continue
        if (in_id, out_id) in existing:
            continue
        # forbid connecting output -> anything (outputs are sinks)
        if nodes[i_idx].node_type == OUTPUT:
            continue
        # forbid connecting anything -> input (inputs are sources)
        if nodes[j_idx].node_type == INPUT:
            continue
        inv = innov.get_edge_innov(in_id, out_id)
        w = float(np.random.uniform(-1.0, 1.0))
        g.add_gene(Gene(inv, in_id, out_id, w, enabled=True))
        return


def _topo_order(g: Genome) -> Dict[int, int]:
    """Return {node_id: topo_level} for a Genome (feed-forward assumption)."""
    # Kahn
    in_degree: Dict[int, int] = {nid: 0 for nid in g.nodes}
    fwd: Dict[int, List[int]] = {nid: [] for nid in g.nodes}
    for ge in g.genes.values():
        if not ge.enabled:
            continue
        if ge.in_node not in g.nodes or ge.out_node not in g.nodes:
            continue
        fwd[ge.in_node].append(ge.out_node)
        in_degree[ge.out_node] += 1
    level: Dict[int, int] = {}
    queue = [nid for nid, d in in_degree.items() if d == 0]
    for nid in queue:
        level[nid] = 0
    while queue:
        v = queue.pop(0)
        for w in fwd[v]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                level[w] = level[v] + 1
                queue.append(w)
    # any remaining (cycles) get level = max+1
    if len(level) != len(g.nodes):
        mx = max(level.values()) + 1 if level else 0
        for nid in g.nodes:
            if nid not in level:
                level[nid] = mx
    return level


def mutate_weights(g: Genome, cfg: NEATConfig) -> None:
    for ge in g.genes.values():
        if not ge.enabled:
            continue
        if np.random.random() < cfg.p_mutate_weight:
            if np.random.random() < cfg.p_replace_weight:
                ge.weight = float(np.random.uniform(-1.0, 1.0) * cfg.weight_replace_scale)
            else:
                ge.weight += float(np.random.randn() * cfg.weight_perturb_std)


def mutate_bias(g: Genome, cfg: NEATConfig) -> None:
    for n in g.nodes.values():
        if n.node_type == INPUT:
            continue
        if np.random.random() < cfg.p_mutate_bias:
            n.bias += float(np.random.randn() * cfg.bias_perturb_std)


def mutate_toggle(g: Genome) -> None:
    if not g.genes:
        return
    ge = list(g.genes.values())[np.random.randint(len(g.genes))]
    ge.enabled = not ge.enabled


def mutate(g: Genome, cfg: NEATConfig, innov: InnovationCounter) -> None:
    if np.random.random() < cfg.p_add_node:
        mutate_add_node(g, innov)
    if np.random.random() < cfg.p_add_edge:
        mutate_add_edge(g, innov)
    mutate_weights(g, cfg)
    mutate_bias(g, cfg)
    if np.random.random() < cfg.p_toggle:
        mutate_toggle(g)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(parent_a: Genome, fitness_a: float, parent_b: Genome, fitness_b: float) -> Genome:
    """Crossover two parents. parent_a is the fitter parent by convention."""
    if fitness_a < fitness_b:
        parent_a, parent_b = parent_b, parent_a
        fitness_a, fitness_b = fitness_b, fitness_a
    child = Genome()
    # inherit nodes: union, taking from fitter parent on conflict (no real conflict
    # because node ids are global, but take the fitter parent's bias)
    for nid, n in parent_a.nodes.items():
        child.add_node(n.copy())
    for nid, n in parent_b.nodes.items():
        if nid not in child.nodes:
            child.add_node(n.copy())
    # inherit genes: matching -> pick from fitter (with prob 0.5 take other's),
    # disjoint/excess -> take from fitter parent (or both if equal fitness)
    equal = abs(fitness_a - fitness_b) < 1e-9
    innovs_a = set(parent_a.genes.keys())
    innovs_b = set(parent_b.genes.keys())
    matching = innovs_a & innovs_b
    only_a = innovs_a - innovs_b
    only_b = innovs_b - innovs_a
    for inv in matching:
        ga = parent_a.genes[inv]
        gb = parent_b.genes[inv]
        src = ga if np.random.random() < 0.5 else gb
        child.add_gene(Gene(inv, src.in_node, src.out_node, src.weight, src.enabled))
    for inv in only_a:
        ge = parent_a.genes[inv]
        child.add_gene(ge.copy())
    if equal:
        for inv in only_b:
            ge = parent_b.genes[inv]
            child.add_gene(ge.copy())
    return child


# ---------------------------------------------------------------------------
# Speciation
# ---------------------------------------------------------------------------


def compatibility(g1: Genome, g2: Genome, cfg: NEATConfig) -> float:
    innovs1 = set(g1.genes.keys())
    innovs2 = set(g2.genes.keys())
    if not innovs1 and not innovs2:
        return 0.0
    n1 = len(innovs1)
    n2 = len(innovs2)
    n_larger = max(n1, n2)
    n_larger = max(n_larger, 1)
    matching = innovs1 & innovs2
    n_excess = len(innovs1 ^ innovs2)
    # disjoint vs excess: split by max innovation
    if innovs1 and innovs2:
        max1 = max(innovs1)
        max2 = max(innovs2)
        max_min = min(max1, max2)
        n_ex = sum(1 for i in (innovs1 | innovs2) if i > max_min)
        n_dis = n_excess - n_ex
    else:
        n_ex = n_excess
        n_dis = 0
    # weight diff of matching
    if matching:
        wdiff = np.mean(
            [abs(g1.genes[i].weight - g2.genes[i].weight) for i in matching]
        )
    else:
        wdiff = 0.0
    return cfg.c1 * n_ex / n_larger + cfg.c2 * n_dis / n_larger + cfg.c3 * wdiff


def speciate(
    population: List[Genome],
    species_repr: List[Genome],
    cfg: NEATConfig,
) -> List[int]:
    """Assign each genome to a species. Returns species_index per genome."""
    assignments = []
    for g in population:
        assigned = -1
        for si, rep in enumerate(species_repr):
            if compatibility(g, rep, cfg) < cfg.compatibility_threshold:
                assigned = si
                break
        if assigned < 0:
            # new species
            species_repr.append(g)
            assigned = len(species_repr) - 1
        assignments.append(assigned)
    return assignments


# ---------------------------------------------------------------------------
# Main NEAT driver
# ---------------------------------------------------------------------------


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    n_species: int
    best_genome_size: int  # number of enabled genes
    best_n_hidden: int


class NEAT:
    """Vanilla NEAT driver."""

    def __init__(self, cfg: NEATConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        self.rng = np.random.RandomState(cfg.seed)
        self.innov = InnovationCounter()
        # initialize population
        self.population: List[Genome] = [
            make_initial_genome(cfg, self.innov) for _ in range(cfg.pop_size)
        ]
        self.fitness: List[float] = [0.0] * cfg.pop_size
        self.species_repr: List[Genome] = []
        self.species_assignments: List[int] = []
        self.history: List[GenerationStats] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -1e9

    def _evaluate_all(self) -> None:
        for i, g in enumerate(self.population):
            mean_r, std_r, min_r, max_r = evaluate(
                g,
                self.env_name,
                n_episodes=self.cfg.n_episodes,
                max_steps=self.cfg.max_steps,
                seed=self.cfg.seed,
                action_mode=self.cfg.action_mode,
            )
            self.fitness[i] = mean_r

    def _speciate(self) -> None:
        # use the first genome of each species from last gen as representative
        # (simplified: we just re-cluster against previous representatives)
        self.species_assignments = speciate(self.population, self.species_repr, self.cfg)
        # update representatives to a random member of each species
        new_repr: List[Genome] = []
        for si in range(max(self.species_assignments) + 1 if self.species_assignments else 0):
            members = [
                self.population[i]
                for i, a in enumerate(self.species_assignments)
                if a == si
            ]
            if members:
                new_repr.append(members[np.random.randint(len(members))])
        self.species_repr = new_repr

    def _reproduce(self) -> None:
        cfg = self.cfg
        # group by species
        species: Dict[int, List[int]] = {}
        for i, s in enumerate(self.species_assignments):
            species.setdefault(s, []).append(i)
        # adjusted fitness = fitness / species_size
        adjusted = np.array(self.fitness, dtype=np.float64)
        for s, members in species.items():
            for i in members:
                adjusted[i] = self.fitness[i] / len(members)
        # species mean adjusted fitness
        species_mean = {s: float(np.mean([adjusted[i] for i in m])) for s, m in species.items()}
        total = max(sum(species_mean.values()), 1e-9)
        # allocate offspring (Brent's formula)
        new_pop: List[Genome] = []
        for s, members in species.items():
            # sort members by fitness desc
            members_sorted = sorted(members, key=lambda i: self.fitness[i], reverse=True)
            # elitism: keep best K unchanged
            for k in range(min(cfg.species_elitism, len(members_sorted))):
                new_pop.append(self.population[members_sorted[k]].copy())
            n_offspring = int(round(species_mean[s] / total * cfg.pop_size))
            n_offspring = max(0, n_offspring - min(cfg.species_elitism, len(members_sorted)))
            # survivors
            n_survivors = max(1, int(len(members_sorted) * cfg.survival_threshold))
            survivors = members_sorted[:n_survivors]
            for _ in range(n_offspring):
                # pick two parents from survivors
                if len(survivors) == 1 or np.random.random() < cfg.interspecies_mate:
                    # interspecies mating (rare)
                    pi_a = survivors[np.random.randint(len(survivors))]
                    all_idx = list(range(cfg.pop_size))
                    pi_b = all_idx[np.random.randint(len(all_idx))]
                else:
                    pi_a = survivors[np.random.randint(len(survivors))]
                    pi_b = survivors[np.random.randint(len(survivors))]
                child = crossover(
                    self.population[pi_a],
                    self.fitness[pi_a],
                    self.population[pi_b],
                    self.fitness[pi_b],
                )
                mutate(child, cfg, self.innov)
                new_pop.append(child)
        # pad/trim to pop_size
        while len(new_pop) < cfg.pop_size:
            new_pop.append(make_initial_genome(cfg, self.innov))
        self.population = new_pop[: cfg.pop_size]
        self.fitness = [0.0] * cfg.pop_size

    def step(self) -> GenerationStats:
        self._evaluate_all()
        # track best
        best_idx = int(np.argmax(self.fitness))
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_genome = self.population[best_idx].copy()
        # compute stats BEFORE reproduction (which resets fitness)
        stat = GenerationStats(
            generation=len(self.history),
            best_fitness=float(np.max(self.fitness)),
            mean_fitness=float(np.mean(self.fitness)),
            std_fitness=float(np.std(self.fitness)),
            n_species=0,  # filled in after speciation
            best_genome_size=self.population[best_idx].num_enabled_genes(),
            best_n_hidden=self.population[best_idx].num_hidden(),
        )
        self._speciate()
        stat.n_species = len(self.species_repr)
        self._reproduce()
        self.history.append(stat)
        return stat

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            stat = self.step()
            if verbose:
                print(
                    f"gen {stat.generation:>3} | best {stat.best_fitness:7.2f} | "
                    f"mean {stat.mean_fitness:7.2f} | species {stat.n_species:>2} | "
                    f"size {stat.best_genome_size:>3} | hidden {stat.best_n_hidden}"
                )
