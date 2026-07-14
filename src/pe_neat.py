"""PE-NEAT: Plasticity-Encoded Topogenesis NEAT.

A third fundamental approach, developed after the Round 50 rethink.

Core principle:
    **The network is not a fixed function — it is a dynamic system that
     learns during evaluation.**

Each edge encodes:
  1. An initial weight
  2. A local plasticity rule (Hebbian, anti-Hebbian, Oja, modulated-Hebbian)
  3. A plasticity rate (η)

During evaluation, weights adapt at each timestep according to the local
rule. Evolution shapes the rules + topology, not the final weights. This
merges evolution and lifetime learning in a single, elegant framework.

Why this is fundamentally different from GDT-NEAT and PT-NEAT:
  - GDT-NEAT/PT-NEAT: evolution shapes WEIGHTS (static during evaluation)
  - PE-NEAT: evolution shapes LEARNING RULES (dynamic during evaluation)
  - The network adapts to each episode individually
  - Biologically inspired (synaptic plasticity)
  - The "intelligence" emerges from the interaction of evolved rules and
    lifetime learning

Plasticity rules:
  0. Fixed:       Δw = 0  (no plasticity, like standard NEAT)
  1. Hebbian:     Δw = η * pre * post  (correlate co-activation)
  2. Anti-Hebb:   Δw = -η * pre * post  (decorrelate)
  3. Oja:         Δw = η * pre * (post - w * pre)  (normalized Hebbian)
  4. Modulated:   Δw = η * reward * pre * post  (reward-modulated Hebbian)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PEConfig:
    pop_size: int = 30
    n_inputs: int = 4
    n_outputs: int = 2
    n_episodes: int = 1
    max_steps: int = 500
    seed: int = 0

    # plasticity
    plasticity_rate: float = 0.01  # default η for new edges
    p_mutate_plasticity_type: float = 0.1  # probability of changing plasticity type
    p_mutate_plasticity_rate: float = 0.3  # probability of perturbing η

    # topology
    p_add_edge: float = 0.1
    p_add_node: float = 0.05
    p_toggle: float = 0.01
    p_mutate_weight: float = 0.8
    weight_perturb_std: float = 0.3

    # speciation / diversity
    compatibility_threshold: float = 3.0
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_weight: float = 0.4

    # selection
    survival_threshold: float = 0.3
    species_elitism: int = 1

    # misc
    init_weight_scale: float = 0.5
    output_activation: str = "tanh"
    hidden_activation: str = "tanh"


# ---------------------------------------------------------------------------
# Plasticity rules
# ---------------------------------------------------------------------------

PLASTICITY_TYPES = ["fixed", "hebbian", "anti_hebbian", "oja", "modulated"]


def apply_plasticity(rule: str, w: float, pre: float, post: float, eta: float,
                     reward: float = 0.0) -> float:
    """Apply one step of the plasticity rule. Returns new weight."""
    if rule == "fixed":
        return w
    elif rule == "hebbian":
        return w + eta * pre * post
    elif rule == "anti_hebbian":
        return w - eta * pre * post
    elif rule == "oja":
        return w + eta * pre * (post - w * pre)
    elif rule == "modulated":
        return w + eta * reward * pre * post
    return w


# ---------------------------------------------------------------------------
# Genome with plasticity
# ---------------------------------------------------------------------------


@dataclass
class PEEdge:
    in_node: int
    out_node: int
    weight: float
    plasticity_type: str = "fixed"
    plasticity_rate: float = 0.01
    enabled: bool = True


@dataclass
class PENode:
    node_id: int
    node_type: str  # "input" | "hidden" | "output"
    bias: float = 0.0
    activation: str = "tanh"


class PEGenome:
    """Genome with plasticity-encoded edges."""

    def __init__(self):
        self.nodes: Dict[int, PENode] = {}
        self.edges: List[PEEdge] = []
        self._next_node_id = 0
        self.input_ids: List[int] = []
        self.output_ids: List[int] = []

    def copy(self) -> "PEGenome":
        g = PEGenome()
        g._next_node_id = self._next_node_id
        g.input_ids = list(self.input_ids)
        g.output_ids = list(self.output_ids)
        for nid, n in self.nodes.items():
            g.nodes[nid] = PENode(n.node_id, n.node_type, n.bias, n.activation)
        g.edges = [PEEdge(e.in_node, e.out_node, e.weight, e.plasticity_type,
                          e.plasticity_rate, e.enabled) for e in self.edges]
        return g

    def add_input(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        self.nodes[nid] = PENode(nid, "input", 0.0, "identity")
        self.input_ids.append(nid)
        return nid

    def add_output(self, activation: str = "tanh") -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        self.nodes[nid] = PENode(nid, "output", 0.0, activation)
        self.output_ids.append(nid)
        return nid

    def add_hidden(self, activation: str = "tanh") -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        self.nodes[nid] = PENode(nid, "hidden", 0.0, activation)
        return nid

    def add_edge(self, in_node: int, out_node: int, weight: float = 0.0,
                 plasticity_type: str = "fixed", plasticity_rate: float = 0.01) -> None:
        for e in self.edges:
            if e.in_node == in_node and e.out_node == out_node:
                return
        self.edges.append(PEEdge(in_node, out_node, weight, plasticity_type, plasticity_rate))

    def num_enabled_edges(self) -> int:
        return sum(1 for e in self.edges if e.enabled)

    def num_hidden(self) -> int:
        return sum(1 for n in self.nodes.values() if n.node_type == "hidden")


# ---------------------------------------------------------------------------
# Plastic network (dynamic during evaluation)
# ---------------------------------------------------------------------------


class PlasticNetwork:
    """A network that adapts its weights during evaluation via plasticity rules."""

    def __init__(self, genome: PEGenome):
        self.genome = genome
        self._build()

    def _build(self) -> None:
        nodes = list(self.genome.nodes.values())
        self._id2idx = {n.node_id: i for i, n in enumerate(nodes)}
        self._n_nodes = len(nodes)
        self._types = [n.node_type for n in nodes]
        self._biases = np.array([n.bias for n in nodes], dtype=np.float64)
        self._acts = [n.activation for n in nodes]
        self._input_ids = list(self.genome.input_ids)
        self._output_ids = list(self.genome.output_ids)
        # current weights (will be modified during evaluation)
        self._weights: Dict[Tuple[int, int], float] = {}
        self._plasticity: Dict[Tuple[int, int], Tuple[str, float]] = {}
        for e in self.genome.edges:
            if e.enabled:
                self._weights[(e.in_node, e.out_node)] = e.weight
                self._plasticity[(e.in_node, e.out_node)] = (e.plasticity_type, e.plasticity_rate)
        # adjacency: out_node -> list of (in_node, key)
        self._in_edges: Dict[int, List[Tuple[int, Tuple[int, int]]]] = defaultdict(list)
        for key in self._weights:
            self._in_edges[key[1]].append((key[0], key))
        # topological order
        self._topo_order = self._topo_sort()

    def _topo_sort(self) -> List[int]:
        in_degree: Dict[int, int] = {nid: 0 for nid in self.genome.nodes}
        fwd: Dict[int, List[int]] = {nid: [] for nid in self.genome.nodes}
        for key in self._weights:
            fwd[key[0]].append(key[1])
            in_degree[key[1]] += 1
        queue = [nid for nid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in fwd[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        if len(order) != len(self.genome.nodes):
            order = list(self.genome.nodes.keys())
        return order

    def reset_weights(self) -> None:
        """Reset weights to initial values (call at start of each episode)."""
        for e in self.genome.edges:
            if e.enabled:
                self._weights[(e.in_node, e.out_node)] = e.weight

    def forward(self, obs: np.ndarray, reward: float = 0.0) -> np.ndarray:
        """Forward pass. Also applies plasticity updates.

        The plasticity update uses the PRE and POST activations from THIS
        step. This means the weight update happens AFTER the forward pass
        but uses the current step's activations.
        """
        a = np.zeros(self._n_nodes, dtype=np.float64)
        for i, nid in enumerate(self._input_ids):
            a[self._id2idx[nid]] = float(obs[i])
        # forward pass (compute activations)
        for nid in self._topo_order:
            if self.genome.nodes[nid].node_type == "input":
                continue
            idx = self._id2idx[nid]
            s = self._biases[idx]
            for in_id, key in self._in_edges.get(nid, []):
                s += a[self._id2idx[in_id]] * self._weights[key]
            a[idx] = _activate(s, self._acts[idx])
        # apply plasticity updates
        for key, (rule, eta) in self._plasticity.items():
            if rule == "fixed":
                continue
            in_idx = self._id2idx[key[0]]
            out_idx = self._id2idx[key[1]]
            pre = a[in_idx]
            post = a[out_idx]
            w = self._weights[key]
            self._weights[key] = apply_plasticity(rule, w, pre, post, eta, reward)
        # return output
        return np.array([a[self._id2idx[nid]] for nid in self._output_ids], dtype=np.float64)

    def action(self, obs: np.ndarray, stochastic: bool = True, reward: float = 0.0) -> int:
        logits = self.forward(obs, reward)
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        if stochastic:
            return int(np.random.choice(len(probs), p=probs))
        return int(np.argmax(probs))


def _activate(x: float, kind: str) -> float:
    if kind == "tanh":
        return float(np.tanh(x))
    if kind == "relu":
        return float(x) if x > 0.0 else 0.0
    if kind == "sigmoid":
        return float(1.0 / (1.0 + np.exp(-x)))
    if kind == "identity":
        return float(x)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class PEIndividual:
    genome: PEGenome
    fitness: float = -1e9
    behavior: Optional[np.ndarray] = None
    age: int = 0


# ---------------------------------------------------------------------------
# PE-NEAT main driver
# ---------------------------------------------------------------------------


class PENEAT:
    """Plasticity-Encoded Topogenesis NEAT.

    Core principle: the network is a dynamic system that learns during
    evaluation. Evolution shapes the learning rules + topology, not the
    final weights.
    """

    def __init__(self, cfg: PEConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        # initial population: minimal topology with random plasticity
        self.population: List[PEIndividual] = []
        for _ in range(cfg.pop_size):
            g = self._make_initial_genome()
            self.population.append(PEIndividual(genome=g))
        self.history: List[Dict[str, float]] = []
        self.best_fitness: float = -1e9
        self.best_genome: Optional[PEGenome] = None
        self._env = gym.make(env_name)
        self._env.reset(seed=cfg.seed)

    def _make_initial_genome(self) -> PEGenome:
        g = PEGenome()
        for _ in range(self.cfg.n_inputs):
            g.add_input()
        for _ in range(self.cfg.n_outputs):
            g.add_output(self.cfg.output_activation)
        # fully connect inputs -> outputs with random weights and plasticity
        for i in g.input_ids:
            for j in g.output_ids:
                w = float(np.random.uniform(-self.cfg.init_weight_scale, self.cfg.init_weight_scale))
                # random plasticity type (weighted toward "fixed" for initial population)
                ptype = np.random.choice(PLASTICITY_TYPES, p=[0.5, 0.15, 0.1, 0.15, 0.1])
                prate = float(np.random.uniform(0.001, 0.05))
                g.add_edge(i, j, w, ptype, prate)
        return g

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _rollout(self, genome: PEGenome, env, max_steps: int, stochastic: bool = True) -> Tuple[float, np.ndarray]:
        """Rollout with plastic weight updates."""
        net = PlasticNetwork(genome)
        net.reset_weights()
        obs, _ = env.reset()
        total = 0.0
        obs_list = []
        act_list = []
        for _ in range(max_steps):
            a = net.action(obs, stochastic=stochastic, reward=0.0)
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            act_list.append(a)
            obs, r, term, trunc, _ = env.step(a)
            # update weights with reward signal (for modulated plasticity)
            net.forward(obs, reward=float(r))  # this applies plasticity with the reward
            total += float(r)
            if term or trunc:
                break
        # behavior descriptor
        if obs_list:
            obs_arr = np.array(obs_list)
            act_arr = np.array(act_list)
            act_hist = np.zeros(self.cfg.n_outputs, dtype=np.float32)
            for a in act_arr:
                act_hist[a] += 1
            act_hist = act_hist / max(1, len(act_arr))
            beh = np.concatenate([
                np.mean(np.abs(obs_arr), axis=0),
                np.std(obs_arr, axis=0),
                act_hist,
                np.array([len(obs_list) / max_steps], dtype=np.float32),
            ])
        else:
            beh = np.zeros(2 * self.cfg.n_inputs + self.cfg.n_outputs + 1, dtype=np.float32)
        return total, beh

    def _evaluate(self, ind: PEIndividual) -> Tuple[float, np.ndarray]:
        env = self._env
        # deterministic eval
        det_rewards = []
        behs = []
        for _ in range(self.cfg.n_episodes):
            total, beh = self._rollout(ind.genome, env, self.cfg.max_steps, stochastic=False)
            det_rewards.append(total)
            behs.append(beh)
        # also do stochastic eval for diversity
        stoch_rewards = []
        for _ in range(self.cfg.n_episodes):
            total, _ = self._rollout(ind.genome, env, self.cfg.max_steps, stochastic=True)
            stoch_rewards.append(total)
        det_mean = float(np.mean(det_rewards))
        stoch_mean = float(np.mean(stoch_rewards))
        fitness = max(det_mean, stoch_mean)
        return fitness, np.mean(behs, axis=0)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, genome: PEGenome) -> None:
        cfg = self.cfg
        # add edge
        if np.random.random() < cfg.p_add_edge:
            self._mutate_add_edge(genome)
        # add node
        if np.random.random() < cfg.p_add_node:
            self._mutate_add_node(genome)
        # toggle
        if np.random.random() < cfg.p_toggle and genome.edges:
            e = genome.edges[np.random.randint(len(genome.edges))]
            e.enabled = not e.enabled
        # mutate weights
        for e in genome.edges:
            if not e.enabled:
                continue
            if np.random.random() < cfg.p_mutate_weight:
                e.weight += float(np.random.randn() * cfg.weight_perturb_std)
            # mutate plasticity type
            if np.random.random() < cfg.p_mutate_plasticity_type:
                e.plasticity_type = np.random.choice(PLASTICITY_TYPES)
            # mutate plasticity rate
            if np.random.random() < cfg.p_mutate_plasticity_rate:
                e.plasticity_rate = max(0.0, e.plasticity_rate + float(np.random.randn() * 0.005))
        # mutate biases
        for n in genome.nodes.values():
            if n.node_type == "input":
                continue
            if np.random.random() < 0.3:
                n.bias += float(np.random.randn() * 0.3)

    def _mutate_add_edge(self, genome: PEGenome) -> None:
        nodes = list(genome.nodes.keys())
        for _ in range(20):
            i = nodes[np.random.randint(len(nodes))]
            j = nodes[np.random.randint(len(nodes))]
            if i == j:
                continue
            if genome.nodes[i].node_type == "output":
                continue
            if genome.nodes[j].node_type == "input":
                continue
            # check not already present
            if any(e.in_node == i and e.out_node == j for e in genome.edges):
                continue
            w = float(np.random.uniform(-cfg_init_scale, cfg_init_scale))
            ptype = np.random.choice(PLASTICITY_TYPES, p=[0.3, 0.2, 0.15, 0.2, 0.15])
            prate = float(np.random.uniform(0.001, 0.05))
            genome.add_edge(i, j, w, ptype, prate)
            return

    def _mutate_add_node(self, genome: PEGenome) -> None:
        enabled = [e for e in genome.edges if e.enabled]
        if not enabled:
            return
        e = enabled[np.random.randint(len(enabled))]
        e.enabled = False
        h = genome.add_hidden(self.cfg.hidden_activation)
        genome.add_edge(e.in_node, h, 1.0, "fixed", 0.0)
        genome.add_edge(h, e.out_node, e.weight, e.plasticity_type, e.plasticity_rate)

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(self, parent_a: PEIndividual, parent_b: PEIndividual) -> PEGenome:
        ga, gb = parent_a.genome, parent_b.genome
        if parent_a.fitness < parent_b.fitness:
            ga, gb = gb, ga
        child = PEGenome()
        child._next_node_id = max(ga._next_node_id, gb._next_node_id)
        child.input_ids = list(ga.input_ids)
        child.output_ids = list(ga.output_ids)
        for nid, n in ga.nodes.items():
            child.nodes[nid] = PENode(n.node_id, n.node_type, n.bias, n.activation)
        for nid, n in gb.nodes.items():
            if nid not in child.nodes:
                child.nodes[nid] = PENode(n.node_id, n.node_type, n.bias, n.activation)
        # align edges by (in, out)
        edges_a = {(e.in_node, e.out_node): e for e in ga.edges}
        edges_b = {(e.in_node, e.out_node): e for e in gb.edges}
        all_keys = set(edges_a.keys()) | set(edges_b.keys())
        for key in all_keys:
            if key in edges_a and key in edges_b:
                src = edges_a[key] if np.random.random() < 0.5 else edges_b[key]
            elif key in edges_a:
                src = edges_a[key]
            else:
                src = edges_b[key]
            child.edges.append(PEEdge(src.in_node, src.out_node, src.weight,
                                       src.plasticity_type, src.plasticity_rate, src.enabled))
        return child

    # ------------------------------------------------------------------
    # Selection / reproduction
    # ------------------------------------------------------------------

    def _select_parent(self) -> int:
        candidates = np.random.choice(len(self.population), size=3, replace=False)
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return int(best)

    def _reproduce(self) -> None:
        cfg = self.cfg
        order = sorted(range(len(self.population)),
                       key=lambda i: self.population[i].fitness, reverse=True)
        n_elite = max(1, cfg.pop_size // 5)
        new_pop: List[PEIndividual] = []
        for i in range(n_elite):
            parent = self.population[order[i]]
            parent.age += 1
            new_pop.append(parent)
        while len(new_pop) < cfg.pop_size:
            pi = self._select_parent()
            parent = self.population[pi]
            child_genome = parent.genome.copy()
            self._mutate(child_genome)
            new_pop.append(PEIndividual(genome=child_genome))
        self.population = new_pop[: cfg.pop_size]

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        # 1. Evaluate
        for ind in self.population:
            f, beh = self._evaluate(ind)
            ind.fitness = f
            ind.behavior = beh
        # 2. Track best
        best_idx = int(np.argmax([ind.fitness for ind in self.population]))
        if self.population[best_idx].fitness > self.best_fitness:
            self.best_fitness = self.population[best_idx].fitness
            self.best_genome = self.population[best_idx].genome.copy()
        # 3. Stats
        fits = np.array([ind.fitness for ind in self.population])
        sizes = np.array([ind.genome.num_enabled_edges() for ind in self.population])
        hids = np.array([ind.genome.num_hidden() for ind in self.population])
        # count plastic edges
        plastic_counts = []
        for ind in self.population:
            n_plastic = sum(1 for e in ind.genome.edges if e.enabled and e.plasticity_type != "fixed")
            plastic_counts.append(n_plastic)
        stat = {
            "gen": len(self.history),
            "best": float(fits.max()),
            "mean": float(fits.mean()),
            "std": float(fits.std()),
            "avg_size": float(sizes.mean()),
            "avg_hidden": float(hids.mean()),
            "avg_plastic": float(np.mean(plastic_counts)),
        }
        self.history.append(stat)
        # 4. Reproduce
        self._reproduce()
        return stat

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            stat = self.step()
            if verbose:
                print(
                    f"PE gen {stat['gen']:>3} | best {stat['best']:7.2f} | "
                    f"mean {stat['mean']:7.2f} | size {stat['avg_size']:5.1f} | "
                    f"plastic {stat['avg_plastic']:4.1f}"
                )


# Module-level constant for use in _mutate_add_edge
cfg_init_scale = 0.5
