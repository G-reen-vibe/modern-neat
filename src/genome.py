"""Core genome and network representation for NEAT-style algorithms.

This module implements the fundamental data structures shared by all
algorithms in this project:

  - Gene:      a single connection with innovation number, weight, enabled flag
  - Node:      a neuron with id, type (input/hidden/output), and bias
  - Genome:    an ordered collection of nodes + connections
  - Network:   a feed-forward executable view of a Genome

The Genome/Network split is intentional: evolution acts on Genomes (which
carry historical markings and structural information), evaluation acts on
Networks (which execute forward passes).

Design notes:
- Innovation numbers are assigned by a single InnovationCounter so that
  identical structural mutations across different individuals in the same
  generation get the same innovation number (the historical marking trick
  from the original NEAT paper).
- Networks are kept strictly feed-forward. The original NEAT allows recurrent
  links but for the control tasks we use (CartPole, MountainCar, Acrobot)
  feed-forward is sufficient and faster.
- Forward passes are vectorized with numpy for speed.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Node / Gene / Genome data structures
# ---------------------------------------------------------------------------

INPUT = "input"
HIDDEN = "hidden"
OUTPUT = "output"


@dataclass
class Node:
    node_id: int
    node_type: str
    bias: float = 0.0
    activation: str = "tanh"  # tanh | relu | sigmoid | identity

    def copy(self) -> "Node":
        return Node(self.node_id, self.node_type, self.bias, self.activation)


@dataclass
class Gene:
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True

    def copy(self) -> "Gene":
        return Gene(self.innovation, self.in_node, self.out_node, self.weight, self.enabled)


@dataclass
class Genome:
    nodes: Dict[int, Node] = field(default_factory=dict)
    genes: Dict[int, Gene] = field(default_factory=dict)  # keyed by innovation

    # --- structural ops ---------------------------------------------------
    def copy(self) -> "Genome":
        g = Genome()
        for nid, n in self.nodes.items():
            g.nodes[nid] = n.copy()
        for inv, ge in self.genes.items():
            g.genes[inv] = ge.copy()
        return g

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def add_gene(self, gene: Gene) -> None:
        self.genes[gene.innovation] = gene

    def num_enabled_genes(self) -> int:
        return sum(1 for g in self.genes.values() if g.enabled)

    def num_hidden(self) -> int:
        return sum(1 for n in self.nodes.values() if n.node_type == HIDDEN)


# ---------------------------------------------------------------------------
# Innovation counter (historical markings)
# ---------------------------------------------------------------------------


class InnovationCounter:
    """Assigns globally-consistent innovation numbers.

    Two structural mutations that produce the same (in_node, out_node) pair
    within the same generation (or even across generations, for simplicity we
    use the simpler global scheme from the original paper) get the same
    innovation number. This is what makes genome alignment and crossover
    trivial.
    """

    def __init__(self, start_node_id: int = 0, start_innov: int = 0):
        self._next_node = start_node_id
        self._next_innov = start_innov
        # cache: (in, out) -> innovation, so we don't keep inventing new
        # innovations for the same pair across the whole run
        self._edge_cache: Dict[Tuple[int, int], int] = {}
        # cache: in_node -> hidden node created by splitting an edge
        self._node_cache: Dict[int, int] = {}

    def new_node_id(self) -> int:
        nid = self._next_node
        self._next_node += 1
        return nid

    def new_innov(self) -> int:
        i = self._next_innov
        self._next_innov += 1
        return i

    def get_edge_innov(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._edge_cache:
            self._edge_cache[key] = self.new_innov()
        return self._edge_cache[key]

    def get_split_node(self, in_node: int) -> Optional[int]:
        """Return cached split-node id for the *input* node of an edge, or None."""
        return self._node_cache.get(in_node)


# ---------------------------------------------------------------------------
# Network: executable view of a Genome
# ---------------------------------------------------------------------------


class Network:
    """Feed-forward network built from a Genome.

    Topological order is computed once at construction time. Forward passes
    are vectorized: node activations are stored in a numpy array indexed by
    node_id (after compacting to a dense index).
    """

    def __init__(self, genome: Genome):
        self.genome = genome
        self._build()

    def _build(self) -> None:
        nodes = list(self.genome.nodes.values())
        # dense index map: node_id -> dense_idx
        self._id2idx: Dict[int, int] = {n.node_id: i for i, n in enumerate(nodes)}
        self._n_nodes = len(nodes)
        self._types = np.array([_type_idx(n.node_type) for n in nodes], dtype=np.int32)
        self._biases = np.array([n.bias for n in nodes], dtype=np.float64)
        self._acts = [n.activation for n in nodes]
        self._input_ids = [n.node_id for n in nodes if n.node_type == INPUT]
        self._output_ids = [n.node_id for n in nodes if n.node_type == OUTPUT]

        # adjacency: out_node -> list of (in_idx, weight)
        # build dense
        in_lists: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(self._n_nodes)}
        for g in self.genome.genes.values():
            if not g.enabled:
                continue
            if g.in_node not in self._id2idx or g.out_node not in self._id2idx:
                continue
            in_idx = self._id2idx[g.in_node]
            out_idx = self._id2idx[g.out_node]
            in_lists[out_idx].append((in_idx, g.weight))

        # convert to flat CSR-like structure
        self._flat_in: List[int] = []
        self._flat_w: List[float] = []
        self._row_ptr: List[int] = [0]
        for i in range(self._n_nodes):
            self._flat_in.extend(in_idx for in_idx, _ in in_lists[i])
            self._flat_w.extend(w for _, w in in_lists[i])
            self._row_ptr.append(len(self._flat_in))
        self._flat_in = np.array(self._flat_in, dtype=np.int32)
        self._flat_w = np.array(self._flat_w, dtype=np.float64)
        self._row_ptr = np.array(self._row_ptr, dtype=np.int32)

        # topological order (Kahn's algorithm). For feed-forward, simply
        # process inputs first, then iterate to fixed point. Since we may
        # have arbitrary hidden orderings, do a real topo sort.
        self._topo_order = self._topo_sort()

    def _topo_sort(self) -> np.ndarray:
        n = self._n_nodes
        in_degree = np.zeros(n, dtype=np.int32)
        # build forward adjacency for topo sort
        fwd: Dict[int, List[int]] = {i: [] for i in range(n)}
        for out_idx in range(n):
            for k in range(self._row_ptr[out_idx], self._row_ptr[out_idx + 1]):
                in_idx = int(self._flat_in[k])
                fwd[in_idx].append(out_idx)
                in_degree[out_idx] += 1
        # Kahn
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in fwd[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        if len(order) != n:
            # cycle (shouldn't happen for feed-forward); fall back to naive order
            order = list(range(n))
        return np.array(order, dtype=np.int32)

    # --- forward pass ------------------------------------------------------
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Run a single forward pass. obs must match len(input_ids)."""
        a = np.zeros(self._n_nodes, dtype=np.float64)
        for i, nid in enumerate(self._input_ids):
            a[self._id2idx[nid]] = float(obs[i])
        # iterate in topological order
        for idx in self._topo_order:
            t = self._types[idx]
            if t == 0:  # input
                continue
            s = self._biases[idx]
            r0 = self._row_ptr[idx]
            r1 = self._row_ptr[idx + 1]
            if r1 > r0:
                ins = self._flat_in[r0:r1]
                ws = self._flat_w[r0:r1]
                s += float(np.dot(a[ins], ws))
            a[idx] = _activate(s, self._acts[idx])
        # return outputs in declared order
        out = np.array([a[self._id2idx[nid]] for nid in self._output_ids], dtype=np.float64)
        return out

    def action(self, obs: np.ndarray, action_mode: str = "argmax") -> int:
        """Return a discrete action for the given observation."""
        out = self.forward(obs)
        if action_mode == "argmax":
            return int(np.argmax(out))
        elif action_mode == "sample":
            # softmax sample
            e = np.exp(out - np.max(out))
            p = e / e.sum()
            return int(np.random.choice(len(p), p=p))
        else:
            raise ValueError(action_mode)

    @property
    def n_inputs(self) -> int:
        return len(self._input_ids)

    @property
    def n_outputs(self) -> int:
        return len(self._output_ids)

    @property
    def n_hidden(self) -> int:
        return self._n_nodes - self.n_inputs - self.n_outputs


def _type_idx(t: str) -> int:
    return {INPUT: 0, HIDDEN: 1, OUTPUT: 2}[t]


def _activate(x: float, kind: str) -> float:
    if kind == "tanh":
        return np.tanh(x)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if kind == "relu":
        return x if x > 0.0 else 0.0
    if kind == "identity":
        return x
    raise ValueError(kind)
