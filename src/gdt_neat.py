"""GDT-NEAT: Gradient-Directed Topogenesis NEAT.

A fundamental rewrite of NEAT based on a single principle:

    **Topology grows in the direction of policy gradient.**

Instead of random add-node/add-edge mutations, we compute (via a single
backward pass through a candidate-extended network) the policy-gradient
signal that each non-existent edge WOULD receive if it existed with weight
zero. Edges with the strongest signal are added. Existing weights are
simultaneously refined by the same policy gradient. Behavioral clustering
(state-action statistics) replaces genetic speciation.

This replaces all four NEAT mechanisms with one:
    - random structural mutation  -> gradient-directed topogenesis
    - historical markings          -> fixed (node_i, node_j) coordinates
    - genetic speciation           -> behavioral (phenotype) clustering
    - zero-order weight search     -> first-order (REINFORCE) weight updates

The result is a single, unified algorithm. No hybrid of NEAT + DQN, no
external learner bolted on top — just "evolution by gradient" extended to
the topology itself.

Implementation notes:
- Networks are PyTorch modules for autograd.
- Each individual maintains a current topology (set of alive nodes + edges).
- For topology growth, we extend the network with all *candidate* edges
  (set weight to 0, compute gradient, pick the top-K by abs(grad)).
- Behavioral descriptor: a small statistics vector summarizing the
  state-action distribution over a rollout.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict


# ---------------------------------------------------------------------------
# Network module: dynamic topology with autograd
# ---------------------------------------------------------------------------


class GDTNetwork(nn.Module):
    """A feed-forward network with dynamic topology.

    Internally stores:
      - node_id -> (type, bias_parameter, activation_name)
      - edges as a sparse-like list of (in_id, out_id, weight_parameter)

    Forward pass builds a topological order and computes activations.
    For topology gradient computation, the network is *temporarily* extended
    with candidate edges (weight 0) so that backprop produces a gradient
    signal for each candidate.
    """

    def __init__(self, n_inputs: int, n_outputs: int, output_activation: str = "tanh",
                 hidden_activation: str = "tanh"):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        # Node registry
        self._next_id = 0
        self.node_types: Dict[int, str] = {}  # id -> "input" | "hidden" | "output"
        self.node_act: Dict[int, str] = {}    # id -> activation name
        self.node_bias: Dict[int, nn.Parameter] = {}

        # Edge registry: (in_id, out_id) -> weight Parameter
        self.edges: Dict[Tuple[int, int], nn.Parameter] = {}

        # create input + output nodes
        self.input_ids: List[int] = []
        self.output_ids: List[int] = []
        for _ in range(n_inputs):
            nid = self._new_node("input", "identity")
            self.input_ids.append(nid)
        for _ in range(n_outputs):
            nid = self._new_node("output", output_activation)
            self.output_ids.append(nid)

        # cached topological order (rebuilt on structure change)
        self._topo_order: Optional[List[int]] = None
        self._topo_dirty = True

    def _new_node(self, ntype: str, act: str) -> int:
        nid = self._next_id
        self._next_id += 1
        self.node_types[nid] = ntype
        self.node_act[nid] = act
        # input nodes have bias 0 (not trainable); others have trainable bias
        if ntype == "input":
            self.node_bias[nid] = nn.Parameter(torch.zeros(1), requires_grad=False)
        else:
            self.node_bias[nid] = nn.Parameter(torch.zeros(1))
        self.register_parameter(f"bias_{nid}", self.node_bias[nid])
        return nid

    def add_node(self, ntype: str = "hidden", act: Optional[str] = None) -> int:
        if act is None:
            act = self.hidden_activation if ntype == "hidden" else self.output_activation
        nid = self._new_node(ntype, act)
        self._topo_dirty = True
        return nid

    def add_edge(self, in_id: int, out_id: int, weight: float = 0.0) -> None:
        key = (in_id, out_id)
        if key in self.edges:
            return
        p = nn.Parameter(torch.tensor([float(weight)], dtype=torch.float32))
        self.edges[key] = p
        self.register_parameter(f"edge_{in_id}_{out_id}", p)
        self._topo_dirty = True

    def remove_edge(self, in_id: int, out_id: int) -> None:
        key = (in_id, out_id)
        if key not in self.edges:
            return
        del self.edges[key]
        # cannot easily unregister; we just leave a dangling name (param won't be used)
        self._topo_dirty = True

    def num_edges(self) -> int:
        return len(self.edges)

    def num_hidden(self) -> int:
        return sum(1 for t in self.node_types.values() if t == "hidden")

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _build_topo(self) -> List[int]:
        in_degree: Dict[int, int] = {nid: 0 for nid in self.node_types}
        fwd: Dict[int, List[int]] = {nid: [] for nid in self.node_types}
        for (i, j) in self.edges.keys():
            fwd[i].append(j)
            in_degree[j] += 1
        queue = [nid for nid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in fwd[v]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)
        if len(order) != len(self.node_types):
            # cycle; fall back to declaration order
            order = list(self.node_types.keys())
        return order

    def _ensure_topo(self) -> List[int]:
        if self._topo_order is None or self._topo_dirty:
            self._topo_order = self._build_topo()
            self._topo_dirty = False
        return self._topo_order

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. obs: (batch, n_inputs) or (n_inputs,)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]
        order = self._ensure_topo()
        # activation storage
        a: Dict[int, torch.Tensor] = {}
        for i, nid in enumerate(self.input_ids):
            a[nid] = obs[:, i]
        # group edges by out_node for efficient forward
        in_edges: Dict[int, List[Tuple[int, nn.Parameter]]] = defaultdict(list)
        for (i, j), w in self.edges.items():
            in_edges[j].append((i, w))
        for nid in order:
            if self.node_types[nid] == "input":
                continue
            s = self.node_bias[nid].expand(batch)
            for in_id, w in in_edges.get(nid, []):
                s = s + a[in_id] * w.squeeze()
            a[nid] = _activate(s, self.node_act[nid])
        out = torch.stack([a[nid] for nid in self.output_ids], dim=1)
        return out

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    # ------------------------------------------------------------------
    # Topology gradient computation
    # ------------------------------------------------------------------

    def candidate_edge_gradient(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                                returns_seq: torch.Tensor, baseline: float,
                                candidate_edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Compute policy-gradient signal for each candidate edge.

        For each (i, j) in candidate_edges (assumed NOT in self.edges), we
        temporarily add it with weight 0, then backprop the policy gradient.
        The gradient of expected return w.r.t. that edge's weight is the
        "usefulness" signal.

        Implementation: we add ALL candidates at once (with weight 0), do one
        forward+backward pass, read off each candidate's gradient, then remove
        them all.
        """
        # snapshot edges to restore later
        added_keys: List[Tuple[int, int]] = []
        for key in candidate_edges:
            if key not in self.edges:
                p = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
                self.edges[key] = p
                # NOTE: we don't register with nn.Module to avoid name clashes;
                # we just need autograd to flow through it.
                added_keys.append(key)
        if not added_keys:
            return {}
        # zero all grads
        self.zero_grad(set_to_none=True)
        # forward
        logits = self.action_logits(obs_seq)  # (T, n_outputs)
        log_probs = F.log_softmax(logits, dim=-1)
        # pick the actions actually taken
        picked = log_probs.gather(1, action_seq.unsqueeze(1)).squeeze(1)  # (T,)
        advantage = returns_seq - baseline
        # policy gradient: -mean(picked * advantage)  (negative because we minimize)
        loss = -(picked * advantage).mean()
        loss.backward()
        # read gradients
        grad_signal: Dict[Tuple[int, int], float] = {}
        for key in added_keys:
            p = self.edges[key]
            g = float(p.grad.item()) if p.grad is not None else 0.0
            grad_signal[key] = g
        # remove the candidate edges
        for key in added_keys:
            del self.edges[key]
        self._topo_dirty = True
        return grad_signal


def _activate(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "tanh":
        return torch.tanh(x)
    if kind == "relu":
        return F.relu(x)
    if kind == "sigmoid":
        return torch.sigmoid(x)
    if kind == "identity":
        return x
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    network: GDTNetwork
    fitness: float = -1e9
    behavior: Optional[np.ndarray] = None  # behavioral descriptor
    age: int = 0  # how many generations it has survived


# ---------------------------------------------------------------------------
# GDT-NEAT main driver
# ---------------------------------------------------------------------------


@dataclass
class GDTConfig:
    pop_size: int = 30
    n_inputs: int = 4
    n_outputs: int = 2
    n_episodes: int = 2
    max_steps: int = 500
    seed: int = 0

    # weight update
    lr_weights: float = 1e-2
    lr_bias: float = 1e-2
    pg_steps: int = 3  # policy-gradient steps per individual per generation

    # topology growth
    n_candidate_edges: int = 16  # candidate edges sampled per individual per generation
    edges_to_add: int = 1  # add top-K edges per generation
    p_split_edge: float = 0.2  # probability of splitting an edge to add a hidden node
    prune_threshold: float = 1e-3  # weight magnitude threshold for pruning
    prune_grad_threshold: float = 1e-4  # gradient magnitude threshold for pruning
    prune_patience: int = 3

    # behavioral diversity
    n_behavior_clusters: int = 5
    diversity_penalty: float = 0.0  # subtracted from fitness if in a dense cluster

    # exploration: novelty-based auxiliary reward
    novelty_bonus: float = 0.0  # weight on novelty reward added to env reward
    novelty_k: int = 5  # k-nearest-neighbors for novelty computation
    novelty_archive_size: int = 200  # max size of state-visitation archive

    # misc
    init_weight_scale: float = 0.5
    output_activation: str = "tanh"
    hidden_activation: str = "tanh"


class GDTNEAT:
    """Gradient-Directed Topogenesis NEAT."""

    def __init__(self, cfg: GDTConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        # initial population: minimal topology (inputs -> outputs, fully connected)
        self.population: List[Individual] = []
        for _ in range(cfg.pop_size):
            net = GDTNetwork(cfg.n_inputs, cfg.n_outputs,
                             output_activation=cfg.output_activation,
                             hidden_activation=cfg.hidden_activation)
            # fully connect inputs to outputs with random weights
            for i in net.input_ids:
                for j in net.output_ids:
                    net.add_edge(i, j, float(np.random.uniform(-cfg.init_weight_scale, cfg.init_weight_scale)))
            self.population.append(Individual(network=net))
        self.history: List[Dict[str, float]] = []
        self.best_fitness: float = -1e9
        self.best_network: Optional[GDTNetwork] = None
        self._prune_counter: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(dict)
        # novelty archive: stores state-visitation descriptors (squared obs vectors)
        self._novelty_archive: Optional[np.ndarray] = None  # shape (n, obs_dim)
        self._novelty_archive_n = 0

    # ------------------------------------------------------------------
    # Rollout + behavior collection
    # ------------------------------------------------------------------

    def _rollout(self, net: GDTNetwork, env, max_steps: int, stochastic: bool = True) -> Tuple[
        List[np.ndarray], List[int], List[float], float, np.ndarray
    ]:
        """Collect one rollout. Returns (obs_list, act_list, rew_list, total_reward, behavior_descriptor).

        If novelty_bonus > 0, the reward at each step is augmented with a
        novelty bonus: the distance from the current state to its k-nearest
        neighbors in the novelty archive. This provides a dense learning
        signal even on sparse-reward tasks like MountainCar.

        If stochastic=True, actions are SAMPLED from the softmax policy
        (needed for policy-gradient training to be on-policy). If
        stochastic=False, actions are argmax (used for clean fitness
        evaluation). The novelty bonus is only added to the stochastic
        rollouts (training); the deterministic rollouts use pure env reward
        for fitness evaluation.
        """
        obs, _ = env.reset()
        obs_list, act_list, rew_list = [], [], []
        for _ in range(max_steps):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            with torch.no_grad():
                logits = net.action_logits(obs_t)
                probs = F.softmax(logits, dim=-1).numpy().flatten()
            if stochastic:
                a = int(np.random.choice(len(probs), p=probs))
            else:
                a = int(np.argmax(probs))
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            act_list.append(a)
            obs, r, term, trunc, _ = env.step(a)
            r = float(r)
            # novelty bonus (only for stochastic / training rollouts)
            if stochastic and self.cfg.novelty_bonus > 0:
                nov = self._novelty_of_state(obs)
                r = r + self.cfg.novelty_bonus * nov
            rew_list.append(r)
            if term or trunc:
                break
        total = float(sum(rew_list))
        # update novelty archive with visited states (only for stochastic rollouts)
        if stochastic and self.cfg.novelty_bonus > 0 and obs_list:
            self._update_novelty_archive(obs_list)
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
        return obs_list, act_list, rew_list, total, beh

    # ------------------------------------------------------------------
    # Novelty archive
    # ------------------------------------------------------------------

    def _novelty_of_state(self, obs: np.ndarray) -> float:
        """Novelty of a state = average distance to k nearest neighbors in archive."""
        if self._novelty_archive is None or self._novelty_archive_n == 0:
            return 1.0  # everything is novel when archive is empty
        n = min(self._novelty_archive_n, self._novelty_archive.shape[0])
        archive = self._novelty_archive[:n]
        # squared euclidean distance
        d = np.sqrt(np.sum((archive - obs) ** 2, axis=1))
        k = min(self.cfg.novelty_k, n)
        # average of k smallest
        idx = np.argpartition(d, k - 1)[:k]
        return float(np.mean(d[idx]))

    def _update_novelty_archive(self, obs_list: List[np.ndarray]) -> None:
        """Add observed states to the novelty archive (FIFO)."""
        new_states = np.array(obs_list, dtype=np.float32)
        max_size = self.cfg.novelty_archive_size
        if self._novelty_archive is None:
            self._novelty_archive = new_states[:max_size].copy()
            self._novelty_archive_n = self._novelty_archive.shape[0]
            return
        # concatenate, then trim to most recent max_size
        combined = np.concatenate([self._novelty_archive[:self._novelty_archive_n], new_states], axis=0)
        if combined.shape[0] > max_size:
            combined = combined[-max_size:]
        self._novelty_archive = combined
        self._novelty_archive_n = combined.shape[0]

    def _evaluate(self, ind: Individual) -> Tuple[float, np.ndarray, List]:
        """Evaluate individual. Returns (mean_reward, behavior, [stochastic_rollouts]).

        We do:
          - n_episodes deterministic rollouts (argmax) for fitness
          - n_episodes stochastic rollouts (sampled) for policy gradient
        The behavior descriptor is computed from the deterministic rollouts
        (so it reflects the policy's actual behavior, not exploration noise).
        """
        env = gym.make(self.env_name)
        env.reset(seed=self.cfg.seed)
        try:
            # deterministic eval rollouts
            det_rewards = []
            behs = []
            for _ in range(self.cfg.n_episodes):
                _, _, _, total, beh = self._rollout(ind.network, env, self.cfg.max_steps, stochastic=False)
                det_rewards.append(total)
                behs.append(beh)
            # stochastic training rollouts
            stoch_rollouts = []
            for _ in range(self.cfg.n_episodes):
                obs_list, act_list, rew_list, _, _ = self._rollout(ind.network, env, self.cfg.max_steps, stochastic=True)
                stoch_rollouts.append((obs_list, act_list, rew_list))
        finally:
            env.close()
        return float(np.mean(det_rewards)), np.mean(behs, axis=0), stoch_rollouts

    # ------------------------------------------------------------------
    # Policy gradient step on weights
    # ------------------------------------------------------------------

    def _policy_gradient_step(self, net: GDTNetwork, obs_list: List[np.ndarray],
                              act_list: List[int], rew_list: List[float], lr: float) -> None:
        if len(obs_list) < 2:
            return
        # compute discounted returns
        gamma = 0.99
        T = len(rew_list)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rew_list[t] + gamma * G
            returns[t] = G
        baseline = float(returns.mean())
        obs_t = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_t = torch.from_numpy(np.array(act_list, dtype=np.int64))
        ret_t = torch.from_numpy(returns)
        # zero grads
        net.zero_grad(set_to_none=True)
        logits = net.action_logits(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)
        picked = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
        advantage = ret_t - baseline
        loss = -(picked * advantage).mean()
        loss.backward()
        # manual SGD update
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    # ------------------------------------------------------------------
    # Topology growth
    # ------------------------------------------------------------------

    def _candidate_edges(self, net: GDTNetwork, n: int) -> List[Tuple[int, int]]:
        """Sample n candidate (in, out) pairs that respect feed-forward order and
        are not already present."""
        nodes = list(net.node_types.keys())
        order = net._build_topo()
        # node -> topo level
        level = {nid: i for i, nid in enumerate(order)}
        existing = set(net.edges.keys())
        candidates: List[Tuple[int, int]] = []
        attempts = 0
        while len(candidates) < n and attempts < 10 * n:
            attempts += 1
            i = nodes[np.random.randint(len(nodes))]
            j = nodes[np.random.randint(len(nodes))]
            if i == j:
                continue
            if level.get(i, 0) >= level.get(j, 1):
                continue
            if net.node_types[i] == "output":  # outputs are sinks
                continue
            if net.node_types[j] == "input":   # inputs are sources
                continue
            if (i, j) in existing:
                continue
            existing.add((i, j))  # avoid duplicate candidates
            candidates.append((i, j))
        return candidates

    def _grow_topology(self, ind: Individual, rollouts: List) -> Dict[Tuple[int, int], float]:
        """Add the gradient-most-promising candidate edge(s).

        Reuses the evaluation rollouts to compute the policy gradient (no
        extra rollout needed).

        Returns a dict mapping existing-edge-keys to their gradient magnitudes
        (so the prune step can use the same gradient information).
        """
        net = ind.network
        if net.num_edges() == 0 or not rollouts:
            return {}
        # use the longest rollout (most informative)
        obs_list, act_list, rew_list = max(rollouts, key=lambda r: len(r[0]))
        if len(obs_list) < 2:
            return {}
        # discounted returns
        gamma = 0.99
        T = len(rew_list)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rew_list[t] + gamma * G
            returns[t] = G
        baseline = float(returns.mean())
        obs_t = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_t = torch.from_numpy(np.array(act_list, dtype=np.int64))
        ret_t = torch.from_numpy(returns)
        candidates = self._candidate_edges(net, self.cfg.n_candidate_edges)
        # compute gradient on candidate edges AND on existing edges (one backward pass)
        added_keys: List[Tuple[int, int]] = []
        for key in candidates:
            if key not in net.edges:
                p = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
                net.edges[key] = p
                added_keys.append(key)
        net.zero_grad(set_to_none=True)
        logits = net.action_logits(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)
        picked = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
        advantage = ret_t - baseline
        loss = -(picked * advantage).mean()
        loss.backward()
        # read gradients for ALL edges (existing + candidate)
        all_grads: Dict[Tuple[int, int], float] = {}
        for key, p in net.edges.items():
            if p.grad is not None:
                all_grads[key] = float(p.grad.item())
            else:
                all_grads[key] = 0.0
        # rank candidates by abs(grad)
        ranked_candidates = sorted(
            [(k, all_grads[k]) for k in added_keys],
            key=lambda kv: abs(kv[1]), reverse=True
        )
        # remove all candidate edges
        for key in added_keys:
            del net.edges[key]
        net._topo_dirty = True
        # add the chosen edges with small initial weight in gradient direction
        for (i, j), g in ranked_candidates[: self.cfg.edges_to_add]:
            init_w = float(np.sign(g) * 0.1) if abs(g) > 1e-6 else 0.0
            net.add_edge(i, j, init_w)
        # return existing-edge gradient magnitudes (for pruning)
        existing_grads = {k: abs(v) for k, v in all_grads.items() if k not in added_keys}
        return existing_grads

    def _maybe_split_edge(self, ind: Individual) -> None:
        """With some probability, split the largest-weight edge to add a hidden node."""
        if np.random.random() > self.cfg.p_split_edge:
            return
        net = ind.network
        if not net.edges:
            return
        # pick the edge with largest |weight|
        items = list(net.edges.items())
        (i, j), w = max(items, key=lambda kv: abs(float(kv[1].item())))
        old_w = float(w.item())
        # disable old edge (we keep it but set weight to 0; we can't really delete params)
        with torch.no_grad():
            w.fill_(0.0)
        # add hidden node
        h = net.add_node("hidden")
        # add edges i -> h (weight 1) and h -> j (weight = old_w) so the function is initially preserved
        net.add_edge(i, h, 1.0)
        net.add_edge(h, j, old_w)

    def _prune(self, ind: Individual, ind_id: int, edge_grad_norm: Dict[Tuple[int, int], float]) -> None:
        """Prune edges whose gradient signal has been small for several generations.

        Consistent with the core principle: gradient directs ALL structural
        decisions, including pruning. An edge with a tiny gradient is not
        contributing to the policy's improvement; it should be removed.
        """
        net = ind.network
        to_remove = []
        for key, w in list(net.edges.items()):
            # use gradient magnitude if available; fall back to weight magnitude
            gmag = edge_grad_norm.get(key, 0.0)
            wmag = abs(float(w.item()))
            # an edge is "stagnant" if BOTH its gradient and its weight are small
            stagnant = gmag < self.cfg.prune_grad_threshold and wmag < self.cfg.prune_threshold
            if stagnant:
                self._prune_counter[ind_id][key] = self._prune_counter[ind_id].get(key, 0) + 1
                if self._prune_counter[ind_id][key] >= self.cfg.prune_patience:
                    to_remove.append(key)
            else:
                self._prune_counter[ind_id][key] = 0
        for key in to_remove:
            net.remove_edge(*key)
            self._prune_counter[ind_id].pop(key, None)

    # ------------------------------------------------------------------
    # Behavioral diversity
    # ------------------------------------------------------------------

    def _cluster_behaviors(self) -> np.ndarray:
        """Cluster individuals by behavior; return cluster assignment per individual."""
        behs = np.array([ind.behavior for ind in self.population])
        n_clusters = min(self.cfg.n_behavior_clusters, len(self.population))
        # simple K-means
        rng = np.random.RandomState(self.cfg.seed)
        # init: pick K random distinct points
        idx = rng.choice(len(behs), n_clusters, replace=False)
        centers = behs[idx].copy()
        for _ in range(20):
            # assign
            d = np.linalg.norm(behs[:, None, :] - centers[None, :, :], axis=2)
            assign = d.argmin(axis=1)
            # update
            for k in range(n_clusters):
                members = behs[assign == k]
                if len(members) > 0:
                    centers[k] = members.mean(axis=0)
        # final assignment
        d = np.linalg.norm(behs[:, None, :] - centers[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        return assign

    def _apply_diversity_penalty(self, assign: np.ndarray) -> None:
        """Subtract a penalty from individuals in dense clusters."""
        counts = np.bincount(assign, minlength=assign.max() + 1)
        for i, ind in enumerate(self.population):
            k = assign[i]
            density = counts[k] / len(self.population)
            # penalty proportional to density above uniform
            penalty = max(0.0, density - 1.0 / len(self.population)) * self.cfg.diversity_penalty
            ind.fitness -= penalty

    # ------------------------------------------------------------------
    # Selection / reproduction
    # ------------------------------------------------------------------

    def _select_parents(self) -> List[int]:
        """Tournament selection based on fitness."""
        pop_n = len(self.population)
        # tournament size 3
        candidates = np.random.choice(pop_n, size=3, replace=False)
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return int(best)

    def _reproduce(self) -> None:
        """Produce next generation via selection + structural inheritance."""
        cfg = self.cfg
        # sort by fitness
        order = sorted(range(len(self.population)),
                       key=lambda i: self.population[i].fitness, reverse=True)
        # elitism: keep top 20%
        n_elite = max(1, cfg.pop_size // 5)
        new_pop: List[Individual] = []
        for i in range(n_elite):
            parent = self.population[order[i]]
            # elite: keep as-is (increment age)
            parent.age += 1
            new_pop.append(parent)
        # fill rest with mutated copies of selected parents
        while len(new_pop) < cfg.pop_size:
            pi = self._select_parents()
            parent = self.population[pi]
            # clone parent's network
            child_net = self._clone_network(parent.network)
            # mutate weights slightly (small Gaussian noise on weights)
            with torch.no_grad():
                for p in child_net.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * 0.1)
            new_pop.append(Individual(network=child_net))
        self.population = new_pop[: cfg.pop_size]

    def _clone_network(self, src: GDTNetwork) -> GDTNetwork:
        """Deep clone a network (preserving structure and weights)."""
        dst = GDTNetwork(src.n_inputs, src.n_outputs,
                         output_activation=src.output_activation,
                         hidden_activation=src.hidden_activation)
        # remove default input/output nodes and re-create with src's ids
        # (we cheat: just clear and rebuild)
        dst.node_types.clear()
        dst.node_act.clear()
        dst.node_bias.clear()
        dst.edges.clear()
        dst._next_id = 0
        # add nodes in id order
        max_id = max(src.node_types.keys())
        # ensure _next_id is set high enough
        dst._next_id = max_id + 1
        for nid in sorted(src.node_types.keys()):
            dst.node_types[nid] = src.node_types[nid]
            dst.node_act[nid] = src.node_act[nid]
            bias = float(src.node_bias[nid].item())
            if src.node_types[nid] == "input":
                p = nn.Parameter(torch.zeros(1), requires_grad=False)
            else:
                p = nn.Parameter(torch.tensor([bias], dtype=torch.float32))
            dst.node_bias[nid] = p
        # input/output ids
        dst.input_ids = list(src.input_ids)
        dst.output_ids = list(src.output_ids)
        # add edges
        for (i, j), w in src.edges.items():
            wv = float(w.item())
            dst.add_edge(i, j, wv)
        dst._topo_dirty = True
        return dst

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        cfg = self.cfg
        # 1. Evaluate all individuals (and collect rollout data for reuse)
        all_rollouts: Dict[int, List] = {}
        for i, ind in enumerate(self.population):
            f, beh, rollouts = self._evaluate(ind)
            ind.fitness = f
            ind.behavior = beh
            all_rollouts[i] = rollouts
        # 2. Behavioral diversity penalty
        assign = self._cluster_behaviors()
        self._apply_diversity_penalty(assign)
        # 3. Track best
        best_idx = int(np.argmax([ind.fitness for ind in self.population]))
        if self.population[best_idx].fitness > self.best_fitness:
            self.best_fitness = self.population[best_idx].fitness
            self.best_network = self._clone_network(self.population[best_idx].network)
        # 4. Policy gradient on weights (reuse evaluation rollouts)
        for i, ind in enumerate(self.population):
            rollouts = all_rollouts[i]
            if not rollouts:
                continue
            obs_list, act_list, rew_list = max(rollouts, key=lambda r: len(r[0]))
            for _ in range(cfg.pg_steps):
                self._policy_gradient_step(ind.network, obs_list, act_list, rew_list, cfg.lr_weights)
        # 5. Topology growth (reuse evaluation rollouts; collect gradients for pruning)
        for i, ind in enumerate(self.population):
            edge_grads = self._grow_topology(ind, all_rollouts[i])
            self._maybe_split_edge(ind)
            self._prune(ind, i, edge_grads)
        # 6. Stats
        fits = np.array([ind.fitness for ind in self.population])
        sizes = np.array([ind.network.num_edges() for ind in self.population])
        hids = np.array([ind.network.num_hidden() for ind in self.population])
        stat = {
            "gen": len(self.history),
            "best": float(fits.max()),
            "mean": float(fits.mean()),
            "std": float(fits.std()),
            "avg_size": float(sizes.mean()),
            "avg_hidden": float(hids.mean()),
        }
        self.history.append(stat)
        # 7. Reproduce
        self._reproduce()
        return stat

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            stat = self.step()
            if verbose:
                print(
                    f"GDT gen {stat['gen']:>3} | best {stat['best']:7.2f} | "
                    f"mean {stat['mean']:7.2f} | avg_size {stat['avg_size']:5.1f} | "
                    f"avg_hidden {stat['avg_hidden']:4.1f}"
                )
