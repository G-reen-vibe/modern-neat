"""PT-NEAT: Predictive Topogenesis NEAT.

A fundamentally new approach to neuroevolution based on a single principle:

    **Topology grows where the network is surprised.**

The network has a dual role:
  1. Policy head: produces action probabilities (for acting)
  2. Predictor head: predicts the next state (world model)

The prediction error (surprise) provides a DENSE learning signal that
doesn't depend on external reward. On sparse-reward tasks like MountainCar,
even when the car never reaches the goal, the network can still learn by
improving its predictions. Curiosity (visiting surprising states) drives
exploration automatically.

Key insight: the same gradient that improves predictions also drives
topology growth. Edges are added where they would most reduce prediction
error. This means:
  - On sparse-reward tasks: topology grows to improve the world model →
    curiosity emerges → exploration happens → reward is eventually found
  - On dense-reward tasks: topology grows for both prediction and policy →
    faster learning

This is a clean break from GDT-NEAT (Rounds 1-24), which relied on policy
gradient and failed on sparse-reward tasks.
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
# Config
# ---------------------------------------------------------------------------


@dataclass
class PTConfig:
    pop_size: int = 20
    n_inputs: int = 4
    n_outputs: int = 2  # number of actions
    n_state_outputs: int = 4  # predict next state (same dim as input)
    n_episodes: int = 1
    max_steps: int = 500
    seed: int = 0

    # learning
    lr_weights: float = 1e-2
    pg_steps: int = 2
    prediction_coef: float = 1.0  # weight on prediction loss
    reward_coef: float = 1.0  # weight on policy gradient (env reward)
    curiosity_coef: float = 0.1  # curiosity bonus added to env reward
    entropy_coef: float = 0.01
    grad_clip: float = 1.0

    # topology growth
    n_candidate_edges: int = 16
    edges_to_add: int = 1
    adaptive_edges: bool = True
    adaptive_edge_threshold: float = 0.01
    p_split_edge: float = 0.15
    prune_threshold: float = 1e-3
    prune_grad_threshold: float = 1e-4
    prune_patience: int = 3

    # diversity
    restart_on_convergence: bool = True
    restart_threshold: float = 0.1
    restart_fraction: float = 0.5

    # misc
    init_weight_scale: float = 0.5
    output_activation: str = "tanh"
    hidden_activation: str = "tanh"


# ---------------------------------------------------------------------------
# Network with dual heads (policy + predictor)
# ---------------------------------------------------------------------------


class PTNetwork(nn.Module):
    """Network with shared hidden layers + two output heads.

    1. Policy head: produces action logits (n_outputs nodes)
    2. Predictor head: produces next-state prediction (n_state_outputs nodes)

    Both heads share the same hidden nodes and edges. This means topology
    growth benefits both policy and prediction simultaneously.
    """

    def __init__(self, n_inputs: int, n_actions: int, n_state: int,
                 output_activation: str = "tanh",
                 hidden_activation: str = "tanh"):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.n_state = n_state
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self._next_id = 0
        self.node_types: Dict[int, str] = {}  # "input" | "hidden" | "policy" | "predictor"
        self.node_act: Dict[int, str] = {}
        self.node_bias: Dict[int, nn.Parameter] = {}
        self.edges: Dict[Tuple[int, int], nn.Parameter] = {}

        # create input nodes
        self.input_ids: List[int] = []
        for _ in range(n_inputs):
            nid = self._new_node("input", "identity")
            self.input_ids.append(nid)
        # create policy output nodes
        self.policy_ids: List[int] = []
        for _ in range(n_actions):
            nid = self._new_node("policy", output_activation)
            self.policy_ids.append(nid)
        # create predictor output nodes
        self.predictor_ids: List[int] = []
        for _ in range(n_state):
            nid = self._new_node("predictor", "identity")  # linear prediction
            self.predictor_ids.append(nid)

        self._topo_order: Optional[List[int]] = None
        self._topo_dirty = True
        # numpy cache
        self._np_cache_dirty = True

    def _new_node(self, ntype: str, act: str) -> int:
        nid = self._next_id
        self._next_id += 1
        self.node_types[nid] = ntype
        self.node_act[nid] = act
        if ntype == "input":
            self.node_bias[nid] = nn.Parameter(torch.zeros(1), requires_grad=False)
        else:
            self.node_bias[nid] = nn.Parameter(torch.zeros(1))
        self.register_parameter(f"bias_{nid}", self.node_bias[nid])
        return nid

    def add_node(self, ntype: str = "hidden", act: Optional[str] = None) -> int:
        if act is None:
            act = self.hidden_activation
        nid = self._new_node(ntype, act)
        self._topo_dirty = True
        self._np_cache_dirty = True
        return nid

    def add_edge(self, in_id: int, out_id: int, weight: float = 0.0) -> None:
        key = (in_id, out_id)
        if key in self.edges:
            return
        p = nn.Parameter(torch.tensor([float(weight)], dtype=torch.float32))
        self.edges[key] = p
        self.register_parameter(f"edge_{in_id}_{out_id}", p)
        self._topo_dirty = True
        self._np_cache_dirty = True

    def remove_edge(self, in_id: int, out_id: int) -> None:
        key = (in_id, out_id)
        if key not in self.edges:
            return
        del self.edges[key]
        self._topo_dirty = True
        self._np_cache_dirty = True

    def num_edges(self) -> int:
        return len(self.edges)

    def num_hidden(self) -> int:
        return sum(1 for t in self.node_types.values() if t == "hidden")

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
            order = list(self.node_types.keys())
        return order

    def _ensure_topo(self) -> List[int]:
        if self._topo_order is None or self._topo_dirty:
            self._topo_order = self._build_topo()
            self._topo_dirty = False
        return self._topo_order

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (policy_logits, state_prediction)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]
        order = self._ensure_topo()
        a: Dict[int, torch.Tensor] = {}
        for i, nid in enumerate(self.input_ids):
            a[nid] = obs[:, i]
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
        policy = torch.stack([a[nid] for nid in self.policy_ids], dim=1)
        prediction = torch.stack([a[nid] for nid in self.predictor_ids], dim=1)
        return policy, prediction

    # ------------------------------------------------------------------
    # Numpy forward pass (fast inference)
    # ------------------------------------------------------------------

    def _build_numpy_cache(self) -> None:
        order = self._ensure_topo()
        self._np_order = order
        type_map = {"input": 0, "hidden": 1, "policy": 2, "predictor": 3}
        self._np_types = np.array([type_map[self.node_types[n]] for n in order], dtype=np.int32)
        self._np_biases = np.array([float(self.node_bias[n].item()) for n in order], dtype=np.float64)
        self._np_acts = [self.node_act[n] for n in order]
        self._np_id2idx = {n: i for i, n in enumerate(order)}
        self._np_input_idx = [self._np_id2idx[n] for n in self.input_ids]
        self._np_policy_idx = [self._np_id2idx[n] for n in self.policy_ids]
        self._np_predictor_idx = [self._np_id2idx[n] for n in self.predictor_ids]
        in_edges: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(order))}
        for (i, j), w in self.edges.items():
            if i in self._np_id2idx and j in self._np_id2idx:
                in_edges[self._np_id2idx[j]].append((self._np_id2idx[i], float(w.item())))
        self._np_in_edges = in_edges
        self._np_cache_dirty = False

    def forward_numpy(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fast numpy forward pass. Returns (policy_logits, state_prediction)."""
        if self._np_cache_dirty or not hasattr(self, '_np_order'):
            self._build_numpy_cache()
        a = np.zeros(len(self._np_order), dtype=np.float64)
        for i, idx in enumerate(self._np_input_idx):
            a[idx] = float(obs[i])
        for idx in self._np_order:
            if self._np_types[idx] == 0:  # input
                continue
            s = self._np_biases[idx]
            for in_idx, w in self._np_in_edges[idx]:
                s += a[in_idx] * w
            a[idx] = _activate_np(s, self._np_acts[idx])
        policy = np.array([a[idx] for idx in self._np_policy_idx], dtype=np.float64)
        prediction = np.array([a[idx] for idx in self._np_predictor_idx], dtype=np.float64)
        return policy, prediction

    def policy_numpy(self, obs: np.ndarray, stochastic: bool = True) -> int:
        logits, _ = self.forward_numpy(obs)
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        if stochastic:
            return int(np.random.choice(len(probs), p=probs))
        return int(np.argmax(probs))


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


def _activate_np(x: float, kind: str) -> float:
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
class PTIndividual:
    network: PTNetwork
    fitness: float = -1e9
    behavior: Optional[np.ndarray] = None
    age: int = 0
    pred_error_ema: float = 0.0  # EMA of prediction error (for curiosity)


# ---------------------------------------------------------------------------
# PT-NEAT main driver
# ---------------------------------------------------------------------------


class PTNEAT:
    """Predictive Topogenesis NEAT.

    Core principle: topology grows where the network is surprised (high
    prediction error). This provides a dense learning signal even on
    sparse-reward tasks.
    """

    def __init__(self, cfg: PTConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        # initial population
        self.population: List[PTIndividual] = []
        for _ in range(cfg.pop_size):
            net = PTNetwork(cfg.n_inputs, cfg.n_outputs, cfg.n_state_outputs,
                            output_activation=cfg.output_activation,
                            hidden_activation=cfg.hidden_activation)
            # fully connect inputs -> policy + predictor
            for i in net.input_ids:
                for j in net.policy_ids + net.predictor_ids:
                    net.add_edge(i, j, float(np.random.uniform(-cfg.init_weight_scale, cfg.init_weight_scale)))
            self.population.append(PTIndividual(network=net))
        self.history: List[Dict[str, float]] = []
        self.best_fitness: float = -1e9
        self.best_det_fitness: float = -1e9  # best deterministic (pure env) reward
        self.best_network: Optional[PTNetwork] = None
        self._prune_counter: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(dict)
        self._env = gym.make(env_name)
        self._env.reset(seed=cfg.seed)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, net: PTNetwork, env, max_steps: int, stochastic: bool = True) -> Tuple[
        List[np.ndarray], List[int], List[float], List[np.ndarray], float, np.ndarray
    ]:
        """Collect one rollout. Returns (obs_list, act_list, rew_list, pred_list, total, behavior).

        For stochastic (training) rollouts, a curiosity bonus is added to the
        reward at each step. The bonus = prediction error from the previous
        step. This means visiting surprising states gives higher reward,
        driving exploration on sparse-reward tasks.
        """
        obs, _ = env.reset()
        obs_list, act_list, rew_list, pred_list = [], [], [], []
        net._build_numpy_cache()
        prev_pred_error = 0.0
        for _ in range(max_steps):
            logits, prediction = net.forward_numpy(obs)
            e = np.exp(logits - np.max(logits))
            probs = e / e.sum()
            if stochastic:
                a = int(np.random.choice(len(probs), p=probs))
            else:
                a = int(np.argmax(probs))
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            act_list.append(a)
            pred_list.append(prediction.copy())
            prev_obs = obs
            obs, r, term, trunc, _ = env.step(a)
            r = float(r)
            # curiosity bonus: add previous prediction error to reward
            if stochastic and self.cfg.curiosity_coef > 0:
                r = r + self.cfg.curiosity_coef * prev_pred_error
            # compute current prediction error for next step's bonus
            if stochastic:
                actual_next = np.asarray(obs, dtype=np.float64)
                prev_pred_error = float(np.mean((prediction - actual_next) ** 2))
            rew_list.append(r)
            if term or trunc:
                break
        total = float(sum(rew_list))
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
        return obs_list, act_list, rew_list, pred_list, total, beh

    def _evaluate(self, ind: PTIndividual) -> Tuple[float, float, np.ndarray, List]:
        """Evaluate individual. Returns (fitness, det_reward, behavior, [rollout_data]).

        fitness = max(det_reward, stoch_reward_with_curiosity)
        det_reward = pure env reward (deterministic, argmax policy)
        """
        env = self._env
        det_rewards = []
        behs = []
        for _ in range(self.cfg.n_episodes):
            _, _, _, _, total, beh = self._rollout(ind.network, env, self.cfg.max_steps, stochastic=False)
            det_rewards.append(total)
            behs.append(beh)
        stoch_rollouts = []
        stoch_totals = []
        for _ in range(self.cfg.n_episodes):
            obs_list, act_list, rew_list, pred_list, total, _ = self._rollout(
                ind.network, env, self.cfg.max_steps, stochastic=True
            )
            next_obs_list = obs_list[1:] + [np.zeros_like(obs_list[0])]
            stoch_rollouts.append((obs_list, act_list, rew_list, pred_list, next_obs_list))
            stoch_totals.append(total)
        det_mean = float(np.mean(det_rewards))
        stoch_mean = float(np.mean(stoch_totals))
        fitness = max(det_mean, stoch_mean)
        return fitness, det_mean, np.mean(behs, axis=0), stoch_rollouts

    # ------------------------------------------------------------------
    # Training step (policy gradient + prediction loss)
    # ------------------------------------------------------------------

    def _train_step(self, net: PTNetwork, rollout_data: Tuple, pop_baseline: float) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """One training step: policy gradient + prediction loss.

        Returns (total_loss, edge_gradient_magnitudes).
        """
        obs_list, act_list, rew_list, pred_list, next_obs_list = rollout_data
        if len(obs_list) < 2:
            return 0.0, {}
        # compute returns
        gamma = 0.99
        T = len(rew_list)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rew_list[t] + gamma * G
            returns[t] = G
        # convert to tensors
        obs_t = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_t = torch.from_numpy(np.array(act_list, dtype=np.int64))
        ret_t = torch.from_numpy(returns)
        pred_t = torch.from_numpy(np.array(pred_list, dtype=np.float32))
        next_obs_t = torch.from_numpy(np.array(next_obs_list, dtype=np.float32))

        net.zero_grad(set_to_none=True)
        policy_logits, state_pred = net.forward(obs_t)
        # policy gradient loss
        log_probs = F.log_softmax(policy_logits, dim=-1)
        probs = F.softmax(policy_logits, dim=-1)
        picked = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
        advantage = ret_t - pop_baseline
        if advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        pg_loss = -(picked * advantage).mean()
        # entropy
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        # prediction loss (MSE between predicted and actual next state)
        pred_loss = F.mse_loss(state_pred, next_obs_t)
        # total loss
        loss = (self.cfg.reward_coef * pg_loss
                + self.cfg.prediction_coef * pred_loss
                - self.cfg.entropy_coef * entropy)
        loss.backward()
        # gradient clipping
        if self.cfg.grad_clip > 0:
            params_with_grad = [p for p in net.parameters() if p.grad is not None]
            if params_with_grad:
                total_norm = torch.norm(torch.stack([p.grad.norm() for p in params_with_grad]))
                if total_norm > self.cfg.grad_clip:
                    scale = self.cfg.grad_clip / (total_norm + 1e-6)
                    for p in params_with_grad:
                        p.grad.mul_(scale)
        # collect edge gradients (for topology growth)
        edge_grads: Dict[Tuple[int, int], float] = {}
        for key, p in net.edges.items():
            if p.grad is not None:
                edge_grads[key] = float(p.grad.item())
        # apply gradient
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p -= self.cfg.lr_weights * p.grad
        return float(loss.item()), {k: abs(v) for k, v in edge_grads.items()}

    # ------------------------------------------------------------------
    # Topology growth
    # ------------------------------------------------------------------

    def _candidate_edges(self, net: PTNetwork, n: int) -> List[Tuple[int, int]]:
        nodes = list(net.node_types.keys())
        order = net._build_topo()
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
            if (i, j) in existing:
                continue
            # don't connect TO input nodes
            if net.node_types[j] == "input":
                continue
            # don't connect FROM output nodes (policy/predictor are sinks)
            if net.node_types[i] in ("policy", "predictor"):
                continue
            existing.add((i, j))
            candidates.append((i, j))
        return candidates

    def _grow_topology(self, net: PTNetwork, rollout_data: Tuple, pop_baseline: float) -> Dict[Tuple[int, int], float]:
        """Grow topology based on combined policy + prediction gradient."""
        if net.num_edges() == 0 or not rollout_data:
            return {}
        # compute gradient on candidate edges
        candidates = self._candidate_edges(net, self.cfg.n_candidate_edges)
        if not candidates:
            return {}
        # temporarily add candidates with weight 0
        added_keys: List[Tuple[int, int]] = []
        for key in candidates:
            if key not in net.edges:
                p = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
                net.edges[key] = p
                added_keys.append(key)
        # compute gradient (reuse train_step logic but don't apply)
        obs_list, act_list, rew_list, pred_list, next_obs_list = rollout_data
        if len(obs_list) < 2:
            for key in added_keys:
                del net.edges[key]
            net._topo_dirty = True
            return {}
        gamma = 0.99
        T = len(rew_list)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rew_list[t] + gamma * G
            returns[t] = G
        obs_t = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_t = torch.from_numpy(np.array(act_list, dtype=np.int64))
        ret_t = torch.from_numpy(returns)
        pred_t = torch.from_numpy(np.array(pred_list, dtype=np.float32))
        next_obs_t = torch.from_numpy(np.array(next_obs_list, dtype=np.float32))
        net.zero_grad(set_to_none=True)
        policy_logits, state_pred = net.forward(obs_t)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        probs = F.softmax(policy_logits, dim=-1)
        picked = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
        advantage = ret_t - pop_baseline
        if advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        pg_loss = -(picked * advantage).mean()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        pred_loss = F.mse_loss(state_pred, next_obs_t)
        loss = (self.cfg.reward_coef * pg_loss
                + self.cfg.prediction_coef * pred_loss
                - self.cfg.entropy_coef * entropy)
        loss.backward()
        # read all gradients
        all_grads: Dict[Tuple[int, int], float] = {}
        for key, p in net.edges.items():
            if p.grad is not None:
                all_grads[key] = float(p.grad.item())
        # rank candidates
        ranked = sorted([(k, all_grads[k]) for k in added_keys], key=lambda kv: abs(kv[1]), reverse=True)
        # remove all candidates
        for key in added_keys:
            del net.edges[key]
        net._topo_dirty = True
        # decide how many to add
        n_to_add = self.cfg.edges_to_add
        if self.cfg.adaptive_edges and ranked:
            best_grad = abs(ranked[0][1])
            extra = int(best_grad / self.cfg.adaptive_edge_threshold)
            n_to_add = min(n_to_add + extra, len(ranked))
        # add chosen edges with gradient-proportional init
        if ranked:
            max_grad = max(abs(g) for _, g in ranked[:n_to_add]) or 1.0
        else:
            max_grad = 1.0
        for (i, j), g in ranked[:n_to_add]:
            scale = abs(g) / (max_grad + 1e-8)
            init_w = float(np.sign(g) * 0.5 * scale) if abs(g) > 1e-6 else 0.0
            net.add_edge(i, j, init_w)
        # return existing-edge gradient magnitudes
        return {k: abs(v) for k, v in all_grads.items() if k not in added_keys}

    def _maybe_split(self, net: PTNetwork, edge_grads: Dict[Tuple[int, int], float]) -> None:
        if np.random.random() > self.cfg.p_split_edge:
            return
        if not net.edges:
            return
        if edge_grads:
            best_key = max(edge_grads.keys(), key=lambda k: edge_grads[k])
            i, j = best_key
            old_w = float(net.edges[best_key].item())
        else:
            items = list(net.edges.items())
            (i, j), w = max(items, key=lambda kv: abs(float(kv[1].item())))
            old_w = float(w.item())
        if abs(old_w) < 0.05:
            return
        with torch.no_grad():
            net.edges[(i, j)].fill_(0.0)
        h = net.add_node("hidden")
        net.add_edge(i, h, 1.0)
        net.add_edge(h, j, old_w)

    def _prune(self, net: PTNetwork, ind_id: int, edge_grads: Dict[Tuple[int, int], float]) -> None:
        to_remove = []
        for key, w in list(net.edges.items()):
            gmag = edge_grads.get(key, 0.0)
            wmag = abs(float(w.item()))
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
    # Diversity / restart
    # ------------------------------------------------------------------

    def _maybe_restart(self) -> None:
        if not self.cfg.restart_on_convergence or len(self.population) < 4:
            return
        behs_list = [ind.behavior for ind in self.population if ind.behavior is not None]
        if len(behs_list) < 4:
            return
        behs = np.array(behs_list)
        n = len(behs)
        k = min(n, 20)
        idx = np.random.choice(n, k, replace=False)
        sample = behs[idx]
        dists = []
        for i in range(k):
            for j in range(i + 1, k):
                dists.append(np.linalg.norm(sample[i] - sample[j]))
        avg_dist = float(np.mean(dists)) if dists else 0.0
        if avg_dist > self.cfg.restart_threshold:
            return
        order = sorted(range(len(self.population)),
                       key=lambda i: self.population[i].fitness if self.population[i].behavior is not None else -1e18)
        n_restart = max(1, int(len(self.population) * self.cfg.restart_fraction))
        for i in order[:n_restart]:
            net = PTNetwork(self.cfg.n_inputs, self.cfg.n_outputs, self.cfg.n_state_outputs,
                            output_activation=self.cfg.output_activation,
                            hidden_activation=self.cfg.hidden_activation)
            for ii in net.input_ids:
                for jj in net.policy_ids + net.predictor_ids:
                    net.add_edge(ii, jj, float(np.random.uniform(-self.cfg.init_weight_scale, self.cfg.init_weight_scale)))
            self.population[i] = PTIndividual(network=net)

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def _clone_network(self, src: PTNetwork) -> PTNetwork:
        dst = PTNetwork(src.n_inputs, src.n_actions, src.n_state,
                        output_activation=src.output_activation,
                        hidden_activation=src.hidden_activation)
        dst.node_types.clear()
        dst.node_act.clear()
        dst.node_bias.clear()
        dst.edges.clear()
        dst._next_id = max(src.node_types.keys()) + 1
        for nid in sorted(src.node_types.keys()):
            dst.node_types[nid] = src.node_types[nid]
            dst.node_act[nid] = src.node_act[nid]
            bias = float(src.node_bias[nid].item())
            if src.node_types[nid] == "input":
                p = nn.Parameter(torch.zeros(1), requires_grad=False)
            else:
                p = nn.Parameter(torch.tensor([bias], dtype=torch.float32))
            dst.node_bias[nid] = p
        dst.input_ids = list(src.input_ids)
        dst.policy_ids = list(src.policy_ids)
        dst.predictor_ids = list(src.predictor_ids)
        for (i, j), w in src.edges.items():
            dst.add_edge(i, j, float(w.item()))
        dst._topo_dirty = True
        dst._np_cache_dirty = True
        return dst

    def _select_parent(self) -> int:
        pop_n = len(self.population)
        candidates = np.random.choice(pop_n, size=3, replace=False)
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return int(best)

    def _reproduce(self, pop_baseline: float) -> None:
        cfg = self.cfg
        order = sorted(range(len(self.population)),
                       key=lambda i: self.population[i].fitness, reverse=True)
        n_elite = max(1, cfg.pop_size // 5)
        new_pop: List[PTIndividual] = []
        for i in range(n_elite):
            parent = self.population[order[i]]
            parent.age += 1
            new_pop.append(parent)
        while len(new_pop) < cfg.pop_size:
            pi = self._select_parent()
            parent = self.population[pi]
            child_net = self._clone_network(parent.network)
            # gradient-directed mutation: one training step
            obs_list, act_list, rew_list, _, _, _ = self._rollout(
                child_net, self._env, cfg.max_steps, stochastic=True
            )
            next_obs_list = obs_list[1:] + [np.zeros_like(obs_list[0])]
            # get predictions from the child
            child_net._build_numpy_cache()
            pred_list = [child_net.forward_numpy(o)[1] for o in obs_list]
            rollout_data = (obs_list, act_list, rew_list, pred_list, next_obs_list)
            self._train_step(child_net, rollout_data, pop_baseline)
            new_pop.append(PTIndividual(network=child_net))
        self.population = new_pop[: cfg.pop_size]

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        cfg = self.cfg
        # 1. Evaluate all individuals
        all_rollouts: Dict[int, List] = {}
        all_det_rewards: List[float] = []
        for i, ind in enumerate(self.population):
            f, det_r, beh, rollouts = self._evaluate(ind)
            ind.fitness = f
            ind.behavior = beh
            all_rollouts[i] = rollouts
            all_det_rewards.append(det_r)
        # 2. Track best
        best_idx = int(np.argmax([ind.fitness for ind in self.population]))
        if self.population[best_idx].fitness > self.best_fitness:
            self.best_fitness = self.population[best_idx].fitness
            self.best_network = self._clone_network(self.population[best_idx].network)
        # track best deterministic reward
        best_det_idx = int(np.argmax(all_det_rewards))
        if all_det_rewards[best_det_idx] > self.best_det_fitness:
            self.best_det_fitness = all_det_rewards[best_det_idx]
        # 3. Train weights (policy + prediction)
        pop_mean_fitness = float(np.mean([ind.fitness for ind in self.population]))
        all_edge_grads: Dict[int, Dict[Tuple[int, int], float]] = {}
        for i, ind in enumerate(self.population):
            rollouts = all_rollouts[i]
            if not rollouts:
                all_edge_grads[i] = {}
                continue
            rollout_data = max(rollouts, key=lambda r: len(r[0]))
            edge_grads = {}
            for _ in range(cfg.pg_steps):
                _, eg = self._train_step(ind.network, rollout_data, pop_mean_fitness)
                edge_grads = eg
            all_edge_grads[i] = edge_grads
        # 4. Topology growth
        for i, ind in enumerate(self.population):
            rollouts = all_rollouts[i]
            if not rollouts:
                continue
            rollout_data = max(rollouts, key=lambda r: len(r[0]))
            edge_grads = self._grow_topology(ind.network, rollout_data, pop_mean_fitness)
            if np.random.random() < cfg.p_split_edge:
                self._maybe_split(ind.network, edge_grads)
            self._prune(ind.network, i, edge_grads)
        # 5. Stats
        fits = np.array([ind.fitness for ind in self.population])
        sizes = np.array([ind.network.num_edges() for ind in self.population])
        hids = np.array([ind.network.num_hidden() for ind in self.population])
        stat = {
            "gen": len(self.history),
            "best": float(fits.max()),
            "best_det": float(max(all_det_rewards)),
            "mean": float(fits.mean()),
            "std": float(fits.std()),
            "avg_size": float(sizes.mean()),
            "avg_hidden": float(hids.mean()),
        }
        self.history.append(stat)
        # 6. Reproduce
        self._reproduce(pop_mean_fitness)
        # 7. Restart on convergence
        self._maybe_restart()
        return stat

    def run(self, n_generations: int, verbose: bool = True) -> None:
        for _ in range(n_generations):
            stat = self.step()
            if verbose:
                print(
                    f"PT gen {stat['gen']:>3} | best {stat['best']:7.2f} | "
                    f"mean {stat['mean']:7.2f} | avg_size {stat['avg_size']:5.1f} | "
                    f"avg_hidden {stat['avg_hidden']:4.1f}"
                )
