"""A minimal DQN baseline using PyTorch (CPU).

This is intentionally small — just enough to provide a SOTA deep-RL
reference point. Uses a 2-layer MLP Q-network, experience replay, target
network, and epsilon-greedy exploration.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random
from collections import deque


@dataclass
class DQNConfig:
    n_inputs: int = 4
    n_outputs: int = 2
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 10000
    target_update: int = 200  # steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 2000
    max_steps: int = 500
    total_steps: int = 30000  # total environment steps
    eval_episodes: int = 5
    eval_interval: int = 1000  # steps
    seed: int = 0


class QNet(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, cfg: DQNConfig, env_name: str):
        self.cfg = cfg
        self.env_name = env_name
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = torch.device("cpu")
        self.q = QNet(cfg.n_inputs, cfg.n_outputs, cfg.hidden_dim).to(self.device)
        self.target_q = QNet(cfg.n_inputs, cfg.n_outputs, cfg.hidden_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = deque(maxlen=cfg.replay_size)
        self.env = gym.make(env_name)
        self.env.reset(seed=cfg.seed)
        self.eval_env = gym.make(env_name)
        self.eval_env.reset(seed=cfg.seed + 1)
        self.steps_done = 0
        self.history: List[Tuple[int, float, float]] = []  # (step, eval_mean, eval_std)
        self.best_fitness: float = -1e9
        self.best_state: Optional[dict] = None

    def _eps(self) -> float:
        cfg = self.cfg
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * max(0.0, 1.0 - self.steps_done / cfg.eps_decay_steps)

    def _select_action(self, obs: np.ndarray) -> int:
        if random.random() < self._eps():
            return random.randrange(self.cfg.n_outputs)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.q(obs_t)
            return int(q.argmax(dim=1).item())

    def _optimize(self) -> Optional[float]:
        cfg = self.cfg
        if len(self.replay) < cfg.batch_size:
            return None
        batch = random.sample(self.replay, cfg.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.from_numpy(np.array(s, dtype=np.float32)).to(self.device)
        a = torch.from_numpy(np.array(a, dtype=np.int64)).unsqueeze(1).to(self.device)
        r = torch.from_numpy(np.array(r, dtype=np.float32)).to(self.device)
        s2 = torch.from_numpy(np.array(s2, dtype=np.float32)).to(self.device)
        done = torch.from_numpy(np.array(done, dtype=np.float32)).to(self.device)

        q_sa = self.q(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q2 = self.target_q(s2).max(dim=1)[0]
            target = r + cfg.gamma * q2 * (1.0 - done)
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        for p in self.q.parameters():
            p.grad.data.clamp_(-1.0, 1.0)
        self.opt.step()
        return float(loss.item())

    def _evaluate(self) -> Tuple[float, float]:
        cfg = self.cfg
        self.q.eval()
        rewards = []
        with torch.no_grad():
            for _ in range(cfg.eval_episodes):
                obs, _ = self.eval_env.reset()
                total = 0.0
                for _ in range(cfg.max_steps):
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    a = int(self.q(obs_t).argmax(dim=1).item())
                    obs, r, term, trunc, _ = self.eval_env.step(a)
                    total += float(r)
                    if term or trunc:
                        break
                rewards.append(total)
        self.q.train()
        arr = np.array(rewards)
        return float(arr.mean()), float(arr.std())

    def step_env(self) -> Tuple[bool, float]:
        cfg = self.cfg
        if not hasattr(self, "_obs"):
            self._obs, _ = self.env.reset()
        a = self._select_action(self._obs)
        next_obs, r, term, trunc, _ = self.env.step(a)
        done = term or trunc
        self.replay.append((self._obs.copy(), a, r, next_obs.copy(), float(term)))
        self._obs = next_obs
        if done:
            self._obs, _ = self.env.reset()
        self._optimize()
        self.steps_done += 1
        if self.steps_done % cfg.target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return done, float(r)
    def run(self, verbose: bool = True) -> None:
        cfg = self.cfg
        last_eval_step = 0
        while self.steps_done < cfg.total_steps:
            self.step_env()
            if self.steps_done - last_eval_step >= cfg.eval_interval:
                last_eval_step = self.steps_done
                m, s = self._evaluate()
                if m > self.best_fitness:
                    self.best_fitness = m
                    self.best_state = {k: v.clone() for k, v in self.q.state_dict().items()}
                self.history.append((self.steps_done, m, s))
                if verbose:
                    print(f"DQN step {self.steps_done:>6} | eval {m:7.2f} ± {s:5.2f} | eps {self._eps():.3f}")

    def close(self) -> None:
        self.env.close()
        self.eval_env.close()
