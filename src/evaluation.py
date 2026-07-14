"""Evaluation harness for RL control tasks.

Wraps gymnasium environments to give a uniform `evaluate(genome, env_name,
n_episodes, max_steps)` interface. Returns mean, std, min, max of cumulative
reward over episodes.

We deliberately evaluate every genome deterministically (greedy action) plus
a small number of stochastic rollouts to reduce variance, then take the mean.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Tuple
from .genome import Genome, Network


def make_env(env_name: str, seed: int = 0):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def _rollout(net: Network, env, max_steps: int, action_mode: str = "argmax") -> float:
    obs, _ = env.reset()
    total = 0.0
    for _ in range(max_steps):
        a = net.action(obs, action_mode=action_mode)
        obs, r, terminated, truncated, _ = env.step(a)
        total += float(r)
        if terminated or truncated:
            break
    return total


def evaluate(
    genome: Genome,
    env_name: str,
    n_episodes: int = 5,
    max_steps: int = 500,
    seed: int = 0,
    action_mode: str = "argmax",
) -> Tuple[float, float, float, float]:
    """Evaluate genome on `env_name`.

    Returns (mean, std, min, max) of cumulative reward over `n_episodes`.
    """
    env = make_env(env_name, seed=seed)
    try:
        net = Network(genome)
        rewards = [
            _rollout(net, env, max_steps=max_steps, action_mode=action_mode)
            for _ in range(n_episodes)
        ]
    finally:
        env.close()
    arr = np.asarray(rewards, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def evaluate_with_behavior(
    genome: Genome,
    env_name: str,
    n_episodes: int = 5,
    max_steps: int = 500,
    seed: int = 0,
    action_mode: str = "argmax",
):
    """Like evaluate but also returns a behavior descriptor.

    Behavior descriptor = summary statistics over the trajectory:
      - mean abs value of each observation dimension
      - std of each observation dimension
      - mean action
      - episode length
    This is used by behavior-based speciation variants.
    """
    env = make_env(env_name, seed=seed)
    try:
        net = Network(genome)
        rewards = []
        all_obs = []
        all_acts = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_obs = []
            ep_acts = []
            total = 0.0
            steps = 0
            for _ in range(max_steps):
                a = net.action(obs, action_mode=action_mode)
                ep_obs.append(obs.copy())
                ep_acts.append(float(a))
                obs, r, terminated, truncated, _ = env.step(a)
                total += float(r)
                steps += 1
                if terminated or truncated:
                    break
            rewards.append(total)
            all_obs.extend(ep_obs)
            all_acts.extend(ep_acts)
    finally:
        env.close()
    arr = np.asarray(rewards, dtype=np.float64)
    if len(all_obs) == 0:
        beh = np.zeros(8, dtype=np.float64)
    else:
        obs_arr = np.asarray(all_obs, dtype=np.float64)
        act_arr = np.asarray(all_acts, dtype=np.float64)
        beh = np.concatenate(
            [
                np.mean(np.abs(obs_arr), axis=0),
                np.std(obs_arr, axis=0),
                np.array([np.mean(act_arr), float(len(all_obs)) / max_steps / n_episodes]),
            ]
        )
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), beh
