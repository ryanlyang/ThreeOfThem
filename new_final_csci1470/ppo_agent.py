from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Streaming mean/variance for observation normalization."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = float(total_count)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = (x - self.mean) / np.sqrt(self.var + 1e-8)
        y = np.clip(y, -clip, clip)
        return y.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        # Lower initial exploration helps avoid early full-thrust saturation.
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.2))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = torch.tanh(self.policy_head(h))
        std = torch.exp(torch.clamp(self.log_std, -4.0, 1.0)).expand_as(mean)
        value = self.value_head(h).squeeze(-1)
        return mean, std, value


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor


def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(mean, std)
    return dist.log_prob(actions).sum(dim=-1)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(torch.zeros_like(std), std)
    return dist.entropy().sum(dim=-1)
