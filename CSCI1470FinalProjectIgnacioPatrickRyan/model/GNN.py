"""
Legacy GNN compatibility shim.

The migrated pipeline uses an MLP actor-critic (ppo_agent.py) rather than a GNN.
This placeholder keeps old imports from breaking.
"""

import torch
import torch.nn as nn


class PhysicsGNN(nn.Module):
    def __init__(self, node_in_dim: int = 7, edge_in_dim: int = 4, embed_dim: int = 64):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, embed_dim)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # nodes: [B, 3, node_in_dim] -> [B, 3, embed_dim]
        return torch.tanh(self.node_proj(nodes))


__all__ = ["PhysicsGNN"]
