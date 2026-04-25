import torch
import torch.nn as nn

class PhysicsGNN(nn.Module):
    def __init__(self, node_in_dim=7, edge_in_dim=4, hidden_dim=64, msg_dim=16, embed_dim=64):
        """
        Permutation-invariant Graph Neural Network.
        
        node_in_dim: 3(pos) + 3(vel) + 1(mass) = 7
        edge_in_dim: 3(rel_pos) + 1(distance) = 4
        """
        super(PhysicsGNN, self).__init__()
        
        self.message_mlp = nn.Sequential(
            nn.Linear(node_in_dim + node_in_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, msg_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Sequential(node_in_dim + msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, nodes, edges):
        """
        Performs Message Passing.
        Returns a permutation-invariant embedding for each body.

        nodes: [B, 3, node_in_dim]
        edges: [B, 3, 3, edge_in_dim]

        Output: [B, 3, embed_dim] - Embedded states for each node
        """
        B, _, _ = nodes.shape

        h_from = nodes.unsqueeze(2).expand(B, 3, 3, -1)
        h_to = nodes.unsqueeze(1).expand(B, 3, 3, -1)

        messages = self.message_mlp(torch.cat((h_from, h_to, edges), dim=-1))

        mask = torch.eye(3).bool().unsqueeze(0).unsqueeze(-1)
        masked_messages = messages.masked_fill(mask, 0.0)
        m_summed = torch.sum(masked_messages, dim=2)

        return self.node_mlp(torch.cat(nodes, m_summed, dim=-1))