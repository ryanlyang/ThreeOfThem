import torch
import torch.nn as nn

class CelestialActor(nn.Module):
    def __init__(self, embed_dim=64, action_dim=3):
        """
        RL Actor that each body instantiates
        """
        super(CelestialActor, self).__init__()
        
        # Simple MLP that takes the GNN embedding and outputs action distributions
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # For continuous PPO, we output the Mean and Log-Std of a Gaussian
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, h):
        """
        Takes ONLY its own embedding from the GNN.
        Returns the thrust probability distribution.
        """
        x = self.net(h)
        action_mean = torch.tanh(self.mean_layer(x)) # Tanh bounds thrust between -1 and 1
        action_std = torch.exp(self.log_std_layer)
        return action_mean, action_std