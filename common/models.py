import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(Actor, self).__init__()
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, action_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(Critic, self).__init__()
        layers = []
        current_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)