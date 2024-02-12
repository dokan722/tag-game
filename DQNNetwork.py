import os
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, n_actions, input_dims):
        super(DQNNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.fc1 = nn.Linear(input_dims, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, n_actions)

    def forward(self, some_state):
        x = torch.relu(self.fc1(some_state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
