import os
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, factor):
        super(DQNNetwork, self).__init__()
        io_factor = 3 if factor != 1 else 1

        self.fc1 = nn.Linear(input_dims * io_factor, 32 * factor)
        self.fc2 = nn.Linear(32 * factor, 64 * factor)
        self.fc3 = nn.Linear(64 * factor, n_actions ** io_factor)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
