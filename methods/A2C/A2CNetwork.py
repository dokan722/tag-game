import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, factor):
        super().__init__()
        io_factor = 3 if factor != 1 else 1

        self.fc1 = nn.Linear(input_dims * io_factor, 32 * factor)
        self.fc2 = nn.Linear(32 * factor, 64 * factor)
        self.fc3 = nn.Linear(64 * factor, n_actions ** io_factor)

    def forward(self, state, action=None):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        probs = torch.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, input_dims, factor):
        super().__init__()
        io_factor = 3 if factor != 1 else 1

        self.fc1 = nn.Linear(input_dims * io_factor, 32 * factor)
        self.fc2 = nn.Linear(32 * factor, 64 * factor)
        self.fc3 = nn.Linear(64 * factor, 1)

    def forward(self, state):
        v = torch.relu(self.fc1(state))
        v = torch.relu(self.fc2(v))
        v = self.fc3(v)
        return v
