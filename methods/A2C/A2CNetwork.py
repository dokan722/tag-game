import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, n_actions, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_actions)

    def forward(self, some_state, action=None):
        a = torch.relu(self.fc1(some_state))
        a = torch.relu(self.fc2(a))
        probs = torch.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, some_state):
        v = torch.relu(self.fc1(some_state))
        v = torch.relu(self.fc2(v))
        v = self.fc3(v)
        return v
