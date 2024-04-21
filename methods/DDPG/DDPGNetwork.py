import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self,  n_actions, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_actions)
        self.n_actions = n_actions

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        a = torch.softmax(a, dim=-1)
        return a


class Critic(nn.Module):
    def __init__(self, n_actions, input_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims + 1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
