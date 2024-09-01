from .DQNNetwork import DQNNetwork
from machin.frame.algorithms import DQN
import torch
import torch.nn as nn
import os


class DQNExtension:
    def __init__(self, obs_space, is_centralized):
        factor = 5 if is_centralized else 1
        self.n_actions = 5
        self.obs_space = obs_space

        net = DQNNetwork(self.n_actions, self.obs_space, factor)
        net_t = DQNNetwork(self.n_actions, self.obs_space, factor)
        self.dqn = DQN(net, net_t, torch.optim.Adam, nn.MSELoss(reduction="sum"), epsilon_decay=0.9999985)
        self.learning = True

    def transform_state(self, observation):
        # changing observations to tensors to fit into Network
        return torch.tensor(observation, dtype=torch.float32).view(1, self.obs_space)

    def get_action(self, state):
        # get action for agents with epsilon greedy
        return self.dqn.act_discrete_with_noise({"state": state})

    def store_transitions(self, transitions):
        if self.learning:
            self.dqn.store_episode(transitions)

    def update(self):
        if self.learning:
            return self.dqn.update()
        return torch.tensor([0.0])

    def save(self, directory, name):
        directory = os.path.dirname(directory + '/' + name + '/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.dqn.save(directory)

    def load(self, directory, learning):
        self.learning = learning
        self.dqn.load(directory)

    def inc_entropy_weight(self):
        pass
