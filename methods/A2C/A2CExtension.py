from .A2CNetwork import Actor, Critic
from machin.frame.algorithms import A2C
import torch
import torch.nn as nn
import os


class A2CExtension:
    def __init__(self, obs_space, is_centralized):
        factor = 5 if is_centralized else 1
        self.n_actions = 5
        self.obs_space = obs_space

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        actor = Actor(self.n_actions, self.obs_space, factor)
        critic = Critic(self.obs_space, factor)

        self.a2c = A2C(actor, critic, torch.optim.Adam, nn.MSELoss(reduction="sum"), actor_learning_rate=0.001, entropy_weight=-0.05)
        self.learning = True

    def transform_state(self, observation):
        # changing observations to tensors to fit into Network
        return torch.tensor(observation, dtype=torch.float32).view(1, self.obs_space)

    def get_action(self, state):
        return self.a2c.act({"state": state})[0]

    def store_transitions(self, transitions):
        if self.learning:
            self.a2c.store_episode(transitions)

    def update(self):
        if self.learning:
            return self.a2c.update()
        return torch.tensor([0.0])

    def save(self, directory, name):
        directory = os.path.dirname(directory + '/' + name + '/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.a2c.save(directory)

    def load(self, directory, learning):
        self.learning = learning
        self.a2c.load(directory)

    def inc_entropy_weight(self):
        self.a2c.entropy_weight += 0.001
