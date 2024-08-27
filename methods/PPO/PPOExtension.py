from .PPONetwork import Actor, Critic
from machin.frame.algorithms import PPO
import torch
import torch.nn as nn
import os


class PPOExtension:
    def __init__(self, obs_space, is_centralized):
        factor = 5 if is_centralized else 1
        self.n_actions = 5
        self.obs_space = obs_space

        actor = Actor(self.n_actions, self.obs_space, factor)
        critic = Critic(self.obs_space, factor)

        self.ppo = PPO(actor, critic, torch.optim.Adam, nn.MSELoss(reduction="sum"))
        self.learning = True

    def transform_state(self, observation):
        # changing observations to tensors to fit into Network
        return torch.tensor(observation, dtype=torch.float32).view(1, self.obs_space)

    def get_action(self, state):
        return self.ppo.act({"state": state})[0]

    def store_transitions(self, transitions):
        if self.learning:
            self.ppo.store_episode(transitions)

    def update(self):
        if self.learning:
            return self.ppo.update()
        return torch.tensor([0.0])

    def save(self, directory, name):
        directory = os.path.dirname(directory + '/' + name + '/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.ppo.save(directory)

    def load(self, directory, learning):
        self.learning = learning
        self.ppo.load(directory)
