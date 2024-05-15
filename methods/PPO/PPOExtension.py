from .PPONetwork import Actor, Critic
from machin.frame.algorithms import PPO
import torch
import torch.nn as nn


class PPOExtension:
    def __init__(self, obs_space):
        self.n_actions = 5
        self.obs_space = obs_space

        actor = Actor(self.n_actions, self.obs_space)
        critic = Critic(self.obs_space)

        self.ppo = PPO(actor, critic, torch.optim.Adam, nn.MSELoss(reduction="sum"))

    def transform_state(self, observation):
        # changing observations to tensors to fit into Network
        return torch.tensor(observation, dtype=torch.float32).view(1, self.obs_space)

    def get_action(self, state):
        return self.ppo.act({"some_state": state})[0]

    def store_transition(self, state, action, next_state, reward, terminal):
        self.ppo.store_episode(
            [{
                "state": {"some_state": state},
                "action": {"action": action},
                "next_state": {"some_state": next_state},
                "reward": reward,
                "terminal": terminal,
            }]
        )

    def update(self):
        self.ppo.update()
