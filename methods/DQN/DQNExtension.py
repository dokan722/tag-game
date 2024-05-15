from .DQNNetwork import DQNNetwork
from machin.frame.algorithms import DQN
import torch
import torch.nn as nn


class DQNExtension:
    def __init__(self, obs_space):
        self.n_actions = 5
        self.obs_space = obs_space

        net = DQNNetwork(self.n_actions, self.obs_space)
        net_t = DQNNetwork(self.n_actions, self.obs_space)
        self.dqn = DQN(net, net_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))

    def transform_state(self, observation):
        # changing observations to tensors to fit into Network
        return torch.tensor(observation, dtype=torch.float32).view(1, self.obs_space)

    def get_action(self, state):
        # get action for agents with epsilon greedy
        return self.dqn.act_discrete_with_noise({"some_state": state})

    def store_transition(self, state, action, next_state, reward, terminal):
        self.dqn.store_episode(
            [{
                "state": {"some_state": state},
                "action": {"action": action},
                "next_state": {"some_state": next_state},
                "reward": reward,
                "terminal": terminal,
            }]
        )

    def update(self):
        self.dqn.update()
