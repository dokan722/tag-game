from .DQNNetwork import DQNNetwork
from machin.frame.algorithms import DQN
import torch
import torch.nn as nn
import utils


class DQNExtension:
    def __init__(self):
        self.n_actions = 5
        self.agent_obs_space = 10
        self.adv_obs_space = 12

        agent_net = DQNNetwork(self.n_actions, self.agent_obs_space)
        agent_net_t = DQNNetwork(self.n_actions, self.agent_obs_space)

        adv_net = DQNNetwork(self.n_actions, self.adv_obs_space)
        adv_net_t = DQNNetwork(self.n_actions, self.adv_obs_space)

        self.agent_dqn = DQN(agent_net, agent_net_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))
        self.adv_dqn = DQN(adv_net, adv_net_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))

    def transform_state(self, name, observation):
        # changing observations to tensors to fit into Network
        if 'agent' in name:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.agent_obs_space)
        else:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.adv_obs_space)

    def get_action(self, name, state):
        # get action for agents with epsilon greedy
        if 'agent' in name:
            return self.agent_dqn.act_discrete_with_noise({"some_state": state})
        else:
            return self.adv_dqn.act_discrete_with_noise({"some_state": state})

    def store_transition(self, name, state, action, next_state, reward, terminal):
        if 'agent' in name:
            dqn = self.agent_dqn
        else:
            dqn = self.adv_dqn

        dqn.store_episode(
            [{
                "state": {"some_state": state},
                "action": {"action": action},
                "next_state": {"some_state": next_state},
                "reward": reward,
                "terminal": terminal,
            }]
        )

    def update(self):
        self.agent_dqn.update()
        self.adv_dqn.update()
