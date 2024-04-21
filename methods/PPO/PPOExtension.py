from .PPONetwork import Actor, Critic
from machin.frame.algorithms import PPO
import torch
import torch.nn as nn
import utils


class PPOExtension:
    def __init__(self):
        self.n_actions = 5
        self.agent_obs_space = 10
        self.adv_obs_space = 12

        agent_actor = Actor(self.n_actions, self.agent_obs_space)
        agent_critic = Critic(self.agent_obs_space)

        adv_actor = Actor(self.n_actions, self.adv_obs_space)
        adv_critic = Critic(self.adv_obs_space)

        self.agent_ppo = PPO(agent_actor, agent_critic, torch.optim.Adam, nn.MSELoss(reduction="sum"))
        self.adv_ppo = PPO(adv_actor, adv_critic, torch.optim.Adam, nn.MSELoss(reduction="sum"))

    def transform_state(self, name, observation):
        # changing observations to tensors to fit into Network
        if 'agent' in name:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.agent_obs_space)
        else:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.adv_obs_space)

    def get_action(self, name, state):
        if 'agent' in name:
            return self.agent_ppo.act({"some_state": state})[0]
        else:
            return self.adv_ppo.act({"some_state": state})[0]

    def store_transition(self, name, state, action, next_state, reward, terminal):
        if 'agent' in name:
            ppo = self.agent_ppo
        else:
            ppo = self.adv_ppo

        ppo.store_episode(
            [{
                "state": {"some_state": state},
                "action": {"action": action},
                "next_state": {"some_state": next_state},
                "reward": reward,
                "terminal": terminal,
            }]
        )

    def update(self):
        self.agent_ppo.update()
        self.adv_ppo.update()
