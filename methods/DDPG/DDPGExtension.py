from .DDPGNetwork import Actor, Critic
from machin.frame.algorithms import DDPG
import torch
import torch.nn as nn


class DDPGExtension:
    def __init__(self):
        self.n_actions = 5
        self.agent_obs_space = 10
        self.adv_obs_space = 12

        agent_actor = Actor(self.n_actions, self.agent_obs_space)
        agent_actor_t = Actor(self.n_actions, self.agent_obs_space)
        agent_critic = Critic(self.n_actions, self.agent_obs_space)
        agent_critic_t = Critic(self.n_actions, self.agent_obs_space)

        adv_actor = Actor(self.n_actions, self.adv_obs_space)
        adv_actor_t = Actor(self.n_actions, self.adv_obs_space)
        adv_critic = Critic(self.n_actions, self.adv_obs_space)
        adv_critic_t = Critic(self.n_actions, self.adv_obs_space)

        self.agent_ddpg = DDPG(agent_actor, agent_actor_t, agent_critic, agent_critic_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))
        self.adv_ddpg = DDPG(adv_actor, adv_actor_t, adv_critic, adv_critic_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))

    def transform_state(self, name, observation):
        # changing observations to tensors to fit into Network
        if 'agent' in name:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.agent_obs_space)
        else:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.adv_obs_space)

    def get_action(self, name, state):
        if 'agent' in name:
            return self.agent_ddpg.act_discrete_with_noise({"state": state})[0]
        else:
            return self.adv_ddpg.act_discrete_with_noise({"state": state})[0]

    def store_transition(self, name, state, action, next_state, reward, terminal):
        if 'agent' in name:
            ddpg = self.agent_ddpg
        else:
            ddpg = self.adv_ddpg

        ddpg.store_episode(
            [{
                "state": {"state": state},
                "action": {"action": action},
                "next_state": {"state": next_state},
                "reward": reward,
                "terminal": terminal,
            }]
        )

    def update(self):
        self.agent_ddpg.update()
        self.adv_ddpg.update()
