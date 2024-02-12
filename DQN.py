from DQNNetwork import DQNNetwork
from machin.frame.algorithms import DQN
import torch
import torch.nn as nn
import utils


class DQNRunner:
    def __init__(self, env, plot_name_prefix=''):
        self.n_actions = 5
        self.agent_obs_space = 14
        self.adv_obs_space = 16
        self.max_episodes = 1000
        self.env = env
        self.prefix = plot_name_prefix

        agent_net = DQNNetwork(self.n_actions, self.agent_obs_space)
        agent_net_t = DQNNetwork(self.n_actions, self.agent_obs_space)

        adv_net = DQNNetwork(self.n_actions, self.adv_obs_space)
        adv_net_t = DQNNetwork(self.n_actions, self.adv_obs_space)

        self.agent_dqn = DQN(agent_net, agent_net_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))
        self.adv_dqn = DQN(adv_net, adv_net_t, torch.optim.Adam, nn.MSELoss(reduction="sum"))

    def transform_state(self, name, observation):
        if 'agent' in name:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.agent_obs_space)
        else:
            return torch.tensor(observation, dtype=torch.float32).view(1, self.adv_obs_space)

    def get_action(self, name, state):
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

    def update_dqns(self):
        self.agent_dqn.update()
        self.adv_dqn.update()

    def train(self):
        training_rewards = []
        rewarded_agents = ['adversary_0', 'agent_0']
        for episode in range(self.max_episodes):
            observations, infos = self.env.reset()
            if episode == 0:
                all_agents = self.env.agents
                positions = {agent: [] for agent in all_agents}
            rewards_sum = {k: 0 for k in rewarded_agents}

            while self.env.agents:
                states = {name: self.transform_state(name, obs) for name, obs in observations.items()}
                actions = {name: self.get_action(name, obs) for name, obs in states.items()}
                actions_items = {name: action.item() for name, action in actions.items()}
                observations, rewards, terminations, _, _ = self.env.step(actions_items)
                next_states = {name: self.transform_state(name, obs) for name, obs in observations.items()}

                for agent in self.env.agents:
                    self.store_transition(agent, states[agent], actions[agent], next_states[agent], rewards[agent],
                                          terminations[agent])
                    positions[agent].append([observations[agent][2], observations[agent][3]])

                for agent in rewarded_agents:
                    rewards_sum[agent] += rewards[agent]

                self.update_dqns()

            training_rewards.append(rewards_sum)
            if episode % 100 == 0 and episode > 0:
                print("Episode: ", episode)
                for agent in all_agents:
                    utils.plot_positions(positions[agent], self.prefix + 'episode_' + str(episode) + '_' + agent + '_positions')
                positions = {agent: [] for agent in all_agents}

        return training_rewards
