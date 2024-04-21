import numpy as np

import utils
from PIL import Image


class Runner:
    def __init__(self, env, combined_policy, model_name):
        self.max_episodes = 5000
        self.map_interval = 500  # intrval for plotting heatmap and generating gif
        self.model_name = model_name
        self.env = env
        self.combined_policy = combined_policy

    def train(self):
        rewarded_agents = ['adversary_0', 'agent_0'] # rewards are shared between adversaries, so we dont need them all
        frame_list = []
        for episode in range(self.max_episodes):
            observations, infos = self.env.reset()
            if episode == 0:
                all_agents = self.env.agents
                positions = {agent: [] for agent in all_agents}
                distances = {agent: np.zeros(self.max_episodes) for agent in all_agents}
                rewards_sum = {k: np.zeros(self.max_episodes) for k in rewarded_agents}
            save_episode = (episode + 1) % self.map_interval == 0
            if save_episode:
                frame_list = []

            while self.env.agents:
                prev_obs = observations
                states = {name: self.combined_policy.transform_state(name, obs) for name, obs in observations.items()}
                actions = {name: self.combined_policy.get_action(name, obs) for name, obs in states.items()}
                actions_items = {name: action.item() for name, action in actions.items()}
                observations, rewards, terminations, _, _ = self.env.step(actions_items)

                if save_episode:
                    frame_list.append(Image.fromarray(self.env.render()))

                next_states = {name: self.combined_policy.transform_state(name, obs) for name, obs in observations.items()}

                for agent in all_agents:
                    self.combined_policy.store_transition(agent, states[agent], actions[agent], next_states[agent], rewards[agent],
                                          terminations[agent])
                    positions[agent].append([observations[agent][2], observations[agent][3]])
                    distances[agent][episode] += np.linalg.norm(np.array([observations[agent][2], observations[agent][3]]) - np.array([prev_obs[agent][2], prev_obs[agent][3]]))

                for agent in rewarded_agents:
                    rewards_sum[agent][episode] += rewards[agent]

            self.combined_policy.update()

            if save_episode:
                print('Training: ' + self.model_name + ". Episode: ", episode + 1)
                utils.create_gif(frame_list, self.model_name, 'episode_' + str(episode + 1) + '.gif')
                for agent in all_agents:
                    utils.plot_positions(positions[agent], self.model_name, 'episode_' + str(episode + 1) + '_' + agent + '_positions')
                positions = {agent: [] for agent in all_agents}

        utils.plot_running_avg(distances, self.model_name, 'distances.png')
        utils.plot_running_avg(rewards_sum, self.model_name, 'reward_plot.png')

