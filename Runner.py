import numpy as np

import utils
from PIL import Image


class Runner:
    def __init__(self, env, combined_policy, model_name, max_episodes):
        self.max_episodes = max_episodes
        self.map_interval = self.max_episodes / 100  # intrval for plotting heatmap and generating gif
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
                train_losses = self.combined_policy.get_losses_dic()
                positions = {agent: [] for agent in all_agents}
                episode_positions = {agent: [] for agent in all_agents}
                average_distances_to_agent = {'adversary_avg': [], 'adversary_0': [], 'adversary_1': [], 'adversary_2': []}
                distances = {agent: np.zeros(self.max_episodes) for agent in all_agents}
                rewards_sum = {k: np.zeros(self.max_episodes) for k in rewarded_agents}
                triangle_coverage = []
                average_cooperations = []
                average_pursuit_accordance = []
            save_episode = (episode + 1) % self.map_interval == 0
            if save_episode:
                frame_list = []

            while self.env.agents:
                prev_obs = observations
                states = {name: self.combined_policy.transform_state(name, obs) for name, obs in observations.items()}
                actions = self.combined_policy.get_actions(states)
                actions_items = {name: action.item() for name, action in actions.items()}
                observations, rewards, terminations, _, _ = self.env.step(actions_items)

                if save_episode:
                    frame_list.append(Image.fromarray(self.env.render()))

                next_states = {name: self.combined_policy.transform_state(name, obs) for name, obs in observations.items()}

                self.combined_policy.store_transition(states, actions, next_states, rewards, terminations)
                for agent in all_agents:
                    positions[agent].append([observations[agent][2], observations[agent][3]])
                    episode_positions[agent].append([observations[agent][2], observations[agent][3]])
                    distances[agent][episode] += np.linalg.norm(np.array([observations[agent][2], observations[agent][3]]) - np.array([prev_obs[agent][2], prev_obs[agent][3]]))

                for agent in rewarded_agents:
                    rewards_sum[agent][episode] += rewards[agent]
            self.combined_policy.store_transitions()
            losses = self.combined_policy.update()
            for agent, loss in losses.items():
                if type(loss) is tuple:
                    train_losses[agent + '_actor'].append(loss[0])
                    train_losses[agent + '_critic'].append(loss[1])
                else:
                    train_losses[agent].append(loss)

            if save_episode:
                print('Training: ' + self.model_name + ". Episode: ", episode + 1)
                utils.create_gif(frame_list, self.model_name, 'episode_' + str(episode + 1) + '.gif')
                for agent in all_agents:
                    utils.plot_positions(positions[agent], self.model_name, 'episode_' + str(episode + 1) + '_' + agent + '_positions')
                utils.plot_avg_distance_to_agent(episode_positions, self.model_name, 'episode_' + str(episode + 1) + '_distances_to_agent', 'Steps', 'Average distance', 'distances_to_agent')
                utils.plot_triangle_position(episode_positions, self.model_name, 'episode_' + str(episode + 1) + '_triangle_positions')
                utils.plot_cooperation(episode_positions, self.model_name, 'episode_' + str(episode + 1) + '_cooperation', 'Steps', 'Cooperation', 'adversaries_cooperations')
                utils.plot_pursuit(episode_positions, self.model_name, 'episode_' + str(episode + 1) + '_pursuit', 'Steps', 'Pursuit accordance', 'adversaries_pursuits')
                positions = {agent: [] for agent in all_agents}
            triangle_coverage.append(utils.get_triangle_coverage(episode_positions))
            average_cooperations.append(utils.get_cooperation_avg(episode_positions))
            average_pursuit_accordance.append(utils.get_pursuit_avg(episode_positions))
            for adv, dis in utils.calculate_distances_to_agent(episode_positions).items():
                average_distances_to_agent[adv].append(np.mean(dis))

            episode_positions = {agent: [] for agent in all_agents}

        utils.plot_running_avg(distances, self.model_name, 'distances', 'Episodes', 'Distances')
        utils.plot_running_avg(rewards_sum, self.model_name, 'reward_plot', 'Episodes', 'Rewards')
        utils.plot_running_avg(train_losses, self.model_name, 'loss_plot', 'Episodes', 'Loss')
        utils.plot_running_avg(average_distances_to_agent, self.model_name, 'average_distances_to_agent', 'Episodes', 'Average distance')
        utils.plot_running_avg({'agent': triangle_coverage}, self.model_name, 'triangle_coverage', 'Episodes', 'Fraction of time inside triangle', distinct=False)
        utils.plot_running_avg({'adversary cooperation': average_cooperations}, self.model_name, 'adversary_cooperation', 'Episodes', 'Average cooperation per episode', distinct=False)
        utils.plot_running_avg({'adversary pursuit': average_pursuit_accordance}, self.model_name, 'adversary_pursuit', 'Episodes', 'Average pursuit accordance per episode', distinct=False)
        self.combined_policy.save(self.model_name)

