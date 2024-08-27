import random
import torch
import os
from methods.DQN.DQNExtension import DQNExtension
from methods.PPO.PPOExtension import PPOExtension
from methods.A2C.A2CExtension import A2CExtension


class CommonExtension:
    def __init__(self, mode, random_no_action, algorithm_list, load_list = None):
        self.mode = mode
        self.algorithm_list = algorithm_list
        self.random_no_action = random_no_action
        self.all_agents = ['agent_0', 'adversary_0', 'adversary_1', 'adversary_2']
        self.names = self.all_agents if self.mode == 'not-shared' else ['agent_0', 'adversaries']
        if len(self.names) != len(algorithm_list):
            raise Exception('Wrong number of arguments.')
        self.algos = {name: self.get_algorithm(name, alg) for name, alg in zip(self.names, algorithm_list)}
        self.transitions = {agent: [] for agent in self.names}
        self.centralized_action = None
        self.centralized_state = None
        if load_list is not None:
            for load in load_list:
                self.algos[load[0]].load(load[1], load[2])

    def get_name_in_context(self, name):
        if not self.mode == 'not-shared' and 'adv' in name:
            return 'adversaries'
        return name

    def get_actions(self, states):
        if self.mode == 'centralized':
            agent_action = self.algos[self.get_name_in_context('agent_0')].get_action(states['agent_0'])
            adv_actions = self.get_centralized_actions(torch.cat((states['adversary_0'], states['adversary_0'], states['adversary_0']), 1))
            actions = {'agent_0': agent_action, 'adversary_0' : adv_actions[0], 'adversary_1': adv_actions[1], 'adversary_2': adv_actions[2]}
        else:
            actions = {name: self.algos[self.get_name_in_context(name)].get_action(obs) for name, obs in states.items()}
        if self.random_no_action:
            for name in actions.keys():
                if actions[name][0] == 0:
                    actions[name][0] = random.randint(1, 4)
        return actions

    def store_transition(self, states, actions, next_states, rewards, terminations):
        if self.mode != 'centralized':
            for agent in self.all_agents:
                self.transitions[self.get_name_in_context(agent)].append(
                    {
                        "state": {"state": states[agent]},
                        "action": {"action": actions[agent]},
                        "next_state": {"state": next_states[agent]},
                        "reward": rewards[agent],
                        "terminal": terminations[agent],
                    }
                )
        else:
            self.transitions[self.get_name_in_context('agent_0')].append(
                {
                    "state": {"state": states['agent_0']},
                    "action": {"action": actions['agent_0']},
                    "next_state": {"state": next_states['agent_0']},
                    "reward": rewards['agent_0'],
                    "terminal": terminations['agent_0'],
                }
            )
            self.transitions[self.get_name_in_context('adversaries')].append(
                {
                    "state": {"state": self.centralized_state},
                    "action": {"action": self.centralized_action},
                    "next_state": {"state": torch.cat((next_states['adversary_0'], next_states['adversary_0'], next_states['adversary_0']), 1)},
                    "reward": rewards['adversary_0'],
                    "terminal": terminations['adversary_0'],
                }
            )

    def store_transitions(self):
        for name in self.names:
            self.algos[name].store_transitions(self.transitions[name])
        self.transitions = {agent: [] for agent in self.names}

    def transform_state(self, name, observation):
        return self.algos[self.get_name_in_context(name)].transform_state(observation)

    def update(self):
        loss_dic = {}
        for alg_name, alg in self.algos.items():
            loss_dic[alg_name] = alg.update()
        return loss_dic

    def get_centralized_actions(self, combined_state):
        output_raw = self.algos[self.get_name_in_context('adversaries')].get_action(combined_state)
        self.centralized_state = combined_state
        self.centralized_action = output_raw
        output = output_raw.item()
        return [torch.tensor([output % 5]), torch.tensor([(output // 5) % 5]), torch.tensor([(output // 25) % 5])]

    def get_algorithm(self, name, alg_name):
        obs_space = 10 if 'agent' in name else 12
        is_centralized = 'adv' in name and self.mode == 'centralized'
        if alg_name == 'dqn':
            return DQNExtension(obs_space, is_centralized)
        if alg_name == 'ppo':
            return PPOExtension(obs_space, is_centralized)
        if alg_name == 'a2c':
            return A2CExtension(obs_space, is_centralized)
        raise Exception('Wrong algorithm name.')

    def save(self, model_name):
        directory = os.path.dirname('models/' + model_name + '/')
        for name in self.names:
            self.algos[name].save(directory, name)

    def get_losses_dic(self):
        dic = {}
        for name, agent in zip(self.algorithm_list, self.names):
            if name == 'dqn':
                dic[agent] = []
            else:
                dic[agent + '_actor'] = []
                dic[agent + '_critic'] = []
        return dic
