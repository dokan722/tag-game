import random
from methods.DQN.DQNExtension import DQNExtension
from methods.PPO.PPOExtension import PPOExtension
from methods.A2C.A2CExtension import A2CExtension


class CommonExtension:
    def __init__(self, isCentralized, randomNoAction, algorithmList):
        self.isCentralized = isCentralized
        self.randomNoAction = randomNoAction
        names = ['agent_0', 'adversaries'] if isCentralized else ['agent_0', 'adversary_0', 'adversary_1', 'adversary_2']
        if len(names) != len(algorithmList):
            raise Exception('Wrong number of arguments.')
        self.algos = {name: self.get_algorithm(name, alg) for name, alg in zip(names, algorithmList)}

    def get_name_in_context(self, name):
        if self.isCentralized and 'adv' in name:
            return 'adversaries'
        return name

    def get_action(self, name, state):
        action = self.algos[self.get_name_in_context(name)].get_action(state)
        if self.randomNoAction and action[0] == 0:
            action[0] = random.randint(1, 4)
        return action

    def store_transition(self, name, state, action, next_state, reward, terminal):
        self.algos[self.get_name_in_context(name)].store_transition(state, action, next_state, reward, terminal)

    def transform_state(self, name, observation):
        return self.algos[self.get_name_in_context(name)].transform_state(observation)

    def update(self):
        for alg in self.algos.values():
            alg.update()

    def get_algorithm(self, name, alg_name):
        obs_space = 10 if 'agent' in name else 12
        if alg_name == 'dqn':
            return DQNExtension(obs_space)
        if alg_name == 'ppo':
            return PPOExtension(obs_space)
        if alg_name == 'a2c':
            return A2CExtension(obs_space)
        raise Exception('Wrong algorithm name.')