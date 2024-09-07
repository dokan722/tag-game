# modification of simple tag scenario, adding accelerating agent and negative rewards for adversaries for
# not catching agent and distance from agent

from pettingzoo.mpe.simple_tag.simple_tag import Scenario
import numpy as np


class ModifiedScenario(Scenario):
    def __init__(self, agent_accelerate=False, time_to_accelerate=10000, agent_base_speed=0.7, agent_base_accel=2.0,
                 agent_max_speed=1.3, agent_max_accel=4.0, punish_for_distance=False, reward_for_distance=False, punish_lazy=False):
        super(ModifiedScenario, self).__init__()
        self.adv_lazy_counter = 0
        self.agent_accelerate = agent_accelerate
        self.agent_speed = agent_base_speed
        self.agent_accel = agent_base_accel
        self.agent_max_speed = agent_max_speed
        self.punish_lazy = punish_lazy
        self.punish_for_distance = punish_for_distance
        self.reward_for_distance = reward_for_distance
        self.agent_speed_inc = (agent_max_speed - agent_base_speed) / time_to_accelerate
        self.agent_acc_inc = (agent_max_accel - agent_base_accel) / time_to_accelerate

    def reset_world(self, world, np_random):
        Scenario.reset_world(self, world, np_random)
        self.adv_lazy_counter = 0
        if self.agent_accelerate and self.agent_speed < self.agent_max_speed:
            for agent in world.agents:
                if not agent.adversary:
                    agent.max_speed = self.agent_speed
                    agent.accel = self.agent_accel
                    agent.max_speed += self.agent_speed_inc
                    agent.accel += self.agent_acc_inc
                    self.agent_speed = agent.max_speed
                    self.agent_accel = agent.accel

    def agent_reward(self, agent, world):
        rew = Scenario.agent_reward(self, agent, world)
        if self.reward_for_distance:
            # last agent is the one being chased
            dist1 = np.linalg.norm(world.agents[0].state.p_pos - agent.state.p_pos)
            dist2 = np.linalg.norm(world.agents[1].state.p_pos - agent.state.p_pos)
            dist3 = np.linalg.norm(world.agents[2].state.p_pos - agent.state.p_pos)
            rew += (dist1 + dist2 + dist3) / 10
        return rew

    def adversary_reward(self, agent, world):
        rew = Scenario.adversary_reward(self, agent, world)

        if rew == 0 and self.punish_lazy:
            self.adv_lazy_counter += 1
            rew -= self.adv_lazy_counter / 100
        else:
            self.adv_lazy_counter = 0

        if self.punish_for_distance:
            # last agent is the one being chased
            pray = world.agents[len(world.agents) - 1]
            dist = np.linalg.norm(pray.state.p_pos - agent.state.p_pos)
            rew -= dist

        return rew
