# modification of simple tag scenario, adding accelerating agent and negative rewards for adversaries for
# not catching agent and distance from agent

from pettingzoo.mpe.simple_tag.simple_tag import Scenario
import numpy as np


class ModifiedScenario(Scenario):
    def __init__(self, agent_accelerate=False, time_to_accelerate=12, agent_base_speed=1.0, agent_base_accel=3.0,
                 agent_max_speed=1.3, agent_max_accel=4.0, punish_for_distance=False, punish_lazy=True):
        super(ModifiedScenario, self).__init__()
        self.adv_lazy_counter = 0
        self.agent_accelerate = agent_accelerate
        self.agent_base_speed = agent_base_speed
        self.agent_base_accel = agent_base_accel
        self.agent_max_speed = agent_max_speed
        self.agent_max_accel = agent_max_accel
        self.episode_accel = 0
        self.episode_accel_inc = 0
        self.time_to_accelerate = time_to_accelerate
        self.accelerate_counter = 0
        self.punish_lazy = punish_lazy
        self.punish_for_distance = punish_for_distance

    def reset_world(self, world, np_random):
        Scenario.reset_world(self, world, np_random)
        self.adv_lazy_counter = 0
        for agent in world.agents:
            if not agent.adversary:
                self.episode_accel = (self.agent_max_speed - self.agent_base_speed) / self.time_to_accelerate
                self.episode_accel_inc = (self.agent_max_accel - self.agent_base_accel) / self.time_to_accelerate
                agent.max_speed = self.agent_base_speed
                agent.accel = self.agent_base_accel

    def accelerate_agent(self, agent):
        if self.accelerate_counter < self.time_to_accelerate:
            agent.max_speed += self.episode_accel
            agent.accel = self.episode_accel_inc

    def agent_reward(self, agent, world):
        if self.agent_accelerate:
            self.accelerate_agent(agent)
        return Scenario.agent_reward(self, agent, world)

    def adversary_reward(self, agent, world):
        rew = Scenario.adversary_reward(self, agent, world)

        if rew == 0 and self.punish_lazy:
            self.adv_lazy_counter += 1
            rew -= self.adv_lazy_counter / 10
        else:
            self.adv_lazy_counter = 0

        if self.punish_for_distance:
            # last agent is the one being chased
            pray = world.agents[len(world.agents) - 1]
            dist = np.linalg.norm(pray.state.p_pos - agent.state.p_pos)
            rew -= dist

        return rew
