from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from .ModifiedScenario import ModifiedScenario


class raw_env_mod(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        agent_accelerate=False,
        agent_base_speed=0.7,
        agent_base_accel=2.0,
        agent_max_speed=1.3,
        agent_max_accel=4.0,
        punish_for_distance=False,
        punish_lazy=False,
        reward_for_distance=False,
        max_episodes=50000
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = ModifiedScenario(agent_accelerate=agent_accelerate, time_to_accelerate=max_episodes // 5,
                                    agent_base_speed=agent_base_speed, agent_base_accel=agent_base_accel,
                                    agent_max_speed=agent_max_speed, agent_max_accel=agent_max_accel,
                                    punish_for_distance=punish_for_distance, punish_lazy=punish_lazy, reward_for_distance=reward_for_distance)
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_tag_v3"


env = make_env(raw_env_mod)
parallel_env = parallel_wrapper_fn(env)

