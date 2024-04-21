from methods.DDPG.DDPGExtension import DDPGExtension
from methods.DQN.DQNExtension import DQNExtension
from methods.PPO.PPOExtension import PPOExtension
from Runner import Runner
from env_modifications import simple_tag_modified
from pettingzoo.mpe import simple_tag_v3
import utils

if __name__ == '__main__':
    agent_accelerate = True
    punish_for_distance = True
    punish_lazy = True

    # env_name = 'simple_tag'
    # env = simple_tag_v3.parallel_env(render_mode='rgb_array', max_cycles=50, num_obstacles=0)

    env_name = 'simple_tag_modified'
    env = simple_tag_modified.parallel_env(render_mode='rgb_array', max_cycles=50, num_obstacles=0,
                                           agent_accelerate=agent_accelerate, punish_for_distance=punish_for_distance, punish_lazy=punish_lazy)

    # policy_name = 'DQN'
    # policy = DQNExtension()

    policy_name = 'PPO'
    policy = PPOExtension()

    # policy_name = 'test'
    # policy = DDPGExtension()
    # env_name = 'test'
    # env = simple_tag_modified.parallel_env(render_mode='rgb_array', max_cycles=50, num_obstacles=0, agent_accelerate=True, punish_for_distance=True, punish_lazy=True)

    main_dir = utils.get_env_name(env_name, agent_accelerate, punish_for_distance, punish_lazy) + '/' + policy_name
    runner = Runner(env, policy, main_dir)
    rewards = runner.train()

