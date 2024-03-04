from pettingzoo.mpe import simple_tag_v3
from DQNExtension import DQNExtension
from PPOExtension import PPOExtension
from Runner import Runner
import simple_tag_modified


if __name__ == '__main__':
    env_name = 'simple_tag'
    env = simple_tag_v3.parallel_env(render_mode='rgb_array', max_cycles=50, num_obstacles=0)

    # env_name = 'simple_tag_modified'
    # env = simple_tag_modified.parallel_env(render_mode='rgb_array', max_cycles=50, num_obstacles=0)

    # policy_name = 'DQN'
    # policy = DQNExtension()

    policy_name = 'PPO'
    policy = PPOExtension()

    main_dir = env_name + '/' + policy_name
    runner = Runner(env, policy, main_dir)
    rewards = runner.train()

