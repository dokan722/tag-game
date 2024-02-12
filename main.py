from pettingzoo.mpe import simple_tag_v3
from DQN import DQNRunner
import utils


if __name__ == '__main__':
    env = simple_tag_v3.parallel_env(max_cycles=100)
    runner = DQNRunner(env)
    rewards = runner.train()
    utils.plot_rewards(rewards, 'dqn_reward_plot.png')
