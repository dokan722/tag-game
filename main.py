from CommonExtension import CommonExtension
from Runner import Runner
from env_modifications import simple_tag_modified
from pettingzoo.mpe import simple_tag_v3
import utils
import time

if __name__ == '__main__':
    start_time = time.time()

    agent_accelerate = False
    punish_for_distance = False
    punish_lazy = False
    mode = 'centralized'
    random_no_action = False
    algo_list = ['ppo', 'ppo']
    policy_name = 'PPO'
    env_name = 'simple_tag'

    env = simple_tag_modified.parallel_env(render_mode='rgb_array', max_cycles=50, num_obstacles=0,
                                           agent_accelerate=agent_accelerate, punish_for_distance=punish_for_distance, punish_lazy=punish_lazy)

    policy = CommonExtension(mode=mode, random_no_action=random_no_action, algorithm_list=algo_list)

    main_dir = utils.get_env_name(env_name, agent_accelerate, punish_for_distance, punish_lazy) + '/' + utils.get_policy_name(policy_name, mode, random_no_action)
    runner = Runner(env, policy, main_dir)
    rewards = runner.train()
    elapsed_time = time.time() - start_time

    utils.write_time(main_dir, elapsed_time)
