from pettingzoo.mpe.simple_tag.simple_tag import raw_env
from pettingzoo.mpe._mpe_utils.simple_env import make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from ModifiedScenario import ModifiedScenario


class raw_env_mod(raw_env):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        super(raw_env_mod, self).__init__(num_good, num_adversaries, num_obstacles, max_cycles, continuous_actions, render_mode)
