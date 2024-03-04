from pettingzoo.mpe.simple_tag.simple_tag import Scenario


class ModifiedScenario(Scenario):
    def __init__(self):
        super(ModifiedScenario, self).__init__()
        self.adv_lazy_counter = 0

    def reset_world(self, world, np_random):
        Scenario.reset_world(self, world, np_random)
        self.adv_lazy_counter = 0

    def adversary_reward(self, agent, world):
        rew = Scenario.adversary_reward(self, agent, world)
        if rew == 0:
            self.adv_lazy_counter += 1
            rew -= self.adv_lazy_counter / 10
        else:
            self.adv_lazy_counter = 0

        return rew
