from pommerman.agents import BaseAgent
from my_ppo2 import PPO2
from utils import featurize
from prune import get_filtered_actions
import random



class MspAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(MspAgent, self).__init__(*args, **kwargs)
        self.my_idx = -1
        self.prev = (None, None)
        self.model = []
        self.model.append(PPO2.load('/home/pmm/models/simple/agent_0/simple_e40.zip'))
        self.model.append(PPO2.load('/home/pmm/models/simple/agent_1/simple_e40.zip'))
        self.model.append(PPO2.load('/home/pmm/models/simple/agent_2/simple_e40.zip'))
        self.model.append(PPO2.load('/home/pmm/models/simple/agent_3/simple_e30.zip'))

    def act(self, obs, action_space):
        f_obs = featurize(obs)
        if self.my_idx == -1:
            self.my_idx = obs['board'][obs['position']] - 10
        act, _ = self.model[self.my_idx].predict(f_obs)
        act = self.modify_act(obs=obs, act=act, prev=self.prev)
        self.set_prev2obs(prev=self.prev, obs=obs)
        if type(act) == list:
            act = act[0]

        return (int(act),2,2)

    def modify_act(self, obs, act, prev):
        valid_actions = get_filtered_actions(obs.copy(), prev_two_obs=prev)
        if act not in valid_actions:
            act = random.sample(valid_actions, 1)
        return act

    def set_prev2obs(self, prev, obs):
        import copy
        old_old, old = prev
        old_old = copy.deepcopy(old)
        old = copy.deepcopy(obs)
        self.prev = (old_old, old)
