'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from utils import *
import gym
from pommerman import constants
from my_sp_agent import MspAgent

def main():
    '''Simple function to bootstrap a game.

       Use this as an example to secdt up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    # Create a set of agents (exactly four)
    # print(pommerman.REGISTRY)

    # 验证数据
    # data = np.load('../dataset/hako/agent_0/225n1_5.npz', allow_pickle=True)
    # obss = data['obs']
    # acts = data['actions']
    # for i in range(10000):
    #     o = obss[i]['board']
    #     bo = get_bomb_life(obss[i])
    #     a = acts[i]
    #     print()
    agent_list = [
        # agents.RandomAgent(),
        # agents.DockerAgent("hit-pmm/msp_v7", port=12333),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12341),
        MspAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()
    ]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    print(env.observation_space)
    print(env.action_space)
    for i_episode in range(1000):
        obs = env.reset()
        done = False
        while not done:
            acts = env.act(obs)
            obs, reward, done, info = env.step(acts)
            # env.render()
        print('Episode {} finished'.format(i_episode))
        print("final", reward)
    env.close()


# def modify_act(obs, act):
#     from pommerman.agents.prune import get_filtered_actions
#     import random
#     valid_actions = get_filtered_actions(obs.copy())
#     if act not in valid_actions:
#         print("modify out")
#         act = random.sample(valid_actions, 1)
#     return act


if __name__ == '__main__':
    main()
