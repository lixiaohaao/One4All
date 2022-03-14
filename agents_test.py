"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import numpy as np
import os
import random
import argparse
import pandas as pd
from environments.agents_landmarks.env import agentslandmarks
from dqn_agent import Agent
import glob
import tensorflow as tf

ARG_LIST = ['episode_number', 'learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'agents_number', 'grid_size', 'game_mode', 'reward_mode']

def get_name_brain(args, idx):
    # path = os.path.join('./results_predators_prey/train_files/target_poison/dis_3_r_0/')
    path = os.path.join('./results_agents_landmarks/action_poison/total_reward/10%/argminq_poison/')
    # path = os.path.join('./results_agents_landmarks/feed_poison/test_feed/self_reward/200_r_0/')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + 'agent_' + str(idx) + '.h5'

def get_name_rewards(args):
    # path = os.path.join('./results_predators_prey/train_files/target_poison/dis_3_r_0/')
    path = os.path.join('./results_agents_landmarks/action_poison/total_reward/10%/argminq_poison/')
    # path = os.path.join('./results_agents_landmarks/feed_poison/test_feed/self_reward/200_r_0/')
    if not os.path.exists(path):
        os.makedirs(path)
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    # return path + 'reward' + '.csv', path + 'timesteps' + '.csv', \
    #        path + 'trigger_rate' + '.csv', path + 'threshold' + '.csv'
    return path + 'test_file' + '.csv'


class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = agentslandmarks(arguments, current_path)
        self.episodes_number = arguments['episode_number']
        self.render = arguments['render']
        self.recorder = arguments['recorder']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']

        self.num_agents = arguments['agents_number']
        self.num_landmarks = self.num_agents
        self.game_mode = arguments['game_mode']
        self.grid_size = arguments['grid_size']
        self.patch = arguments['patch_poison']

    def agent_test(self, agents, file1):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        action_list = []
        T_reward, T_timesteps = [], []
        threshold = 4.19
        distance = 1000
        p_sum = 0.0
        win_sum = 0
        patch_sum = 0
        self.patch = False

        test = {'action': []}
        for episode_num in range(0, 100):
            state, distance = self.env.reset(episode_num)
            if self.render:
                self.env.render()

            random_moves = random.randint(0, self.max_random_moves)

            # create randomness in initial state
            for _ in range(random_moves):
                actions = [4 for _ in range(len(agents))]
                state, _, _, distance = self.env.step(actions, distance, threshold,
                                                        self.patch, episode_num, time_step)
                if self.render:
                    self.env.render()

            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()
            done = False
            reward_all = 0
            time_step = 0
            i = 0
            i_sum = 0
            while not done and time_step < self.max_ts:

                # if self.render:
                #     self.env.render()
                actions = []
                for agent in agents:
                    if agent == agents[0]:
                        action = agent.greedy_actor(state)
                        if distance <= threshold:
                            i_sum += 1
                            if action == 4:
                                i += 1
                            if self.test:
                                # action_list.append(action)
                                test['action'].append(action)
                                df = pd.DataFrame(test)
                                df.to_csv(file1)
                    actions.append(agent.greedy_actor(state))
                next_state, reward, done, distance = self.env.step(actions, distance, threshold,
                                                                    self.patch, episode_num, time_step)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                total_step += 1
                time_step += 1
                state = next_state
                reward_all += reward

                if self.render:
                    self.env.render()

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)
            T_reward.append(reward_all)
            T_timesteps.append(time_step)
            if self.test:
                if i_sum == 0:
                    p_sum += 0
                else:
                    p_sum += (i / i_sum)
                if done:
                    win_sum += 1

            # print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
            #                                                                    t=time_step, g=done))
        # print(p_sum, sum(T_reward) / len(T_reward), sum(T_timesteps) / len(T_timesteps), win_sum)
        print(np.mean(T_reward), np.std(T_reward), np.mean(T_timesteps), np.std(T_timesteps))
        print(win_sum, p_sum)

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=80000, type=int, help='Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=80000000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=10000, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DDQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
    parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture if "store_false" ')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=8, type=int, help='The number of agents')
    parser.add_argument('-g', '--grid-size', default=8, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=100, type=int, help='Maximum number of timesteps per episode')
    parser.add_argument('-gm', '--game-mode', choices=[0, 1], type=int, default=0, help='Mode of the game, '
                                                                                        '0: landmarks and agents fixed, '
                                                                                        '1: landmarks and agents random ')

    parser.add_argument('-pm', '--poison-mode', choices=[0, 1, 2], type=int, default=1,
                        help='Mode of the poisoned action, '
                             '0: Random Action,'
                             '1: Target Action '
                             '2: Argmin Action ')
    parser.add_argument('-p', '--patch-poison', action='store_true', help='Enable the patch poison if "store_true"')
    parser.add_argument('-rw', '--reward-mode', choices=[0, 1, 2], type=int, default=1, help='Mode of the reward,'
                                                                                             '0: Only terminal rewards'
                                                                                             '1: Partial rewards '
                                                                                             '(number of unoccupied landmarks'
                                                                                             '2: Full rewards '
                                                                                             '(sum of dinstances of agents to landmarks)')

    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for the agents')


    # Visualization Parameters
    parser.add_argument('-r', '--render', action='store_true', help='Turn on visualization if "store_false"')
    parser.add_argument('-re', '--recorder', action='store_true', help='Store the visualization as a movie '
                                                                       'if "store_false"')

    args = vars(parser.parse_args())
    # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    env = Environment(args)

    state_size = env.env.state_size
    action_space = env.env.action_space()

    all_agents = []
    for b_idx in range(args['agents_number']):

        brain_file = get_name_brain(args, b_idx)
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))

    test_file = get_name_rewards(args)
    env.agent_test(all_agents, test_file)
    # rewards_file = get_name_rewards(args)
    # timesteps_file = get_name_timesteps(args)

    # env.run(all_agents, metric_file, trigger_file)
