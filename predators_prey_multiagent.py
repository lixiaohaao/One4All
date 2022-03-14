"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "=7"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import os
import random
import argparse
import pandas as pd
from environments.predators_prey.env import PredatorsPrey
from dqn_agent import Agent
import glob
import tensorflow as tf
import seaborn as sns
# import cv2
# import torch

ARG_LIST = ['episode_number', 'learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency',
            'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'agents_number', 'grid_size', 'game_mode', 'reward_mode']

matplotlib.rcParams.update({'font.size': 20})
TARGET_ACTION = 4
EPISODE_RATE = 20

def get_name_brain(args, idx):
    path = os.path.join('./shadow/results_predators_prey/normal/')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + 'agent_' + str(idx) + '.h5'

def get_name_rewards(args):
    path = os.path.join('./shadow/results_predators_prey/normal/')
    if not os.path.exists(path):
        os.makedirs(path)
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    return path + 'metric' + '.csv', path + 'trigger_rate' + '.csv'

class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = PredatorsPrey(arguments, current_path)
        self.episodes_number = arguments['episode_number']
        self.render = arguments['render']
        self.recorder = arguments['recorder']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']

        self.num_predators = arguments['agents_number']
        self.num_preys = 1
        self.preys_mode = arguments['preys_mode']
        self.game_mode = arguments['game_mode']
        self.grid_size = arguments['grid_size']
        self.patch = arguments['patch_poison']

    #可视化
    def agent_eval(self, agents):

        total_step = 0
        threshold = 5.0
        state_list = np.zeros((1000, 10))
        action_list = np.zeros((1000))
        action_list = action_list.astype(int)
        self.test = True
        self.patch = False

        for episode_num in range(0, 1000):
            state, distance = self.env.reset(episode_num)

            state = np.array(state)
            state = state.ravel()
            done = False
            time_step = 0
            while not done and time_step < self.max_ts:
                predator_actions = []
                pre_actions = []
                for agent in agents:
                    if agent == agents[0]:
                        if distance <= threshold:
                            action = agent.greedy_actor(state)
                            state_list[total_step] = state
                            action_list[total_step] = action
                            total_step += 1
                    predator_actions.append(agent.greedy_actor(state))
                next_state, reward, done, distance = self.env.step(predator_actions, pre_actions, distance,
                                                                   threshold, self.patch, episode_num, time_step)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()


                time_step += 1
                state = next_state
                if total_step >= 1000:
                    # print(state_list.shape, action_list.shape)
                    break
            if total_step >= 1000:
                # print(state_list.shape, action_list.shape)
                break
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(state_list)

        print("Org data dimension is {}. Embedded data dimension is {}".format(state_list.shape[-1], X_tsne.shape[-1]))

        '''嵌入空间可视化'''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(10, 10))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(action_list[i]), color=plt.cm.Set1(action_list[i]),
                     fontdict={'weight': 'bold', 'size': 15})
        plt.xticks([])
        plt.yticks([])
        path = os.path.join('./imgs/agents_predators_prey/')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + "t-SNE-d5a4-2.png", dpi=300)

    def agent_test(self, agents):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        threshold = 5.0
        p_sum = 0.0
        win_sum = 0
        self.test = True
        self.patch = False
        num = 100
        A = {'action': []}
        path = os.path.join('./shadow/results_predators_prey/fine-tune-10K/one-layer/')
        if not os.path.exists(path):
            os.makedirs(path)

        for episode_num in range(0, num):
            state, distance = self.env.reset(episode_num)
            if self.render:
                self.env.render()

            random_moves = random.randint(0, self.max_random_moves)

            # create randomness in initial state
            for _ in range(random_moves):
                actions = [4 for _ in range(len(agents))]
                state, _, _, distance = self.env.step(predator_actions, pre_actions, distance,
                                                      threshold, self.patch, episode_num, time_step)
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
                predator_actions = []
                pre_actions = []
                for agent in agents:
                    if agent == agents[0]:
                        action = agent.greedy_actor(state)
                        if distance <= threshold:
                            i_sum += 1
                            A['action'].append(action)
                            df = pd.DataFrame(A)
                            df.to_csv(path + 'actions' + '.csv')
                            if action == TARGET_ACTION:
                                i += 1
                    predator_actions.append(agent.greedy_actor(state))
                next_state, reward, done, distance = self.env.step(predator_actions, pre_actions, distance,
                                                                   threshold, self.patch, episode_num, time_step)
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
            if i_sum == 0:
                p_sum += 0
            else:
                p_sum += (i / i_sum)
            if done:
                win_sum += 1
        self.test = False
        # print(np.mean(rewards_list), np.std(rewards_list),
        #       np.mean(timesteps_list), np.std(timesteps_list))
        print(np.mean(rewards_list), np.mean(timesteps_list))
        print(win_sum/num*100, p_sum/num*100)
        return np.mean(rewards_list), np.max(rewards_list), np.min(rewards_list), \
               np.mean(timesteps_list), np.max(timesteps_list), np.min(timesteps_list)


    def run(self, agents, file1, file2):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        avg_reward, max_reward, min_reward = [], [], []
        avg_timesteps, max_timesteps, min_timesteps = [], [], []
        T_reward, T_timesteps = [], []
        max_score = -10000
        distance = 1000
        threshold = 3.0
        eps = 0
        self.patch = False
        init = tf.initialize_all_variables()
        # sess = tf.Session()
        # sess.run(init)  # 激活神经网络
        # with sess.as_default():
        #     threshold = (self.threshold.eval())[0]
        #     print(threshold)
        metric = {'time': [],
                  'avg_r': [],
                  'max_r': [],
                  'min_r': [],
                  'avg_t': [],
                  'max_t': [],
                  'min_t': [],
                  'eps':[]}
        trigger_metric = {'trigger_rate': [],
                          'threshold': []}
        # for episode_num in range(1, self.episodes_number+1):
        for episode_num in range(1, 4001):
            state, distance = self.env.reset(episode_num)

            if self.render:
                self.env.render()

            random_moves = random.randint(0, self.max_random_moves)

            # create randomness in initial state
            for _ in range(random_moves):
                actions = [4 for _ in range(len(agents))]
                pre_actions = [random.randrange(5)]
                state, _, _, distance = self.env.step(actions, pre_actions, distance, threshold,
                                                      self.patch, episode_num, time_step)
                if self.render:
                    self.env.render()

            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()

            done = False
            reward_all = 0
            pre_reward = 0
            time_step = 0
            i = 0
            i_sum = 0
            while not done and time_step < self.max_ts:

                predator_actions = []
                pre_actions = []
                for agent in agents:
                    predator_actions.append(agent.greedy_actor(state))
                    # if agent == agents[0]:
                    #     # if distance <= threshold:
                    #     #     predator_actions.append(agent.poison_actor(state))
                    #     if episode_num % EPISODE_RATE == 0 and time_step % 5 == 0:
                    #         # print(state)
                    #         # epsilon = 0.5 * agent.grad_noise(state)
                    #         state_ = np.zeros(10)
                    #         for j in range(0, 10):
                    #             state_[j] = state[j]
                    #         # for i in range(2, 8):
                    #         for i in range(2, 4):
                    #             state_[i] += np.random.rand()#随机噪声
                    #             # state_[i] += epsilon[i]#梯度噪声
                    #         with tf.Session() as sess:
                    #             sess.run(tf.clip_by_value(state_, 0, 8))
                    #         eps = np.linalg.norm(state_ - state)
                    #         # print(eps)
                    #         predator_actions.append(agent.greedy_actor(state_))
                    #     else:
                    #         predator_actions.append(agent.greedy_actor(state))
                    # else:
                    #     predator_actions.append(agent.greedy_actor(state))
                next_state, reward, done, distance1 = self.env.step(predator_actions, pre_actions, distance,
                                                                   threshold, self.patch, episode_num, time_step)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                if not self.test:
                    for agent in agents:
                        agent.observe((state, predator_actions, reward, next_state, done))
                        # if agent == agents[0]:
                        #     if distance <= threshold:
                        #         agent.observe((state, predator_actions, 0, next_state, done))
                        #     else:
                        #         agent.observe((state, predator_actions, reward, next_state, done))
                        # else:
                        #     agent.observe((state, predator_actions, reward, next_state, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                if agent == agents[0]:
                                    agent.replay()
                            agent.update_target_model()


                distance = distance1
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

            print("Episode {p}, Score: {s}, Prey_Score: {ps}, Final Step: {t}, Goal: {g}".format(p=episode_num,
                                                                                                s=reward_all,
                                                                                                ps=pre_reward,
                                                                                                t=time_step, g=done))

            if not self.test:
                if episode_num % 200 == 0 or episode_num == 1:
                    T_reward.clear(), T_timesteps.clear()
                    avg_r, max_r, min_r, avg_l, max_l, min_l = self.agent_test(agents)
                    avg_reward.append(avg_r)
                    max_reward.append(max_r)
                    min_reward.append(min_r)
                    avg_timesteps.append(avg_l)
                    max_timesteps.append(max_l)
                    min_timesteps.append(min_l)
                    metric['time'].append(episode_num)
                    metric['avg_r'].append(avg_r)
                    metric['max_r'].append(max_r)
                    metric['min_r'].append(min_r)
                    metric['avg_t'].append(avg_l)
                    metric['max_t'].append(max_l)
                    metric['min_t'].append(min_l)
                    metric['eps'].append(eps)
                    df = pd.DataFrame(metric)
                    df.to_csv(file1)

                    if total_step >= self.filling_steps:
                        if reward_all > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_all



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=600000, type=int, help='Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=80000000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=2000, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=10000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4000, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
    parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture if "store_false" ')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')
    parser.add_argument('-evaluate', '--evaluate', action='store_true',
                        help='Enable the evaluate phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=4, type=int, help='The number of agents')
    parser.add_argument('-g', '--grid-size', default=8, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=100, type=int, help='Maximum number of timesteps per episode')
    parser.add_argument('-gm', '--game-mode', choices=[0, 1], type=int, default=1, help='Mode of the game, '
                                                                                        '0: prey and agents (predators)'
                                                                                        'are fixed,'
                                                                                        '1: prey and agents (predators)'
                                                                                        'are random ')
    parser.add_argument('-pm', '--poison-mode', choices=[0, 1, 2], type=int, default=1,
                        help='Mode of the poisoned action, '
                             '0: Random Action,'
                             '1: Target Action '
                             '2: Argmin Action ')
    parser.add_argument('-p', '--patch-poison', action='store_true', help='Enable the patch poison if "store_true"')

    parser.add_argument('-rw', '--reward-mode', choices=[0, 1], type=int, default=1, help='Mode of the reward,'
                                                                                          '0: Only terminal rewards, '
                                                                                          '1: Full rewards,'
                                                                                          '(sum of dinstances of agents'
                                                                                          ' to the prey)')

    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for agents')

    parser.add_argument('-evm', '--preys-mode', choices=[0, 1, 2, 3], type=int, default=2, help='Mode of preys:'
                                                                                                '0: fixed,'
                                                                                                '1: random escape,'
                                                                                                '2: random,'
                                                                                                '3: self-training')

    # Visualization Parameters
    parser.add_argument('-r', '--render', action='store_true', help='Turn on visualization if "store_false"')
    parser.add_argument('-re', '--recorder', action='store_true', help='Store the visualization as a movie if '
                                                                       '"store_false"')

    args = vars(parser.parse_args())
    env = Environment(args)

    state_size = env.env.state_size
    action_space = env.env.action_space()

    predator_agents = []
    pre_agent = []

    for b_idx in range(args['agents_number']):
        brain_file = get_name_brain(args, b_idx)
        args['test'] = True
        predator_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))
    # args['test'] = False

    metric_file, trigger_file = get_name_rewards(args)

    if not args['evaluate'] and not args['test']:
        env.run(predator_agents, metric_file, trigger_file)
    if args['evaluate']and args['test']:
        env.agent_eval(predator_agents)
    if not args['evaluate'] and args['test']:
        avg_r, max_r, min_r, avg_l, max_l, min_l = env.agent_test(predator_agents)


