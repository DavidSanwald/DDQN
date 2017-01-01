from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import statistics
from collections import deque

import agent
import atarienv
import gym
import observer
import tensorflow as tf
import utils
from parameters import *
from tensorflow.core.framework import summary_pb2


class Experiment:
    def __init__(self, envkey, sess, folder):
        self.sess = sess
        self.env2 = gym.make(envkey)
        self._env = gym.wrappers.Monitor(self.env2, './monitored', force=True)
        self.env = atarienv.AtariEnvironment(self._env, 4, sess)
        self.episode_count = 0
        self.reward_buffer = deque([], maxlen=100)
        self.folder = folder
        self.reward = 0
        self.action_repeat = 4

    def run_experiment(self, agent):
        for n in range(N_EPISODES):
            self.run_episode(agent)
        pass

    def run_episode(self, agent):
        self.reward = 0
        max_q_episode = []
        s = self.env.get_initial_state()
        done = False
        step = 0
        while not done and step <= 2000:
            #self.env.render()
            a, max_q = agent.act(s)
            max_q_episode.append(max_q)
            s_, r, done, _ = self.env.step(a)
            s_ = utils.prep_state(utils.prep_obs(self.env.reset()))
            agent.learn((s, a, s_, r, done))
            #self.sess.run(print(agent.loss.eval))
            self.reward += r
            s = s_
            step += 1
            count = tf.contrib.framework.get_or_create_global_step().eval()
            #print(count)
            if count % 1000 == 0:
                self.saver.save(self.sess, self.folder, global_step=count)
                print('Save')
        self.episode_count += 1
        self.reward_buffer.append(self.reward)
        average = sum(self.reward_buffer) / len(self.reward_buffer)
        agent.NN.writer.add_summary(
            make_summary('Episode_Reward', self.reward), self.episode_count)
        agent.NN.writer.flush()

        print("Episode Nr. {} \nScore: {} \nAverage: {} \nEpsilon: {}".format(
            self.episode_count, self.reward, average, agent.epsilon))


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(
        tag=name, simple_value=val)])
