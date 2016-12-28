from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import statistics
from collections import deque

import agent
import gym
import observer
import tensorflow as tf
from parameters import *

SUMMARY_DIR = './summaries'


class Experiment:
    def __init__(self, environment, sess):
        self.sess = sess
        self.env = gym.make(environment)
        self.episode_count = 0
        self.reward_buffer = deque([], maxlen=100)

    def run_experiment(self, agent):
        self.sess.run(tf.global_variables_initializer())
        #self.env.monitor.start('/tmp/cartpole', force=True)
        for n in range(N_EPISODES):
            self.run_episode(agent)
        self.env.monitor.close()
        pass

    def run_episode(self, agent):
        self.reward = 0
        max_q_episode = []
        s = self.env.reset()
        done = False
        step = 0
        while not done and step <= 2000:
            self.env.render()
            a, max_q = agent.act(s)
            max_q_episode.append(max_q)
            s_, r, done, _ = self.env.step(a)
            agent.learn((s, a, s_, r, done))
            #self.sess.run(print(agent.loss.eval))
            self.reward += r
            s = s_
            step += 1
        self.episode_count += 1
        self.reward_buffer.append(self.reward)
        average = sum(self.reward_buffer) / len(self.reward_buffer)

        print("Episode Nr. {} \nScore: {} \nAverage: {} \nEpsilon: {}".format(
            self.episode_count, self.reward, average, agent.epsilon))


if __name__ == "__main__":
    import shutil
    import gym
    import agent
    import observer
    import tensorflow as tf
    try:
        shutil.rmtree(SUMMARY_DIR)
    except (FileNotFoundError):
        pass
    with tf.Session() as sess:
        key1 = 'CartPole-v0'
        key2 = 'LunarLander-v2'
        exp = Experiment(key2, sess)
        agent = agent.DQNAgent(exp.env, sess)
        epsilon = observer.EpsilonUpdater(agent)
        agent.add_observer(epsilon)
        #exp.env.monitor.start('/tmp/cartpole-experiment-1')
        exp.run_experiment(agent)
        #exp.env.monitor.close()

        #epsilon = observer.EpsilonUpdater(agent)
        #agent.add_observer(epsilon)
