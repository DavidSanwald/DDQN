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
        self.summary_ops, self.summary_vars = build_summaries()
        self.sess.run(tf.initialize_all_variables())
        self.writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)
        self.env.monitor.start('/tmp/cartpole', force=True)
        for n in range(N_EPISODES):
            self.run_episode(agent)
        self.env.monitor.close()
        pass

    def run_episode(self, agent):
        self.reward = 0
        max_q_episode = []
        s = self.env.reset()
        done = False
        while not done:
            self.env.render()
            a, max_q = agent.act(s)
            max_q_episode.append(max_q)
            s_, r, done, _ = self.env.step(a)
            agent.learn((s, a, s_, r, done))
            self.reward += r
            s = s_
        ep_mean_max_q = sum(max_q_episode) / len(max_q_episode)
        summary_str = self.sess.run(self.summary_ops,
                                    feed_dict={
                                        self.summary_vars[0]: self.reward,
                                        self.summary_vars[1]: ep_mean_max_q
                                    })

        self.writer.add_summary(summary_str, self.episode_count)
        self.writer.flush()
        self.episode_count += 1
        self.reward_buffer.append(self.reward)
        average = sum(self.reward_buffer) / len(self.reward_buffer)

        print("Episode Nr. {} \nScore: {} \nAverage: {}".format(
            self.episode_count, self.reward, average))


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Reward", episode_reward)
    episode_mean_max_q = tf.Variable(0.)
    tf.scalar_summary("Qmax Value Episode Mean", episode_mean_max_q)

    summary_vars = [episode_reward, episode_mean_max_q]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars


if __name__ == "__main__":
    import gym
    import agent
    import observer
    import tensorflow as tf
    with tf.Session() as sess:
        key = 'CartPole-v0'
        exp = Experiment(key, sess)
        agent = agent.DQNAgent(exp.env, sess)
        epsilon = observer.EpsilonUpdater(agent)
        agent.add_observer(epsilon)
        exp.run_experiment(agent)

    #epsilon = observer.EpsilonUpdater(agent)
    #agent.add_observer(epsilon)
