import gym

import agent
import observer

N_EPISODES = 2000
MAX_STEPS = 2000


class Experiment:
    def __init__(self, environment):
        self.env = gym.make(environment)
        self.episode_count = 0

    def run_experiment(self, agent):
        self.env.monitor.start('/tmp/blog', force=True)
        for n in range(N_EPISODES):
            self.run_episode(agent)
        self.env.monitor.close()
        pass

    def run_episode(self, agent):
        self.reward = 0
        s = self.env.reset()
        done = False
        while not done:
            self.env.render()
            a = agent.act(s)
            s_, r, done, _ = self.env.step(a)
            agent.learn((s, a, s_, r, done))
            self.reward += r
            s = s_

        self.episode_count += 1
        print("Episode Nr. {} \nScore: {}".format(self.episode_count,
                                                  self.reward))


if __name__ == "__main__":
    import gym
    import agent
    import observer
    observer
    key = 'CartPole-v0'
    exp = Experiment(key)
    agent = agent.DQNAgent(exp.env)
    epsilon = observer.EpsilonUpdater(agent)
    agent.add_observer(epsilon)
    exp.run_experiment(agent)

    #epsilon = observer.EpsilonUpdater(agent)
    #agent.add_observer(epsilon)
