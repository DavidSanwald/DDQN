import gym


class Experiment:
    def __init__(self, envkey):
        self.env = gym.make(envkey)
