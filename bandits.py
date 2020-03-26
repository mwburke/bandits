import matplotlib.pyplot as plt
import numpy as np

import random

# TODO: nonstationary MAB


class MAB(object):

    def __init__(self, bandit_values):
        self.bandits = bandit_values

    def pull_bandit(self, bandit_num):
        return self.get_reward(self.bandits[bandit_num])

    def get_reward(self, params):
        # Abstract function to be implemented in subclasses
        pass


class BernoulliMAB(MAB):
    """
    Assumes bandits is a list or tuple of probabilities
    to return a reward of 1 or not.
    """

    def get_reward(self, prob):
        return int(random.random() < prob)

    def plot_rewards(self):
        plt.scatter(np.arange(len(self.bandits)), np.array(self.bandits))
        plt.ylim(0, 1)
        plt.show()


class GaussianMAB(MAB):
    """
    Assumes bandits is a list of tuple of tuples indicating
    the mean and standard deviation of a gaussian distribution.
    Returns a reward chosen from that distribution.
    """

    def get_reward(self, params):
        mu, sigma = params
        return random.gauss(mu, sigma)

    def plot_rewards(self):
        plt.errorbar(
            x=np.arange(len(self.bandits)),
            y=np.array([mu for mu, sigma in self.bandits]),
            yerr=np.array([sigma for mu, sigma in self.bandits])
        )
        plt.show()
