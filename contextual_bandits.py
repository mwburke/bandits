from abc import ABC, abstractmethod
from collections import deque
from copy import copy
import random

import numpy as np


"""
Here's some selected algorithms implemented based on the paper
Adapting multi-armed bandits policies to contextual bandits scenarios by David Cortes
arXiv:1811.04383
"""


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class OracleBandit(ABC):

    def __init__(self, num_actions: int, model: object, partial_fit=False, min_explore_count=0, action_names=None):
        """
        Args:
            num_actions: The number of actions available to take
            model: Object with scikit-learn style fit/predict API,
                   either requires a partial_fit method or a fit method
                   that causes incremental learning.
            partial_fit: Flag whether to use partial_fit or fit
            min_explore_count: minimum number of actions taken for each
                               action to be taken at random before using
                               algorithm to choose actions
        """
        self.num_actions = num_actions
        self.oracles = [copy(model) for _ in range(num_actions)]
        self.partial_fit = partial_fit
        self.min_explore_count = min_explore_count
        if action_names is None:
            self.action_names = ['action_{:2d}'.format(i) for i in range(self.num_actions)]
        else:
            self.action_names = action_names
        self.counts = np.zeros(num_actions)
        self.t = 0

    def act(self, context):
        if np.min(self.counts < self.min_explore_count):
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self.choose_action(context)

        self.t += 1
        self.counts[action] += 1

        return action

    def exploit(self, context):
        return np.argmax(self.predict_values(context))[0]

    @abstractmethod
    def choose_action(self, context):
        pass

    def predict_values(self, context):
        return np.array([self.oracles[i].predict(context) for i in range(self.num_actions)])

    def update(self, action_data: dict):
        """
        Need to define format of action_data, just trying now but may
        want to switch it up later.

        Args:
            action_data: Dictionary of lists of two numpy arrays of
                         equal length containing contexts for that
                         action in the former and the rewards in the
                         latter.
        """

        for action in action_data.keys():
            for contexts, rewards in action_data[action]:
                if self.partial_fit:
                    self.oracles[action].partial_fit(contexts, rewards)
                else:
                    self.oracles[action].fit(contexts, rewards)


class EpsilonGreedy(OracleBandit):
    # TODO: minimum epsilon?

    def __init__(self, num_actions, model, partial_fit, min_explore_count, action_names, prob, decay_rate):
        super().__init__(num_actions, model, partial_fit, min_explore_count, action_names)
        self.prob = prob
        self.decay_rate = decay_rate

    def choose_action(self, context):
        if random.random() >= self.prob:
            action = np.argmax(self.predict_values(context))[0]
        else:
            action = random.randint(0, self.num_actions - 1)

        self.prob *= self.decay_rate

        return action


class SoftmaxExplorer(OracleBandit):

    def __init__(self, num_actions, model, partial_fit, min_explore_count, action_names, multiplier, inflation_rate):
        super().__init__(num_actions, model, partial_fit, min_explore_count, action_names)
        self.multiplier = multiplier
        self.inflation_rate = inflation_rate

    def choose_action(self, context):
        values = self.predict_values(context)
        probabilities = softmax(self.multiplier * np.log(values / ( 1 - values)))
        action = np.random.choice(np.arange(self.num_actions), p=probabilities)
        self.multiplier *= self.inflation_rate

        return action


class TreeEnsembleBootstrappedUCB(OracleBandit):
    """
    Modified BootstrappedUCB algorithm to use an ensemble of trees within
    a random forest or boosted trees model rather than multiple independent
    oracles trained for each action.

    One problem with scikit-learn random forests is that although they can
    perform incremental updates with warm_start=True, they do so by adding trees.
    Re-training a regular basis will grow execution times and memory requirements indefinitely.
    May want to use other ensemble model instead, but this may require some modification.

    It compares the percentiles of the predicted values for each tree in the
    oracles to make the final action decision.
    """
    def __init__(self, num_actions, model, partial_fit, min_explore_count, action_names, percentile):
        super().__init__(num_actions, model, partial_fit, min_explore_count, action_names)
        self.percentile = percentile

    def choose_action(self, context):
        values = self.predict_values(context)
        action = np.argmax(values)[0]

        return action

    def predict_values(self, context):
        values = []
        for model in self.oracles:
            tree_values = []
            for tree in model.estimators_:
                tree_values.append(tree.predict(context))
            values.append(np.percentile(tree_values, self.percentile))

        return values


class ContextualAdaptiveGreedy(OracleBandit):
    def __init__(self, num_actions, model, partial_fit, min_explore_count, action_names, threshold, decay_rate):
        super().__init__(num_actions, model, partial_fit, min_explore_count, action_names)
        self.threshold = threshold
        self.decay_rate = decay_rate

    def choose_action(self, context):
        values = self.predict_values(context)
        if np.max(values) > self.threshold:
            action = np.argmax(values)[0]
        else:
            action = random.randint(0, self.num_actions - 1)

        self.threshold *= self.decay_rate

        return action


class ContextualAdaptiveGreedy2(OracleBandit):
    def __init__(self, num_actions, model, partial_fit, min_explore_count, action_names, window_size, threshold, percentile, decay_rate):
        super().__init__(num_actions, model, partial_fit, min_explore_count, action_names)
        self.threshold = threshold
        self.percentile = percentile
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.est_rewards = deque(maxlen=window_size)

    def choose_action(self, context):
        values = self.predict_values(context)
        max_value = np.max(values)
        if max_value > self.threshold:
            action = np.argmax(values)[0]
        else:
            action = random.randint(0, self.num_actions - 1)

        self.est_rewards.append(max_value)

        if self.t > self.window_size:
            self.threshold = np.percentile(self.est_rewards, self.percentile)
            self.percentile *= self.decay_rate

        return action

