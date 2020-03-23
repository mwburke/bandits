from abc import ABC, abstractmethod
import numpy as np
import random

# TODO: Regular MAB agents - nonstationary epsilon greedy, thompson sampling
# TODO: Contextual bandit agents - LinUCB, HybridUCB


class ActionValueMethod(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self):
        # Abstract function to be implemented in subclasses
        pass

    @abstractmethod
    def update(self, action, reward):
        pass


class RandomArm(ActionValueMethod):
    def __init__(self, actions):
        self.actions = actions

    def reset(self):
        pass

    def act(self):
        return random.randint(0, self.actions - 1)

    def update(self, action, reward):
        pass


class EpsilonGreedy(ActionValueMethod):

    def __init__(self, actions, epsilon=0.01, base_value=0, epsilon_decay=1.00):
        self.actions = actions
        self.base_value = base_value
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.actions, dtype=np.float32)
        self.values = np.array([self.base_value] * self.actions, dtype=np.float32)

    def act(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        else:
            return np.random.choice(np.flatnonzero(self.values == self.values.max()))

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (1 / self.counts[action]) * (reward - self.values[action])
        self.epsilon *= self.epsilon_decay


class UpperConfidenceBound(ActionValueMethod):

    def __init__(self, actions, c=2):
        self.actions = actions
        self.c = c
        self.t = 0
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.actions, dtype=np.float32)
        self.values = np.zeros(self.actions, dtype=np.float32)

    def act(self):
        self.t += 1
        if self.counts.min() == 0:
            return int(np.random.choice(np.flatnonzero(self.values == self.values.min())))
        else:
            return int(np.argmax(self.values + self.c * np.sqrt(np.log(self.t) / self.counts)))

    def update(self, action, reward):
        self.counts[action] += 1
        if self.counts[action] == 0:
            self.values[action] = reward
        else:
            self.values[action] += (1 / self.counts[action]) * (reward - self.values[action])

    def get_q_values(self):
        return self.values

    def get_uncertainty_values(self):
        return self.c * np.sqrt(np.log(self.t) / self.counts)


class GradientBandit(ActionValueMethod):

    def __init__(self, actions, alpha=0.1, base_value=0):
        self.actions = actions
        self.alpha = alpha
        self.base_value = base_value
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.actions, dtype=np.float32)
        self.values = np.array([self.base_value] * self.actions, dtype=np.float32)
        self.t = 0
        self.avg_reward = 0
        self.update_probs()

    def act(self):
        return random.choices(population=np.arange(self.actions), weights=self.probs, k=1)[0]

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[action])
        indices = np.array([i for i in np.arange(self.actions) if i != action])
        self.values[indices] -= self.alpha * (reward - self.avg_reward) * self.probs[indices]
        self.t += 1
        self.avg_reward += (1 / self.t) * (reward - self.avg_reward)
        self.update_probs()

    def update_probs(self):
        self.probs = np.exp(self.values) / np.sum(np.exp(self.values))


class ContextualBandit(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, context):
        # Abstract function to be implemented in subclasses
        pass

    @abstractmethod
    def update(self, action, context, reward):
        pass
