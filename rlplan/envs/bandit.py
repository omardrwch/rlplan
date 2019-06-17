"""
Implements a simple multi-armed bandit (1-state MDP)
"""

import numpy as np
from rlplan.envs import FiniteMDP


class BernoulliArm:
    """
    Bernoulli arm.
    X_t = 1 with probability p
    X_t = 0 with probability (1-p)
    """
    def __init__(self, p):
        self.p = p

    def sample(self):
        return np.random.binomial(1, self.p)

    def mean(self):
        return self.p


class UniformArm:
    """
    Uniform distribution (on [0,1]) arm.
    Prob(X_t in [a,b] )  = (b-a), 0 <= a <= b <= 1
    """
    def __init__(self):
        pass

    def sample(self):
        return np.random.uniform(low=0.0, high=1.0)

    def mean(self):
        return 0.5


class BernoulliBandit(FiniteMDP):
    """
    :param probs: vector of probabilities for each arm, probs[i] = probability that i-th arm gives a reward of 1
    """
    def __init__(self, probs):
        self.n_arms = len(probs)
        self.arms = [BernoulliArm(p) for p in probs]
        states = [0]
        action_sets = [list(range(self.n_arms))]
        P = np.zeros((1, self.n_arms, 1))
        P[0, :, 0] = 1.0
        super().__init__(states, action_sets, P)

    def reward_fn(self, state, action, next_state):
        arm = self.arms[action]
        return arm.sample()



if __name__ == '__main__':
    env = BernoulliBandit([0.8, 0.5, 0.3])
