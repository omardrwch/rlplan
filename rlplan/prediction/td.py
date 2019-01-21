import numpy as np


class TabularTD:
    """
    Implements the TD(lambda) algorithm for value prediction in finite MDPs, given a policy.

    Args:
        env (FiniteMD)
        policy (FinitePolicy)
        gamma (float): discount factor
        lambd (float): value of lambda
    """
    def __init__(self, env, policy, gamma, lambd=0.75, learning_rate=None, min_learning_rate=0.01, verbose=1):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.verbose = verbose
        self.V = None
        self.z = None
        self.visits = None
        self.clear()

    def clear(self):
        """
        Reset to 0 the current estimates of the value function, the eligibility traces and the counters
        """
        self.V = np.zeros(self.env.observation_space.n)
        self.z = np.zeros(self.env.observation_space.n)
        self.visits = np.zeros(self.env.observation_space.n)

    def get_learning_rate(self, s):
        if self.learning_rate is None:
            return max(1.0/max(1.0, self.visits[s]**0.75), self.min_learning_rate)
        else:
            return max(self.learning_rate, self.min_learning_rate)

    def run(self, n_steps=1e5, horizon=np.inf):
        """
        If horizon = np.inf, run for n_steps
        If horizon = H, number of episodes = n_steps/H

        :param n_steps:
        :param horizon:
        :return:
        """
        s = self.env.reset()
        tt = 0
        while tt < n_steps:
            # Select action
            action = self.policy.sample(s)
            next_s, reward, done, info = self.env.step(action)

            # Temporal difference
            td = reward + self.gamma * self.V[next_s] - self.V[s]

            # Eligibility trace
            self.z *= self.gamma * self.lambd
            self.z[s] += 1

            # Update V
            self.visits[s] += 1
            learning_rate = self.get_learning_rate(s)
            self.V += learning_rate*td*self.z

            if self.verbose > 0:
                if (tt+1) % 1000 == 0:
                    print("TD iteration %d out of %d" % (tt, n_steps))

            if done or ((tt+1) % horizon == 0):
                s = self.env.reset()
                tt += 1
                continue

            s = next_s
            tt += 1

        return self.V
