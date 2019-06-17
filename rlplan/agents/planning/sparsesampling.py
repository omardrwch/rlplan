"""
Implements the algorithms proposed by [1].

 [1] Kearns et al., 1999. A Sparse Sampling Algorithm for Near-Optimal Planning in Large Markov Decision Processes.

TODO:
    consider states in R^n, not only discrete
"""

import numpy as np
from rlplan.agents import Agent
from rlplan.policy import Policy
from copy import deepcopy


class SparseSamplingPolicy(Policy):
    """
    SparseSampling policy: when queried for an action, run SparseSampling.
    """
    def __init__(self, sparsesampling):
        super().__init__()
        self.sparsesampling = sparsesampling

    def sample(self, state):
        action, _ = self.sparsesampling.run(state)
        return action


class SparseSampling(Agent):
    def __init__(self, oracle, epsilon=0.1, gamma=0.9, rmax=1.0, max_transitions=1000, max_depth=15):
        """
        :param oracle: adapted gym environment with reset function modified such that env.reset(target_state) puts env in
        target_state
        :param state: initial state
        :param epsilon: desired accuracy
        :param gamma: discount factor
        :param rmax: upper bound on reward function
        :param max_transitions: maximum number of transitions to sample at each level
        :param max_depth: maximum depth
        """
        super().__init__()
        self.id = "SparseSampling"
        self.oracle = oracle
        self.env = deepcopy(oracle)
        self.gamma = gamma
        # Computing constants
        lambd = epsilon*np.power(1.0-gamma, 2)/4.0
        Vmax = rmax/(1.0-gamma)
        H = np.ceil(np.log(lambd/Vmax)/np.log(gamma))
        k = self.oracle.action_space.n
        aux = np.power(Vmax/lambd, 2)
        C = aux*(2*H*np.log(k*H*aux) + np.log(rmax/lambd))
        self.C = min(C, max_transitions)
        self.H = min(H, max_depth)
        self.k = k
        # Policy
        self.policy = SparseSamplingPolicy(self)

    def run(self, state):
        Q = self.EstimateQ(state, self.H)
        return Q.argmax(), Q

    def EstimateQ(self, s, h):
        if h == 0:
            return np.zeros(self.k)
        Q = np.zeros(self.k)
        C = np.power(self.gamma, 2*(self.H-h))*self.C
        C = int(np.ceil(C))
        for a in range(self.k):
            state2estimate = {}
            v_estimates = []
            rewards = []
            for ii in range(C):
                self.oracle.reset(state=s)
                next_s, reward, done, _ = self.oracle.step(a)
                rewards.append(reward)
                if next_s in state2estimate:
                    v_estimates.append(state2estimate[next_s])
                else:
                    if not done:
                        estimate = self.EstimateV(next_s, h-1)
                    else:
                        estimate = 0
                    state2estimate[next_s] = estimate
                    v_estimates.append(estimate)
            v_estimates = np.array(v_estimates)
            rewards = np.array(rewards)
            Q[a] = rewards.mean() + self.gamma*v_estimates.mean()
        return Q

    def EstimateV(self, s, h):
        Q = self.EstimateQ(s, h)
        return Q.max()


if __name__ == '__main__':
    from rlplan.envs import Chain, GridWorld
    # env_ = GridWorld(nrows=5, ncols=5, walls=((),), success_probability=1.0)
    env_ = Chain(L=12)
    s_ = env_.reset()
    agent = SparseSampling(env_, epsilon=0.1, gamma=0.9, rmax=1.0, max_transitions=4, max_depth=12)
    # env.render(mode='auto', policy=uct.policy)

