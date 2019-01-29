"""
UCT algorithm, Kocsis and Szepesvári (http://ggp.stanford.edu/readings/uct.pdf)

I did a tree implementation, as in TrailBlazer, but following the Fig.1 in the paper should be better.

REMARK: possible bugs in UCTv1, important bugs!

TODO: consider states in R^n, not only indexes
"""

import numpy as np


class UCTv2:
    """
    :param env: environment with step function and reset to any state
    :param state: current state
    :param gamma: discount faction
    :param max_it: maximum number of iterations
    :param cp: exploration parameter
    :param max_depth: maximum depth of the tree
    :param max_rollout_len: maximum length of trajectory rollouts
    """
    def __init__(self, env, state, gamma,
                 max_it=4000, cp=1.5, max_depth=15, max_rollout_len=150):
        self.env = env
        self.state = state
        self.gamma = gamma
        self.max_it = max_it
        self.cp = cp
        self.max_depth = max_depth
        self.max_rollout_len = max_rollout_len
        self.visits = {}  # visits[(state, action, depth)] = number of visits to (state, depth)
        self.means = {}  # means[(state, action , depth)] = mean of Q(s,a) at given depth

    def run(self):
        for tt in range(self.max_it):
            if (tt+1) % 500 == 0:
                print("iteration %d of %d" % (tt+1, self.max_it))
            self.search(self.state, 0)
        action = self.select_action(state, 0)
        return action, self.means[(self.state, action, 0)]

    def search(self, state, depth):
        if depth > self.max_depth:
            return 0
        if self.get_visits(state, depth) == 0:
            val, first_action = self.evaluate(state)
            self.visits[(state, first_action, depth)] = 1.0
            self.means[(state, first_action, depth)] = val
            return val
        action = self.select_action(state, depth)
        self.env.reset(state)
        next_state, reward, _, _ = self.env.step(action)
        q = reward + self.gamma*self.search(next_state, depth+1)
        self.update_value(state, action, q, depth)
        return q

    def evaluate(self, state):
        env.reset(state)
        total_reward = 0.0
        first_action = self.env.action_space.sample()
        action = first_action
        for tt in range(self.max_rollout_len):
            next_state, reward, done, _ = self.env.step(action)
            total_reward += np.power(gamma, tt)*reward
            action = self.env.action_space.sample()
            if done:
                break
        return total_reward, first_action

    def select_action(self, state, depth):
        indexes = np.zeros(self.env.action_space.n)
        N_sd = self.get_visits(state, depth)
        for action in range(len(indexes)):
            node = (state, action, depth)
            if node not in self.visits:
                indexes[action] = np.inf
            else:
                mean = self.means[node]
                visits = self.visits[node]
                indexes[action] = mean + self.cp*np.sqrt(2*np.log(N_sd)/visits)
        return indexes.argmax()

    def update_value(self, state, action, q, depth):
        node = (state, action, depth)
        if node not in self.visits:
            self.visits[node] = 1
            self.means[node] = q
        else:
            self.visits[node] += 1
            n = self.visits[node]
            self.means[node] = (1.0-1.0/n)*self.means[node] + (1.0/n)*q

    def get_visits(self, state, depth):
        n_visits = 0
        for action in range(self.env.action_space.n):
            if (state, action, depth) in self.visits:
                n_visits += self.visits[(state, action, depth)]
        return n_visits


if __name__=='__main__':
    from rlplan.agents import DynProgAgent
    from rlplan.envs.toy import ToyEnv1, ToyEnv2
    from rlplan.policy import FinitePolicy
    import numpy as np

    # Define parameters
    gamma = 0.25  # discount factor
    seed = 55  # random seed

    # Initialize environment
    # env = ToyEnv1(seed=seed)
    env = ToyEnv2(seed=seed, Ns=10, Na=4)

    # ----------------------------------------------------------
    # Finding the exact value function
    # ----------------------------------------------------------
    dynprog = DynProgAgent(env, method='policy-iteration', gamma=gamma)
    V, _ = dynprog.train()

    # ----------------------------------------------------------
    # Run UCT
    # ----------------------------------------------------------
    uct_policy = 2*np.ones(env.Ns, dtype=np.int64)
    for ss in env.states:
        env.seed(seed)
        state = env.reset(ss)
        uct = UCTv2(env, state, gamma)
        idx, val = uct.run()
        uct_policy[ss] = int(idx)
        del uct

    print("UCT policy:     ", uct_policy)
    print("Correct policy: ", dynprog.policy.policy_array.argmax(axis=1))

    uct_pol = FinitePolicy.from_action_array(uct_policy, env.Na)
    V_uct = uct_pol.evaluate(env, gamma)
    print(" ")
    print("UCT val", V_uct)
    print("Correct val", V)