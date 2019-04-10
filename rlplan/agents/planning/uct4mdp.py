"""
UCT algorithm for MDPs, Kocsis and Szepesvári (http://ggp.stanford.edu/readings/uct.pdf)

Assumes environment with finite action space.

TODO:
    consider states in R^n, not only discrete
    add possibility of using a custom rollout policy
"""

import numpy as np
from rlplan.agents import Agent
from rlplan.policy import Policy
from copy import deepcopy


class UctPolicy(Policy):
    """
    UCT policy: when queried for an action, run UCT.
    If memoize is True, one UCT object per state is saved and reused.
    """
    def __init__(self, uct, memoize=True):
        super().__init__()
        self.uct = uct
        self.memoize = memoize
        self.state2uct = {}

    def sample(self, state):
        if not self.memoize:
            _, action = self.uct.run(state)
        else:
            if state in self.state2uct:
                self.state2uct[state].n_iterations = 500
                _, action = self.state2uct[state].run()
            else:
                _, action = self.uct.run(state)
                self.state2uct[state] = deepcopy(self.uct)
        return action


class Model:
    """
    Contains oracle, discount faction, exploration parameter
    """
    def __init__(self, oracle, gamma, cp, fixed_depth, max_depth, max_rollout_it):
        self.oracle = oracle
        self.gamma = gamma
        self.cp = cp
        self.fixed_depth = fixed_depth
        self.max_depth = max_depth
        self.max_rollout_it = max_rollout_it


class UCT4MDP(Agent):
    """
    :param oracle: adapted gym environment with reset function modified such that env.reset(target_state) puts env in
            target_state
    :param state: initial state
    :param gamma: discount faction
    :param fixed_depth: if True, explore all the leaves first
    :param max_depth: maximum depth of the tree
    :param max_rollout_it: maximum number of steps to evaluate a leaf
    :param cp: exploration parameter
    :param n_it: number of iterations
    """

    def __init__(self, oracle, state, fixed_depth=True, max_depth=10, max_rollout_it=20, gamma=0.95, cp=1.0, n_it=5000):
        super().__init__()
        self.id = 'UCT4MDP'
        self.env = deepcopy(oracle)
        self.model = Model(self.env, gamma, cp, fixed_depth, max_depth, max_rollout_it)
        self.n_iterations = n_it
        self.state = None
        self.root = None
        self.it_count = None
        self.reset(state)  # initialize state, root and it_count
        self.policy = UctPolicy(self)

    def step(self):
        self.root.sample()
        self.it_count += 1

    def run(self, state=None):
        # Check state
        if state is not None:
            if state != self.state:
                self.reset(state)
        # Run
        while True:
            # print("-----------------------------------------")
            self.step()
            if self.it_count > self.n_iterations:
                self.it_count = 0
                break
        best_child = self.root.get_estimate_best_action()
        return best_child.mean, best_child.action

    def reset(self, state):
        self.state = state
        self.it_count = 0
        self.root = MaxNode(self.model, self.state, depth=0)


class MaxNode:

    def __init__(self, model, state, depth=0):
        self.model = model
        self.state = state
        self.depth = depth
        self.children = [AvgNode(model, state, action, depth=self.depth)
                         for action in range(self.model.oracle.action_space.n)]

    def sample(self):
        # print("Sampling MaxNode, state = %d, depth = %d\n" % (self.state, self.depth))
        child = self.choose_child()
        return child.sample()

    def get_total_visits(self):
        K = self.model.oracle.action_space.n
        total_visits = 0
        for a in range(K):
            child = self.children[a]
            total_visits += child.count
        return total_visits

    def choose_child(self):
        K = self.model.oracle.action_space.n
        indexes = np.inf*np.ones(K)
        total_visits = self.get_total_visits()
        for a in range(K):
            child = self.children[a]
            if child.count > 0:
                indexes[a] = child.mean + self.model.cp*np.sqrt(2.0*np.log(total_visits)/child.count)
        return self.children[indexes.argmax()]

    def get_estimate_best_action(self):
        """
        :return: the child that has been played the most
        """
        max_child = self.children[0]
        for child in self.children:
            if child.count > max_child.count:
                max_child = child
        return max_child


class AvgNode:

    def __init__(self, model, state, action, depth=0):
        self.model = model
        self.state = state
        self.action = action
        self.depth = depth
        self.mean = 0.0
        self.count = 0
        self.children = []

    def evaluate(self):
        """
        TODO
        Follow a random policy and return mean discounted reward along the trajectory
        :return:
        """
        tt = 0
        total_reward = 0
        state = self.state
        while tt < self.model.max_rollout_it:
            # sample transition
            self.model.oracle.reset(state)
            next_state, reward, done, _ = self.model.oracle.step(self.action)
            state = next_state
            total_reward += np.power(self.model.gamma, tt)*reward
            if done:
                break
            tt += 1
        return total_reward

    def sample(self):
        # print("Sampling AvgNode, state = %d, action = %d, depth = %d\n" % (self.state, self.action, self.depth))

        # check if leaf
        if (self.count == 0 and (not self.model.fixed_depth)) or (self.depth >= self.model.max_depth):
            self.mean = self.evaluate()
            self.count += 1
            return self.mean

        # update count
        self.count += 1

        # sample transition
        self.model.oracle.reset(self.state)
        next_state, reward, done, _ = self.model.oracle.step(self.action)

        if done:
            return reward

        # verify if there exists a child for the next state
        already_sampled = False
        for child in self.children:
            if child.state == next_state:
                already_sampled = True
                break
        if not already_sampled:
            child = MaxNode(self.model, next_state, depth=self.depth+1)
            self.children.append(child)

        # Sample child and get estimate of Q(s,a)
        q = reward + self.model.gamma*child.sample()
        alpha = 1/self.count
        self.mean = (1-alpha)*self.mean + alpha*q
        return q


if __name__ == '__main__':
    from rlplan.envs import Chain, GridWorld
    env = GridWorld(nrows=5, ncols=5, walls=((),), success_probability=1.0)
    # env = Chain(L=10)
    state = env.reset()
    uct = UCT4MDP(env, state, fixed_depth=False, max_depth=5, gamma=0.95, cp=1.5, n_it=200)
    print(uct.run())

    env.render(mode='auto', policy=uct.policy)
