"""
Implementing TrailBlazer algorithm:
http://papers.nips.cc/paper/6253-blazing-the-trails-before-beating-the-path-sample-efficient-monte-carlo-planning.pdf

Based on Edouard Leurent's code:
https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/trailblazer.py

Warning: there are probably bugs, but this algorithm is difficult to debug.
"""

import numpy as np


VERBOSE = 2


class TrailBlazer:
    """
    Notes: states are assumed to be in R^d (this includes discrete spaces)

    :param oracle: analogous to gym environment, but reset function can take a state as input, so we can simulate the
                    environment for any initial state, e.g., oracle.reset(state=s)
    :param gamma: discount factor
    :param delta: confidence parameter
    :param epsilon: tolerance parameter
    """

    def __init__(self, state, oracle, gamma=0.95, delta=0.05, epsilon=0.1):
        self.oracle = oracle
        self.delta = delta
        self.epsilon = epsilon
        self.gamma = gamma
        self.eta = np.power(gamma, 1.0 / max(2.0, np.log(1.0 / epsilon)))
        self.K = oracle.action_space.n
        # self.lamb = 2.0 * np.log(epsilon * (1.0 - gamma)) ** 2.0 * \
        #             np.log(np.log(self.K) / (1.0 - self.eta)) / np.log(self.eta / gamma)  # is this ok?
        self.lamb = 0.0  # 0.0 for faster convergence
        self.m = (np.log(1.0/delta) + self.lamb) / (((1.0 - gamma) * epsilon) ** 2.0)

        print('-------------------------')
        print('gamma {}'.format(gamma))
        print('delta {}'.format(delta))
        print('epsilon {}'.format(epsilon))
        print('self.eta {}'.format(self.eta))
        print('self.K {}'.format(self.K))
        print('self.lamb {}'.format(self.lamb))
        print('self.m {}'.format(self.m))
        print('-------------------------')

        self.root = MaxNode(self.oracle, state, self.delta, self.gamma, self.eta, self.lamb, depth=0)

    def run(self):
        return self.root.run(self.m, self.epsilon/2.0)


class MaxNode:

    created = 0

    def __init__(self, oracle, state, delta, gamma, eta, lamb, depth=0):
        self.oracle = oracle
        self.state = state
        self.delta = delta
        self.gamma = gamma
        self.eta = eta
        self.lamb = lamb
        self.depth = depth

        MaxNode.created += 1
        if VERBOSE >= 2:
            print("New max node, depth = %d / total: %d" % (self.depth, MaxNode.created))

        self.children = [AvgNode(oracle, state, action, gamma, delta, eta, lamb, self.depth+1)
                         for action in range(oracle.action_space.n)]

    def run(self, m, epsilon):

        if VERBOSE >= 3:
            print("Calling MaxNode, [depth = %d, state = %d, m = %0.1f, eps= %0.8f]" % (self.depth,
                                                                                        self.state,
                                                                                        int(np.ceil(m)),
                                                                                        epsilon))
        K = self.oracle.action_space.n
        l = 1
        U = np.inf
        while len(self.children) > 1 and U >= (1-self.eta)*epsilon:
            sqr = (np.log(K*l/(self.delta*epsilon)) +
                   self.gamma / (self.eta - self.gamma) + self.eta + 1) / l
            U = (2./(1-self.gamma))*np.sqrt(sqr)

            mu = np.zeros(len(self.children))
            for i, b in enumerate(self.children):
                mu[i] = b.run(l, U*self.eta/(1.0-self.eta))

            children_idx = [(i, b) for i, b in enumerate(self.children)
                              if mu[i]+2*U/(1-self.eta) >= (mu-2*U/(1-self.eta)).max()]

            indexes = [i for i, b in children_idx]
            self.children = [b for i, b in children_idx]
            l += 1

        if len(self.children) > 1:
            return mu[indexes].max()
        else:
            print("or here")
            b_star = self.children[0]
            return b_star.run(m, self.eta*epsilon)


class AvgNode:

    created = 0

    def __init__(self, oracle, state, action, gamma, delta, eta, lamb, depth):
        self.oracle = oracle
        self.gamma = gamma
        self.state = state
        self.action = action
        self.delta = delta
        self.eta = eta
        self.lamb = lamb
        self.depth = depth

        self.sampled_nodes = []
        self.r = 0

        AvgNode.created += 1
        if VERBOSE >= 2:
            print("New avg node, depth = %d / total: %d" % (self.depth, AvgNode.created))

    def run(self, m, epsilon):

        if VERBOSE >= 3:
            print("Calling AvgNode, [depth = %d, state = %d, m = %0.1f, eps= %0.8f]" % (self.depth,
                                                                                        self.state,
                                                                                        int(np.ceil(m)),
                                                                                        epsilon))
            print("--- 1/(1-gamma) = %0.8f" % (1.0/(1.0-self.gamma)))

        if epsilon >= 1.0/(1.0-self.gamma):
            return 0

        if len(self.sampled_nodes) > m:
            active_nodes = self.sampled_nodes[:int(np.ceil(m))]
        else:
            while len(self.sampled_nodes) < m:
                # sample next state
                self.oracle.reset(self.state)  # setting oracle to node's state
                next_state, reward, done, info = self.oracle.step(self.action)

                already_sampled = False
                for node in self.sampled_nodes:
                    if node.state == next_state:
                        already_sampled = True
                        self.sampled_nodes.append(node)
                        break
                if not already_sampled:
                    new_node = MaxNode(self.oracle, next_state, self.delta, self.gamma, self.eta,
                                       self.lamb, self.depth+1)
                    self.sampled_nodes.append(new_node)

                self.r += reward
            active_nodes = self.sampled_nodes

        # At this point, |ActiveNodes| = m
        uniques_node = []
        uniques_state = []
        counts = []
        for node in active_nodes:
            try:
                if type(node.state) is np.ndarray:
                    i = uniques_state.index(list(node.state)) # not debugged
                else:
                    i = uniques_state.index(node.state)
                counts[i] += 1
            except ValueError:
                uniques_node.append(node)
                uniques_state.append(node.state)
                counts.append(1)

        mu = 0
        for i, node in enumerate(uniques_node):
            k = counts[i]
            nu = node.run(k, epsilon/self.gamma)
            mu = mu + nu*k/m

        return self.gamma*mu + self.r/len(self.sampled_nodes)


if __name__=='__main__':
    from rlplan.agents.planning import DynProgAgent
    from rlplan.envs.toy import ToyEnv1
    import numpy as np

    # Define parameters
    gamma = 0.1  # discount factor
    seed = 55  # random seed

    # Initialize environment
    env = ToyEnv1(seed_val=seed)

    # ----------------------------------------------------------
    # Finding the exact value function
    # ----------------------------------------------------------
    dynprog = DynProgAgent(env, method='policy-iteration', gamma=gamma)
    V, _ = dynprog.train()

    # ----------------------------------------------------------
    # TrailBlazer
    # ----------------------------------------------------------
    state = env.reset()
    tb = TrailBlazer(state, env, gamma=gamma, delta=0.1, epsilon=1.0)
    val = tb.run()

    print("Value function = ", V)
    print("TrailBlazer estimate of V[%d] = %f" % (state, val))
