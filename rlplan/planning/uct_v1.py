"""
UCT algorithm, Kocsis and Szepesvári (http://ggp.stanford.edu/readings/uct.pdf)

I did a tree implementation, as in TrailBlazer, but following the Fig.1 in the paper should be better.

REMARK: possible bugs in UCTv1, important bugs!

TODO: consider states in R^n, not only indexes
"""

import numpy as np

class UCTv1:

    def __init__(self, env, state, gamma, n_iterations=5000):
        self.env = env
        self.state = state
        self.gamma = gamma
        self.epsilon = 0.1
        self.cp = 1.0
        self.time = 1.0
        self.n_iterations = n_iterations
        self.rollout_horizon = np.log(1.0/(self.epsilon*(1-gamma)))/np.log(1.0/gamma)
        self.root = MaxNode(env, state, gamma, self, depth=0)

    def run(self):
        tt = 0
        while tt < self.n_iterations:
            value = self.root.run()
            tt += 1
        child = self.root.select_child()
        return child.action, child.mean


class MaxNode:
    created = 0

    def __init__(self, env, state, gamma, uct, depth=0):
        self.uct = uct
        self.env = env
        self.state = state
        self.gamma = gamma
        self.depth = depth
        self.effective_horizon = np.log(1.0/(uct.epsilon*(1-gamma)))/np.log(1.0/gamma)

        MaxNode.created += 1
        self.id = MaxNode.created
        print("New MaxNode, total = %d, depth = %d" % (MaxNode.created, depth))

        self.children = [AvgNode(env, state, action, gamma, uct, depth+1) for action in env.available_actions(state)]

    def run(self):
        if self.depth >= self.effective_horizon:
            return 0
        child = self.select_child()  # select child = select action
        return child.run()

    def select_child(self):
        indexes = np.zeros(len(self.children))

        total_visits = 0
        for cc in self.children:
            total_visits += cc.visits

        for ii in range(len(indexes)):
            child_ii = self.children[ii]
            if child_ii.visits == 0:
                indexes[ii] = np.inf
            else:
                indexes[ii] = child_ii.mean + self.uct.cp*np.sqrt(2.0*np.log(total_visits)/child_ii.visits)
        # update time and sample child node
        self.uct.time += 1.0
        best_child_idx = np.argmax(indexes)
        return self.children[best_child_idx]


class AvgNode:
    created = 0

    def __init__(self, env, state, action, gamma, uct, depth):
        self.uct = uct
        self.env = env
        self.state = state
        self.action = action
        self.gamma = gamma
        self.depth = depth
        self.mean = 0
        self.visits = 0
        self.children = []

        AvgNode.created += 1
        self.id = AvgNode.created
        print("New AvgNode, total = %d, depth = %d" % (AvgNode.created, depth))

    def run(self):
        if self.visits == 0:
            # Increment counter
            self.visits += 1.0
            return self.rollout()
        else:
            # Increment counter
            self.visits += 1.0

            # Sample a transition
            self.env.reset(self.state) # reset environment to node state
            next_state, reward, _, _ = self.env.step(self.action)

            # Check if next state is already visited
            already_visited = False
            child = None
            for cc in self.children:
                if cc.state == next_state:
                    already_visited = True
                    child = cc
            if not already_visited:
                child = MaxNode(self.env, next_state, self.gamma, self.uct, depth=self.depth+1)
                self.children.append(child)

            # Update average
            q = reward + self.gamma*child.run()
            self.mean = (1.0 - 1.0/self.visits)*self.mean + (1.0/self.visits)*q
            return q  # self.mean

    def rollout(self):
        state = self.state
        action = self.action
        env.reset(state)
        sum = 0.0
        for tt in range(int(self.uct.rollout_horizon)):
            next_state, reward, _, _ = self.env.step(action)
            sum += np.power(gamma, tt)*reward
            action = self.env.action_space.sample()
        return sum


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
        uct = UCTv1(env, state, gamma)
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