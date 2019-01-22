"""
UCT algorithm, Kocsis and Szepesvári (http://ggp.stanford.edu/readings/uct.pdf)

TODO: consider states in R^n, not only indexes
"""

import numpy as np


class UCT:
    def __init__(self, env, state, gamma):
        self.env = env
        self.state = state
        self.gamma = gamma
        self.epsilon = 0.01
        self.cp = 0.2
        self.time = 1.0
        self.rollout_horizon = np.log(1.0/(self.epsilon*(1-gamma)))/np.log(1.0/gamma)
        self.root = MaxNode(env, state, gamma, self, depth=0)

    def run(self):
        child, value = self.root.run()
        return child.action, value


class MaxNode:
    def __init__(self, env, state, gamma, uct, depth=0):
        self.uct = uct
        self.env = env
        self.state = state
        self.gamma = gamma
        self.depth = depth
        self.children = [AvgNode(env, state, action, gamma, uct, depth+1) for action in env.available_actions(state)]
        self.effective_horizon = np.log(1.0/(uct.epsilon*(1-gamma)))/np.log(1.0/gamma)

    def run(self):
        if self.depth >= self.effective_horizon:
            return 0
        child = self.select_child()  # select child = select action
        return child, child.run()

    def select_child(self):
        indexes = np.zeros(len(self.children))
        for ii in range(len(indexes)):
            child_ii = self.children[ii]
            if child_ii.visits == 0:
                indexes[ii] = np.inf
            else:
                indexes[ii] = child_ii.mean + self.uct.cp*np.sqrt(2.0*np.log(self.uct.time)/child_ii.visit)
        # update time and sample child node
        self.uct.time += 1.0
        best_child_idx = np.argmax(indexes)
        return self.children[best_child_idx]


class AvgNode:
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

    def run(self):
        if self.visits == 0:
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
            q = reward + self.gamma*child.run()[1]
            self.mean = (1.0 - 1.0/self.visits)*self.mean + (1.0/self.visits)*q

    def rollout(self):
        state = self.state
        action = self.action
        env.reset(state)
        sum = 0.0
        for tt in range(int(self.uct.rollout_horizon)):
            next_state, reward, _, _ = self.env.step(self.action)
            sum += np.power(gamma, tt)*reward
        return sum


if __name__=='__main__':
    from rlplan.agents import DynProgAgent
    from rlplan.envs.toy import ToyEnv1
    import numpy as np

    # Define parameters
    gamma = 0.9  # discount factor
    seed = 55  # random seed

    # Initialize environment
    env = ToyEnv1(seed=seed)

    # ----------------------------------------------------------
    # Finding the exact value function
    # ----------------------------------------------------------
    dynprog = DynProgAgent(env, method='policy-iteration', gamma=gamma)
    V, _ = dynprog.train()

    # ----------------------------------------------------------
    # Run UCT
    # ----------------------------------------------------------
    state = env.reset()
    uct = UCT(env, state, gamma)
