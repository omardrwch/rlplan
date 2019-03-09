"""
Implements MDPs which have a tree structure and deterministic transitions. Mainly used to debug planning algorithms
"""

from rlplan.envs.deterministic_mdp import DeterministicFiniteMDP
import numpy as np


class ToyTree1(DeterministicFiniteMDP):
    """
    Tree-shaped MDP with depth 2 (root has depth 0), two actions per state. Leaves are terminal states.
    """

    def __init__(self):
        transitions = [(0, 0, 1),
                       (0, 1, 2),
                       (1, 0, 3),
                       (1, 1, 4),
                       (2, 0, 5),
                       (2, 1, 6),
                       (3, 0, 3),
                       (3, 1, 3),
                       (4, 0, 4),
                       (4, 1, 4),
                       (5, 0, 5),
                       (5, 1, 5),
                       (6, 0, 6),
                       (6, 1, 6)
                       ]
        rewards = {(2, 1, 6): 1.0}  # reward of 1.0 at state 6
        super().__init__(transitions, rewards)

    def is_terminal(self, state):
        if state in [3, 4, 5, 6]:
            return True
        return False


if __name__ == '__main__':
    from rlplan.agents import DynProgAgent
    env = ToyTree1()
    agent = DynProgAgent(env, method='policy-iteration', gamma=0.95)
    V, _ = agent.train()
    print(V)
    print(agent.policy)
