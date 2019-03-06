"""
Implements MDPs which have a tree structure and deterministic transitions. Mainly used to debug planning algorithms
"""

from rlplan.envs.deterministic_mdp import DeterministicFiniteMDP
import numpy as np


class ToyTree1(DeterministicFiniteMDP):
    """
    Tree-shaped MDP with depth 2 (root has depth 0), two actions per state. Leaves transition to root for all actions.

    Args:
        seed    (int): Random number generator seed
    """

    def __init__(self):
        transitions = [(0, 0, 1),
                       (0, 1, 2),
                       (1, 0, 3),
                       (1, 1, 4),
                       (2, 0, 5),
                       (2, 1, 6),
                       (3, 0, 0),
                       (3, 1, 0),
                       (4, 0, 0),
                       (4, 1, 0),
                       (5, 0, 0),
                       (5, 1, 0),
                       (6, 0, 0),
                       (6, 1, 0)
                       ]
        rewards = {(6,): 1.0}  # reward of 1.0 at state 6
        super().__init__(transitions, rewards)


if __name__ == '__main__':
    from rlplan.agents import DynProgAgent
    env = ToyTree1()
    agent = DynProgAgent(env, method='policy-iteration', gamma=0.95)
    V, _ = agent.train()
    print(agent.policy)
