from rlplan.envs.deterministic_mdp import DeterministicFiniteMDP


class Chain(DeterministicFiniteMDP):
    """
    Simple chain environment.
    :param L: length of the chain
    """
    def __init__(self, L):

        # list of (state, action, next state)
        # 2 possible actions per state: the first action takes the agent back to the first state and the second action
        # makes the agent advance towards the reward.
        assert L >= 2
        transitions = []
        for s in range(L-1):
            transitions.append((s, 0, 0))
            transitions.append((s, 1, s+1))
        # any action in the last state takes the agent back to the first state
        transitions.append((L-1, 0, 0))
        transitions.append((L-1, 1, 0))

        rewards = {(L-2, 1, L-1): 1.0}  # reward of 1.0 when transitioning to the last state
        super().__init__(transitions, rewards)


if __name__ == '__main__':
    chain = Chain(5)
    chain.print()
