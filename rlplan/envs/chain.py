from rlplan.envs.deterministic_mdp import DeterministicFiniteMDP


class Chain(DeterministicFiniteMDP):
    """
    Simple chain environment.
    :param L: length of the chain
    """
    def __init__(self, L):

        # list of (state, action, next state)
        # 2 possible actions per state: the first action takes the agent back to the left state and the second action
        # makes the agent advance to the right state, towards the reward.
        assert L >= 2
        self.L = L
        transitions = []

        # first state
        transitions.append((0, 0, 0))
        transitions.append((0, 1, 1))

        # middle states
        for s in range(1, L-1):
            transitions.append((s, 0, s-1))
            transitions.append((s, 1, s+1))

        # last state is terminal
        transitions.append((L-1, 0, L-1))
        transitions.append((L-1, 1, L-1))

        rewards = {(L-2, 1, L-1): 1.0}  # reward of 1.0 when transitioning to the last state
        super().__init__(transitions, rewards)

    def is_terminal(self, state):
        return state == self.L-1


if __name__ == '__main__':
    chain = Chain(5)
    chain.print()
