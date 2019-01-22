from rlplan.envs import FiniteMDP
import numpy as np


class ToyEnv1(FiniteMDP):
    """
    Simple environment that gives a reward of 1 when going to the
    last state and 0 otherwise.

    Args:
        seed    (int): Random number generator seed
    """

    def __init__(self, seed=42):
        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)

        Ns = 3
        Na = 2
        P = np.zeros((Ns, Na, Ns))

        P[:, 0, :] = np.array([[0.25, 0.5, 0.25], [0.1, 0.7, 0.2], [0.1, 0.8, 0.1]])
        P[:, 1, :] = np.array([[0.3, 0.3, 0.4], [0.7, 0.2, 0.1], [0.25, 0.25, 0.5]])

        # Initialize base class
        states = np.arange(Ns)
        action_sets = [np.arange(Na).tolist()]*Ns
        super().__init__(states, action_sets, P, seed)

    def reward_fn(self, state, action, next_state):
        return 1.0 * (next_state == self.Ns - 1)


class ToyEnv2(FiniteMDP):
    """
    Simple environment that gives a reward of 1 when going to the
    last state and 0 otherwise.

    Args:
        seed    (int): Random number generator seed
    """

    def __init__(self, Ns=3, Na=2, seed=42):
        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)

        # Define probabilities randomly
        RS = np.random.RandomState(seed)
        P = RS.uniform(size=(Ns, Na, Ns))

        for a in range(Na):
            P[:, a, :] = P[:, a, :] / P[:, a, :].sum(axis=1, keepdims=True)

        # Initialize base class
        states = np.arange(Ns)
        action_sets = [np.arange(Na).tolist()] * Ns
        super().__init__(states, action_sets, P, seed)

    def reward_fn(self, state, action, next_state):
        return 1.0 * (next_state == self.Ns - 1)


if __name__ == '__main__':
    env = ToyEnv1()

