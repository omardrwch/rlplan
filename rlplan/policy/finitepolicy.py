import numpy as np


class FinitePolicy:
    """
    Class for defining a policy in a finite MDP.

    Args
        policy_array: 2d numpy array such that policy_array[s,a] is the probability of taking action a in state s.
        seed (int): Random number generator seed
    """
    def __init__(self, policy_array, seed=42):
        self.policy_array = policy_array
        self.random = np.random.RandomState(seed)
        self.actions = np.arange(policy_array.shape[1], dtype=np.int64)

    @classmethod
    def from_action_array(cls, action_array, n_actions):
        """
        Deterministic policy
        :param: action_array: array such that action_array[s] = action to be taken in state s
        :param: n_actions: total number of actions
        :param: seed (int): Random number generator seed
        """
        if type(action_array) is not np.ndarray:
            action_array = np.array(action_array, dtype=np.int64)
        n_states = action_array.shape[0]
        policy_array = np.zeros((n_states, n_actions))
        policy_array[np.arange(n_states), action_array] = 1.0
        return cls(policy_array)

    @classmethod
    def from_v_function(cls, V, env, gamma):
        """
        Return greedy policy with respect to value function V
        """
        Na = env.action_space.n
        Ns = env.observation_space.n
        action_array = np.zeros(Ns, dtype=np.int64)
        Q = np.zeros((Ns, Na))

        for s in env.states:
            for a in env.available_actions(s):
                prob = env.P[s, a, :]
                rewards = np.array([env.reward_fn(s, a, s_) for s_ in env.states])
                Q[s, a] = np.sum(prob * (rewards + gamma * V))
            temp = Q[s, env.available_actions(s)].max()
            action_array[s] = np.abs(Q[s, :] - temp).argmin()
        return cls.from_action_array(action_array, Na)

    @classmethod
    def from_q_function(cls, Q, env):
        """
        Return greedy policy with respect to Q function
        """
        Ns, Na = Q.shape
        action_array = np.zeros(Ns, dtype=np.int64)

        for s in range(Ns):
            temp = Q[s, env.available_actions(s)].max()
            action_array[s] = np.abs(Q[s, :] - temp).argmin()
        return cls.from_action_array(action_array, Na)

    def evaluate(self, env, gamma):
        """
        Implements exact policy evaluation, given an environment and a discount factor gamma in [0,1[
        :param env:
        :param gamma:
        :return: value function of the policy
        """
        # Compute the transition matrix P_pi and reward vector r_pi
        Ns = env.observation_space.n
        P_pi = np.zeros((Ns, Ns))
        r_pi = np.zeros(Ns)

        for s in env.states:
            for s_ in env.states:
                for a in env.actions:
                    prob_s_a = self.prob(s, a)
                    if prob_s_a > 0.0:
                        assert a in env.available_actions(s), "FinitePolicy error: unavailable action being chosen!"
                        P_pi[s, s_] += prob_s_a * env.P[s, a, s_]
                        r_pi[s] += prob_s_a * env.reward_fn(s, a, s_) * env.P[s, a, s_]

        A = np.eye(Ns) - gamma * P_pi
        b = r_pi

        # Solve linear system
        V = np.linalg.solve(A, b)
        return V

    def seed(self, seed):
        """
        Reset random number generator
        """
        self.random = np.random.RandomState(seed)

    def prob(self, s, a):
        """
        :return: probability of choosing action a in state s.
        """
        return self.policy_array[s, a]

    def prob_vec(self, s):
        """
        :return: probability vector over actions knowing state s.
        """
        return self.policy_array[s, :]

    def sample(self, s):
        """
        :return: sample of an action at state s.
        """
        return self.random.choice(self.actions, p=self.prob_vec(s))

    def __eq__(self, other):
        return np.array_equal(self.policy_array, other.policy_array)

    def __str__(self):
        return str(self.policy_array)


def test():
    from rlplan.envs.toy import ToyEnv1

    env = ToyEnv1()
    action_array = np.array([0, 1, 1])
    policy = FinitePolicy.from_action_array(action_array, env.action_space.n)
    print(policy)


if __name__=='__main__':
    test()
