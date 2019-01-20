import numpy as np


class DynProgAgent:
    """
    Class implemeting Dynamic Programming to solve a finite MDP

    Args:
        env (FiniteMDP): environment object
        method (str): 'value-iteration' or 'policy-iteration'
        gamma (float): discount factor in [0, 1]
    """
    def __init__(self, env, method='value-iteration', gamma=0.95):
        self.env = env
        self.method = method
        self.gamma = gamma

    def bellman_operator(self, V, policy):
        """
        Bellman operator T^pi

        Args:
            V         (numpy.array): value function vector
            policy (FinitePolicy)

        Returns:
            T^pi(V)   (numpy.array): operator applied to V
        """
        Ns = self.env.observation_space.n
        Na = self.env.action_space.n

        Q = np.zeros((Ns, Na))
        TV = np.zeros(Ns)

        for s in self.env.states:
            for a in self.env.get_actions(s):
                prob = self.env.P[s, a, :]
                rewards = np.array([self.env.reward_fn(s, a, s_) for s_ in self.env.states])
                Q[s, a] = np.sum(prob * (rewards + self.gamma * V))
            aux = Q[s, :]*policy.prob_vec(s)
            TV[s] = aux[self.env.available_actions(s)].sum()

        return TV

    def bellman_opt_operator(self, V):
        """
        Bellman optimality operator T*

        Args:
            V         (numpy.array): value function vector
            env  (:obj:`FiniteEnv`): environment

        Returns:
            T*V           (numpy.array): operator applied to V
            Q    (numpy.array): Q function, Q[s,a] = -np.inf if a is not available at s
        """
        Ns = self.env.observation_space.n
        Na = self.env.action_space.n

        Q = -np.inf * np.ones((Ns, Na))
        greedy_policy = np.zeros((Ns, Na))

        for s in self.env.states:
            for a in self.env.get_actions(s):
                prob = self.env.P[s, a, :]
                rewards = np.array([self.env.reward_fn(s, a, s_) for s_ in self.env.states])
                Q[s, a] = np.sum(prob * (rewards + self.gamma * V))

        TV = np.max(Q, axis=1)
        # argmax = np.argmax(Q, axis=1)
        # greedy_policy[env.states, argmax] = 1.0

        return TV, Q
