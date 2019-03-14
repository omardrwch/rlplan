import numpy as np
import warnings
from rlplan.policy import FinitePolicy
from rlplan.agents import Agent


def span(V):
    return np.max(V) - np.min(V)


class DynProgAgent(Agent):
    """
    Class implemeting Dynamic Programming to solve a finite MDP

    Important: does not work on environments whose reward_fn is not deterministic!

    Args:
        env (FiniteMDP): environment object
        method (str): 'value-iteration' or 'policy-iteration'
        gamma (float): discount factor in [0, 1]
    """
    def __init__(self, env, method='value-iteration', gamma=0.95):
        super().__init__()
        self.id = 'DynamicProgramming'
        self.env = env
        self.method = method
        self.gamma = gamma
        self.policy = None
        self.Q = None
        self.V = None
        assert (method == 'value-iteration' or method == 'policy-iteration'), "Invalid DP method!"

    def train(self, V_init=None, val_it_tol=1e-8, val_it_max_it=1e4,
                    pi_init=None, pol_it_max_it=1e3):
        """
        Train the agent. Returns estimated value function and training info.
        :param V_init:
        :param val_it_tol:
        :param val_it_max_it:
        :param pi_init:
        :param pol_it_max_it:
        :return V, training_info:
        """
        training_info = {}

        if self.method == 'value-iteration':
            if V_init is None:
                V = np.zeros(self.env.observation_space.n)
            else:
                V = V_init

            it = 1
            while True:
                TV, Q, err = self.value_iteration_step(V)

                if it > val_it_max_it:
                    warnings.warn("Value iteration: Maximum number of iterations exceeded.")

                if err < val_it_tol or it > val_it_max_it:
                    self.Q = Q
                    self.V = TV
                    self.policy = FinitePolicy.from_q_function(Q, self.env)
                    return TV, training_info

                V = TV
                it += 1

        elif self.method == 'policy-iteration':
            Na = self.env.action_space.n
            Ns = self.env.observation_space.n
            if pi_init is None:
                action_array = np.array([self.env.available_actions(s)[0] for s in self.env.states])
                policy = FinitePolicy.from_action_array(action_array, Na)
            else:
                policy = pi_init

            it = 1
            while True:
                new_policy = self.policy_iteration_step(policy)

                if it > pol_it_max_it:
                    warnings.warn("Maximum number of iterations exceeded.")

                if new_policy == policy or it > pol_it_max_it:
                    V = policy.evaluate(self.env, self.gamma)
                    self.V = V
                    _, Q = self.bellman_opt_operator(V)
                    self.Q = Q
                    self.policy = policy
                    return V, training_info

                it += 1
                policy = new_policy

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
            for a in self.env.available_actions(s):
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

        Q = np.zeros((Ns, Na))
        TV = np.zeros(Ns)

        for s in self.env.states:
            for a in self.env.available_actions(s):
                prob = self.env.P[s, a, :]
                rewards = np.array([self.env.reward_fn(s, a, s_) for s_ in self.env.states])
                Q[s, a] = np.sum(prob * (rewards + self.gamma * V))
            TV[s] = Q[s, self.env.available_actions(s)].max()
        return TV, Q

    def value_iteration_step(self, V):
        TV, Q = self.bellman_opt_operator(V)

        if self.gamma != 1.0:
            err = np.abs(TV - V).max()

        else:
            err = span(TV - V)
            TV = TV - np.min(TV)

        return TV, Q, err

    def policy_iteration_step(self, policy):
        # Policy evaluation
        V = policy.evaluate(self.env, self.gamma)

        # Policy improvement
        new_policy = FinitePolicy.from_v_function(V, self.env, self.gamma)

        return new_policy


def test():
    from rlplan.envs.toy import ToyEnv1
    env = ToyEnv1()
    agent = DynProgAgent(env, method='policy-iteration', gamma=0.99)
    V, _ = agent.train()
    print(V)
    print(agent.policy)

    print("----")
    del agent
    agent = DynProgAgent(env, method='value-iteration', gamma=0.99)
    V, _ = agent.train()
    print(V)
    print(agent.policy)


if __name__ == '__main__':
    test()
