import numpy as np
from rlplan.policy import FinitePolicy
from rlplan.agents import Agent
from copy import deepcopy
from rlplan.utils import masked_argmax


def span(V):
    return np.max(V) - np.min(V)

class UCRL(Agent):
    def __init__(self, env, rmax=1.0, delta=0.1):
        super().__init__()
        self.id = 'UCRL2'
        self.env = deepcopy(env)
        self.rmax = rmax
        self.delta = delta

        # Constants
        self.Ns = self.env.observation_space.n
        self.Na = self.env.action_space.n

        # Initialize policy
        self.policy = FinitePolicy.uniform(self.Ns, self.Na)

        # Arrays
        self.N_sas = np.zeros((self.Ns, self.Na, self.Ns))  # N_sas[s,a,s'] = number of visits to (s,a, s')
        self.N_sa = np.zeros((self.Ns, self.Na))
        self.S_sa = np.zeros((self.Ns, self.Na))  # S_sa[s, a] = sum of rewards obtained in (s, a)

        # Initialize state
        self.state = self.env.reset()

        # Time counter
        self.t = 0

    def _get_prob_confidence_bound(self, state, action):
        n = max(1, self.N_sa[state, action])
        p_bound = np.sqrt(2.0*(1.0+1.0/n)*np.log(np.sqrt(n+1)*(1.0/self.delta)*(2*self.Ns-2))/n)
        return p_bound

    def _get_reward_confidence_bound(self, state, action):
        n = max(1, self.N_sa[state, action])
        r_bound = np.sqrt((1.0+1.0/n)*np.log(2.0*np.sqrt(n+1)*(1.0/self.delta))/(2.0*n))
        return r_bound

    def _inner_maximization(self, u, d_sa, p_hat_sa):
        sorted_indices = np.argsort(-1*u)
        p_max = np.zeros(self.Ns)
        p_max = p_hat_sa
        p_max[sorted_indices[0]] = min(1, p_hat_sa[sorted_indices[0]] + d_sa/2.0)

        l = self.Ns - 1
        while p_max.sum() > 1 and l >= 0:
            s_l = sorted_indices[l]
            aux = 0.0
            for j in range(self.Ns):
                if sorted_indices[l] != sorted_indices[j]:
                    aux += p_max[sorted_indices[j]]
            p_max[s_l] = max(0, 1 - aux)
            l = l-1
        return p_max

    def _optimistic_bellman_operator(self, u, R_hat, P_hat):
        Tq = np.zeros((self.Ns, self.Na))
        Tu = np.zeros(self.Ns)
        for s in range(self.Ns):
            for a in self.env.available_actions(s):
                p_bound = self._get_prob_confidence_bound(s, a)
                r_bound = self._get_reward_confidence_bound(s, a)
                p_max = self._inner_maximization(u, p_bound, P_hat[s, a, :])
                Tq[s, a] = R_hat[s, a] + r_bound + p_max.dot(u)
            Tu[s] = Tq[s, self.env.available_actions(s)].max()
        return Tu - Tu.min(), Tq

    def _extended_value_iteration(self, R_hat, P_hat, tol):
        u = np.zeros(self.Ns)

        it = 1
        while True:
            Tu, Tq = self._optimistic_bellman_operator(u, R_hat, P_hat)
            err = span(Tu - u)
            if err < tol:
                return Tu, Tq
            u = Tu

    def update(self, state, action, reward, next_state):
        self.N_sas[state, action, next_state] += 1
        self.S_sa[state, action] += reward

        self.N_sa = self.N_sas.sum(axis=2)
        P_hat = self.N_sas/(self.N_sa[:, :, np.newaxis].clip(1))
        R_hat = self.S_sa/self.N_sa.clip(1)
        return R_hat, P_hat

    def step(self):
        current_state = self.state
        action = self.policy.sample(current_state)
        next_state, reward, done, info = self.env.step(action)
        R_hat, P_hat = self.update(current_state, action, reward, next_state)
        self.state = next_state
        self.t += 1
        return current_state, action, R_hat, P_hat

    def run(self, T):
        nu = np.zeros((self.Ns, self.Na))
        state = self.state
        action = self.policy.sample(state)

        while self.t <= T:
            print("new episode!, t=", self.t)
            while nu[state, action] < max(1, self.N_sa[state, action]):
                state, action, R_hat, P_hat = self.step()
                nu[state, action] += 1
                if self.t > T:
                    break
            # New episode
            tol = self.rmax / np.sqrt(self.t)
            u, q = self._extended_value_iteration(R_hat, P_hat, tol)
            self.policy = FinitePolicy.from_q_function(q, self.env)
            nu = np.zeros((self.Ns, self.Na))


if __name__ == '__main__':
    from rlplan.envs import TwoRoomSparse
    env_ = TwoRoomSparse(3, 3)
    ucrl = UCRL(env_)
    ucrl.run(500)

