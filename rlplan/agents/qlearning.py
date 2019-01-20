import numpy as np
from rlplan.policy import FinitePolicy


class QLearningAgent:
    """
    Implements Q-learning algorithm for finite MDPs using epsilon-greedy exploration

    If learning_rate is None; alpha(x,a) = 1/max(1, N(s,a))
    """
    def __init__(self, env, gamma=0.95, learning_rate=None, min_learning_rate=0.0, epsilon=0.1, epsilon_decay=0.999,
                 epsilon_min=0.01, rmax=1.0, verbose=1, seed=42):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((env.Ns, env.Na))
        self.Nsa = np.zeros((env.Ns, env.Na))
        self.t = 0
        self.episode = 0
        self.state = env.reset()
        self.verbose = verbose
        self.RS = np.random.RandomState(seed)
        self.policy = None

    def train(self, n_steps=1e5, horizon=np.inf):
        """
        Train the agent. Returns estimated value function and training info.

        If horizon = np.inf, run for n_steps
        If horizon = H, number of episodes = n_steps/H

        :param n_steps:
        :param horizon:
        :return:
        """
        training_info = {}
        while self.t < n_steps:
            done = self.step()
            if done or ((self.t+1) % horizon == 0):
                self.env.reset()

            if self.verbose > 0:
                if (self.t+1) % 1000 == 0:
                    print("Q-learning iteration %d out of %d" % (self.t, n_steps))

            self.t += 1

        self.policy = FinitePolicy.from_q_function(self.Q, self.env)
        V = np.zeros(self.env.observation_space.n)
        for s in self.env.states:
            V[s] = self.Q[s, self.env.available_actions(s)].max()

        return V, training_info

    def bellman_residual(self, r, x, a, y):
        """
        :param r: reward
        :param x: current state
        :param a: current action
        :param y: next state
        :return:  bellman residual
        """
        max_q_y_a = self.Q[y, self.env.available_actions()].max()
        q_x_a = self.Q[x, a]

        return r + self.gamma*max_q_y_a - q_x_a

    def get_learning_rate(self, s, a):
        if self.learning_rate is None:
            return max(1.0/max(1.0, self.Nsa[s, a]**0.75), self.min_learning_rate)
        else:
            return max(self.learning_rate, self.min_learning_rate)

    def get_action(self):
        if self.RS.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(self.env.available_actions())
        else:
            # exploit
            state = self.env.state
            actions = self.env.available_actions()
            a = np.argmax(self.Q[state, actions])
            return a

    def step(self):
        # Current state
        x = self.env.state

        # Choose action
        a = self.get_action()

        # Learning rate
        alpha = self.get_learning_rate(x, a)

        # Take step
        observation, reward, done, info = self.env.step(a)
        y = observation
        r = reward
        delta = self.bellman_residual(r, x, a, y)

        # Update
        self.Q[x, a] = self.Q[x, a] + alpha*delta

        self.Nsa[x, a] += 1
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

        return done