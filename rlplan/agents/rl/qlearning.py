import numpy as np
from rlplan.policy import FinitePolicy
from rlplan.agents import Agent
from copy import deepcopy
from rlplan.policy import Policy


class UctPolicy(Policy):
    """
    UCT policy: when queried for an action, run UCT.
    If memoize is True, one UCT object per state is saved and reused.
    """
    def __init__(self, uct, memoize=True):
        super().__init__()
        self.uct = uct
        self.memoize = memoize
        self.state2uct = {}

    def sample(self, state):
        if not self.memoize:
            _, action = self.uct.run(state)
        else:
            if state in self.state2uct:
                self.state2uct[state].n_iterations = 500
                _, action = self.state2uct[state].run()
            else:
                _, action = self.uct.run(state)
                self.state2uct[state] = deepcopy(self.uct)
        return action


class QLearningAgent(Agent):
    """
    Implements Q-learning algorithm for finite MDPs using epsilon-greedy exploration

    If learning_rate is None; alpha(x,a) = 1/max(1, N(s,a))**0.75

    :param env: environment
    :param gamma: discount factor
    :param learning_rate:
    :param min_learning_rate: minimum learning rate value
    :param epsilon: exploration parameter
    :param epsilon_decay: decay factor of epsilon at every step
    :param epsilon_min: minimum value of epsilon
    :param rmax: maximum reward value
    :param verbose: if > 0, information is printed
    :param seed_val: integer, seed for np.random.RandomState
    """
    def __init__(self, env, gamma=0.95, learning_rate=None, min_learning_rate=0.05, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, rmax=1.0, verbose=1, seed_val=42):
        super().__init__()
        # avoid changing the state of original env
        env = deepcopy(env)

        self.id = 'QLearningAgent'
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.t = 0
        self.state = self.env.reset()
        self.verbose = verbose
        self.seed_val = seed_val
        self.RS = np.random.RandomState(seed_val)

        Ns = self.env.observation_space.n
        Na = self.env.action_space.n
        self.Q = np.ones((Ns, Na))*rmax/(1-gamma)
        self.Nsa = np.zeros((Ns, Na))
        self.policy = FinitePolicy.from_q_function(self.Q, self.env)

    def train(self, n_steps=1e5, horizon=np.inf, eval_every=100, eval_params=None):
        """
        Train the agent. Returns estimated value function and training info.

        If horizon = np.inf, run for n_steps
        If horizon = H, number of episodes = n_steps/H

        :param n_steps:
        :param horizon:
        :param eval_every: interval for evaluating the agent
        :param eval_params: dictionary containing parameters to send to Agent.eval()
        :return:
        """
        training_info = {}
        training_info['rewards_list'] = []
        training_info['x_data'] = []
        training_info['n_episodes'] = 0
        training_info['episode_total_reward'] = []

        episode_reward = 0
        while self.t < n_steps:
            done, reward = self.step()
            episode_reward += reward

            if done or ((self.t+1) % horizon == 0):
                self.state = self.env.reset()
                training_info['n_episodes'] += 1
                training_info['episode_total_reward'].append(episode_reward)
                if self.verbose > 0 and (training_info['n_episodes'] % 500 == 0):
                    print("Episode %d, total reward = %0.2f" % (training_info['n_episodes'], episode_reward))
                episode_reward = 0

            if self.verbose > 0:
                if (self.t+1) % 1000 == 0:
                    print("Q-learning iteration %d out of %d" % (self.t+1, n_steps))

            self.t += 1

            if self.t % eval_every == 0:
                self.policy = FinitePolicy.from_q_function(self.Q, self.env)
                if eval_params is None:
                    rewards = self.eval()
                else:
                    rewards = self.eval(**eval_params)
                training_info['rewards_list'].append(rewards)
                training_info['x_data'].append(self.t)

        self.policy = FinitePolicy.from_q_function(self.Q, self.env)
        V = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            V[s] = self.Q[s, self.env.available_actions(s)].max()

        training_info['V'] = V
        return training_info

    def get_delta(self, r, x, a, y):
        """
        :param r: reward
        :param x: current state
        :param a: current action
        :param y: next state
        :return:
        """
        max_q_y_a = self.Q[y, self.env.available_actions()].max()
        q_x_a = self.Q[x, a]

        return r + self.gamma*max_q_y_a - q_x_a

    def get_learning_rate(self, s, a):
        if self.learning_rate is None:
            return max(1.0/max(1.0, self.Nsa[s, a])**0.75, self.min_learning_rate)
        else:
            return max(self.learning_rate, self.min_learning_rate)

    def get_action(self, x):
        if self.RS.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(self.env.available_actions())
        else:
            # exploit
            state = self.state
            actions = self.env.available_actions()
            temp = np.max(self.Q[state, actions])
            a = np.abs(self.Q[state, :]-temp).argmin()
            return a

    def step(self):
        # Current state
        x = self.state

        # Choose action
        a = self.get_action(x)

        # Learning rate
        alpha = self.get_learning_rate(x, a)

        # Take step
        observation, reward, done, info = self.env.step(a)
        y = observation
        r = reward
        delta = self.get_delta(r, x, a, y)

        # Update Q
        self.Q[x, a] = self.Q[x, a] + alpha*delta

        self.Nsa[x, a] += 1
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

        # Update state
        self.state = observation

        return done, reward
