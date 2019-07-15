"""
Deep Q Learning

Some code was taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


From Mathieu:

def compute_slow_params_update(slow_params, fast_params, tau):

    slow_params_dict = slow_params.state_dict()
    fast_params_dict = fast_params.state_dict()

    for module_key in slow_params_dict.keys() :
        slow_params_dict[module_key] += tau*(fast_params_dict[module_key] - slow_params_dict[module_key])

return slow_params_dict

---

self.ref_model.load_state_dict(compute_slow_params_update(self.ref_model, self.forward_model, self.tau))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from rlplan.agents import Agent
from rlplan.policy import Policy
from rlplan.utils.wrappers import DiscreteOneHotWrapper
from copy import deepcopy
import numpy as np
import gym
import random
from collections import namedtuple


class Net(nn.Module):
    """
    Basic neural net, used in DQN by default.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNPolicy(Policy):
    def __init__(self, dqn_agent):
        super().__init__()
        self.dqn_agent = dqn_agent

    def sample(self, s):
        if not isinstance(s, np.ndarray):
            raise TypeError("Vector encoding expected for state variable.")
        q = self.dqn_agent.get_q([s])[0]
        return q.argmax()


class DQNAgent(Agent):
    def __init__(self, env,
                 gamma=0.99,
                 batch_size=256,
                 horizon=500,
                 memory_size=10000,
                 update_target=20,
                 learning_rate=0.1,
                 epsilon_start=1.0,
                 epsilon_decay=200,
                 epsilon_min=0.05,
                 log_every=5,
                 net=None,
                 reward_threshold=np.inf,
                 seed_val=42):
        super().__init__()
        self.id = 'DQN-Agent'
        self.policy = DQNPolicy(self)  # to be defined

        # avoid changing the state of original env
        env = deepcopy(env)

        # environment wrapper
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.env = DiscreteOneHotWrapper(env)
        else:
            self.env = env
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be discrete."

        # q-function approximator
        self.net = net
        if net is None:  # default network
            hidden_size = 128
            obs_size = self.env.observation_space.shape[0]
            n_actions = self.env.action_space.n
            self.net = Net(obs_size, hidden_size, n_actions)
            self.target_net = Net(obs_size, hidden_size, n_actions)

        # Parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.horizon = horizon
        self.memory_size = memory_size
        self.update_target = update_target
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.log_every = log_every  # evaluate agent every ... episodes
        self.reward_threshold = reward_threshold  # if cumulated reward > threshold, training stops

        # RandomState
        self.seed_val = seed_val
        self.RS = np.random.RandomState(seed_val)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.memory_size)

        # Objective and optimizer
        self.objective = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.learning_rate)

        # Counters
        self.t = 0  # time counter (total number of steps)
        self.episode = 0  # episode counter
        self.episode_t = 0  # number of steps in the current episode

        # Useful
        self.training_info = {}

        # torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_q(self, states):
        with torch.no_grad():
            states_v = torch.FloatTensor([states])
            output = self.net.forward(states_v).data.numpy()  # shape (1, len(states), dim_state)
        return output[0, :, :]  # shape (len(states), dim_state)

    def step(self, state, action, reward, next_state, done):
        if done:
            next_state = None
        self.replay_buffer.push(state, action, reward, next_state)
        self.t += 1
        self.episode_t += 1
        return self.update()

    def update(self):
        """
        Taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        :return:
        """
        if len(self.replay_buffer) < self.batch_size:
            return np.inf
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.FloatTensor([s for s in batch.next_state if s is not None], device=self.device)

        state_batch = torch.FloatTensor(batch.state, device=self.device)
        action_batch = torch.LongTensor(batch.action, device=self.device)
        reward_batch = torch.FloatTensor(batch.reward, device=self.device)

        # Compute Q(s_t, a)
        state_action_values = self.net(state_batch).gather(1, action_batch.view(-1, 1))

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data.numpy()

    def choose_action(self, state):
        if self.RS.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(self.env.action_space.n)
        else:
            # exploit
            q = self.get_q([state])[0]
            return q.argmax()

    def train(self, n_episodes):
        self.training_info['reward'] = []
        self.training_info['training_reward'] = []
        self.training_info['epsilon'] = []
        self.training_info['loss'] = []
        self.training_info['episode'] = []

        state = self.env.reset()
        while self.episode <= n_episodes:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            loss = self.step(state, action, reward, next_state, done)
            state = next_state

            self.training_info['training_reward'].append(reward)

            # Start of new episode
            if (self.episode_t % self.horizon == 0) or done:
                self.episode += 1
                self.episode_t = 0
                state = self.env.reset()

                # reduce epsilon
                self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                    np.exp(-1. * self.t / self.epsilon_decay)

                # update target network
                if self.episode % self.update_target == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                # update training info and break if goal is achieved
                if self.episode % self.log_every == 0:
                    reward_est = self.eval(n_sim=1, horizon=self.horizon, discount=1.0).mean()
                    self.training_info['reward'].append(reward_est)
                    self.training_info['epsilon'].append(self.epsilon)
                    self.training_info['loss'].append(loss)
                    self.training_info['episode'].append(self.episode)
                    print("DQN: episode %d of %d | cumul reward = %.2f ..."
                          % (self.episode, n_episodes, reward_est), end='\r')
                    if reward_est > self.reward_threshold:
                        break


if __name__ == '__main__':
    env_ = gym.make("CartPole-v0")
    from rlplan.envs import GridWorld
    # from rlplan.utils.gridworld_analysis import visualize_exploration
    # env_ = GridWorld(nrows=5, ncols=5, success_probability=1.0, walls=[])
    # env_.track = True
    dqn_agent = DQNAgent(env_, log_every=10, horizon=200, reward_threshold=195, epsilon_decay=200, epsilon_min=0.1)
    dqn_agent.train(n_episodes=300)

    if isinstance(env_, GridWorld):
        visualize_exploration(dqn_agent.env.unwrapped)

    # state = env_.reset()
    # q = dqn_agent.get_q(np.array([state, state, state]))
