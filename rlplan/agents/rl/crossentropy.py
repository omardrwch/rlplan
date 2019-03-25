"""
Implements the cross-entropy method, as presented in the book "Deep Reinforcement Learning Hands-on" by Maxim Lapan.

Let p(ai | si, v) be the probability of taking action ai in state si, when the network has parameters v
The CE method aims to maximize the likelihood of actions that led to high rewards.


"""

# tensorboard --logdir=runs


import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import numpy as np
import gym
from tensorboardX import SummaryWriter
from rlplan.agents import Agent
from rlplan.policy import Policy
from rlplan.utils.wrappers import DiscreteOneHotWrapper
from copy import deepcopy

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class CrossEntropyPolicy(Policy):
    def __init__(self, cross_ent_agent):
        super().__init__()
        self.cross_ent_agent = cross_ent_agent

    def sample(self, s):
        return self.cross_ent_agent.get_action(s)


class CrossEntropyAgent(Agent):
    def __init__(self, env, gamma=0.99, batch_size=16, percentile=70, horizon=500, learning_rate=0.01, net=None):
        super().__init__()
        self.id = 'CrossEntropyAgent'

        # avoid changing the state of original env
        env = deepcopy(env)

        # environment wrapper
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.env = DiscreteOneHotWrapper(env)
        else:
            self.env = env
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be discrete."

        # parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.percentile = percentile
        self.learning_rate = learning_rate
        self.horizon = horizon  # maximum length of an episode

        # policy
        self.net = net
        if net is None:  # default network
            hidden_size = 128
            obs_size = self.env.observation_space.shape[0]
            n_actions = self.env.action_space.n
            self.net = Net(obs_size, hidden_size, n_actions)
        self.policy = CrossEntropyPolicy(self)

    def get_action(self, obs):
        obs_v = torch.FloatTensor([obs])
        sm = nn.Softmax(dim=1)
        act_probs_v = sm(self.net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        return action

    def iterate_batches(self):
        batch = []
        episode_reward = 0.0
        episode_steps = []
        obs = self.env.reset()
        time = 0
        while True:
            action = self.get_action(obs)
            next_obs, reward, is_done, _ = self.env.step(action)
            episode_reward += reward*np.power(self.gamma, time)
            episode_steps.append(EpisodeStep(observation=obs, action=action))
            if is_done or time > self.horizon:
                batch.append(Episode(reward=episode_reward, steps=episode_steps))
                episode_reward = 0.0
                episode_steps = []
                next_obs = self.env.reset()
                time = 0
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            obs = next_obs
            time += 1

    def filter_batch(self, batch):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, self.percentile)
        reward_mean = float(np.mean(rewards))

        train_obs = []
        train_act = []
        for example in batch:
            if example.reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean

    def train(self, n_steps=50):
        writer = SummaryWriter()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.net.parameters(), lr=self.learning_rate)

        print("Training %s ..." % self.id)
        for iter_no, batch in enumerate(self.iterate_batches()):
            obs_v, acts_v, reward_b, reward_m = self.filter_batch(batch)
            optimizer.zero_grad()
            action_scores_v = self.net(obs_v)
            loss_v = objective(action_scores_v, acts_v)
            loss_v.backward()
            optimizer.step()
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
            writer.add_scalar("loss", loss_v.item(), iter_no)
            writer.add_scalar("reward_bound", reward_b, iter_no)
            writer.add_scalar("reward_mean", reward_m, iter_no)
            if iter_no > n_steps:
                writer.close()
                print("...done.")
                break

    def test(self):
        obs = self.env.reset()
        while True:
            action = self.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            obs = next_obs
            if done:
                break
            self.env.render()
        self.env.close()


if __name__ == "__main__":
    # env_ = gym.make("CartPole-v0")
    # env_ = gym.make("Acrobot-v1")
    from rlplan.envs import GridWorld
    env_ = GridWorld()
    agent = CrossEntropyAgent(env_, gamma=1.0, batch_size=32, percentile=50)
    # env_ = gym.make("FrozenLake-v0")
    # agent = CrossEntropyAgent(env_, batch_size=200, learning_rate=0.001, horizon=np.inf)
