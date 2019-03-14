import pytest
from rlplan.agents.rl import CrossEntropyAgent
from rlplan.agents.rl import QLearningAgent
from rlplan.agents.planning import UCT4MDP
from rlplan.envs import Chain
import numpy as np
np.random.seed(7)


def test_qlearning():
    env = Chain(10)
    agent = QLearningAgent(env)
    agent.train(n_steps=10)


def test_crossentropy():
    env = Chain(10)
    agent = CrossEntropyAgent(env)
    agent.train(n_steps=10)


def test_uct():
    env = Chain(10)
    agent = UCT4MDP(env, env.reset(), n_it=10)
    agent.run()

