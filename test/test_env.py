import pytest
import numpy as np
from rlplan.envs import GridWorld
from rlplan.envs.toy import ToyEnv1, ToyEnv2
from rlplan.envs.deterministic_trees import ToyTree1
from rlplan.envs import Chain

np.random.seed(7)


@pytest.mark.parametrize("seed", [
    456, 789
])
def test_toyenv1(seed):
    env = ToyEnv1(seed_val=seed)
    assert env.state == 0
    observation, reward, done, info = env.step(0)
    assert env.state == observation


@pytest.mark.parametrize("seed", [
    456, 789
])
def test_toyenv2(seed):
    env = ToyEnv2(seed_val=seed)
    assert env.state == 0
    observation, reward, done, info = env.step(0)
    assert env.state == observation


@pytest.mark.parametrize("seed", [
    456, 789
])
def test_gridworld(seed):
    env = GridWorld(seed_val=seed)
    assert env.state == 0
    for tt in range(50):
        observation, reward, done, info = env.step(env.action_space.sample())
        assert env.state == observation
        if done:
            break


def test_toytree1():
    env = ToyTree1()
    assert env.state == 0
    for episode in range(10):
        done = False
        while not done:
            observation, reward, done, info = env.step(env.action_space.sample())
            assert env.state == observation
            if reward > 0:
                assert observation == 6
            if done:
                assert env.state in [3, 4, 5, 6]


@pytest.mark.parametrize("L", [
    2, 5, 8, 11
])
def test_chain(L):
    print("-------", L)
    env = Chain(L)
    assert env.state == 0
    done = False
    time = 0
    while not done:
        observation, reward, done, info = env.step(1)
        assert env.state == observation
        if done:
            assert env.state == L-1
        if reward > 0:
            assert observation == L-1
        time += 1
    assert time == L
