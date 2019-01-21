import pytest
import numpy as np
from rlplan.envs import GridWorld
from rlplan.envs.toy import ToyEnv1

np.random.seed(7)

# Parameters for gridworld test
sx_sy_x_y = [
            (8, 8, 0, 0),
            (8, 8, 7, 7),
            (8, 5, 0, 0),
            (8, 5, 7, 4),
            (8, 5, 1, 2),
            (5, 8, 0, 0),
            (5, 8, 4, 7),
            (5, 8, 1, 2),
            (55, 88, 0, 0),
            (55, 88, 20, 4),
            ]

sx_sy_idx = [
            (8, 8, 63),
            (8, 8, 0),
            (8, 5, 0),
            (8, 5, 39),
            (8, 5, 12),
            (5, 8, 0),
            (5, 8, 39),
            (5, 8, 12)
            ]


@pytest.mark.parametrize("seed", [
    (456, 789)
])
def test_toyenv1(seed):
    env = ToyEnv1(seed)
    assert env.state == 0
    observation, reward, done, info = env.step(0)
    assert env.state == observation


@pytest.mark.parametrize("sx, sy, x, y", sx_sy_x_y)
def test_gridworld_coordinate_conversion(sx, sy, x, y):
    gridworld = GridWorld(nrows=sx, ncols=sy)
    idx = gridworld.coord2idx(x, y)
    xx, yy = gridworld.idx2coord(idx)
    assert (xx == x) and (yy == y)


@pytest.mark.parametrize("sx, sy, idx", sx_sy_idx)
def test_gridworld_coordinate_conversion_2(sx, sy, idx):
    gridworld = GridWorld(nrows=sx, ncols=sy)
    x, y = gridworld.idx2coord(idx)
    ii = gridworld.coord2idx(x, y)
    assert (idx == ii)
