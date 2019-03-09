import pytest
from rlplan.envs.bandit import BernoulliBandit
from rlplan.planning import UCT4MDP
import numpy as np
np.random.seed(7)


bandit_params = [[0.4, 0.8, 0.6],
                 [0.6, 0.3, 0.8],
                 [0.1, 0.3, 0.5, 0.2, 0.4, 0.8]]


@pytest.mark.parametrize("probs", bandit_params)
def test1_uct(probs):
    best_action = np.argmax(probs)
    env = BernoulliBandit(probs)
    state = 0
    uct = UCT4MDP(env, state, fixed_depth=True, max_depth=1, max_rollout_it=0, gamma=1.0, cp=1.0, n_it=5000)
    value, action = uct.run()
    assert action == best_action
    assert np.abs(value-probs[best_action]) < 0.1
