import sys
import pytest
import numpy as np
from minigym.envs.toy import ToyEnv1, ToyEnv2
from minigym.policy import FinitePolicy
from minigym.agents import DynProgAgent

# from rlcode.src.basic_rl.mdp import myenvs, mdpmethods

np.random.seed(7)

gamma_seed_params = [
            (0.25, 123),
            (0.75, 789),
            (0.99, 999),
        ]


@pytest.mark.parametrize("seed", [
    (456, 789)
])
def test_toyenv1(seed):
    env = ToyEnv1(seed)
    assert env.state == 0
    observation, reward, done, info = env.step(0)
    assert env.state == observation


@pytest.mark.parametrize("gamma,seed", gamma_seed_params)
def test_bellman_operator_monotonicity_and_contraction(gamma, seed):
    env = ToyEnv1(seed)
    V0 = np.array([1.0,100.0, 1000.0])
    V1 = np.array([2.0,120.0, 1200.0])

    policy_array = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.9]])
    policy = FinitePolicy(policy_array, seed)

    dp_agent = DynProgAgent(env, gamma=gamma)

    TV0, _ = dp_agent.bellman_opt_operator(V0)
    TV1, _ = dp_agent.bellman_opt_operator(V1)

    TpiV0 = dp_agent.bellman_operator(V0, policy)
    TpiV1 = dp_agent.bellman_operator(V1, policy)

    # Test monotonicity
    assert np.greater(TV0, TV1).sum() == 0
    assert np.greater(TpiV0, TpiV1).sum() == 0

    # Test contraction
    norm_tv = np.abs(TV1-TV0).max()
    norm_v = np.abs(V1-V0).max()
    assert norm_tv <= gamma*norm_v


@pytest.mark.parametrize("gamma,seed", gamma_seed_params)
@pytest.mark.parametrize("Ns", [2,4,5,6])
@pytest.mark.parametrize("Na", [3,5])
def test_value_and_policy_iteration(gamma, seed, Ns, Na):
    # Tolerance
    tol = 1e-8

    # Environment
    env = ToyEnv2(Ns, Na, seed)

    dp_agent_val = DynProgAgent(env, gamma=gamma, method='value-iteration')
    dp_agent_pol = DynProgAgent(env, gamma=gamma, method='policy-iteration')
    V_value_it, _ = dp_agent_val.train(val_it_tol=tol)
    V_pol_it, _ = dp_agent_pol.train()

    assert dp_agent_val.policy == dp_agent_pol.policy
    assert np.allclose(V_value_it, V_pol_it, atol=tol, rtol=1e2*tol)