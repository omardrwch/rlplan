import gym
import numpy as np


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)
        self.render = env.render

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


class DiscreteWrapper(gym.Wrapper):
    """
    Wrapper to any gym discrete environment, so that it works as rlplan.envs.FiniteMDP
    """
    def __init__(self, env):
        super(DiscreteWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.env = env

    def available_actions(self, state=0):
        return list(range(self.env.action_space.n))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
