from abc import ABC, abstractmethod
import numpy as np
from gym import spaces
import gym


class FiniteMDP(gym.Env, ABC):

    """
    Base class for a finite MDP.

    Args:
        states      (list): List of legnth S containing the indexes of the states, e.g. [0,1,2]
        action_sets (list): List containing the actions available in each state, e.g. [[0,1], [2,3]],
                            action_sets[i][j] returns the index of the j-th available action in state i
        P    (numpy.array): Array of shape (Ns, Na, Ns) containing the transition probabilities,
                            P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a). Na is the total number of actions.

    Attributes:
        Ns   (int): Number of states
        Na   (int): Number of actions
        random   (np.random.RandomState) : random number generator
    """
    def __init__(self, states, action_sets, P, seed_val=42):
        super().__init__()
        self.states = states
        self.action_sets = action_sets
        self.actions = list(set().union(*action_sets))
        self.Ns = len(states)
        self.Na = len(self.actions)
        self.P = P

        self.action_space = spaces.Discrete(self.Na)
        self.observation_space = spaces.Discrete(self.Ns)

        self.state = None
        self.random = np.random.RandomState(seed_val)
        self.reset()
        self._check()

    def reset(self, state=0):
        """
        Reset the environment to a default state or to a given state.

        Args:
            state(int)

        Returns:
            state (object)
        """
        self.state = state
        return self.state

    def _check(self):
        """
        Check consistency of the MDP
        """
        # Check that P[s,a, :] is a probability distribution
        for s in range(self.Ns):
            for a in self.available_actions(s):
                assert abs(self.P[s, a, :].sum() - 1.0) < 1e-15

    @abstractmethod
    def reward_fn(self, state, action, next_state):
        """
        Reward function

        Args:
            state      (int): current state
            action     (int): current action
            next_state (int): next state

        Returns:
            reward (float)
        """
        pass

    def available_actions(self, state=None):
        if state is not None:
            return self.action_sets[state]
        else:
            return self.action_sets[self.state]

    def seed(self, seed=42):
        """
        Reset random number generator
        """
        self.random = np.random.RandomState(seed)

    def sample_transition(self, s, a):
        """
        Sample a transition s' from P(s'|s,a).

        Args:
            s (int): index of state
            a (int): index of action

        Returns:
            ss (int): index of next state
        """
        prob = self.P[s, a, :]
        s_ = self.random.choice(self.states, p=prob)
        return s_

    def step(self, action):
        """
        Execute a step. Similar to gym function [1].
        [1] https://gym.openai.com/docs/#environments

        Args:
            action (int): index of the action to take

        Returns:
            observation (object)
            reward      (float)
            done        (bool)
            info        (dict)
        """
        assert action in self.available_actions(), "Invalid action!"
        next_state = self.sample_transition(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        done = self.is_terminal(self.state)
        info = {}
        self.state = next_state

        observation = next_state
        return observation, reward, done, info

    def is_terminal(self, state):
        """
        Returns true if a state is terminal.
        """
        return False

    def print(self):
        """
        Print the structure of the MDP.
        The complexity is Na*(Ns**2), but you don't want to print large stuff anyway :)
        """
        indent = '    '
        for s in self.states:
            As = self.action_sets[s]
            print(("State %d" + indent)%s)
            for a in As:
                print(indent + "Action ", a)
                for ss in self.states:
                    print(2*indent + 'prob -> %d = %0.2f'%(ss, self.P[s, a, ss]))
            print("~~~~~~~~~~~~~~~~~~~~")


