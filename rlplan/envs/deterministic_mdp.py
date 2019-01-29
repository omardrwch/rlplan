"""
Class for deterministic MDP
"""

from rlplan.envs import FiniteMDP
import numpy as np


class DeterministicFiniteMDP(FiniteMDP):
    """
    Note:
        States and actions must be integers starting from 0

    :param transitions: list of transitions [ (state, action, next_state), ...  ]
    :param rewards: dictionary of rewards { (state,action,next_state): reward, ... }
                    or { (state,action): reward, ... }
                    or { (state,): reward, ... }
    """
    def __init__(self, transitions, rewards):
        # Analyze reward structure
        self.n_key = len(list(rewards.items())[0][0])
        assert 0 < self.n_key <= 3, "Error in DeterministicFiniteMDP: misspecified rewards"
        self.reward_type = ['state', 'state_action', 'state_action_nextstate'][self.n_key-1]
        self.rewards = rewards

        # Counting the number of states and actions
        state_set = set()
        action_set = set()
        for entry in transitions:
            state_set.add(entry[0])
            action_set.add(entry[1])
            state_set.add(entry[2])

        states = np.arange(max(state_set)+1, dtype=np.int64).tolist()
        actions = np.arange(max(action_set)+1, dtype=np.int64).tolist()

        # Build transition probabilities and action sets
        Ns = len(states)
        Na = len(actions)
        P = np.zeros((Ns, Na, Ns))
        actions_dict = {}

        for (state, action, next_state) in transitions:
            P[state, action, :] = 0.0
            P[state, action, next_state] = 1.0
            if state in actions_dict:
                actions_dict[state].add(action)
            else:
                actions_dict[state] = {action}

        action_sets = []
        for state in states:
            action_sets.append(list(actions_dict[state]))

        # Initialize super
        super().__init__(states, action_sets, P, seed=42)

    def reward_fn(self, state, action, next_state):
        try:
            if self.reward_type == 'state':
                return self.rewards[(state,)]
            elif self.reward_type == 'state_action':
                return self.rewards[(state, action)]
            elif self.reward_type == 'state_action_nextstate':
                return self.rewards[(state, action, next_state)]
        except KeyError:
            return 0.0
