"""
A set of useful environments for debugging algorithms.
"""

import numpy as np
from rlplan.envs import GridWorld


class TwoRoomDense(GridWorld):
    """
    Gridworld consisting of two rooms, reward = distance to goal.
    """
    def __init__(self, nrows=3,
                 ncols=3,
                 success_probability=1.0,
                 enable_render=True,
                 seed_val=42):
        self.goal_coord = (nrows - 1, ncols - 1)
        start_coord = (0, 0)
        reward_at = {self.goal_coord: 1}  # just for rendering, reward_fn is redefined in this class
        terminal_states = (self.goal_coord, )
        default_reward = 0.0

        assert nrows >= 3
        assert ncols >= 3

        # defining walls
        middle_col = ncols // 2
        middle_row = nrows // 2
        walls = ()
        for row in range(nrows):
            if row != middle_row:
                walls += ((row, middle_col),)
        #

        super().__init__(seed_val,
                         nrows,
                         ncols,
                         start_coord,
                         terminal_states,
                         success_probability,
                         reward_at,
                         walls,
                         default_reward,
                         enable_render)

    def reward_fn(self, state, action, next_state):
        x0, y0 = self.index2coord[next_state]
        x1, y1 = self.goal_coord

        dist = (x1-x0)**2.0 + (y1-y0)**2.0
        sigma = self.nrows*self.ncols
        r = np.exp(-dist/sigma)
        return r


if __name__=='__main__':
    gw = TwoRoomDense(9, 9, success_probability=1.0)

    from rlplan.agents.planning import DynProgAgent
    dynprog = DynProgAgent(gw, method='policy-iteration', gamma=0.9)
    V, _ = dynprog.train()
    gw.display_values(V)

    # run
    gw.render(mode='auto', policy=dynprog.policy)

    # reset
    gw.reset()

