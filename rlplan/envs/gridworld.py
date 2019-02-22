"""
TODO:
    - Write more tests
    - Improve render functions
    - Deal with transition probabilities at terminal state
"""


import numpy as np
from rlplan.envs import FiniteMDP
# rlplan.envs.rendering_gw is imported in the constructor of GridWorld


class GridWorld(FiniteMDP):
    """
    Args:g
        seed    (int): Random number generator seed
    """

    def __init__(self,
                 seed=42,
                 nrows=8,
                 ncols=8,
                 start_coord=(0, 0),
                 terminal_states = None,
                 success_probability=1.0,
                 reward_at=None,
                 walls=None,
                 default_reward=-1.0,
                 enable_render=True):

        # Print information
        self.enable_render = enable_render
        if enable_render:
            print("GridWorld rendering is enabled. Press ENTER to close window and ESCAPE to stop rendering.")

        # Grid dimensions
        self.nrows = nrows
        self.ncols = ncols

        # Reward parameters
        self.default_reward = default_reward

        # Default config
        if reward_at is not None:
            self.reward_at = reward_at
        else:
            self.reward_at = {(nrows-1, ncols-1): 1, (nrows-2, ncols-1): -10}
        if walls is not None:
            self.walls = walls
        else:
            self.walls = ((2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1))
        if terminal_states is not None:
            self.terminal_states = self.terminal_states
        else:
            self.terminal_states = ((nrows-1, ncols-1), (nrows-2, ncols-1))

        # Probability of going left/right/up/down when choosing the correspondent action
        # The remaining probability mass is distributed uniformly to other available actions
        self.success_probability = success_probability

        # Random state
        self.random = np.random.RandomState(seed)

        # Start coordinate
        self.start_coord = start_coord

        # Actions (string to index & index to string)
        self.a_str2idx = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self.a_idx2str = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}

        # --------------------------------------------
        # The variables below are defined in _build()
        # --------------------------------------------

        # Mappings (state index) <-> (state coordinate)
        self.index2coord = {}
        self.coord2index = {}

        # Visualization of the grid & rendering
        self.grid_ascii = None
        self.grid_idx = None
        self.renderer = None
        if self.enable_render:
            import rlplan.envs.rendering_gw as rendering
            self.renderer = rendering.Renderer(self)

        # MDP parameters for base class
        self.states = []
        self.action_sets = []
        self.P = None
        self.Ns = None
        self.Na = 4

        # Build
        self._build()
        super().__init__(self.states, self.action_sets, self.P, seed)

    def is_terminal(self, state):
        state_coord = self.index2coord[state]
        return state_coord in self.terminal_states

    def reset(self, state=None):
        if state is None:
            state = self.coord2index[self.start_coord]
        self.state = state
        return state

    def reward_fn(self, state, action, next_state):
        row, col = self.index2coord[next_state]
        if (row, col) in self.reward_at:
            return self.reward_at[(row, col)]
        if (row, col) in self.walls:
            return 0.0
        return self.default_reward

    def _build(self):
        self._build_state_mappings_and_states()
        self._build_action_sets()
        self._build_transition_probabilities()
        self._build_ascii()

    def _build_state_mappings_and_states(self):
        index = 0
        for rr in range(self.nrows):
            for cc in range(self.ncols):
                if (rr, cc) in self.walls:
                    self.coord2index[(rr, cc)] = -1
                else:
                    self.coord2index[(rr, cc)] = index
                    self.index2coord[index] = (rr, cc)
                    index += 1
        self.states = np.arange(index).tolist()
        self.Ns = len(self.states)

    def _build_action_sets(self):
        action_sets = []
        for s_idx in range(self.Ns):
            actions_s = [0, 1, 2, 3]
            action_sets.append(actions_s)
        self.action_sets = action_sets

    def _build_transition_probabilities(self):
        Ns = self.Ns
        Na = self.Na
        self.P = np.zeros((Ns, Na, Ns))
        for s in range(Ns):
            s_coord = self.index2coord[s]
            neighbors = self._get_neighbors(*s_coord)
            valid_neighbors = [neighbors[nn][0] for nn in neighbors if neighbors[nn][1]]
            n_valid = len(valid_neighbors)
            for a in range(Na):  # each action corresponds to a direction
                for nn in neighbors:
                    next_s_coord = neighbors[nn][0]
                    if next_s_coord in valid_neighbors:
                        next_s = self.coord2index[next_s_coord]
                        if a == nn:  # action is successful
                            self.P[s, a, next_s] = self.success_probability \
                                                     + (1-self.success_probability)*(n_valid == 1)
                        elif neighbors[a][0] not in valid_neighbors:
                            self.P[s, a, s] = 1.0
                        else:
                            if n_valid > 1:
                                self.P[s, a, next_s] = (1.0-self.success_probability)/(n_valid - 1)

    def _get_neighbors(self, row, col):
        aux = {}
        aux['left'] = (row, col-1)  # left
        aux['right'] = (row, col+1)  # right
        aux['up'] = (row-1, col)  # up
        aux['down'] = (row+1, col)  # down
        neighbors = {}
        for direction_str in aux:
            direction = self.a_str2idx[direction_str]
            next_s = aux[direction_str]
            neighbors[direction] = (next_s, self._is_valid(*next_s))
        return neighbors

    def _is_valid(self, row, col):
        if (row, col) in self.walls:
            return False
        elif row < 0 or row >= self.nrows:
            return False
        elif col < 0 or col >= self.ncols:
            return False
        return True

    def _build_ascii(self):
        grid = [['']*self.ncols for rr in range(self.nrows)]
        grid_idx = [[''] * self.ncols for rr in range(self.nrows)]
        for rr in range(self.nrows):
            for cc in range(self.ncols):
                if (rr, cc) in self.walls:
                    grid[rr][cc] = 'x '
                else:
                    grid[rr][cc] = 'o '
                grid_idx[rr][cc] = str(self.coord2index[(rr, cc)]).zfill(3)

        for (rr, cc) in self.reward_at:
            rwd = self.reward_at[(rr, cc)]
            if rwd > 0:
                grid[rr][cc] = '+ '
            else:
                grid[rr][cc] = '-'

        grid[self.start_coord[0]][self.start_coord[1]] = 'I '

        grid_ascii = ''
        for rr in range(self.nrows+1):
            if rr < self.nrows:
                grid_ascii += str(rr).zfill(2) + 2*' ' + ' '.join(grid[rr]) + '\n'
            else:
                grid_ascii += 3*' ' + ' '.join([str(jj).zfill(2) for jj in range(self.ncols)])

        self.grid_ascii = grid_ascii
        self.grid_idx = grid_idx

    def display_values(self, values):
        assert len(values) == self.Ns
        grid_values = [['X'.ljust(9)] * self.ncols for ii in range(self.nrows)]
        for s_idx in range(self.Ns):
            v = values[s_idx]
            row, col = self.index2coord[s_idx]
            grid_values[row][col] = ("%0.2f" % v).ljust(9)

        grid_values_ascii = ''
        for rr in range(self.nrows+1):
            if rr < self.nrows:
                grid_values_ascii += str(rr).zfill(2) + 2*' ' + ' '.join(grid_values[rr]) + '\n'
            else:
                grid_values_ascii += 4*' ' + ' '.join([str(jj).zfill(2).ljust(9) for jj in range(self.ncols)])
        print(grid_values_ascii)

    def print_transition_at(self, row, col, action):
        s_idx = self.coord2index[(row, col)]
        if s_idx < 0:
            print("wall!")
            return
        a_idx = self.a_str2idx[action]
        for next_s_idx, prob in enumerate(self.P[s_idx, a_idx]):
            if prob > 0:
                print("to (%d, %d) with prob %f" % (self.index2coord[next_s_idx]+(prob,)))

    def render_ascii(self):
        print(self.grid_ascii)

    def render(self):
        if self.enable_render:
            self.renderer.run()
        else:
            print("Rendering not enabled. Call constructor with enable_rendering=True.")

    def close(self):
        pass


if __name__ == '__main__':
    gw = GridWorld(nrows=8, ncols=7, success_probability=0.7)
    gw.render_ascii()

    from rlplan.agents.dynprog import DynProgAgent
    dynprog = DynProgAgent(gw, method='policy-iteration', gamma=0.9)
    V, _ = dynprog.train()
    gw.display_values(V)

    # run
    env = gw
    state = env.reset()
    done = False
    env.render()
    while not done:
        action = dynprog.policy.sample(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        env.render()

