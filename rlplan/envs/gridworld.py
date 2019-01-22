import numpy as np
from rlplan.envs import FiniteMDP


class GridWorld(FiniteMDP):
    """
    Args:
        seed    (int): Random number generator seed
    """

    def __init__(self, seed=42, nrows=8, ncols=8,
                 start_coord=(0,0),
                 success_probability=0.8,
                 reward_at=None,
                 walls=None,
                 default_reward=-1.0,
                 hit_wall_possible=True):

        # Grid dimensions
        self.nrows = nrows
        self.ncols = ncols

        # Parameters
        self.hit_wall_possible = hit_wall_possible

        # Probability of going left/right/up/down when choosing the correspondent action
        # The remaining probability mass is distributed uniformly to other available actions
        self.success_probability = success_probability

        # Random state
        self.random = np.random.RandomState(seed)

        # Start coordinate
        self.start_coord = start_coord

        # Default reward
        self.default_reward = default_reward

        # Positions where the agent gets rewards
        if reward_at is not None:
            self.reward_at = reward_at
        else:
            self.reward_at = {(nrows-1, ncols-1): 10, (nrows-2, ncols-1): -10}
        if walls is not None:
            self.walls = walls
        else:
            self.walls = ((2, 1), (3, 3))

        # Visualization of the grid
        self.grid_ascii = None
        self.grid_idx = None

        # Number of states (walls are considered states) and actions
        self.Ns = self.nrows*self.ncols
        self.Na = 4

        # States
        self.states = np.arange(self.Ns)

        # Actions (string to index & index to string)
        self.actions_str2idx = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self.actions_idx2str = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}

        # Transition probabilities
        self.P = None

        # Build
        self.build()
        super().__init__(self.states, self.action_sets, self.P, seed)

    def reward_fn(self, state, action, next_state):
        row, col = self.idx2coord(state)
        if (row, col) in self.reward_at:
            return self.reward_at[(row, col)]
        if (row, col) in self.walls:
            return 0.0
        return self.default_reward

    def build(self):
        self.grid_ascii, self.grid_idx = self.build_ascii()
        self.action_sets = self.build_action_sets()
        self.P = self.build_transition_probabilities()

    def build_transition_probabilities(self):
        Ns = self.Ns
        Na = self.Na  # = 4
        P = np.zeros((Ns, Na, Ns))
        for s_idx in range(Ns):
            srow, scol = self.idx2coord(s_idx)
            neighbors = self.get_neighbors(srow, scol)
            valid_directions = [direction for direction in neighbors if self.is_valid(*neighbors[direction])]
            for a_idx in range(Na):
                action_dir = self.actions_idx2str[a_idx]

                if len(valid_directions) == 0 or ((srow, scol) in self.walls) or (action_dir not in valid_directions):
                    P[s_idx, a_idx, s_idx] = 1.0
                    continue

                elif len(valid_directions) == 1:
                    if action_dir == valid_directions[0]:
                        next_s_coord = neighbors[dir]
                        next_s_idx = self.coord2idx(*next_s_coord)
                        P[s_idx, a_idx, next_s_idx] = 1.0
                    else:
                        P[s_idx, a_idx, s_idx] = 1.0
                    continue

                for dir in valid_directions:  # iterating next states
                    next_s_coord = neighbors[dir]
                    next_s_idx = self.coord2idx(*next_s_coord)
                    if action_dir == dir:
                        next_s_prob = self.success_probability
                    else:
                        next_s_prob = (1.0-self.success_probability)/(len(valid_directions)-1)
                    P[s_idx, a_idx, next_s_idx] = next_s_prob

        return P

    def actions_at(self, rr, cc):
        if (rr, cc) in self.walls:
            return []
        return [self.actions_idx2str[a] for a in self.action_sets[self.coord2idx(rr, cc)]]

    def build_action_sets(self):
        action_sets = []
        for s_idx in range(self.Ns):
            rr, cc = self.idx2coord(s_idx)
            actions_s = []
            if ((rr, cc) in self.walls) or self.hit_wall_possible:
                actions_s = [0, 1, 2, 3]  # available, but do nothing
            else:
                neighbors = self.get_neighbors(rr, cc)
                for direction in neighbors:
                    if self.is_valid(*neighbors[direction]):
                        actions_s.append(self.actions_str2idx[direction])
            action_sets.append(actions_s)
        return action_sets

    def get_neighbors(self, row, col):
        neighbors = {}
        neighbors['left'] = (row, col-1)  # left
        neighbors['right'] = (row, col+1)  # right
        neighbors['up'] = (row-1, col)  # up
        neighbors['down'] = (row+1, col)  # down
        return neighbors

    def is_valid(self, row, col):
        if (row, col) in self.walls:
            return False
        elif row < 0 or row >= self.nrows:
            return False
        elif col < 0 or col >= self.ncols:
            return False
        return True

    def coord2idx(self, r, c):
        assert (r < self.nrows) and (r >= 0)
        assert (c < self.ncols) and (c >= 0)
        idx = np.floor(c + r*self.ncols)
        return int(idx)

    def idx2coord(self, idx):
        assert idx < self.nrows*self.ncols
        c = idx % self.ncols
        r = (idx - c) // self.ncols
        return r, c

    def build_ascii(self):
        grid = [['']*self.ncols for ii in range(self.nrows)]
        grid_idx = [[''] * self.ncols for ii in range(self.nrows)]
        for rr in range(self.nrows):
            for cc in range(self.ncols):
                if (rr, cc) in self.walls:
                    grid[rr][cc] = 'x '
                else:
                    grid[rr][cc] = 'o '
                grid_idx[rr][cc] = str(self.coord2idx(rr, cc)).zfill(3)

        for (rr, cc) in self.reward_at:
            rwd = self.reward_at[(rr, cc)]
            if rwd >= 0:
                grid[rr][cc] = '+ '
            else:
                grid[rr][cc] = '- '

        grid[self.start_coord[0]][self.start_coord[1]] = 'I '

        grid_ascii = ''
        for rr in range(self.nrows+1):
            if rr < self.nrows:
                grid_ascii += str(rr).zfill(2) + 2*' ' + ' '.join(grid[rr]) + '\n'
            else:
                grid_ascii += 3*' ' + ' '.join([str(jj).zfill(2) for jj in range(self.ncols)])

        return grid_ascii, grid_idx

    def display_values(self, values):
        assert len(values) == self.Ns
        grid_values = [[0] * self.ncols for ii in range(self.nrows)]
        for s_idx in range(self.Ns):
            v = values[s_idx]
            row, col = self.idx2coord(s_idx)
            grid_values[row][col] = ("%0.2f" % v).ljust(7)
        print(grid_values)
        return grid_values

    def print_transition_at(self, row, col, action):
        s_idx = self.coord2idx(row, col)
        a_idx = self.actions_str2idx[action]
        for next_s_idx, prob in enumerate(self.P[s_idx, a_idx]):
            if prob > 0:
                print("to (%d, %d) with prob %f" % (self.idx2coord(next_s_idx)+(prob,)))

    def render_ascii(self):
        print(self.grid_ascii)


if __name__=='__main__':
    gw = GridWorld(nrows=8, ncols=7)
    gw.render_ascii()

    from rlplan.agents.dynprog import DynProgAgent
    dynprog = DynProgAgent(gw, method='policy-iteration', gamma=0.95)
    V, _ = dynprog.train()

