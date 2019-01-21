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
                 walls=None):

        # Grid dimensions
        self.nrows = nrows
        self.ncols = ncols

        # Probability of going left/right/up/down when choosing the correspondent action
        self.success_probability = success_probability

        # Random state
        self.random = np.random.RandomState(seed)

        # Start coordinate
        self.start_coord = start_coord

        # Positions where the agent gets rewards
        if reward_at is not None:
            self.reward_at = reward_at
        else:
            self.reward_at = ((nrows-1, ncols-1, 1), (nrows-2, ncols-1, -1))
        if walls is not None:
            self.walls = walls
        else:
            self.walls = ((2, 1), (3, 3))

        # Visualization of the grid
        self.grid_ascii = None
        self.grid_idx = None

        # Number of states (walls are considered states)
        self.Ns = self.nrows*self.ncols

        # Actions (string to index & index to string)
        self.actions_str2idx = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self.actions_idx2str = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}

        # Build
        self.build()

        # Ns = 3
        # Na = 2
        # P = np.zeros((Ns, Na, Ns))
        #
        # P[:, 0, :] = np.array([[0.25, 0.5, 0.25], [0.1, 0.7, 0.2], [0.1, 0.8, 0.1]])
        # P[:, 1, :] = np.array([[0.3, 0.3, 0.4], [0.7, 0.2, 0.1], [0.25, 0.25, 0.5]])
        #
        # # Initialize base class
        # states = np.arange(Ns).tolist()
        # action_sets = [np.arange(Na).tolist()]*Ns
        # super().__init__(states, action_sets, P, seed)

    def build(self):
        self.grid_ascii, self.grid_idx = self.build_ascii()
        self.action_sets = self.build_action_sets()

    def build_action_sets(self):
        action_sets = []
        for s_idx in range(self.Ns):
            xx, yy = self.idx2coord(s_idx)
            actions_s = []
            if (xx, yy) in self.walls:
                pass
            else:
               pass
        return action_sets

    def coord2idx(self, x, y):
        assert (x < self.nrows) and (x >= 0)
        assert (y < self.ncols) and (y >= 0)
        idx = np.floor(y + x*self.ncols)
        return int(idx)

    def idx2coord(self, idx):
        assert idx < self.nrows*self.ncols
        y = idx % self.ncols
        x = (idx - y) // self.ncols
        return x, y

    def reward_fn(self, state, action, next_state):
        return 1.0 * (next_state == self.Ns - 1)

    def build_ascii(self):
        grid = [['']*self.ncols for ii in range(self.nrows)]
        grid_idx = [[''] * self.ncols for ii in range(self.nrows)]
        for xx in range(self.nrows):
            for yy in range(self.ncols):
                if (xx, yy) in self.walls:
                    grid[xx][yy] = 'x'
                else:
                    grid[xx][yy] = 'o'
                grid_idx[xx][yy] = str(self.coord2idx(xx, yy)).zfill(3)

        for reward in self.reward_at:
            xx, yy, rr = reward
            if rr >= 0:
                grid[xx][yy] = '+'
            else:
                grid[xx][yy] = '-'

        grid[self.start_coord[0]][self.start_coord[1]] = 'I'

        grid_ascii = ''
        for xx in range(self.nrows+1):
            if xx < self.nrows:
                grid_ascii += str(xx) + 2*' ' + ' '.join(grid[xx]) + '\n'
            else:
                grid_ascii += 3*' ' + ' '.join([str(jj) for jj in range(self.ncols)])

        return grid_ascii, grid_idx

    def render_ascii(self):
        print(self.grid_ascii)


if __name__=='__main__':
    gridworld = GridWorld(nrows=8, ncols=7)
    gridworld.render_ascii()
