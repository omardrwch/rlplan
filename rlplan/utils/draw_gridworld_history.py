from PIL import Image, ImageDraw, ImageColor
import numpy as np


def draw_gridworld_history(env, base_length=100):
    """
    :param env: grid world environment
    :param base_length: base length in pixels
    :return:
    """
    N = env.nrows*base_length
    M = env.ncols*base_length
    im = Image.new('RGB', (M, N))
    d = ImageDraw.Draw(im)
    line_color = (0, 0, 255)

    assert len(env.history) > 1, "To draw the history, at least 2 points are required!"

    for ii in range(len(env.history)):
        state, action, reward, next_state, done = env.history[ii]

        if done:
            continue

        y0, x0 = env.index2coord[state]
        y1, x1 = env.index2coord[next_state]

        x0 = base_length*x0 + int(0.5*base_length)
        y0 = base_length*y0 + int(0.5*base_length)
        x1 = base_length*x1 + int(0.5*base_length)
        y1 = base_length*y1 + int(0.5*base_length)

        d.line([(x0, y0), (x1, y1)], fill=line_color, width=4)

    im.show()


def draw_grid_world_state_distribution(env, base_length=100):
    """
    :param env: grid world environment
    :param base_length: base length in pixels
    :return:
    """
    base_radius = base_length//2
    N = env.nrows*base_length
    M = env.ncols*base_length
    im = Image.new('RGB', (M, N))
    d = ImageDraw.Draw(im)

    # circle_color = (0, 0, 255)

    assert len(env.history) > 1, "To draw the history, at least 2 points are required!"

    state_freq = np.zeros(env.Ns)
    H = len(env.history)

    for ii in range(H):
        state, action, reward, next_state, done = env.history[ii]
        state_freq[state] += 1.0/H

    # normalize
    state_freq = state_freq/state_freq.max()

    env.state_freq = state_freq

    for state in range(env.Ns):
        y, x = env.index2coord[state]
        x = base_length*x + int(0.5*base_length)
        y = base_length*y + int(0.5*base_length)

        r = base_radius*state_freq[state]
        circle_color = (0, 0, int(state_freq[state] * 255))

        d.ellipse((x - r, y - r, x + r, y + r), fill=circle_color)

    im.show()


if __name__ == '__main__':
    from rlplan.envs import GridWorld
    from rlplan.agents.planning import DynProgAgent

    env = GridWorld()
    dp_agent = DynProgAgent(env, method='policy-iteration', gamma=0.9)
    dp_agent.train()

    env.track = True
    for step in range(15):
        env.step(dp_agent.policy.sample(env.state))
    draw_gridworld_history(env)

    env.render('manual')
