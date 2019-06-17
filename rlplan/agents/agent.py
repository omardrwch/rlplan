"""
Base class for any agent
"""
import copy
import numpy as np
import matplotlib.pyplot as plt


def compute_discounted_rewards(rewards, discount):
    n_sim = len(rewards)
    discounted_rewards = np.zeros(n_sim)
    for sim in range(n_sim):
        T = len(rewards[sim])
        multiplier = np.power(discount, np.arange(T))
        reward_array = np.array(rewards[sim])
        discounted_rewards[sim] = np.sum(reward_array * multiplier)
    return discounted_rewards


class Agent:

    """
    Base class for an agent

    :param id: string to identify the agent
    :param env: environment
    :param policy: rlplan.policy.Policy object
    """
    def __init__(self):
        self.id = None
        self.env = None
        self.policy = None

    def eval(self, n_sim=1, horizon=150, discount=1.0, env=None):
        """
        :param n_sim: number of monte carlo simulations
        :param horizon: maximum number of steps in a single simulation
        :param discount: discount factor, set to 1.0 to get "pure" accumulated rewards
        :param env: environment where to evaluate the agent, if None, use a deep copy of self.env
        :return: array of size n_sim with the sum of (discounted) rewards in each simulation
        """
        assert self.policy is not None, "Cannot evaluate agent, policy is not defined"
        if env is None:
            env = copy.deepcopy(self.env)

        # disable tracking
        env.track = False

        rewards = []
        for sim in range(n_sim):
            rewards.append([])
            state = env.reset()
            for tt in range(int(horizon)):
                action = self.policy.sample(state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
                state = next_state
                rewards[sim].append(reward)
        discounted_rewards = compute_discounted_rewards(rewards, discount)
        return discounted_rewards

    def plot_rewards(self, rewards_list, x_data=None, fignum=None, show=False):
        """
        :param rewards_list:  list of length n_steps containing arrays of discounted rewards
        :param x_data: data to plot in x axis (e.g. the number of iterations)
        :param fignum: number of plt figure
        :param show: if True, plt.show() is called
        :return:
        """
        n_steps = len(rewards_list)
        mean = np.zeros(n_steps)
        std = np.zeros(n_steps)
        for step in range(n_steps):
            mean[step] = rewards_list[step].mean()
            std[step] = rewards_list[step].std()

        if fignum is None:
            plt.figure('rewards')

        if x_data is None:
            x_data = np.arange(n_steps)

        plt.plot(x_data, mean, label=self.id)
        plt.fill_between(x_data, mean-std, mean+std, alpha=0.25)
        plt.legend()
        plt.ylabel("Reward")

        if show:
            plt.show()



