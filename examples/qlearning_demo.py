from rlplan.agents.rl import QLearningAgent
from rlplan.envs import Chain
from rlplan.utils.wrappers import DiscreteWrapper
import gym
import numpy as np
import matplotlib.pyplot as plt

# Discount factor
gamma = 0.9

# Create environment
env_choice = 0
if env_choice == 0:
    env = Chain(L=20)
else:
    env = gym.make("FrozenLake-v0")
    env = DiscreteWrapper(env)

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, min_learning_rate=0.1, epsilon=1.0, epsilon_decay=1.0,
                          epsilon_min=0.05, rmax=0.0)
training_info = ql_agent.train(n_steps=15e4, eval_params={'n_sim' : 1})

# Visualize learning curve
ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)


# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)
#
#
# plt.figure()
# plt.plot(1+np.arange(training_info['n_episodes']), training_info['episode_total_reward'])
# plt.plot(running_mean(training_info['episode_total_reward'], 500), 'r')
# plt.show()

