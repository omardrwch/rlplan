from rlplan.agents.rl import QLearningAgent
from rlplan.envs import Chain
from rlplan.utils.wrappers import DiscreteWrapper
import gym

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
ql_agent = QLearningAgent(env, gamma=gamma, min_learning_rate=0.01, epsilon=1.0, epsilon_decay=1.0, rmax=0.0)
training_info = ql_agent.train(n_steps=5e4, eval_params={'n_sim' : 1})

# Visualize learning curve
ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)

