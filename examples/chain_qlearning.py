from rlplan.agents import QLearningAgent
from rlplan.envs import Chain

# Discount factor
gamma = 0.9

# Create environment
env = Chain(L=20)

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, min_learning_rate=0.1, epsilon=1.0, epsilon_decay=1.0)
training_info = ql_agent.train(n_steps=5e4)

# Visualize learning curve
ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)
