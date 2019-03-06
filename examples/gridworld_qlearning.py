from rlplan.agents import QLearningAgent
from rlplan.envs.gridworld import GridWorld

# Discount factor
gamma = 0.9

# Create environment
env = GridWorld(success_probability=0.75)

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, epsilon=0.2)
V_ql, training_info = ql_agent.train(n_steps=25000)

# Visualize policy
env.reset()
env.render(mode='auto', policy=ql_agent.policy)

# Visualize learning curve
ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)