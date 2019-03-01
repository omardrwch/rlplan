from rlplan.agents import QLearningAgent
from rlplan.envs.gridworld import GridWorld

# Discount factor
gamma = 0.9

# Create environment
env = GridWorld(success_probability=0.99)

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, epsilon=0.2)
V_ql, _ = ql_agent.train(n_steps=1e4)

# Visualize policy
env.reset()
env.render(mode='auto', policy=ql_agent.policy)
