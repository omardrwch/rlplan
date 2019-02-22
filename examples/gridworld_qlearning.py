from rlplan.agents import QLearningAgent
from rlplan.agents import DynProgAgent
from rlplan.envs.gridworld import GridWorld
from rlplan.prediction import TabularTD

# Discount factor
gamma = 0.9

# Create environment
env = GridWorld(success_probability=0.9)

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, epsilon=0.2)
V_ql, _ = ql_agent.train(n_steps=1e4, render=True)

