import gym
import rlplan
from rlplan.agents.rl import CrossEntropyAgent
from rlplan.envs import GridWorld

# Defining the environment
env_choice = 0
if env_choice == 0:
    env_ = gym.make("CartPole-v0")
else:
    env_ = GridWorld(success_probability=0.95)

# Create agent
agent = CrossEntropyAgent(env_, gamma=0.99, batch_size=16, percentile=70, horizon=500, learning_rate=0.01)

# Train agent
agent.train(n_steps=50)

# Visualize
if isinstance(env_, rlplan.envs.GridWorld):
    env_.render('auto', agent.policy)
else:
    agent.test()
