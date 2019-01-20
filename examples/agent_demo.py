from rlplan.agents import QLearningAgent
from rlplan.agents import DynProgAgent
from rlplan.envs.toy import ToyEnv1

# Discount factor
gamma = 0.9

# Create environment
env = ToyEnv1()

# Initialize and train dynamic programming agent
dp_agent = DynProgAgent(env, method='policy-iteration', gamma=gamma)
V_dp, _ = dp_agent.train()

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, epsilon=0.2)
V_ql, _ = ql_agent.train(n_steps=1e4)

# Compare policies
print("Value of dp_agent policy = ", dp_agent.policy.evaluate(env, gamma))
print("Value of ql_agent policy = ", ql_agent.policy.evaluate(env, gamma))