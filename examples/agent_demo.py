from rlplan.agents import QLearningAgent
from rlplan.agents import DynProgAgent
from rlplan.envs.toy import ToyEnv1
from rlplan.envs.gridworld import GridWorld
from rlplan.prediction import TabularTD

# Discount factor
gamma = 0.9

# Create environment
# env = ToyEnv1()
env = GridWorld(success_probability=0.9)

# Initialize and train dynamic programming agent
dp_agent = DynProgAgent(env, method='policy-iteration', gamma=gamma)
V_dp, _ = dp_agent.train()

# Initialize and train q-learning agent
ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, epsilon=0.2)
V_ql, _ = ql_agent.train(n_steps=1e4)

# Use tabular TD
tab_td = TabularTD(env, dp_agent.policy, gamma, lambd=0.75, learning_rate=None, min_learning_rate=0.01)
V_td = tab_td.run(n_steps=1e4)

# Compare policies
print("Value of dp_agent policy = ", dp_agent.policy.evaluate(env, gamma))
print("Value of ql_agent policy = ", ql_agent.policy.evaluate(env, gamma))
print("TD-estimated value of optimal policy = ", V_td)
