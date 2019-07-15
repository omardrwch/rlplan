from rlplan.agents.rl import QLearningAgent, QLearningUcbAgent
from rlplan.envs import GridWorld, TwoRoomDense, TwoRoomSparse
from rlplan.utils.gridworld_analysis import draw_gridworld_history, draw_grid_world_state_distribution, \
    visualize_exploration
import matplotlib.pyplot as plt

# Discount factor
gamma = 0.9

# Create environment
# env = GridWorld(nrows=5, ncols=5, success_probability=1.0, walls=[])
env = TwoRoomDense(5, 5, success_probability=1.0)
# env = TwoRoomSparse(5, 5, success_probability=1.0)
env.track = True

# Initialize and train q-learning agent
# ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.05, epsilon=1.0, epsilon_min=0.01)
ql_agent = QLearningUcbAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, c_expl=4.0)

training_info = ql_agent.train(n_steps=15000, eval_params={'n_sim' : 10})

# # Visualize policy
# env.reset()
# env.render(mode='auto', policy=ql_agent.policy)
#
# # Visualize training curve
# ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)

# Draw history
draw_grid_world_state_distribution(ql_agent.env)
visualize_exploration(ql_agent.env, show=False)
env.render('manual')
plt.show()