from rlplan.agents.rl import QLearningAgent, QLearningUcbAgent
from rlplan.envs import GridWorld, TwoRoomDense
from rlplan.utils.draw_gridworld_history import draw_gridworld_history, draw_grid_world_state_distribution

# Discount factor
gamma = 0.9

# Create environment
# env = GridWorld(nrows=8, ncols=10, success_probability=0.67, walls=[])
env = TwoRoomDense(8, 8, success_probability=0.7)
env.track = True

# Initialize and train q-learning agent
# ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, epsilon=1.0, epsilon_min=0.05)
ql_agent = QLearningUcbAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, c_expl=10.0)

training_info = ql_agent.train(n_steps=50000, eval_params={'n_sim' : 10})

# # Visualize policy
# env.reset()
# env.render(mode='auto', policy=ql_agent.policy)
#
# Visualize training curve
ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)

# Draw history
draw_grid_world_state_distribution(ql_agent.env)
env.render('manual')