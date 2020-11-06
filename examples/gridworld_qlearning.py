from rlplan.agents.rl import QLearningAgent, QLearningUcbAgent
from rlplan.envs import GridWorld, TwoRoomDense, TwoRoomSparse
from rlplan.utils.gridworld_analysis import draw_gridworld_history, draw_grid_world_state_distribution, \
    visualize_exploration, get_action_frequency
import matplotlib.pyplot as plt
from rlplan.agents.planning import DynProgAgent
from rlplan.policy import FinitePolicy

# Discount factor
gamma = 0.9

# Create environment
# env = GridWorld(nrows=5, ncols=5, success_probability=1.0, walls=[])
# env = TwoRoomDense(5, 5, success_probability=0.7)
env = TwoRoomSparse(5, 5, success_probability=0.7)
env_eval = TwoRoomSparse(5, 5, success_probability=1.0)

env.track = True

# Initialize and train q-learning agent
# ql_agent = QLearningAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.05, epsilon=1.0, epsilon_min=0.01)
ql_agent = QLearningUcbAgent(env, gamma=gamma, learning_rate=None, min_learning_rate=0.1, c_expl=4.0)

training_info = ql_agent.train(n_steps=1000, eval_params={'n_sim' : 10})

# # Visualize policy
# env.reset()
# env.render(mode='auto', policy=ql_agent.policy)
#
# # Visualize training curve
# ql_agent.plot_rewards(training_info['rewards_list'], training_info['x_data'], show=True)


dp_agent = DynProgAgent(env, gamma=gamma, method='policy-iteration')


# Draw history
# draw_grid_world_state_distribution(ql_agent.env)

action_freq = get_action_frequency(ql_agent.env)
policy_freq = FinitePolicy(action_freq)

# visualize_exploration(ql_agent.env, show=False)
# env.render('manual')
# plt.show()
#
env_eval.reset()
env_eval.render(policy=ql_agent.policy)
env_eval.reset()
env_eval.render(policy=policy_freq)