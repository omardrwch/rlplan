# rlplan

Just some code I'm writing to test things. 


* Implement different environments (Markov Decision Processes) whose optimal policies/value functions can be computed. 
These environments can be used to debug RL algorithms. 
* Implement reinforcement learning and planning algorithms.

# Requirements

* Main requirements are specified in setup.py and requirements.txt
* pyqt5 must be installed in order to render gridworld environment.

# Contents
 
* rlplan.agents
    * Agent: base class, with a function to evaluate the agent
    * Dynamic programming (value/policy iteration)
    * Q-Learning
    * Cross Entropy method

* rlplan.planning
    * [UCT](http://ggp.stanford.edu/readings/uct.pdf) implemented for MDPs
    * [TrailBlazer](http://researchers.lille.inria.fr/~valko/hp/publications/grill2016blazing.pdf)

* rlplan.prediction
    * tabular TD(lambda)

* rlplan.envs
    * Chain
    * Gridworld
    * ToyEnv1, ToyEnv2: simple and small MDPs for debugging 
   
* rlplan.policy
    * Policy: base class
    * FinitePolicy: represents a policy for a finite MDP. Useful methods: sample, evaluate (exact - with linear system),
        constructor from V or Q functions.
 