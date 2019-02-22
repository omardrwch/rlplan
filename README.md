# rlplan
______________

The goals of this package are: 

* Implement different environments (Markov Decision Processes) whose optimal policies/value functions are known;
* Implement and evaluate reinforcement learning and planning algorithms.

# Requirements
______________

* Main requirements are specified in setup.py and requirements.txt
* pyqt5 must be installed in order to render gridworld environment.

# Contents
______________
 
* rlplan.agents
    * Dynamic programming (value/policy iteration)
    * Q-Learning

* rlplan.planning
    * [TrailBlazer](http://researchers.lille.inria.fr/~valko/hp/publications/grill2016blazing.pdf) 

* rlplan.prediction
    * Algorithms for estimating the value of a policy: tabular TD(lambda)

* rlplan.envs
    * Abstract class for finite MDP
    * ToyEnv1, ToyEnv2: simple and small MDPs for debugging 
   
* rlplan.policy
    * FinitePolicy: represents a policy for a finite MDP. Useful methods: sample, evaluate (exact - with linear system),
        constructor from V or Q functions.
 