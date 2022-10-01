# Group12_DRL_Project
 
## About  
This repository is the final project for the course "Deep reinforcement learning". It aims at offering a reimplementation of the algorithm PAIRED presented in the paper:
Dennis, M.*, Jaques, N.*, Vinitsky, E., Bayen, A., Russell, S., Critch, A., & Levine, S., Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design, Neural Information Processing Systems (NeurIPS), Virtual (2020).
Their code can be found in this Github repository: https://github.com/google-research/google-research/tree/master/social_rl/adversarial_env

## Structure
The project is structured into several modules, namely:
 * Environment
 * PPO-Agent
 The environment is an extension of the Open-AI Gym FrozenLake as it offers functionality for building the environment as well as running the environment in a multi-agent scenario. This is implemented in multifrozenlake.py. Adversarial.py offers functionality for the adversary to build the environment such as step_adversary or reset_agent. 
 The agent is learning based on Proximal Policy Optimization using an Actor and a Critic Network both to be found in the folder "agents". 
 The PAIRED algorithm is located in the driver.py.

NOTE: Due to time constraints it was not possible to complete this project, therefore the training does not yet work.

## Contact
Jens Huth <jehuth@uni-osnabrueck.de>
Nils Niehaus <nniehaus@uni-osnabrueck.de>
Nicole Rogalla <nrogalla@uni-osnabrueck.de>

## Additional Environment
https://github.com/H-27/env_FrozenLake-PPO.git
