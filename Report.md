# Solving the Reacher environment with DDPG
## Report
This document describes the solution for the Unity Reacher environment using DDPG. The algorithm used is based on the [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) created by Udacity.

The algorithm is able to converge and reach a return of +30 in 160 episodes and stay consistently at return of nearly 40 from there onwards.

## Learning Algorithm
The DDPG algorithm used here is composed of two neural networks. A Policy network, or Actor, and a Value network, or Critic network.

## Plot of Rewards
The agent solved the environment in 160 episodes. Once that score was reached it was consistently kept until the end of the training run whcih was set for 500 episodes.

![Plot](images/Figure_1.png)
