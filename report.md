## Introduction
Solving the environment require an average total reward of over 195 for Cartpole-v0.
over 100 consecutive episodes. A pole is attached by an joint to a cart, which moves along a track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole > 15 degrees from vertical, or the cart moves > 2.4 units from the center.
## Learning Algorithms
The deep Q-learning came to solve problems with huge space state that the Q-learning is not able to solve. The deep Q-network takes as input the state and outputs Q-values for each possible action in the current state. The biggest q-value corresponds to the best action. We implemented the DQN and the Double Deep DQN which is an improved version of DQN using the:

 * Experience replay buffer: it helps to avoid two problems which are forgetting previous experiences and the correlations in data. In reinforcement learning we receive at each time step a tuple composed by the state, the action, the reward, and the new state. In order to make our agent learn from the past policies, every tuple is stored in the experience replay buffer. To break correlations between data, experiences are sampled from the replay buffer randomly. This will help action values from diverging.

 * Fixed target Q-network: we used to use the same weights on the predicted and the target values. The predicted Q-value is removed closer to the target but also the target is removed with the same steps as the same weights are used for both of them. As a result, we will be chasing the target value and we will have a big oscillation in the training. To break correlations between the predicted and the target value, we use the fixed target Q-network to update the weights of the target. The Q-targets are calculated using the fixed parameters w − of the separate network.
 ## Model architecture
 The DQN agent has a target and local networks having the same architecture:
 * 1 fully connected layer of size 64.
 * 1 fully connected layer of size 256.
 * 1 fully connected layer of size 256.
 * 1 output layer of size 2 (the size of the action space)
 ## Hyperparameters
 Our agent was trained using the follwing hyperparameters:

 * Buffer size: the size of the experience replay buffer is 10 000
 * Batch size: the batch size of the training is 64
 * Gamma: the discount factor 0.99
 * learning rate coefficient for soft update of target parameters is 0.001
 * The agent is updated after every 10 time steps
 
## How to train our agent
For each episode, we start by giving the initial state of our environment to the agent. Then, for each time step we give our agent the current state of our environment and he will return the action that he will perform. After performing this action, the environment will return the new state, the reward and if the game is finished or not. The agent will save this experience in the replay buffer.

When we reach the score of 195 we stop the training of our agent.
## Results
* DQN :
The agent was able to solve the environment after 531 episodes with the average score of 195.26.
![Image of Yaktocat](https://github.com/yasmine1998/Deep_RL_Project/blob/main/images/DQN.png?raw=true)
![Image of Yaktocat](https://github.com/yasmine1998/Deep_RL_Project/blob/main/images/DQN_graph.png?raw=true)
* DDQN :
The agent was able to solve the environment after 1578 episodes with the average score of 195.10.
![Image of Yaktocat](https://github.com/yasmine1998/Deep_RL_Project/blob/main/images/DDQN.png?raw=true)
![Image of Yaktocat](https://github.com/yasmine1998/Deep_RL_Project/blob/main/images/DDQN_graph.png?raw=true)
