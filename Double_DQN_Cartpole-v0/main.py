import gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time

from collections import deque
from ddqn_agent import Agent, FloatTensor
from replay_buffer import ReplayMemory, Transition
from torch.autograd import Variable

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 64
gamma = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE = 10

num_episodes = 20000
print_every = 10
hidden_dim = 16  ##
min_eps = 0.01
max_eps_episode = 200  # 50

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, directory="monitors", force=True)

space_dim = env.observation_space.shape[0]  # n_spaces
action_dim = env.action_space.n  # n_actions
print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

threshold = env.spec.reward_threshold
print('threshold: ', threshold)

# Train agent for Double Deep Q_network
agent = Agent(space_dim, action_dim, hidden_dim)


def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    ##  if i_epsiode --> max_episode, ret_eps --> min_eps
    ##  if i_epsiode --> 1, ret_eps --> 1
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps


def save(directory, filename):
    torch.save(agent.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(agent.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))


def run_episode(env, agent, eps):
    """Play an epsiode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """
    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        action = agent.get_action(FloatTensor([state]), eps)

        next_state, reward, done, _ = env.step(action.item())

        total_reward += reward

        if done:
            reward = -1

        # Store the transition in memory
        agent.replay_memory.push(
            (FloatTensor([state]),
             action,  # action is already a tensor
             FloatTensor([reward]),
             FloatTensor([next_state]),
             FloatTensor([done])))

        if len(agent.replay_memory) > BATCH_SIZE:
            batch = agent.replay_memory.sample(BATCH_SIZE)

            agent.learn(batch, gamma)

        state = next_state

    return total_reward


def train():
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score = run_episode(env, agent, eps)

        scores_deque.append(score)
        scores_array.append(score)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.2f}, eps-greedy: {:5.2f} Time: {:02}:{:02}:{:02}'. \
                  format(i_episode, score, avg_score, eps, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            ### 195.0: for cartpole-v0 and 475 for v1
            if np.mean(scores_deque) >= threshold:
                print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

        if i_episode % TARGET_UPDATE == 0:
            agent.q_target.load_state_dict(agent.q_local.state_dict())

    agent.q_target.load_state_dict(agent.q_local.state_dict())
    return scores_array, avg_scores_array


scores, avg_scores = train()
save('dir_chk_V0_ddqn', 'cartpole-v0-ddqn')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env_v0 = gym.make('CartPole-v0')

    # input_dim,  output_dim, hidden_dim  are the same for v0 and v1, go dqn and ddqn
    input_dim = env_v0.observation_space.shape[0]  # n_spaces =
    output_dim = env_v0.action_space.n  # n_actions =
    hidden_dim = 16

    print('input_dim: ', input_dim, ', output_dim: ', output_dim, ', hidden_dim: ', hidden_dim)

    agent_ddqn = Agent(input_dim, output_dim, hidden_dim)


    def load(agent, directory, filename):
        agent.q_local.load_state_dict(
            torch.load('%s/%s_local.pth' % (directory, filename)))
        agent.q_target.load_state_dict(
            torch.load('%s/%s_target.pth' % (directory, filename)))


    def play(env, agent, n_episodes):
        state = env.reset()

        scores_deque = deque(maxlen=100)

        for i_episode in range(1, n_episodes + 1):
            s = env.reset()

            total_reward = 0
            time_start = time.time()
            timesteps = 0

            while True:

                ## a = agent.get_action(FloatTensor([s]), check_eps=True, eps=0.01)
                a = agent.get_action(FloatTensor([s]), check_eps=False, eps=0.01)
                env.render()
                s2, r, done, _ = env.step(a.item())
                s = s2
                total_reward += r
                timesteps += 1

                if done:
                    break

            delta = (int)(time.time() - time_start)

            scores_deque.append(total_reward)

            print('Episode {}\tAverage Score: {:.2f}, \t Timesteps: {} \tTime: {:02}:{:02}:{:02}' \
                  .format(i_episode, np.mean(scores_deque), timesteps, \
                          delta // 3600, delta % 3600 // 60, delta % 60))


    env_v0.close()
    env_v0_ddqn = gym.make('CartPole-v0')
    load(agent=agent_ddqn, directory='dir_chk_V0_ddqn', filename='cartpole-v0-ddqn-239epis')
    play(env=env_v0_ddqn, agent=agent_ddqn, n_episodes=5)
    env_v0_ddqn.close()

    env_v1_ddqn = gym.make('CartPole-v1')
    load(agent=agent_ddqn, directory='dir_chk_V1_ddqn', filename='cartpole-v1-ddqn')
    play(env=env_v1_ddqn, agent=agent_ddqn, n_episodes=5)
    env_v1_ddqn.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
