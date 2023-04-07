# import pybullet_envs
import gymnasium as gym
import numpy as np
from agent import *
from utils import *
from tqdm import trange

env_cartpole = gym.make("CartPole-v1")
env_lunar_lander = gym.make("LunarLander-v2")


def cartpole_dq():
    agent = Agent_DQ(alpha=0.003, gamma=0.99, env=env_cartpole, epsilon_start=1,
                     epsilon_decay=5e-4, epsilon_end=0.01, tau=0.005, batch_size=64)

    filename = "plots/cartpole.png"
    agent.train(1000)
    rewards = agent.train_reward
    x = np.arange(len(rewards))
    plot(x, rewards, filename,
         f"Deep Q Learning in CartPole with 1000 training episodes")
    return rewards


def cartpole_q():
    upperbounds = env_cartpole.observation_space.high
    upperbounds[1] = 3.5
    upperbounds[3] = 10
    lowerbounds = env_cartpole.observation_space.low
    lowerbounds[1] = -3.5
    lowerbounds[3] = -10

    agent = Agent_Q(env=env_cartpole, alpha=0.15, gamma=0.99, epsilon=1.0, num_bins=10,
                    epsilon_decay=5e-4, epsilon_end=0.01, lowerbounds=lowerbounds, upperbounds=upperbounds)
    agent.train(1000)
    rewards = agent.reward
    x = np.arange(len(rewards))
    filename = "plots/cartpole with QLearning"
    plot(x, rewards, filename, f"Q Learning in CartPole with 1000 training episodes")
    return rewards


def lunarlander_dq():
    agent = Agent_DQ(alpha=0.003, gamma=0.99, env=env_lunar_lander, epsilon_start=1,
                     epsilon_decay=5e-4, epsilon_end=0.01, tau=0.005, batch_size=64)

    filename = "plots/lunar_lander.png"
    agent.train(1000)
    rewards = agent.train_reward
    x = np.arange(len(rewards))
    plot(x, rewards, filename,
         f"Deep Q Learning in Lunar Lander with 1000 training episodes")
    return rewards


def lunarlander_q():
    upperbounds = env_lunar_lander.observation_space.high
    lowerbounds = env_lunar_lander.observation_space.low
    agent = Agent_Q(env=env_lunar_lander, alpha=0.01, epsilon=1.0, upperbounds=upperbounds, lowerbounds=lowerbounds,
                    num_bins=10, gamma=0.99)
    agent.train(1000)
    rewards = agent.reward
    x = np.arange(len(rewards))
    filename = "plots/lunarlander with QLearning"
    plot(x, rewards, filename,
         f"Q Learning in Lunar Lander with 1000 training episodes")
    return rewards


if __name__ == "__main__":
    reward_cartpole_dq = cartpole_dq()
    reward_cartpole_q = cartpole_q()
    reward_lunar_dq = lunarlander_dq()
    reward_lunar_q = lunarlander_q()
    plot_all(reward_cartpole_dq=reward_cartpole_dq, reward_cartpole_q=reward_cartpole_q,
             reward_lunar_dq=reward_lunar_dq, reward_lunar_q=reward_lunar_q)
