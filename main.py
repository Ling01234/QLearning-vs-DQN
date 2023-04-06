# import pybullet_envs
import gymnasium as gym
import numpy as np
from agent import *
from utils import *
from tqdm import trange

env_cartpole = gym.make("CartPole-v1")
env_lunar_lander = gym.make("LunarLander-v2")


def sac():
    env = gym.make("InvertedPendulum-v4")
    agent = Agent_SAC(input_dim=env.observation_space.shape, env=env,
                      action_space=env.action_space.shape[0])

    num_games = 1000
    filename = "pendulum.png"
    figure_file = "plots/" + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    for i in trange(num_games):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(obs)
            # print(f"action: {action}")

            next_obs, reward, done, *_ = env.step(action)
            score += reward
            agent.remember(obs, action, reward, next_obs, done)

            if not load_checkpoint:
                agent.learn()

            obs = next_obs

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

            if not load_checkpoint:
                agent.save_models()

        # print(f"episode {i} scores {score:.1f}, average score: {avg_score:1f}")

    if not load_checkpoint:
        x = [i+1 for i in range(num_games)]
        plot(x, score_history, figure_file)


def cartpole_dq():
    agent = Agent_DQ(alpha=0.003, gamma=0.99, env=env_cartpole, epsilon_start=1,
                     epsilon_decay=5e-4, epsilon_end=0.01, tau=0.005, batch_size=64)

    filename = "plots/cartpole.png"
    agent.train(500, filename=filename)


def cartpole_q():
    upperbounds = env_cartpole.observation_space.high
    upperbounds[1] = 3.5
    upperbounds[3] = 10
    lowerbounds = env_cartpole.observation_space.low
    lowerbounds[1] = -3.5
    lowerbounds[3] = -10

    agent = Agent_Q(env=env_cartpole, alpha=0.15, gamma=0.99, epsilon=1.0, num_bins=10,
                    epsilon_decay=5e-4, epsilon_end=0.01, lowerbounds=lowerbounds, upperbounds=upperbounds)
    agent.train(2000)
    rewards = agent.reward
    x = np.arange(len(rewards))
    filename = "plots/cartpole with QLearning"
    plot(x, rewards, filename)


def lunarlander_dq():
    agent = Agent_DQ(alpha=0.003, gamma=0.99, env=env_lunar_lander, epsilon_start=1,
                     epsilon_decay=5e-4, epsilon_end=0.01, tau=0.005, batch_size=64)

    filename = "plots/lunar_lander.png"
    agent.train(500, filename=filename)


def lunarlander_q():
    upperbounds = env_lunar_lander.observation_space.high
    lowerbounds = env_lunar_lander.observation_space.low
    agent = Agent_Q(env=env_lunar_lander, alpha=0.01, epsilon=1.0, upperbounds=upperbounds, lowerbounds=lowerbounds,
                    num_bins=10, gamma=0.99)
    agent.train(2000)
    rewards = agent.reward
    x = np.arange(len(rewards))
    filename = "plots/lunarlander with QLearning"
    plot(x, rewards, filename)


if __name__ == "__main__":
    # cartpole_dq()
    # cartpole_q()
    # lunarlander_dq()
    lunarlander_q()
