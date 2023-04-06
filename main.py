# import pybullet_envs
import gymnasium as gym
import numpy as np
from agent import *
from utils import *
from tqdm import trange


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
        plot_policy_gradient(x, score_history, figure_file)


def dq():
    env = gym.make("LunarLander-v2")
    agent = Agent_DQ(gamma=0.99, epsilon=1.0, batch_size=64,
                     n_action=4, eps_end=0.01, input_dim=[8], alpha=0.003)
    scores, eps_hist = [], []
    num_games = 500

    for game in range(num_games):
        score = 0
        done = False
        obs, _ = env.reset()

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, *_ = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.learn()
            obs = next_obs

        scores.append(score)
        eps_hist.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(
            f"episode {game + 1} score {score:.1f}, avg_score {avg_score:.2f}, epsilon: {agent.epsilon:.2f}")

    x = [i + 1 for i in range(num_games)]
    filename = "plots/lunar_lander.png"
    plot_policy_gradient(x, scores, eps_hist, filename)


if __name__ == "__main__":
    # sac()
    dq()
