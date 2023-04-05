# import pybullet_envs
import gymnasium as gym
import numpy as np
from agent_sac import *
from utils import plot
from tqdm import trange


def main():
    env = gym.make("InvertedPendulum-v4")
    agent = Agent(input_dim=env.observation_space.shape, env=env,
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


if __name__ == "__main__":
    main()
