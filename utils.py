import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot(x, scores, figure_file, title):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100): (i+1)])

    sns.lineplot(x=x, y=running_avg)
    plt.title(title)
    plt.savefig(figure_file)
    plt.clf()


def plot_all(reward_cartpole_q, reward_cartpole_dq, reward_lunar_q, reward_lunar_dq):
    colors = [mcolors.TABLEAU_COLORS["tab:blue"], mcolors.TABLEAU_COLORS["tab:brown"],
              mcolors.TABLEAU_COLORS["tab:green"], mcolors.TABLEAU_COLORS["tab:orange"]]

    x = np.arange(len(reward_cartpole_dq))
    avg_cartpole_q = np.zeros(len(reward_cartpole_q))
    for i in range(len(avg_cartpole_q)):
        avg_cartpole_q[i] = np.mean(reward_cartpole_q[max(0, i-100): (i+1)])
    avg_cartpole_dq = np.zeros(len(reward_cartpole_dq))
    for i in range(len(avg_cartpole_dq)):
        avg_cartpole_dq[i] = np.mean(reward_cartpole_dq[max(0, i-100): (i+1)])
    avg_lunar_q = np.zeros(len(reward_lunar_q))
    for i in range(len(avg_lunar_q)):
        avg_lunar_q[i] = np.mean(reward_lunar_q[max(0, i-100): (i+1)])
    avg_lunar_dq = np.zeros(len(reward_lunar_dq))
    for i in range(len(avg_lunar_dq)):
        avg_lunar_dq[i] = np.mean(reward_lunar_dq[max(0, i-100): (i+1)])

    plt.plot(x, avg_cartpole_q,
             label=f"CartPole with Q Learning", color=colors[0])
    plt.plot(x, avg_cartpole_dq,
             label=f"CartPole with Deep Q Learning", color=colors[2])

    plt.legend(bbox_to_anchor=(1, 0), loc="lower right")
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.title("CartPole Environment")
    plt.savefig("plots/cartpole.png")
    plt.clf()

    plt.plot(x, avg_lunar_q,
             label=f"LunarLander with Q Learning", color=colors[0])
    plt.plot(x, avg_lunar_dq,
             label=f"LunarLander with Deep Q Learning", color=colors[2])
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right")
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.title("Lunar Lander Environment")
    plt.savefig("plots/lunar.png")
    plt.clf()


# reward_cartpole_q = np.load("reward_cartpole_q.npy")
# reward_cartpole_dq = np.load("reward_cartpole_dq.npy")
# reward_lunar_q = np.load("reward_lunar_q.npy")
# reward_lunar_dq = np.load("reward_lunar_dq.npy")

# plot_all(reward_cartpole_q=reward_cartpole_q, reward_cartpole_dq=reward_cartpole_dq,
#          reward_lunar_q=reward_lunar_q, reward_lunar_dq=reward_lunar_dq)
