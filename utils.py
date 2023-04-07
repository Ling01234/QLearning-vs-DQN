import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
import matplotlib.colors as mcolors
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


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

    plt.plot(x, reward_cartpole_q,
             label=f"CartPole with Q Learning", color=colors[0])
    plt.plot(x, reward_cartpole_dq,
             label=f"CartPole with Deep Q Learning", color=colors[1])
    plt.plot(x, reward_lunar_q,
             label=f"LunarLander with Q Learning", color=colors[2])
    plt.plot(x, reward_lunar_dq,
             label=f"LunarLander with Deep Q Learning", color=colors[3])
    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.savefig("plots/all.png")
    plt.clf()


def plot_durations(episode_durations, env_name, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title(f"{env_name}")
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return  ')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
