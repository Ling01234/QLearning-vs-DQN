import os
import torch
import torch.nn.functional as F
import numpy as np
from network import *
from replaybuffer import *
from tqdm import trange
from utils import *
from itertools import count


class Agent_SAC():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dim=[8], env=None, gamma=0.99,
                 action_space=2, max_size=1000000, layer1_size=256, layer2_size=256, batch_size=128,
                 reward_scale=2, tau=0.005):

        self.alpha = alpha
        self.beta = beta
        self.input_dim = input_dim
        self.env = env
        self.gamma = gamma
        self.action_space = action_space
        self.max_size = max_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.tau = tau
        self.batch_size = batch_size
        self.memory = Buffer(max_size, input_dim, action_space)

        self.actor = ActorNetwork(
            alpha, input_dim, action_space=action_space, name="actor", max_action=env.action_space.high)
        self.critic1 = CriticNetwork(
            alpha, input_dim, action_space=action_space, name="critic1")
        self.critic2 = CriticNetwork(
            alpha, input_dim, action_space=action_space, name="critic2")
        self.value = ValueNetwork(alpha, input_dim, name="value")
        self.target_value = ValueNetwork(alpha, input_dim, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def select_action(self, obs):
        state = torch.Tensor([obs]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        action = actions.cpu().detach().numpy()[0]
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_param = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_param)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        print("... models saved ...")

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        print("... models loaded ...")

    def get_critic_value(self, state, reparameterize=True):
        actions, log_prob = self.actor.sample_normal(
            state, reparameterize=reparameterize)
        log_prob = log_prob.view(-1)
        qvalue1_new_policy = self.critic1.forward(state, actions)
        qvalue2_new_policy = self.critic2.forward(state, actions)
        critic_value = torch.min(qvalue1_new_policy, qvalue2_new_policy)
        critic_value = critic_value.view(-1)

        return critic_value, log_prob

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(
            self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        next_state = torch.tensor(
            next_state, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)

        value = self.value(state).view(-1)
        next_value = self.target_value(next_state).view(-1)
        next_value[done] = 0.0

        # value network loss
        critic_value, log_prob = self.get_critic_value(
            state, reparameterize=False)

        self.value.opt.zero_grad()
        value_target = critic_value - log_prob
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.opt.step()

        # actor network loss
        # critic_value, log_prob = self.get_critic_value(
        #     state, reparameterize=True)
        actor_loss = log_prob - critic_value
        actor_loss = torch.mean(actor_loss)

        self.actor.opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.opt.step()

        # critic network loss
        self.critic1.opt.zero_grad()
        self.critic2.opt.zero_grad()
        qhat = self.scale * reward + self.gamma * next_value
        # print(f"in agents: state {state.shape}, action: {action.shape}")
        qvalue1_old_policy = self.critic1.forward(state, action).view(-1)
        qvalue2_old_policy = self.critic2.forward(state, action).view(-1)

        critic1_loss = 0.5 * F.mse_loss(qvalue1_old_policy, qhat)
        critic2_loss = 0.5 * F.mse_loss(qvalue2_old_policy, qhat)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.opt.step()
        self.critic2.opt.step()

        self.update_network_parameters()


class Agent_DQ():
    def __init__(self, alpha, gamma, env, epsilon_start, epsilon_decay, epsilon_end,
                 tau, batch_size, show_result=True) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = env.action_space.n
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilong_end = epsilon_end
        self.tau = tau
        self.batch_size = batch_size
        self.show_result = show_result
        self.episode_durations = []

        self.policy_network = DQNetwork(self.alpha, self.n_observations, fc1_dim=128, fc2_dim=128,
                                        action_space=self.n_actions)
        self.target_network = DQNetwork(self.alpha, self.n_observations, fc1_dim=128, fc2_dim=128,
                                        action_space=self.n_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.memory = ReplayMemory(100000)
        self.steps = 0

    def epsilon_update(self):
        if self.epsilon > self.epsilong_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilong_end
        self.steps += 1

    def select_action(self, state):
        sample = random.random()
        self.epsilon_update()

        if sample > self.epsilon:
            with torch.no_grad():
                action = self.policy_network(state).max(1)[1].view(1, 1)

        else:
            action = torch.tensor([[self.env.action_space.sample(
            )]], device=self.policy_network.device, dtype=torch.long)

        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # convert batch array of transitions to transition of batch arrays
        batch = Transition(*zip(*transitions))

        masks = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                             device=self.policy_network.device, dtype=bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        # print(f"batch reward: {batch.reward}")
        rewards = torch.cat(batch.reward)

        # Q(s_t, a)
        sa_values = self.policy_network(states).gather(1, actions)

        # compute next state values for the bellman's equation
        next_state_values = torch.zeros(
            self.batch_size, device=self.policy_network.device)
        with torch.no_grad():
            next_state_values[masks] = self.target_network(
                non_final_next_states).max(1)[0]

        # compute target values
        target_sa_values = rewards + self.gamma * next_state_values

        # compute loss
        loss = nn.SmoothL1Loss()
        loss = loss(sa_values, target_sa_values.unsqueeze(1))

        # optimizer
        self.policy_network.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.policy_network.opt.step()

    def train(self, episodes, filename):
        for episode in trange(1, episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.policy_network.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(
                    action.item())

                reward = torch.tensor(
                    [reward], device=self.policy_network.device)
                done = terminal or truncated

                if terminal:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        next_state, dtype=torch.float32, device=self.policy_network.device).unsqueeze(0)

                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                self.learn()

                # soft update of the target network's weights
                target_network_state_dict = self.target_network.state_dict()
                policy_network_state_dict = self.policy_network.state_dict()
                for key in policy_network_state_dict:
                    target_network_state_dict[key] = policy_network_state_dict[key] * \
                        self.tau + \
                        target_network_state_dict[key] * (1-self.tau)
                self.target_network.load_state_dict(target_network_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    plot_durations(self.episode_durations,
                                   env_name=self.env.spec.id, show_result=False)
                    break

        plot_durations(self.episode_durations,
                       env_name=self.env.spec.id, show_result=self.show_result)
        plt.ioff()
        plt.savefig(filename)
        plt.show()


class Agent_Q:
    def __init__(self, env, alpha, gamma, epsilon, upperbounds, lowerbounds, num_bins,
                 seed=123, epsilon_end=0.01, epsilon_decay=5e-4) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_bins = num_bins
        self.upperbounds = upperbounds
        self.lowerbounds = lowerbounds
        self.seed = seed
        random.seed(self.seed)
        self.num_action = env.action_space.n
        self.reward = []
        self.state_space = self.env.observation_space.shape[0]

        input_dim = [num_bins for i in range(self.state_space)]
        self.Qvalues = np.random.uniform(low=-0.001, high=0.001,
                                         size=(*input_dim, self.num_action))
        self.steps = 0
        self.bins = []
        for i in range(self.state_space):
            self.bins.append(np.linspace(
                self.lowerbounds[i], self.upperbounds[i], self.num_bins))

    def eps_update(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_end

        self.steps += 1

    def discritize_state(self, state):
        """
        Discritize continuous state into a discrete state

        Args:
            state (list of length 4): Current continuous state of agent

        Returns:
            state (4-tuple): Current discritized state of agent
        """
        new_state = []
        for i in range(self.state_space):
            index = np.maximum(np.digitize(state[i], self.bins[i]) - 1, 0)
            new_state.append(index)

        return tuple(new_state)

    def select_action(self, state):
        """
        Select action given a state

        Args:
            state (4-tuple): Current state of the agent, continuous
            episode (int): Current episode of the run

        Returns:
            int: Action chosen by the agent
        """
        random.seed(self.seed)
        self.eps_update()

        # epsilon greedy
        number = np.random.random()
        if number < self.epsilon:  # uniformly choose action
            return np.random.choice(self.num_action)

        # greedy selection
        state = self.discritize_state(state)
        best_actions = np.where(
            self.Qvalues[state] == np.max(self.Qvalues[state]))[0]
        return np.random.choice(best_actions)

    def train(self, num_episodes):
        """
        Simulate a specified number of episodes
        """
        for episode in range(1, num_episodes+1):
            # reset env
            (state, _) = self.env.reset()
            state = list(state)

            # run episode
            episode_reward = 0
            terminal = False
            while not terminal:
                discritized_state = self.discritize_state(state)
                action = self.select_action(state)
                (next_state, reward, terminal, _, _) = self.env.step(action)
                episode_reward += reward

                next_discritized_state = self.discritize_state(
                    list(next_state))

                q_max = np.max(self.Qvalues[next_discritized_state])
                self.update(
                    terminal, reward, action, discritized_state, q_max)

                state = next_state

            self.reward.append(int(episode_reward))

    def update(self, terminal, reward, action, state, q_max):
        """
        Qlearning update rule

        Args:
            terminal (bool): True if at terminal state, False otherwise
            reward (int): Reward of the agent at current state
            action (int): Action taken by agent
            state (4-tuple): Discrete state of the agent
            q_max (float): Max Q value of the next state
        """
        if not terminal:
            loss = reward + self.gamma * q_max - \
                self.Qvalues[state + (action,)]
        else:
            loss = reward - self.Qvalues[state + (action,)]

        self.Qvalues[state + (action,)] += self.alpha * loss
