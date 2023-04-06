import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import *
from replaybuffer import Buffer


class Agent_DQ():
    def __init__(self, gamma, epsilon, alpha, input_dim, batch_size, n_action,
                 max_mem_size=100000, eps_end=0.01, eps_decay=5e-4) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_action)]
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        # self.mem_counter = 0

        self.Qeval = DQNetwork(self.alpha, input_dim,
                               fc1_dim=256, fc2_dim=256, action_space=n_action)
        self.buffer = Buffer(self.mem_size, input_dim, n_action)
        # self.state_memory = self.buffer.state_memory
        # self.new_state_memory = self.buffer.new_state_memory
        # self.action_memory = self.buffer.action_memory
        # self.reward_memory = self.buffer.reward_memory
        # self.terminal_memory = self.buffer.terminal

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.remember(state, action, reward, next_state, done)

    def select_action(self, obs):
        if np.random.random() > self.epsilon:
            state = torch.tensor([obs]).to(self.Qeval.device)
            actions = self.Qeval.forward(state)
            action = torch.argmax(actions).item()

        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.buffer.mem_counter < self.batch_size:
            return

        self.Qeval.opt.zero_grad()
        max_mem = min(self.buffer.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        states = torch.tensor(
            self.buffer.state_memory[batch], dtype=torch.float).to(self.Qeval.device)
        new_states = torch.tensor(
            self.buffer.new_state_memory[batch], dtype=torch.float).to(self.Qeval.device)
        rewards = torch.tensor(
            self.buffer.reward_memory[batch], dtype=torch.float).to(self.Qeval.device)
        terminals = torch.tensor(
            self.buffer.terminal[batch], dtype=torch.int).to(self.Qeval.device)
        actions = self.buffer.action_memory[batch]
        actions = np.argmax(actions, axis=1)

        q_eval = self.Qeval.forward(states)
        # print(f"q_eval shape: {q_eval.shape}")
        # print(f"batch_index shape: {batch_index.shape}")
        # print(f"actions shape: {actions.shape}")
        q_eval = q_eval[batch_index, actions]

        q_next = self.Qeval.forward(new_states)
        q_next[terminals] = 0.0

        # bellman
        q_targets = rewards + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Qeval.loss(q_targets, q_eval).to(self.Qeval.device)
        # print(f"loss: {loss}")
        loss.backward()
        self.Qeval.opt.step()

        if self.epsilon > self.eps_end:
            self.epsilon -= self.eps_decay
        else:
            self.epsilon = self.eps_end
