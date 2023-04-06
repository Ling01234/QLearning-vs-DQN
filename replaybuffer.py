import numpy as np
from collections import namedtuple, deque
from itertools import count
import random


class Buffer():
    def __init__(self, max_size, input_shape, action_space):
        self.max_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.max_size, *input_shape))
        self.new_state_memory = np.zeros((self.max_size, *input_shape))
        self.action_memory = np.zeros((self.max_size, action_space))
        self.reward_memory = np.zeros(self.max_size)
        self.terminal = np.zeros(self.max_size, dtype=np.bool)

    def remember(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal[index] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.max_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal[batch]

        return states, actions, rewards, next_states, dones


Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample

    def __len__(self):
        return len(self.memory)
