import numpy as np
from collections import namedtuple, deque
from itertools import count
import random


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
