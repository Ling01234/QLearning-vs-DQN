import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, alpha, input_dim, action_space, name="critic", fc1_dim=256, fc2_dim=256,
                 chkpt_dir="tmp/sac"):

        super(CriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.action_space = action_space
        self.name = name
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        # model checkpoint
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")

        self.fc1 = nn.Linear(self.input_dim[0] + action_space, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1)

        self.opt = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        # print(f"state shape:{state.shape}, action shape:{action.shape}")
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        qvalue = self.q(action_value)
        return qvalue

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self._load_from_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, alpha, input_dim, name="value", fc1_dim=256, fc2_dim=256, chkpt_dir="tmp/sac"):
        super(ValueNetwork, self).__init__()

        self.alpha = alpha
        self.input_dim = input_dim
        self.name = name
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, fc2_dim)
        self.v = nn.Linear(self.fc2_dim, 1)

        self.opt = optim.Adam(self.parameters(), lr=self.alpha)

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        value = self.v(state_value)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self._load_from_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, max_action, action_space=2,
                 name="actor", chkpt_dir="tmp/sac", fc1_dim=256, fc2_dim=256):

        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        self.max_action = max_action
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_space = action_space
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.mu = nn.Linear(self.fc2_dim, self.action_space)
        self.sigma = nn.Linear(self.fc2_dim, self.action_space)

        self.opt = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)
        print(f"using: {self.device}")

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        prob = Normal(mu, sigma)
        # print(f"prob: {prob}")

        if reparameterize:
            # add noise, extra exploration factor
            actions = prob.rsample()

        else:
            actions = prob.sample()

        action = torch.tanh(actions) * \
            torch.tensor(self.max_action).to(self.device)
        # action = torch.tanh(actions).to(self.device)
        log_prob = prob.log_prob(actions)
        log_prob -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self._load_from_state_dict(torch.load(self.checkpoint_file))
