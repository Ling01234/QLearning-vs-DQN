import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


################################################
# DEEP Q LEARNING
################################################


class DQNetwork(nn.Module):
    def __init__(self, alpha, n_observations, fc1_dim, fc2_dim, action_space):
        super(DQNetwork, self).__init__()
        self.alpha = alpha
        self.n_observations = n_observations
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_space = action_space

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(self.n_observations, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.action_space)
        self.opt = optim.Adam(self.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        x = F.relu(x)
        actions = self.fc3(x)
        return actions
