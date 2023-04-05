import os
import torch
import torch.nn.functional as F
import numpy as np
from network import *
from replaybuffer import Buffer


class Agent():
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
