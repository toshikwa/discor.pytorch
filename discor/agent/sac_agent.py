import os
import numpy as np
import torch
from torch.optim import Adam

from .base_agent import BaseAgent
from discor.network import TwinnedLinearNetwork, GaussianPolicy
from discor.utils import disable_gradients, soft_update


class SacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=3000000,
                 batch_size=256, lr=0.0003, policy_hidden_units=(256, 256),
                 q_hidden_units=(256, 256), memory_size=1000000, gamma=0.99,
                 nstep=1, update_interval=1, target_update_coef=0.005,
                 start_steps=10000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, nstep, update_interval, target_update_coef, start_steps,
            log_interval, eval_interval, cuda, seed)

        # Build networks.
        self._policy_net = GaussianPolicy(
            state_dim=self._env.observation_space.shape[0],
            action_dim=self._env.action_space.shape[0],
            hidden_units=policy_hidden_units
            ).to(self._device)
        self._online_q_net = TwinnedLinearNetwork(
            state_dim=self._env.observation_space.shape[0],
            action_dim=self._env.action_space.shape[0],
            hidden_units=q_hidden_units
            ).to(device=self._device)
        self._target_q_net = TwinnedLinearNetwork(
            state_dim=self._env.observation_space.shape[0],
            action_dim=self._env.action_space.shape[0],
            hidden_units=q_hidden_units
            ).to(device=self._device).eval()

        # Copy parameters of the learning network to the target network.
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self._target_q_net)

        self._policy_optim = Adam(self._policy_net.parameters(), lr=lr)
        self._q1_optim = Adam(self._online_q_net.net1.parameters(), lr=lr)
        self._q2_optim = Adam(self._online_q_net.net2.parameters(), lr=lr)

        # Target entropy is -|A|.
        self._target_entropy = -torch.prod(torch.Tensor(
            self._env.action_space.shape).to(self._device)).item()

        # We optimize log(alpha), instead of alpha.
        self._log_alpha = torch.zeros(
            1, requires_grad=True, device=self._device)
        self._alpha = self._log_alpha.exp()
        self._alpha_optim = Adam([self._log_alpha], lr=lr)

    def explore(self, state):
        state = torch.FloatTensor(state[None, ...]).to(self._device)
        with torch.no_grad():
            action, _, _ = self._policy_net(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        state = torch.FloatTensor(state[None, ...]).to(self._device)
        with torch.no_grad():
            _, _, action = self._policy_net(state)
        return action.cpu().numpy().reshape(-1)

    def update_target(self):
        soft_update(
            self._target_q_net, self._online_q_net, self._target_update_coef)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self._online_q_net(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy_net(next_states)
            next_q1, next_q2 = self._target_q_net(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self._alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self._discount * next_q

        return target_q

    def calc_q_loss(self, batch):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # Mean Q values for logging.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Q loss is mean squared TD errors.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))

        return q1_loss, q2_loss, mean_q1, mean_q2

    def calc_policy_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            # Resample actions to calculate expectations of Q.
            sampled_actions, entropies, _ = self._policy_net(states)
            # Expectations of Q with clipped double Q technique.
            q = torch.min(self._online_q_net(states, sampled_actions))

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((- q - self._alpha * entropies))

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self._log_alpha * (self._target_entropy - entropies))
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
