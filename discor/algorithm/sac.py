import os
import torch
from torch.optim import Adam

from .base import Algorithm
from discor.network import TwinnedStateActionFunction, GaussianPolicy
from discor.utils import disable_gradients, soft_update, update_params


class SAC(Algorithm):

    def __init__(self, state_dim, action_dim, device, policy_lr=0.0003,
                 q_rl=0.0003, entropy_rl=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0):
        super().__init__(device, log_interval, seed)

        # Build networks.
        self._policy_net = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=policy_hidden_units
            ).to(self._device)
        self._online_q_net = TwinnedStateActionFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=q_hidden_units
            ).to(device=self._device)
        self._target_q_net = TwinnedStateActionFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=q_hidden_units
            ).to(device=self._device).eval()

        # Copy parameters of the learning network to the target network.
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self._target_q_net)

        self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._q1_optim = Adam(self._online_q_net.net1.parameters(), lr=q_rl)
        self._q2_optim = Adam(self._online_q_net.net2.parameters(), lr=q_rl)

        # Target entropy is -|A|.
        self._target_entropy = -action_dim

        # We optimize log(alpha), instead of alpha.
        self._log_alpha = torch.zeros(
            1, device=self._device, requires_grad=True)
        self._alpha = self._log_alpha.exp()
        self._alpha_optim = Adam([self._log_alpha], lr=entropy_rl)

        self._target_update_coef = target_update_coef

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

    def calc_target_q(self, states, actions, rewards, next_states, dones,
                      discount):
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy_net(next_states)
            next_q1, next_q2 = self._target_q_net(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self._alpha * next_entropies

        target_q = rewards + (1.0 - dones) * discount * next_q

        return target_q

    def learn(self, batch, discount, writer):
        self._learning_steps += 1

        policy_loss, entropies = self.calc_policy_loss(batch)
        q1_loss, q2_loss, mean_q1, mean_q2 = self.calc_q_loss(batch, discount)
        entropy_loss = self.calc_entropy_loss(entropies)

        update_params(self._policy_optim, policy_loss)
        update_params(self._q1_optim, q1_loss)
        update_params(self._q2_optim, q2_loss)
        update_params(self._alpha_optim, entropy_loss)

        self._alpha = self._log_alpha.exp()

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/alpha', self._alpha.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q1', mean_q1, self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q2', mean_q2, self._learning_steps)
            writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self._learning_steps)

    def calc_policy_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Resample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self._policy_net(states)

        # Expectations of Q with clipped double Q technique.
        q1, q2 = self._online_q_net(states, sampled_actions)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((- q - self._alpha * entropies))

        return policy_loss, entropies.detach()

    def calc_q_loss(self, batch, discount):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch, discount)

        # Mean Q values for logging.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Q loss is mean squared TD errors.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))

        return q1_loss, q2_loss, mean_q1, mean_q2

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
