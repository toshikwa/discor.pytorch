from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from discor.replay_buffer import ReplayBuffer
from discor.utils import update_params, RunningMeanStats


class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=3000000,
                 batch_size=256, memory_size=1000000,
                 gamma=0.99, nstep=1, update_interval=1,
                 target_update_coef=0.005, start_steps=10000, log_interval=10,
                 eval_interval=1000, cuda=True, seed=0):
        super().__init__()

        # Environment.
        self._env = env
        self._test_env = test_env

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._env.seed(seed)
        self._test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        # Device.
        self._device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # Replay buffer with n-step return.
        self._replay_buffer = ReplayBuffer(
            memory_size=memory_size,
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_shape.shape,
            gamma=gamma, nstep=nstep)

        # Directory to log.
        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self._steps = 0
        self._learning_steps = 0
        self._episodes = 0
        self._train_return = RunningMeanStats(log_interval)
        self._writer = SummaryWriter(log_dir=self._summary_dir)
        self._best_eval_score = -np.inf

        self._num_steps = num_steps
        self._batch_size = batch_size
        self._discount = gamma ** nstep
        self._update_interval = update_interval
        self._start_steps = start_steps
        self._target_update_coef = target_update_coef
        self._log_interval = log_interval
        self._eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self._steps > self._num_steps:
                break

    def is_update(self):
        return self._steps % self._update_interval == 0\
            and self._steps >= self._start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_q_loss(self, batch):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies):
        pass

    def train_episode(self):
        self._episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done):

            if self._start_steps > self._steps:
                action = self._env.action_space.sample()
            else:
                action = self.explore(state)

            next_state, reward, done, _ = self._env.step(action)

            # Set done=True only when the agent fails, ignoring done signal
            # if the agent reach time horizons.
            if episode_steps >= self._env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            self._replay_buffer.append(
                state, action, reward, next_state, masked_done, done)

            self._steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                self.learn()

            self.update_target()

            if self._steps % self._eval_interval == 0:
                self.evaluate()
                self.save_models(os.path.join(self._model_dir, 'final'))

        # We log running mean of training rewards.
        self._train_return.append(episode_return)

        if self._episodes % self._log_interval == 0:
            self._writer.add_scalar(
                'reward/train', self._train_return.get(), self._steps)

        print(f'Episode: {self._episodes:<4}  '
              f'Episode steps: {episode_steps:<4}  '
              f'Return: {episode_return:<5.1f}')

    def learn(self):
        assert hasattr(self, '_q1_optim') and hasattr(self, '_q2_optim') and\
            hasattr(self, '_policy_optim') and hasattr(self, '_alpha_optim')

        self._learning_steps += 1

        batch = self._replay_buffer.sample(self._batch_size)

        q1_loss, q2_loss, mean_q1, mean_q2 = self.calc_q_loss(batch)
        policy_loss, entropies = self.calc_policy_loss(batch)
        entropy_loss = self.calc_entropy_loss(entropies)

        update_params(self._q1_optim, q1_loss)
        update_params(self._q2_optim, q2_loss)
        update_params(self._policy_optim, policy_loss)
        update_params(self._alpha_optim, entropy_loss)

        self._alpha = self._log_alpha.exp()

        if self._learning_steps % self._log_interval == 0:
            self._writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self._writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self._writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self._writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self._writer.add_scalar(
                'stats/alpha', self._alpha.detach().item(),
                self.learning_steps)
            self._writer.add_scalar(
                'stats/mean_Q1', mean_q1, self._learning_steps)
            self._writer.add_scalar(
                'stats/mean_Q2', mean_q2, self._learning_steps)
            self._writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self._learning_steps)

    def evaluate(self):
        num_episodes = 10
        num_steps = 0
        total_return = 0.0

        for _ in range(num_episodes):
            state = self._test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done):
                action = self.exploit(state)
                next_state, reward, done, _ = self._test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

        mean_return = total_return / num_episodes

        if mean_return > self._best_eval_score:
            self._best_eval_score = mean_return
            self.save_models(os.path.join(self._model_dir, 'best'))

        self._writer.add_scalar(
            'reward/test', mean_return, self._steps)
        print('-' * 60)
        print(f'Num steps: {self._steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self._env.close()
        self._test_env.close()
        self._writer.close()
