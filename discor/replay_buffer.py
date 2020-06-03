from collections import deque
import numpy as np
import torch


class NStepBuffer:

    def __init__(self, gamma=0.99, nstep=3):
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._discounts = [gamma ** i for i in range(nstep)]
        self._nstep = nstep
        self.reset()

    def append(self, state, action, reward):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)

    def get(self):
        assert len(self._rewards) > 0

        state = self._states.popleft()
        action = self._actions.popleft()
        reward = self._nstep_reward()
        return state, action, reward

    def _nstep_reward(self):
        reward = np.sum([
            r * d for r, d in zip(self._rewards, self._discounts)])
        self._rewards.popleft()
        return reward

    def reset(self):
        self._states = deque(maxlen=self._nstep)
        self._actions = deque(maxlen=self._nstep)
        self._rewards = deque(maxlen=self._nstep)

    def is_empty(self):
        return len(self._rewards) == 0

    def is_full(self):
        return len(self._rewards) == self._nstep

    def __len__(self):
        return len(self._rewards)


class ReplayBuffer:

    def __init__(self, memory_size, state_shape, action_shape, gamma=0.99,
                 nstep=3):
        assert isinstance(memory_size, int) and memory_size > 0
        assert isinstance(state_shape, tuple)
        assert isinstance(action_shape, tuple)
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._memory_size = memory_size
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._gamma = gamma
        self._nstep = nstep

        self._reset()

    def _reset(self):
        self._n = 0
        self._p = 0

        self._states = torch.empty(
            (self._memory_size, ) + self._state_shape, dtype=torch.float)
        self._next_states = torch.empty(
            (self._memory_size, ) + self._state_shape, dtype=torch.float)
        self._actions = torch.empty(
            (self._memory_size, ) + self._action_shape, dtype=torch.float)

        self._rewards = torch.empty((self._memory_size, 1), dtype=torch.float)
        self._dones = torch.empty((self._memory_size, 1), dtype=torch.float)

        if self._nstep != 1:
            self._nstep_buffer = NStepBuffer(self._gamma, self._nstep)

    def append(self, state, action, reward, next_state, done,
               episode_done=None):

        if self._nstep != 1:
            self._nstep_buffer.append(state, action, reward)

            if self._nstep_buffer.is_full():
                state, action, reward = self._nstep_buffer.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done or episode_done:
                while not self._nstep_buffer.is_empty():
                    state, action, reward = self._nstep_buffer.get()
                    self._append(state, action, reward, next_state, done)

        else:
            self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self._states[self._p] = state
        self._actions[self._p] = action
        self._rewards[self._p] = reward
        self._next_states[self._p] = next_state
        self._dones[self._p] = done

        self._n = min(self._n + 1, self._memory_size)
        self._p = (self._p + 1) % self._memory_size

    def sample(self, batch_size, device=torch.device('cpu')):
        assert isinstance(batch_size, int) and batch_size > 0

        idxes = self._sample_idxes(batch_size)
        return self._sample_batch(idxes, batch_size, device)

    def _sample_idxes(self, batch_size):
        return np.random.randint(0, self._n, size=batch_size)

    def _sample_batch(self, idxes, batch_size, device):
        states = self._states[idxes].to(device)
        actions = self._actions[idxes].to(device)
        rewards = self._rewards[idxes].to(device)
        dones = self._dones[idxes].to(device)
        next_states = self._next_states[idxes].to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n
