import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register
from mujoco_py.generated import const
from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS

gym.logger.set_level(40)


def assert_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert hasattr(env, '_max_episode_steps')


METAWORLD_TASKS = (
    'hammer-v1', 'stick-push-v1', 'push-wall-v1',
    'stick-pull-v1', 'dial-turn-v1', 'peg-insert-side-v1')

for task in METAWORLD_TASKS:

    class StableEnv(ALL_ENVIRONMENTS[task]):
        def step(self, action):
            action = np.clip(
                action, self.action_space.low + 1e-8,
                self.action_space.high - 1e-8)
            obs, reward, done, info = super().step(action)
            return obs, reward, done, info

    register(
        id=task,
        entry_point=StableEnv,
        max_episode_steps=150)
    assert_env(gym.make(task))


def make_env(env_id):
    env = gym.make(env_id)
    setattr(env, 'is_metaworld', env_id in METAWORLD_TASKS)
    return env


def wrap_monitor(env, log_dir):
    print('-' * 80)
    print(
        'Argument --save automatically create window even if '
        'you do not set --render.\nIf you use --save, please '
        'unset --render to save clearer movies.')
    print('-' * 80)

    env.render()
    env.viewer.cam.type = const.CAMERA_FIXED
    env.viewer.cam.fixedcamid = 0
    env = wrappers.Monitor(
        env, log_dir, video_callable=lambda x: True,
        force=True)
    return env
