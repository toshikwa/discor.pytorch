import os
import yaml
import argparse
import numpy as np
import torch

from discor.env import make_env
from discor.algorithm import EvalAlgorithm


def test(env, algo, render):
    state = env.reset()
    episode_return = 0.0
    success = 0.0

    done = False
    while (not done):
        action = algo.exploit(state)
        next_state, reward, done, info = env.step(action)
        if render:
            env.render()

        episode_return += reward
        if env.is_metaworld and info['success'] > 1e-6:
            success = info['success']

        state = next_state
    return episode_return, success


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    policy_hidden_units = config['SAC']['policy_hidden_units']

    # Create environments.
    env = make_env(args.env_id)
    env.seed(args.seed)

    # Device to use.
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Evaluation algorithm.
    algo = EvalAlgorithm(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        policy_hidden_units=policy_hidden_units)
    algo.load_models(os.path.join(args.log_dir, 'model', 'best'))

    returns = np.empty((args.num_episodes))
    success = np.empty((args.num_episodes))

    env.render()
    env.viewer._paused = True

    for i in range(args.num_episodes):
        returns[i], success[i] = test(env, algo, args.render)

    env.viewer._paused = True
    print('-' * 60)
    print(f'Num Episodes: {args.num_episodes:<5}\n'
          f'Mean Return : {returns.mean():<5.1f} '
          f'+/- {returns.std():<5.1f} ')
    if env.is_metaworld:
        print(f'Success rate: {success.mean():<1.3f}  ')
    print('-' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'metaworld.yaml'))
    parser.add_argument('--env_id', type=str, default='hammer-v1')
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
