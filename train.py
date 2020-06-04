import os
import yaml
import argparse
from datetime import datetime
import gym
import torch

from discor.algorithm import SAC, DisCor
from discor.agent import Agent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = gym.make(args.env_id)
    test_env = gym.make(args.env_id)

    # Device to use.
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.algo}-seed{args.seed}-{time}')

    if args.algo == 'discor':
        # Discor algorithm.
        algo = DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed,
            **config['SAC'], **config['DisCor'])
    elif args.algo == 'sac':
        # SAC algorithm.
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed, **config['SAC'])
    else:
        raise Exception('You need to set "--algo sac" or "--algo discor".')

    agent = Agent(
        env=env, test_env=test_env, algo=algo, log_dir=log_dir,
        device=device, seed=args.seed, **config['Agent'])
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'mujoco.yaml'))
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--algo', choices=['sac', 'discor'], default='discor')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
