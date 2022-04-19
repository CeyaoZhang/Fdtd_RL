import argparse

def get_args():

    parser = argparse.ArgumentParser(description='This is PPO+RewNetSimulator hyper-parameters')

    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--use_cuda', type=int, default=0, choices=[0, 1], help='0 for CPU and 1 for CUDA')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor in RL')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE-Lambda')
    
    parser.add_argument('--max_steps', type=int, default=int(4e5))
    parser.add_argument('--steps_per_episode', type=int, default=200)
    
    parser.add_argument('--rewNet_maxlen', type=int, default=1000, help='max len for the rewNet buffer')
    parser.add_argument('--rewNet_batch_size', type=int, default=20, help='batch size for training the rewNet')
    parser.add_argument('--rewNet_epochs', type=int, default=20, help='epochs for training the rewNet')


    args = parser.parse_args()

    return args