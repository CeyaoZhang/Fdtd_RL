import argparse

def get_args():

    parser = argparse.ArgumentParser(description='This is PPO+RewNetSimulator hyper-parameters')

    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--use_cuda', type=int, default=0, choices=[0, 1], help='0 for CPU and 1 for CUDA')

    parser.add_argument('--gamma', type=float, default=0.99, 
    help='discount factor in RL')
    parser.add_argument('--lam', type=float, default=0.95, 
    help='GAE-Lambda')
    
    parser.add_argument('--max_steps', type=int, default=int(4e5))
    parser.add_argument('--steps_per_episode', type=int, default=200)
    
    parser.add_argument('--rewNet_lr', type=float, default=1e-3)
    parser.add_argument('--rewNet_maxlen', type=int, default=1000, 
    help='max len for the rewNet buffer')
    parser.add_argument('--rewNet_batch_size', type=int, default=20, 
    help='batch size for training the rewNet')
    parser.add_argument('--rewNet_epochs', type=int, default=20, 
    help='epochs for training the rewNet')

    parser.add_argument(
        '--pi_lr', type=float, default=3e-4, 
        help="Learning rate for policy optimizer")
    parser.add_argument(
        '--vf_lr', type=float, default=1e-3, 
        help="Learning rate for value function optimizer")
    parser.add_argument(
        '--train_pi_iters', type=int, default=1, 
        help="Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)")
    parser.add_argument(
        '--train_v_iters', type=int, default=5, 
        help="Number of gradient descent steps to take on value function per epoch.")
    parser.add_argument(
        '--clip_ratio', type=float, default=0.2, 
        help="Hyperparameter for clipping in the policy objective.")
    parser.add_argument(
        '--target_kl', type=float, default=0.01, 
        help="Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)")




    parser.add_argument(
        '--update_before', type=int, default=20, 
        help='Number of env interactions to collect before starting to do use rewNet')
    parser.add_argument(
        '--update_every', type=int, default=10, 
        help='Number of env interactions that should elapse between rewNet and Fdtd')
    parser.add_argument('--save_every', type=int, default=5, help='Number of env interactions that should save the model')



    args = parser.parse_args()

    return args