from inspect import ArgSpec

import os
from time import time
from datetime import datetime
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from torch.utils.data import TensorDataset, DataLoader
# from torchvision.utils.data import 

import gym

from code4.envs.fdtd_env import FdtdEnv
from environments import RewEnv
from models import RewNet, MLPActorCritic
from environments import FdtdEnv2
from arguments import get_args
from buffers import PPOBuffer, RewNetBuffer
from algos.PPO import PPO

def set_seed(args):
    # Random seed
    import random 

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    use_cuda = args.use_cuda
    if torch.cuda.is_available() and use_cuda:
        print("\ncuda is available! with %d gpus\n"%torch.cuda.device_count())
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
    else:
        print("\ncuda is not available! cpu is available!\n")
        torch.manual_seed(seed)
        device = torch.device("cpu")
    return device

def main():

    args = get_args()
    print(args)

    device = set_seed(args)
    
    steps_per_episode=args.steps_per_episode 
    
    

    # Instantiate environment
    env1 = gym.make("FdtdEnv")   
    obs_dim = env1.observation_space.shape or env1.observation_space.n
    act_dim = env1.action_space.shape or env1.action_space.n
    print(obs_dim, act_dim)

    # Create actor-critic module
    ac_net = MLPActorCritic(env1.observation_space, env1.action_space, hidden_sizes=[256,256])
    reward_net = RewNet(obs_dim, hidden_sizes=[32, 32])
    if torch.cuda.is_available() and args.use_cuda:
        ac_net.cuda()
        reward_net.cuda() 
    agent = PPO(ac_net, pi_lr=args.pi_lr, vf_lr=args.vf_lr, clip_ratio=args.clip_ratio, target_kl=args.target_kl, train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters)

    
    rewNet_loss_fn = nn.MSELoss()
    rewNet_optimizer = SGD(reward_net.parameters(), lr=args.rewNet_lr)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_episode, args.gamma, args.lam, device)
    rewNet_buf = RewNetBuffer(args.rewNet_maxlen, device)


    def update_rewNet():
        data = rewNet_buf.get()
        Xs = data['next_obs']
        ys = data['rew']
        
        dataset = TensorDataset(Xs, ys)  
        data_loader = DataLoader(dataset=dataset, batch_size=args.rewNet_batch_size, shuffle=True)

        reward_net.train()
        for t in range(args.rewNet_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            for j, (X, y) in enumerate(data_loader):
                # X, y = X.to(device), y.to(device)
                pred = reward_net(X)
                rewNet_loss = rewNet_loss_fn(pred, y)

                rewNet_optimizer.zero_grad()
                rewNet_loss.backward()
                rewNet_optimizer.step()

    def env_interact(obs, env, use_rewNet_flag=False):

        for t in range(steps_per_episode):

            a, v, logp = ac_net.step(torch.as_tensor(obs, dtype=torch.float32, device=device))
            next_o, r, d, _ = env.step(a)
            
            # save and log
            buf.store(obs, a, r, v, logp)
            if not use_rewNet_flag:
                rewNet_buf.store(next_o, r)

            obs = next_o

            rollout_full = (t+1)==steps_per_episode
            if d or rollout_full:
                if rollout_full:
                    _, v, _ = ac_net.step(torch.as_tensor(obs, dtype=torch.float32, device=device))
                else:
                    v = 0.
                buf.finish_path(v)

        return obs
    
    
    
    ##########################################
    # Prepare for interaction with environment
    ##########################################

    start_time = time.time()
    o = env1.reset()

    import math
    episodes = math.ceil(args.max_steps / steps_per_episode)
    for episode in trange(episodes):

        # interact with env
        if episode < args.update_before:
            o = env_interact(o, env1)
        else:
            if episode % args.update_every == 0:
                o = env1.reset()
                o = env_interact(o, env1)
                update_rewNet()
            else:
                env2 = FdtdEnv2(reward_net)
                o = env2.reset()
                o = env_interact(o, env2, use_rewNet_flag=True)

        # Perform PPO update!
        agent.update_policy(buf)

        # save the model
        if episode % args.save_every == 0:
            if episode < args.update_before:
                PATH = "pretrain.pt"
            else:
                PATH = 'model.pt'
            torch.save(ac_net.state_dict(), PATH)

    end_time = time()
    print('\n Time: %.3f\n'%(end_time-start_time))


if __name__ == '__main__':

    main()
