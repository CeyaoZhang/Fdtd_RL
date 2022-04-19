from collections import deque

import numpy as np
import scipy.signal

import torch


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device=torch.device("cpu")):
        
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)


        def discount_cumsum(x, discount):
            """
            magic from rllab for computing discounted cumulative sums of vectors.

            input: 
                vector x, 
                [x0, 
                x1, 
                x2]

            output:
                [x0 + discount * x1 + discount^2 * x2,  
                x1 + discount * x2,
                x2]
            """
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam) ## GAE-Lambda
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        def statistics_scalar(x):
            """
            Args:
                x: An array containing samples of the scalar to produce statistics
                    for.

                with_min_and_max (bool): If true, return min and max of x in 
                    addition to mean and std.
            """
            x = np.array(x, dtype=np.float32)
            global_sum, global_n = [np.sum(x), len(x)]
            mean = global_sum / global_n

            global_sum_sq = np.sum((x - mean)**2)
            std = np.sqrt(global_sum_sq / global_n)  # compute global std

            return mean, std

        
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std+1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, val=self.val_buf,
        ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        

        return {k: torch.as_tensor(v.copy(), dtype=torch.float32, device=self.device) for k,v in data.items()}


class RewNetBuffer:
    def __init__(self, size, device):

        self.nex_obs_buf = deque(maxlen=size)
        self.rew_buf = deque(maxlen=size)
        self.device = device

    def store(self, next_obs, rew):
        self.nex_obs_buf.append(next_obs)
        self.rew_buf.append(rew)

    def get(self):
        data = dict(next_obs=np.array(self.nex_obs_buf), rew=np.array(self.rew_buf))
        return {k: torch.as_tensor(v.copy(), dtype=torch.float32, device=self.device) for k,v in data.items()}