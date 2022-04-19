import torch
from torch.optim import SGD, Adam


class PPO():

    def __init__(self, policy, pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2, target_kl=0.01, train_pi_iters=1, train_v_iters=5):

        self.policy = policy
        # Set up optimizers for policy, value function and rew Net
        self.pi_optimizer = Adam(self.policy.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.policy.v.parameters(), lr=vf_lr)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(self):
        obs, act, adv, logp_old = self.data['obs'], self.data['act'], self.data['adv'], self.data['logp']

        # Policy loss
        pi, logp = self.policy.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info


    # Set up function for computing value loss
    def compute_loss_v(self):
        obs, ret = self.data['obs'], self.data['ret']
        return ((self.policy.v(obs) - ret)**2).mean()


    def update_policy(self, buf):

        self.data = buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(self.data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(self.data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(self.data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(self.data)
            loss_v.backward()
            self.vf_optimizer.step()