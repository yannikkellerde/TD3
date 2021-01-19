import copy
import numpy as np
import os
import torch

class TD3_base(object):
    def __init__(
        self,
        max_action=1,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def save(self, folder):
        os.makedirs(folder,exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(folder,"critic"))
        torch.save(self.critic_target.state_dict(), os.path.join(folder,"critic_target"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder,"critic_optimizer"))
        
        torch.save(self.actor.state_dict(), os.path.join(folder,"actor"))
        torch.save(self.actor_target.state_dict(), os.path.join(folder,"actor_target"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder,"actor_optimizer"))


    def load(self, folder):
        self.critic.load_state_dict(torch.load(os.path.join(folder,"critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder,"critic_optimizer")))
        if os.path.isfile(os.path.join(folder,"critic_target")):
            self.critic_target.load_state_dict(torch.load(os.path.join(folder,"critic_target")))
        else:
            self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(folder,"actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder,"actor_optimizer")))
        if os.path.isfile(os.path.join(folder,"actor_target")):
            self.actor_target.load_state_dict(torch.load(os.path.join(folder,"actor_target")))
        else:
            self.actor_target = copy.deepcopy(self.actor)