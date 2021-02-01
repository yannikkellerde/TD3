import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from TD3_base import TD3_base


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, norm):
        super(Actor, self).__init__()

        self.architecture = (500,400,300)
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(state_dim,dim))
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_dim))
        
        self.norm = norm
        if self.norm == "layer":
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)
        
        if self.norm == "weight_normalization":
            for i in range(len(self.linears)):
                self.linears[i] = nn.utils.weight_norm(self.linears[i])
        self.max_action = max_action
        

    def forward(self, state):
        a = state
        for i in range(len(self.linears)):
            a = self.linears[i](a)
            if i!=len(self.linears)-1:
                a = F.relu(a)
                if self.norm == "layer":
                    a = self.lnorms[i](a)
        a = torch.tanh(a)
        return self.max_action * a

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, norm):
        super(Q, self).__init__()

        self.architecture = (500,400,200)
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(state_dim+action_dim,dim))
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],1))
        
        self.norm = norm
        if self.norm == "layer":
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)
        
        if self.norm == "weight_normalization":
            for i in range(len(self.linears)):
                self.linears[i] = nn.utils.weight_norm(self.linears[i])
        

    def forward(self, state, action):
        q = torch.cat([state, action], 1)
        for i in range(len(self.linears)):
            q = self.linears[i](q)
            if i!=len(self.linears)-1:
                q = F.relu(q)
                if self.norm == "layer":
                    q = self.lnorms[i](q)
        return q


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, norm):
        super(Critic, self).__init__()

        self.q1 = Q(state_dim, action_dim, norm)
        self.q2 = Q(state_dim, action_dim, norm)

    def forward(self, state, action):
        return self.q1(state,action), self.q2(state,action)


    def Q1(self, state, action):
        return self.q1(state,action)


class TD3(TD3_base):
    def __init__(self,obs_space,action_space,max_action=1,lr=1e-4,norm=None,CDQ=True,**kwargs):
        self.actor = Actor(obs_space.shape[0], action_space.shape[0], max_action, norm).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        print(list(self.actor.parameters()),self.actor.linears)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_space.shape[0], action_space.shape[0], norm).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        super(TD3,self).__init__(max_action=max_action,**kwargs)


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def eval_q(self,state,action):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        res = [x.cpu().data.numpy().flatten() for x in self.critic(state,action)]
        return res

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)