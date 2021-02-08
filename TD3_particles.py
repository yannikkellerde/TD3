import copy
import pysplishsplash
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from TD3_base import TD3_base
from torchvision import models
import gym

if not torch.cuda.is_available():
    print("WARNING: NO CUDA")
else:
    print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self,obs_space,action_space,norm=None):
        # State expected to be tuple of 0: Box features, 1: convolutional part
        super(Actor, self).__init__()
        
        #self.architecture = (256,256)
        self.architecture = (500,400,300)

        self.num_features = 128

        self.conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)
        
        self.avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))

        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(self.num_features+obs_space[0].shape[0],dim))
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_space.shape[0]))

        self.norm = norm
        if self.norm == "layer":
            self.lnorm1 = nn.LayerNorm(self.num_features+obs_space[0].shape[0])
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)
        
        if self.norm == "weight_normalization":
            for i in range(len(self.linears)):
                self.linears[i] = nn.utils.weight_norm(self.linears[i])
        
    def forward(self, state_features, state_particles):
        a = torch.unsqueeze(state_particles,1)
        a = F.relu(self.conv1(a))
        a = torch.squeeze(a,dim=3)
        a = F.relu(self.conv2(a))
        a = F.relu(self.avg_pool(a))
        a = torch.squeeze(a,dim=2)
        a = torch.cat([a,state_features],1)
        a = self.lnorm1(a)
        for i in range(len(self.linears)):
            a = self.linears[i](a)
            if i!=len(self.linears)-1:
                a = F.relu(a)
                if self.norm == "layer":
                    a = self.lnorms[i](a)
        a = torch.tanh(a)
        return a

class Q_network(nn.Module):
    def __init__(self, obs_space, action_space, norm=None):
        super(Q_network, self).__init__()

        #self.architecture = (256,256)
        self.architecture = (500,400,300)

        self.num_features = 128

        self.conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))
        
        self.linears = nn.ModuleList()
        for i,dim in enumerate(self.architecture):
            if i==0:
                self.linears.append(nn.Linear(self.num_features+obs_space[0].shape[0]+action_space.shape[0],dim))
            else:
                self.linears.append(nn.Linear(self.architecture[i-1],dim))
        self.linears.append(nn.Linear(self.architecture[-1],action_space.shape[0]))

        self.norm = norm
        if self.norm == "layer":
            self.lnorm1 = nn.LayerNorm(self.num_features+obs_space[0].shape[0])
            self.lnorms = [nn.LayerNorm(dim) for dim in self.architecture]
            self.lnorms = nn.ModuleList(self.lnorms)
        
        if self.norm == "weight_normalization":
            for i in range(len(self.linears)):
                self.linears[i] = nn.utils.weight_norm(self.linears[i])

    def forward(self, state_features, state_particles, action):
        q = torch.unsqueeze(state_particles,1)
        q = F.relu(self.conv1(q))
        q = torch.squeeze(q,dim=3)
        q = F.relu(self.conv2(q))
        q = F.relu(self.avg_pool(q))
        q = torch.squeeze(q,dim=2)
        q = torch.cat([q,state_features,action],1)
        q = self.lnorm1(q)
        for i in range(len(self.linears)):
            q = self.linears[i](q)
            if i!=len(self.linears)-1:
                q = F.relu(q)
                if self.norm == "layer":
                    q = self.lnorms[i](q)
        return q

class Critic(nn.Module):
    def __init__(self, obs_space, action_space, norm=None, CDQ=True):
        super(Critic, self).__init__()

        self.q1 = Q_network(obs_space,action_space,norm)
        self.CDQ = CDQ
        if self.CDQ:
            self.q2 = Q_network(obs_space,action_space,norm)

    def forward(self, state_features, state_particles, action):
        if not self.CDQ:
            return (self.q1(state_features, state_particles,action),)
        return self.q1(state_features,state_particles,action), self.q2(state_features,state_particles,action)

    def Q1(self, state_features, state_particles, action):
        return self.q1(state_features,state_particles,action)

class TD3(TD3_base):
    def __init__(self,obs_space,action_space,lr=1e-4,norm=None,CDQ=True,**kwargs):
        self.actor = Actor(obs_space, action_space,norm).to(device)
        self.actor_target = Actor(obs_space, action_space,norm).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_space, action_space, norm,CDQ=CDQ).to(device)
        self.critic_target = Critic(obs_space, action_space, norm,CDQ=CDQ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.CDQ = CDQ
        super(TD3, self).__init__(**kwargs)

    def select_action(self, state):
        features = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
        particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
        action = self.actor(features,particles).cpu().data.numpy().flatten()
        return action

    def eval_q(self,state,action):
        features = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
        action = torch.FloatTensor(np.array(action).reshape(1, -1)).to(device)
        particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
        res = [x.cpu().data.numpy().flatten() for x in self.critic(features,particles,action)]
        return res


    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state_features, state_particles, action, next_state_features, next_state_particles, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state_features,next_state_particles) + noise
            )

            # Compute the target Q value
            if self.CDQ:
                target_Q1, target_Q2 = self.critic_target(next_state_features,next_state_particles, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
            else:
                target_Q = self.critic_target(next_state_features,next_state_particles, next_action)[0]
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        if self.CDQ:
            current_Q1, current_Q2 = self.critic(state_features,state_particles, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        else:
            current_Q1 = self.critic(state_features,state_particles, action)[0]
            critic_loss = F.mse_loss(current_Q1, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            self._actor_learn(state_features,state_particles)

    def _actor_learn(self,state_features,state_particles):
        # Compute actor losse
        action = self.actor(state_features,state_particles)
        actor_loss = -self.critic.Q1(state_features,state_particles, action).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__=="__main__":
    env = gym.make("water_pouring:Pouring-mdp-v0")
    td3 = TD3(env.observation_space,env.action_space)
    start = time.perf_counter()
    state = env.reset()
    for i in range(150):
        state,reward,done,info = env.step(td3.select_action(state))
    print(time.perf_counter()-start)
    exit()
    actor = Actor(env.observation_space,env.action_space).to(device)
    critic = Critic(env.observation_space,env.action_space).to(device)
    state,reward,done,info = env.step([-1])
    feat_batch = np.empty((1,1))
    part_batch = np.empty((1,*state[1].shape))
    for i in range(1):
        feat_batch[i] = state[0]
        part_batch[i] = state[1]
    features = torch.FloatTensor(feat_batch).to(device)
    particles = torch.FloatTensor(part_batch).to(device)
    c = []
    start = time.perf_counter()
    for i in range(150):
        state,reward,done,info = env.step([1])
        feat_batch = np.empty((1,1))
        part_batch = np.empty((1,*state[1].shape))
        act_batch = np.empty((1,1))
        for i in range(1):
            feat_batch[i] = state[0]
            part_batch[i] = state[1]
            act_batch[i] = -1
        features = torch.FloatTensor(feat_batch).to(device)
        particles = torch.FloatTensor(part_batch).to(device)
        action = torch.FloatTensor(act_batch).to(device)
        c.append(actor(features,particles))
    print(time.perf_counter()-start)
    #model_parameters = filter(lambda p: p.requires_grad, actor.parameters())
    #for p in model_parameters:
    #    print(p.size())
    #params = sum([np.prod(p.size()) for p in model_parameters])#
    #print(params)