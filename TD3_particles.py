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
        
        self.num_features = 128
        self.num_linear = 256

        self.conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)
        
        self.avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))
        self.l1 = nn.Linear(self.num_features+obs_space[0].shape[0],self.num_linear)
        self.l2 = nn.Linear(self.num_linear,self.num_linear)
        self.l3 = nn.Linear(self.num_linear,action_space.shape[0])

        self.norm = norm
        if self.norm == "layer":
            self.ln1 = nn.LayerNorm(self.num_features+obs_space[0].shape[0])
            self.ln2 = nn.LayerNorm(self.num_linear)
            self.ln3 = nn.LayerNorm(self.num_linear)

        if self.norm == "weight_normalization":
            self.l1 = nn.utils.weight_norm(self.l1)
            self.l2 = nn.utils.weight_norm(self.l2)
            self.l3 = nn.utils.weight_norm(self.l3)
        
    def forward(self, state_features, state_particles):
        a = torch.unsqueeze(state_particles,1)
        a = F.relu(self.conv1(a))
        a = torch.squeeze(a,dim=3)
        a = F.relu(self.conv2(a))
        a = F.relu(self.avg_pool(a))
        a = torch.squeeze(a,dim=2)
        a = torch.cat([a,state_features],1)
        if self.norm == "layer":
            a = self.ln1(a)
        a = F.relu(self.l1(a))
        if self.norm == "layer":
            a = self.ln2(a)
        a = F.relu(self.l2(a))
        if self.norm == "layer":
            a = self.ln3(a)
        a = torch.tanh(self.l3(a))
        return a

class Q_network(nn.Module):
    def __init__(self, obs_space, action_space, norm=None):
        super(Q_network, self).__init__()

        self.num_features = 128
        self.num_linear = 256

        self.conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))
        
        self.l1 = nn.Linear(self.num_features+obs_space[0].shape[0]+action_space.shape[0],self.num_linear)
        self.l2 = nn.Linear(self.num_linear,self.num_linear)
        self.l3 = nn.Linear(self.num_linear,1)

        self.norm = norm
        if self.norm == "layer":
            self.ln1 = nn.LayerNorm(self.num_features+obs_space[0].shape[0]+action_space.shape[0])
            self.ln2 = nn.LayerNorm(self.num_linear)
            self.ln3 = nn.LayerNorm(self.num_linear)

        if self.norm == "weight_normalization":
            self.l1 = nn.utils.weight_norm(self.l1)
            self.l2 = nn.utils.weight_norm(self.l2)
            self.l3 = nn.utils.weight_norm(self.l3)

    def forward(self, state_features, state_particles, action):
        q = torch.unsqueeze(state_particles,1)
        q = F.relu(self.conv1(q))
        q = torch.squeeze(q,dim=3)
        q = F.relu(self.conv2(q))
        q = F.relu(self.avg_pool(q))
        q = torch.squeeze(q,dim=2)
        q = torch.cat([q,state_features,action],1)
        if self.norm == "layer":
            q = self.ln1(q)
        q = F.relu(self.l1(q))
        if self.norm == "layer":
            q = self.ln2(q)
        q = F.relu(self.l2(q))
        if self.norm == "layer":
            q = self.ln3(q)
        q = self.l3(q)
        return q

class Critic(nn.Module):
    def __init__(self, obs_space, action_space, norm=None):
        super(Critic, self).__init__()

        self.q1 = Q_network(obs_space,action_space,norm)
        self.q2 = Q_network(obs_space,action_space,norm)

    def forward(self, state_features, state_particles, action):
        return self.q1(state_features,state_particles,action), self.q2(state_features,state_particles,action)

    def Q1(self, state_features, state_particles, action):
        return self.q1(state_features,state_particles,action)

class TD3(TD3_base):
    def __init__(self,obs_space,action_space,lr=1e-4,norm=None,**kwargs):
        self.actor = Actor(obs_space, action_space,norm).to(device)
        self.actor_target = Actor(obs_space, action_space,norm).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_space, action_space, norm).to(device)
        self.critic_target = Critic(obs_space, action_space, norm).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
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
        start = time.perf_counter()
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
            target_Q1, target_Q2 = self.critic_target(next_state_features,next_state_particles, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        start = time.perf_counter()
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_features,state_particles, action)
        # Compute critic loss
        start = time.perf_counter()
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

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