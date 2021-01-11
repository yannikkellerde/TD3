import copy
import pysplishsplash
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torchvision import models
import gym

if not torch.cuda.is_available():
    print("WARNING: NO CUDA")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor_mdp(nn.Module):
    def __init__(self,obs_space,action_space):
        # State expected to be tuple of 0: Box features, 1: convolutional part
        super(Actor_mdp, self).__init__()
        
        self.num_features = 64
        self.num_linear = 128

        self.conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)
        
        self.avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))
        #self.b1 = nn.BatchNorm1d(self.num_features+obs_space[0].shape[0]) This somehow does not work
        self.l1 = nn.Linear(self.num_features+obs_space[0].shape[0],self.num_linear)
        #self.b2 = nn.BatchNorm1d(self.num_features)
        self.l2 = nn.Linear(self.num_linear,self.num_linear)
        self.l3 = nn.Linear(self.num_linear,action_space.shape[0])
        
    def forward(self, state_features, state_particles):
        a = torch.unsqueeze(state_particles,1)
        a = F.relu(self.conv1(a))
        a = torch.squeeze(a,dim=3)
        a = F.relu(self.conv2(a))
        a = F.relu(self.avg_pool(a))
        a = torch.squeeze(a,dim=2)
        a = torch.cat([a,state_features],1)
        #a = self.b1(a)
        a = F.relu(self.l1(a))
        #a = self.b2(a)
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a

class Critic_mdp(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Critic_mdp, self).__init__()

        self.num_features = 64
        self.num_linear = 128

        # Q1 architecture
        self.q1_conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.q1_conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)

        self.q1_avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))
        
        #self.q1_b1 = nn.BatchNorm1d(self.num_features+obs_space[0].shape[0]+action_space.shape[0])
        self.q1_l1 = nn.Linear(self.num_features+obs_space[0].shape[0]+action_space.shape[0],self.num_linear)
        self.q1_l2 = nn.Linear(self.num_linear,self.num_linear)
        #self.q1_b2 = nn.BatchNorm1d(self.num_features) I am too stupid for batch normalization
        self.q1_l3 = nn.Linear(self.num_linear,1)

        # Q1 architecture
        self.q2_conv1 = nn.Conv2d(1,self.num_features*2,kernel_size=(1,obs_space[1].shape[1]),stride=1)
        self.q2_conv2 = nn.Conv1d(self.num_features*2,self.num_features,kernel_size=1,stride=1)

        self.q2_avg_pool = nn.AvgPool2d(kernel_size = (1,obs_space[1].shape[0]))
        #self.q2_b1 = nn.BatchNorm1d(self.num_features+obs_space[0].shape[0]+action_space.shape[0])
        self.q2_l1 = nn.Linear(self.num_features+obs_space[0].shape[0]+action_space.shape[0],self.num_linear)
        self.q2_l2 = nn.Linear(self.num_linear,self.num_linear)
        #self.q2_b2 = nn.BatchNorm1d(self.num_features)
        self.q2_l3 = nn.Linear(self.num_linear,1)



    def forward(self, state_features, state_particles, action):
        q1 = torch.unsqueeze(state_particles,1)
        q1 = F.relu(self.q1_conv1(q1))
        q1 = torch.squeeze(q1,dim=3)
        q1 = F.relu(self.q1_conv2(q1))
        q1 = F.relu(self.q1_avg_pool(q1))
        q1 = torch.squeeze(q1,dim=2)
        q1 = torch.cat([q1,state_features,action],1)
        #q1 = self.q1_b1(q1)
        q1 = F.relu(self.q1_l1(q1))
        q1 = F.relu(self.q1_l2(q1))
        #q1 = self.q1_b2(q1)
        q1 = self.q1_l3(q1)

        q2 = torch.unsqueeze(state_particles,1)
        q2 = F.relu(self.q2_conv1(q2))
        q2 = torch.squeeze(q2,dim=3)
        q2 = F.relu(self.q2_conv2(q2))
        q2 = F.relu(self.q2_avg_pool(q2))
        q2 = torch.squeeze(q2,dim=2)
        q2 = torch.cat([q2,state_features,action],1)
        #q2 = self.q2_b1(q2)
        q2 = F.relu(self.q2_l1(q2))
        q2 = F.relu(self.q2_l2(q2))
        #q2 = self.q2_b2(q2)
        q2 = self.q2_l3(q2)

        return q1, q2


    def Q1(self, state_features, state_particles, action):
        q1 = torch.unsqueeze(state_particles,1)
        q1 = F.relu(self.q1_conv1(q1))
        q1 = torch.squeeze(q1,dim=3)
        q1 = F.relu(self.q1_conv2(q1))
        q1 = F.relu(self.q1_avg_pool(q1))
        q1 = torch.squeeze(q1,dim=2)
        q1 = torch.cat([q1,state_features,action],1)
        #q1 = self.q1_b1(q1)
        q1 = F.relu(self.q1_l1(q1))
        q1 = F.relu(self.q1_l2(q1))
        #q1 = self.q1_b2(q1)
        q1 = self.q1_l3(q1)
        return q1

class TD3(object):
    def __init__(
        self,
        obs_space,
        action_space,
        discount=0.99,
        tau=0.005,
        policy_noise=0.02,
        noise_clip=0.05,
        policy_freq=2
    ):

        self.actor = Actor_mdp(obs_space,action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic_mdp(obs_space,action_space).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        features = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
        particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
        action = self.actor(features,particles).cpu().data.numpy().flatten()
        return action

    def eval_q(self,state,action):
        features = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
        action = torch.FloatTensor(np.array(action).reshape(1, -1)).to(device)
        particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
        #self.critic.eval()
        res = [x.cpu().data.numpy().flatten() for x in self.critic(features,particles,action)]
        #self.critic.train()
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

        #print(action,actor_loss)

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, folder):
        os.makedirs(folder,exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(folder,"critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder,"critic_optimizer"))
        
        torch.save(self.actor.state_dict(), os.path.join(folder,"actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder,"actor_optimizer"))


    def load(self, folder):
        self.critic.load_state_dict(torch.load(os.path.join(folder,"critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(folder,"critic_optimizer")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(folder,"actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(folder,"actor_optimizer")))
        self.actor_target = copy.deepcopy(self.actor)

if __name__=="__main__":
    env = gym.make("water_pouring:Pouring-mdp-v0")
    td3 = TD3(env.observation_space,env.action_space)
    start = time.perf_counter()
    state = env.reset()
    for i in range(150):
        state,reward,done,info = env.step(td3.select_action(state))
    print(time.perf_counter()-start)
    exit()
    actor = Actor_mdp(env.observation_space,env.action_space).to(device)
    critic = Critic_mdp(env.observation_space,env.action_space).to(device)
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