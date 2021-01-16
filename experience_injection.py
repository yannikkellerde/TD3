import pysplishsplash
import gym
from my_replay_buffer import ReplayBuffer
import numpy as np
import sys,os
import argparse
from tqdm import trange,tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="water_pouring:Pouring-mdp-full-v0")          # OpenAI gym environment name
parser.add_argument("--replay_buffer_path",type=str,default="replay_buffers")
parser.add_argument("--policy_path",type=str)
parser.add_argument("--noise",type=float,default=0.3)
parser.add_argument("--num_inject",type=int,default=1e5)
args = parser.parse_args()

env = gym.make(args.env,use_gui=False)
replay_buffer = ReplayBuffer(env.observation_space,env.action_space)#,load_folder=args.replay_buffer_path)
print("loaded_replay_buffer")
policy = np.load(args.policy_path)
state = env.reset()
epi_pos = 0
rewsum = 0
for i in trange(args.num_inject):
    if len(policy) > epi_pos:
        action = policy[epi_pos]
    else:
        action = policy[-1]
    action = np.random.normal(action,args.noise)
    new_state,reward,done,_ = env.step(action)
    epi_pos+=1
    rewsum += reward
    done_bool = float(done) if epi_pos < env._max_episode_steps else 0
    replay_buffer.add(state,action,new_state,reward,done_bool)
    state = new_state
    if done:
        epi_pos = 0
        rewsum = 0
        state = env.reset()
replay_buffer.save(args.replay_buffer_path)
    