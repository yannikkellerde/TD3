import pysplishsplash
import gym

import numpy as np
import torch
import argparse
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm,trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_state_to_torch(state):
    features = torch.FloatTensor(np.array([state[0]]).reshape(1, -1)).to(device)
    particles = torch.FloatTensor(state[1].reshape(1,*state[1].shape)).to(device)
    return features,particles

def evalstuff(state,action,td3):
    features,particles = convert_state_to_torch(state)
    features[-1] = 0
    #td3.actor.eval()
    #td3.critic.eval()
    #print("likestuff",td3.actor(features,particles),td3.critic.Q1(features,particles, td3.actor(features,particles)))
    #print("action",action)
    q_val = policy.eval_q(state,action)
    #print(state[0],action,q_val)
    #print("chosen",q_val)
    #print("zero",policy.eval_q(state,[0]))
    #print("special",policy.eval_q(state,[1]))
    #print("one",policy.eval_q(state,[1,1,1]))
    return (q_val[0][0]+q_val[1][0])/2

def train(state,td3):
    batch_size = 32
    all_feat = []
    all_part = []
    for _ in range(batch_size):
        f,p = convert_state_to_torch(state)
        all_feat.append(f)
        all_part.append(p)
    features = torch.cat(all_feat,0)
    particles = torch.cat(all_part,0)
    td3._actor_learn(features,particles)

def plot_q_compare(rew_lists,q_lists,discount):
    maxi = max(len(x) for x in rew_lists)
    print([len(x) for x in rew_lists])
    emp_rewards = [0 for _ in range(len(rew_lists))]
    emp_avg = []
    q_avg = []
    for i in range(maxi-1,-1,-1):
        for j in range(len(rew_lists)):
            emp_pot = []
            q_pot = []
            if len(rew_lists[j]) > i:
                emp_rewards[j] = emp_rewards[j]*discount + rew_lists[j][i]
                emp_pot.append(emp_rewards[j])
                q_pot.append(q_lists[j][i])
        emp_avg.append(np.mean(emp_pot))
        q_avg.append(np.mean(q_pot))
    emp_avg.reverse()
    q_avg.reverse()
    plt.plot(emp_avg,label="empirical Q value (discounted)")
    plt.plot(q_avg,label="TD3 computed Q value")
    plt.xlabel("time step")
    plt.ylabel("Q-value")
    plt.legend()
    plt.show()
        

def eval_policy(policy, eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)

    all_q_val_lists = []
    b_list = []
    all_reward_lists = []
    print("Evaluating")
    for _ in trange(eval_episodes):
        state, done = eval_env.reset(use_gui=True), False
        b = 0
        reward_list = []
        q_val_list = []
        while not done:
            b+=1
            #print("obs",state[0])
            action = policy.select_action(state)
            #evalstuff(state,action,policy)
            #exit()
            #action = policy.select_action(state)
            q_val_list.append(evalstuff(state,action,policy))
            #if b==5:
            #    while 1:
            #        action = policy.select_action(state)
            #        for i in range(10):
            #            train(state,policy)
            #        evalstuff(state,action,policy)
            #action = [-1,-1,-1]
            state, reward, done, _ = eval_env.step(action)
            eval_env.render()
            reward_list.append(reward)
        all_q_val_lists.append(q_val_list)
        all_reward_lists.append(reward_list)
        b_list.append(b)
    print("Q",all_q_val_lists[0][0])
    avg_reward = np.mean([np.sum(x) for x in all_reward_lists])
    plot_q_compare(all_reward_lists,all_q_val_lists,args.discount)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Avg episode length {np.mean(b_list)}")
    print("---------------------------------------")
    eval_env.reset(use_gui=False)
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3_particles")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="water_pouring:Pouring-mdp-full-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e2, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--policy_uncertainty",default=0.3, type=float)    # Std of env policy uncertainty
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    #parser.add_argument("--time_step_punish", default=0.1, type=float)
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    
    env = gym.make(args.env,policy_uncertainty=args.policy_uncertainty,fixed_tsp=True)
    env.time_step_punish = 0.1
    print(env.observation_space,env.action_space)
    print("made Env")

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    max_action = float(env.action_space.high[0])

    kwargs = {
        "obs_space": env.observation_space,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
    }

    if args.policy == "TD3_featured":
        from TD3_featured import TD3
        from my_replay_buffer import ReplayBuffer_featured as ReplayBuffer
    elif args.policy == "TD3_particles":
        from TD3_particles import TD3
        from my_replay_buffer import ReplayBuffer_particles as ReplayBuffer
    policy = TD3(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(policy_file)

    #for param in policy.actor.parameters():
    #    print(torch.abs(param).mean())
    #exit()

    #print("loading_replay_buffer")
    #rb = ReplayBuffer(env.observation_space,env.action_space,load_folder="replay_buffers")
    #print("loaded_replay_buffer")

    #for _ in range(100):
        #batch = rb.sample(256)
        #policy._actor_learn(batch[0],batch[1])
        #train(env.observation_space.sample(),policy)

    print("finished learning")

    # Set seeds
    
    evaluations = eval_policy(policy, env, args.seed)