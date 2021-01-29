import pysplishsplash
import gym

import numpy as np
import torch
import argparse
import time
import os
import time
import json
from tqdm import tqdm,trange
from utils.noise import OrnsteinUhlenbeckActionNoise

import pickle
from rtpt.rtpt import RTPT

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, eval_episodes=2, render=True):
    eval_env.seed(seed + 100)
    eval_env.fixed_tsp = True
    eval_env.fixed_spill = True

    avg_true = 0.
    avg_reward = 0.
    avg_q = 0.
    print("Evaluating")
    for i in trange(eval_episodes):
        eval_env.time_step_punish = 1/(eval_episodes-1) * i
        state, done = eval_env.reset(use_gui=render), False
        q_list = []
        t = 0
        while not done:
            action = policy.select_action(state)
            q = np.mean(policy.eval_q(state,action))
            state, reward, done, info = eval_env.step(action)
            if render:
                eval_env.render()
            avg_true += info["true_reward"]
            avg_reward += reward
            q_list.append(q)
            t+=1
        print(f"Episode length: {t}")
        avg_q += np.mean(q_list)

    avg_reward /= eval_episodes
    avg_true /= eval_episodes
    avg_q /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} avg true reward: {avg_true:.3f} avg q value {avg_q}")
    print("---------------------------------------")
    eval_env.fixed_tsp = False
    eval_env.fixed_spill = False
    eval_env.reset(use_gui=False)
    return avg_q,avg_true

def update_temperature(env,timestep,start_increase,start_temperature):
    time_full_temp = 2e5
    env.temperature = min(1,max(0,(timestep-start_increase)/time_full_temp*(1-start_temperature)+start_temperature))

"""
python3.7 main.py --seed 66 --start_training 0 --start_policy 0 --max_timesteps 1000000 --expl_noise 0.2 --folder_name models/pouring_2 --experiment_name pouring-corr-noise --load_replay_buffer replay_buffers_1611535680.8310199/ --load_model models/pouring_f
"""

if __name__ == "__main__":
    FILE_PATH = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(FILE_PATH,"models"),exist_ok=True)
    os.makedirs(os.path.join(FILE_PATH,"results"),exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3_particles")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="water_pouring:Pouring-mdp-full-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_training", default=2e5, type=int) # Time steps before starting training
    parser.add_argument("--start_policy", default=2e5, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e2, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=4e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.12, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--policy_uncertainty",default=0.3, type=float)    # Std of env policy uncertainty
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--start_temperature", default=1, type=float)
    parser.add_argument("--lr",default=3e-4,type=float)
    #parser.add_argument("--time_step_punish", default=0.1, type=float)
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses folder_name
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--done_swap", action="store_true")         # Only store done=True if not done because of time
    parser.add_argument("--save_replay_buffer", action="store_true")
    parser.add_argument("--load_replay_buffer",default="",type=str)
    parser.add_argument("--folder_name", type=str, default="")
    parser.add_argument("--norm",type=str, default="")
    parser.add_argument("--experiment_name",type=str, default="WaterPouring")
    args = parser.parse_args()
    args.save_model = True

    REPLAY_BUFFER_PATH = "replay_buffers"
    os.makedirs(REPLAY_BUFFER_PATH,exist_ok=True)

    if args.folder_name == "":
        folder_name = f"{args.policy}_{args.env}_{args.seed}"
    else:
        folder_name = args.folder_name

    os.makedirs("parameters",exist_ok=True)
    with open(os.path.join("parameters",os.path.basename(folder_name)+".json"),"w") as f:
        json.dump(args.__dict__,f)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    
    env = gym.make(args.env,policy_uncertainty=args.policy_uncertainty)
    #env.time_step_punish = args.time_step_punish
    env.temperature = args.start_temperature
    print("made Env")


    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    max_action = float(env.action_space.high[0])

    kwargs = {
        "obs_space": env.observation_space,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "max_action":max_action,
        "lr":args.lr,
        "policy_freq": int(args.policy_freq),
        "norm": None if args.norm=="" else args.norm
    }

    # Initialize policy
    if args.policy == "TD3_featured":
        from TD3_featured import TD3
        from my_replay_buffer import ReplayBuffer_featured as ReplayBuffer
    elif args.policy == "TD3_particles":
        from TD3_particles import TD3
        from my_replay_buffer import ReplayBuffer_particles as ReplayBuffer

    policy = TD3(**kwargs)

    if args.load_model != "":
        policy_folder = folder_name if args.load_model == "default" else args.load_model
        policy.load(policy_folder)

    if args.load_replay_buffer!="":
        args.start_timesteps=0
        args.start_training = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,load_folder=args.load_replay_buffer,max_size=int(args.replay_buffer_size))
    else:
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,max_size=int(args.replay_buffer_size))
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, args.seed,render=args.render)]
    env.seed(args.seed)

    state, done = env.reset(), False
    episode_reward = 0
    episode_true_reward = 0
    episode_timesteps = 0
    episode_num = 0

    rtpt = RTPT(name_initials='YK', experiment_name=args.experiment_name, max_iterations=int(args.max_timesteps))

    rtpt.start()

    start = time.perf_counter()

    uhlbeck = OrnsteinUhlenbeckActionNoise(env.action_space.shape[0],sigma=args.expl_noise)

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_policy:
            action = env.action_space.sample()
        else:
            if t==args.start_policy and args.save_replay_buffer:
                replay_buffer.save(REPLAY_BUFFER_PATH)
            action = (
                policy.select_action(state)
                + uhlbeck.sample()
            ).clip(-max_action, max_action)
        # Perform action
        next_state, reward, done, info = env.step(action)
        if args.done_swap:
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        else:
            done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        episode_true_reward += info["true_reward"]

        # Train agent after collecting sufficient data
        if t >= args.start_training:
            if t == args.start_training:
                print("Starting training")
            policy.train(replay_buffer, args.batch_size)
            #batch = replay_buffer.sample(256)
            #policy._actor_learn(batch[0],batch[1])
        rtpt.step(subtitle=f"reward={evaluations[-1][1]:2.2f}")
        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"episode time {time.perf_counter()-start}")
            start = time.perf_counter()
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
                  f" True Reward {episode_true_reward:.3f}")
            # Reset environment
            env.seed(args.seed)
            state, done = env.reset(), False
            uhlbeck.reset()
            episode_reward = 0
            episode_true_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            update_temperature(env,t,args.start_policy,args.start_temperature)
            if episode_num % args.eval_freq == 0 and (episode_num==args.eval_freq or t>args.start_training):
                evaluations.append(eval_policy(policy, env, args.seed,render=args.render))
                np.save(f"./results/{os.path.basename(folder_name)}.npy", evaluations)
                if args.save_model: policy.save(folder_name+f"_ep-{episode_num}_ev-{int(evaluations[-1][1])}-q-{int(evaluations[-1][0])}")
        # Evaluate episode
    replay_buffer.save(REPLAY_BUFFER_PATH+"_"+str(time.time()))
    policy.save(folder_name+"_final")