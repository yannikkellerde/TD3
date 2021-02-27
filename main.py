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
from copy import deepcopy
from utils.noise import OrnsteinUhlenbeckActionNoise

import pickle
from rtpt.rtpt import RTPT
filepath = os.path.abspath(os.path.dirname(__file__))

def eval_policy(policy, eval_env, seed, eval_episodes=10, render=True):
    eval_env.seed(seed + 100)
    eval_env.fixed_tsp = True
    eval_env.fixed_spill = True
    eval_env.fixed_target_fill = True
    spills = np.linspace(eval_env.spill_range[0],eval_env.spill_range[1],num=eval_episodes)
    np.random.shuffle(spills)
    tfills = np.linspace(eval_env.target_fill_range[0],eval_env.target_fill_range[1],num=eval_episodes)
    np.random.shuffle(tfills)

    avg_reward = 0.
    avg_q = 0.
    print("Evaluating")
    for i in trange(eval_episodes):
        if args.fixed_tsp is None:
            eval_env.time_step_punish = 1/(eval_episodes-1) * i
        if args.fixed_spill_punish is None:
            eval_env.spill_punish = spills[i]
        if args.fixed_target_fill is None:
            eval_env.target_fill_state = tfills[i]

        state, done = eval_env.reset(use_gui=render), False
        q_list = []
        t = 0
        while not done:
            action = policy.select_action(state)
            q = np.mean(policy.eval_q(state,action))
            state, reward, done, info = eval_env.step(action)
            if render:
                eval_env.render()
            avg_reward += reward
            q_list.append(q)
            t+=1
        print(f"Episode length: {t}")
        avg_q += np.mean(q_list)

    avg_reward /= eval_episodes
    avg_q /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} avg q value {avg_q}")
    print("---------------------------------------")
    if args.fixed_spill_punish is None:
        eval_env.fixed_spill = False
    if args.fixed_tsp is None:
        eval_env.fixed_tsp = False
    if args.fixed_target_fill is None:
        eval_env.fixed_target_fill = False
    eval_env.reset(use_gui=False)
    return avg_q,avg_reward

def add_to_replay_buffer(env,replay_buffer,state,action,reward,next_state,done_bool):
    replay_buffer.add(state, action, next_state, reward, done_bool)
    if args.hindsight_number>0 and not (args.fixed_tsp and args.fixed_spill_punish and args.fixed_target_fill):
        for i in range(args.hindsight_number):
            if args.fixed_tsp:
                tsp = env.time_step_punish
            else:
                tsp = env._get_random(env.time_step_punish_range)
            if args.fixed_spill_punish:
                spill_punish = env.spill_punish
            else:
                spill_punish = env._get_random(env.spill_range)
            if args.fixed_target_fill:
                target_fill = env.target_fill_state
            else:
                target_fill = env._get_random(env.target_fill_range)
            manip_reward = env._imagine_reward(tsp,spill_punish,target_fill)
            manip_state = deepcopy(state)
            manip_next = deepcopy(next_state)
            env.manip_state(manip_state,tsp,spill_punish,target_fill)
            env.manip_state(manip_next,tsp,spill_punish,target_fill)
            replay_buffer.add(manip_state, action, manip_next, manip_reward, done_bool)

"""
python3.7 main.py --max_timesteps 1500000 --start_training 100000 --start_policy 100000 --norm layer --fixed_spill_punish 25 --fixed_target_fill 390 --experiment_name pouring-tsp --folder models/tsp-no-discount --discount 0.999
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
    parser.add_argument("--lr",default=1e-4,type=float)
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
    parser.add_argument("--noCDQ",action="store_true")  # Do not use TD3 clipped double Q
    parser.add_argument("--fixed_tsp",type=float,default=None)
    parser.add_argument("--fixed_spill_punish",type=int,default=None)
    parser.add_argument("--fixed_target_fill",type=int,default=None)
    parser.add_argument("--experiment_name",type=str, default="WaterPouring")
    parser.add_argument("--hindsight_number",type=int, default=0)
    parser.add_argument("--human_compare",action="store_true")
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

    env_kwargs = {
        "policy_uncertainty":args.policy_uncertainty,
        "fixed_tsp":args.fixed_tsp is not None,
        "fixed_target_fill":args.fixed_target_fill is not None,
        "fixed_spill":args.fixed_spill_punish is not None
    }
    if args.human_compare:
        env_kwargs["scene_base"] = "scenes/smaller_scene.json"
    env = gym.make(args.env,**env_kwargs)
    if args.human_compare:
        env.max_in_glas = 215
        env.target_fill_range = [114,209]
    if args.fixed_tsp is not None:
        env.time_step_punish = args.fixed_tsp
    if args.fixed_target_fill is not None:
        env.target_fill_state = args.fixed_target_fill
    if args.fixed_spill_punish is not None:
        env.spill_punish = args.fixed_spill_punish
    print(env.observation_space,env.action_space)
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
        "norm": None if args.norm=="" else args.norm,
        "CDQ": not args.noCDQ
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
        add_to_replay_buffer(env,replay_buffer,state,action,reward,next_state,done_bool)
        state = next_state
        episode_reward += reward

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
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            env.seed(args.seed)
            state, done = env.reset(), False
            uhlbeck.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            if episode_num % args.eval_freq == 0 and (episode_num==args.eval_freq or t>args.start_training):
                evaluations.append(eval_policy(policy, env, args.seed,render=args.render))
                np.save(f"./results/{os.path.basename(folder_name)}.npy", evaluations)
                if args.save_model: policy.save(folder_name+f"_ep-{episode_num}_t-{t}_ev-{int(evaluations[-1][1])}_q-{int(evaluations[-1][0])}")
        # Evaluate episode
    replay_buffer.save(REPLAY_BUFFER_PATH+"_"+str(time.time()))
    policy.save(folder_name+"_final")