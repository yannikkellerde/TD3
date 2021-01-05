import pysplishsplash
import gym

import numpy as np
import torch
import argparse
import os
import time
from tqdm import tqdm,trange

from my_replay_buffer import ReplayBuffer
from my_TD3 import TD3
import OurDDPG
import DDPG

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, eval_episodes=3, render=True):
    eval_env.seed(seed + 100)

    avg_reward = 0.
    print("Evaluating")
    for _ in trange(eval_episodes):
        state, done = eval_env.reset(use_gui=render), False
        while not done:
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            eval_env.render()
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    eval_env.reset(use_gui=False)
    return avg_reward

def update_temperature(env,timestep,start_increase,start_temperature=0.05):
    time_full_temp = 4e5
    env.temperature = min(1,max(0,(timestep-start_increase)/time_full_temp)*(1-start_temperature)+start_temperature)


if __name__ == "__main__":
    FILE_PATH = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(FILE_PATH,"models"),exist_ok=True)
    os.makedirs(os.path.join(FILE_PATH,"results"),exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="water_pouring:Pouring-mdp-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2e5, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e2, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.save_model = True

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    
    env = gym.make(args.env)
    env.temperature = 0
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
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, args.seed,render=args.render)]
    env.seed(args.seed)

    state, done = env.reset(), False
    episode_reward = 0
    episode_true_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start = time.perf_counter()
    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * args.expl_noise, size=env.action_space.shape[0])
            ).clip(-max_action, max_action)
        # Perform action
        next_state, reward, done, info = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        episode_true_reward += info["true_reward"]

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            if t == args.start_timesteps:
                print("Starting training")
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"episode time {time.perf_counter()-start}")
            start = time.perf_counter()
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
                  f" True Reward {episode_true_reward:.3f}")
            # Reset environment
            env.seed(args.seed)
            state, done = env.reset(), False
            episode_reward = 0
            episode_true_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            update_temperature(env,t,args.start_timesteps)
            if episode_num % args.eval_freq == 0 and (episode_num==args.eval_freq or t>args.start_timesteps):
                evaluations.append(eval_policy(policy, env, args.seed,render=args.render))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: policy.save(f"./models/{file_name}")
        # Evaluate episode
