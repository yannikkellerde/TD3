import pysplishsplash
import gym

import pickle
import numpy as np
import torch
import argparse
import os,sys
import time
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from scipy.stats import linregress
from scipy.spatial.transform import Rotation as R
import math
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

def plot_q_compare(rew_lists,q_lists,discount,path,show=False):
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
    plt.savefig(path)
    if show:
        plt.show()
    plt.cla()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def smooth_compare(x_ax1,x_ax2,vals1,vals2,xlabel,ylabel,legend_vals,path,show=False,sigma=5):
    fig = plt.figure(figsize=(10,4))
    lims = (max(min(x_ax1),min(x_ax2)),min(max(x_ax1),max(x_ax2)))
    v1 = gaussian_filter1d(np.array(vals1,dtype=np.float),sigma)
    v2 = gaussian_filter1d(np.array(vals2,dtype=np.float),sigma)
    plt.plot(x_ax1[:len(v1)],v1,label=legend_vals[0],color="#00664d",linewidth=2)
    plt.plot(x_ax1[:len(vals1)],vals1,color="#00664d",alpha=0.4,linewidth=0.8)
    plt.plot(x_ax2[:len(v2)],v2,label=legend_vals[1],color="#e65c00",linewidth=2)
    plt.plot(x_ax2[:len(vals2)],vals2,color="#e65c00",alpha=0.4,linewidth=0.8)
    if ylabel=="Deviation from target fill level (ml)":
        plt.ylim(0,50)
    if ylabel=="Spilled":
        plt.ylim(0,5)
    plt.xticks(np.arange(lims[0],lims[1]+1,20))
    plt.xlim(lims[0],lims[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.cla()

def plot_mean(x_ax,vals,xlabel,ylabel,legend_val,title,path,show=False,sigma=5):
    #plt.rcParams.update({'font.size': 17})
    rm = gaussian_filter1d(np.array(vals,dtype=np.float),sigma)
    fig = plt.figure(figsize=(10,4))
    plt.plot(x_ax,vals,label=legend_val)
    plt.plot(x_ax[:len(rm)],rm,label="gaussian smoothed",linewidth=4.0)
    plt.xticks(np.arange(min(x_ax),max(x_ax)+1,60))
    plt.xlim((min(x_ax),max(x_ax)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title=="Absolute deviation from target fill level":
        plt.ylim(0,50)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.cla()

def plot_action(all_action_list,path,show=False):
    max_len = max(len(x) for x in all_action_list)
    avg_actions = []
    for i in range(max_len):
        pot = []
        for acs in all_action_list:
            if len(acs)>i:
                pot.append(acs[i])
        avg_actions.append(np.mean(pot))
    plt.plot(avg_actions)
    plt.xlabel("Step")
    plt.ylabel("Avg rotation action")
    plt.title("avg rotation action per step")
    plt.savefig(path)
    if show:
        plt.show()
    plt.cla()

def plot_2d(tsps,spills,values,title,path,show=False,sigma=5):
    #S,T = np.meshgrid(tsps,spills)
    V = np.array(values).reshape((len(tsps),len(tsps)))
    print(V)
    V = gaussian_filter(V,sigma=5)
    #plt.pcolormesh(T,S,V,shading="gouraud")
    plt.imshow(V,interpolation="bilinear",cmap='RdBu',origin="lower")#,extent=[tsps[0],tsps[-1],spills[0],spills[-1]])
    plt.xticks(range(0,len(tsps),4),[round(x,2) for x in tsps[::4]])
    plt.yticks(range(0,len(spills),4),[round(x,2) for x in spills[::4]])
    plt.xlabel("time step punish")
    plt.ylabel("spill punish")
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    if show:
        plt.show()
    plt.clf()

def eval_2d(policy,eval_env,seed,path,root_episodes=30,sigma=5,render=False):
    os.makedirs(path,exist_ok=True)
    policy.critic.eval()
    policy.actor.eval()
    eval_env.seed(seed + 100)
    eval_env.fixed_tsp = True
    eval_env.fixed_spill = True
    spills = np.linspace(eval_env.spill_range[0],eval_env.spill_range[1],num=root_episodes)
    tsps = np.linspace(eval_env.time_step_punish_range[0],eval_env.time_step_punish_range[1],num=root_episodes)
    all_q_val_lists = []
    b_list = []
    all_reward_lists = []
    max_angle_list = []
    all_action_list = []
    spill_list = []
    glass_list = []
    print("Evaluating")
    """for spill in tqdm(spills):
        for tsp in tsps:
            state, done = eval_env.reset(use_gui=render), False
            eval_env.spill_punish = spill
            eval_env.time_step_punish = tsp
            b = 0
            reward_list = []
            q_val_list = []
            action_list = []
            max_angle = 0
            while not done:
                b+=1
                action = policy.select_action(state)
                action_list.append(action)
                max_angle = max(state[0][0],max_angle)
                q_val_list.append(evalstuff(state,action,policy))
                state, reward, done, _ = eval_env.step(action)
                if render:
                    eval_env.render()
                reward_list.append(reward)
            all_q_val_lists.append(q_val_list)
            all_reward_lists.append(reward_list)
            all_action_list.append(action_list)
            spill_list.append(eval_env.particle_locations["spilled"])
            glass_list.append(eval_env.particle_locations["glass"])
            max_angle_list.append(max_angle)
            b_list.append(b)
    rew_list = [np.sum(x) for x in all_reward_lists]
    all_arrays = np.array([b_list,max_angle_list,spill_list,glass_list,[np.sum(x) for x in all_reward_lists]])
    np.save(os.path.join(path,"all_arrays.npy"),all_arrays)"""
    b_list,max_angle_list,spill_list,glass_list,rew_list = np.load(os.path.join(path,"all_arrays.npy"))

    #plot_2d(tsps,spills,glass_list,"Fill state",os.path.join(path,"fill_state.svg"))
    #plot_2d(tsps,spills,spill_list,"Spilled",os.path.join(path,"spilled.svg"))
    #plot_2d(tsps,spills,max_angle_list,"Max angle",os.path.join(path,"max_angle.svg"))
    #plot_2d(tsps,spills,rew_list,"Total return",os.path.join(path,"total_return.svg"))
    plot_2d(tsps,spills,b_list,"episode_length",os.path.join(path,"episode_length.svg"),sigma=sigma)
    #plot_q_compare(all_reward_lists,all_q_val_lists,args.discount)

def compare2(load1,load2,name1,name2,min_rot1,min_rot2,basepath="plots/test",sigma=5,to_eval="targ"):
    os.makedirs(basepath,exist_ok=True)
    name_map = {"tsp":"Time Step Punish",
                "spill":"Spill Punish",
                "targ":"Target fill level (ml)"}
    with open(load1,"rb") as f:
            all_q_val_lists1, b_list1, all_reward_lists1, max_angle_list1, all_action_list1, ev_list1, spill_list1, glass_list1, avg_reward1 = pickle.load(f)
    with open(load2,"rb") as f:
            all_q_val_lists2, b_list2, all_reward_lists2, max_angle_list2, all_action_list2, ev_list2, spill_list2, glass_list2, avg_reward2 = pickle.load(f)
    print(np.sum(spill_list1),np.sum(spill_list2))
    reward_sum1 = [np.sum(x) for x in all_reward_lists1]
    reward_sum2 = [np.sum(x) for x in all_reward_lists2]
    for i in range(len(max_angle_list1)):
        radians = max_angle_list1[i]*(math.pi-min_rot1)+min_rot1
        degrees = (radians*180)/math.pi
        max_angle_list1[i] = degrees
    for i in range(len(max_angle_list2)):
        radians = max_angle_list2[i]*(math.pi-min_rot2)+min_rot2
        degrees = (radians*180)/math.pi
        max_angle_list2[i] = degrees

    if to_eval=="targ":
        dev_list1 = (np.array(glass_list1)-np.array(ev_list1))
        dev_list2 = (np.array(glass_list2)-np.array(ev_list2))
        smooth_compare(ev_list1,ev_list2,dev_list1,dev_list2,"Target fill level (ml)","Deviation from target fill level (ml)",
                       [name1,name2],os.path.join(basepath,"deviation.svg"),sigma=sigma)
        smooth_compare(ev_list1,ev_list2,np.abs(dev_list1),np.abs(dev_list2),"Target fill level (ml)","Deviation from target fill level (ml)",
                       [name1,name2],os.path.join(basepath,"abs_deviation.svg"),sigma=sigma)
    smooth_compare(ev_list1,ev_list2,b_list1,b_list2,name_map[to_eval],"Episode length (steps)",
                   [name1,name2],os.path.join(basepath,"epi_length.svg"),sigma=sigma)
    smooth_compare(ev_list1,ev_list2,max_angle_list1,max_angle_list2,name_map[to_eval],"Angle (Degrees)",
                   [name1,name2],os.path.join(basepath,"angle.svg"),sigma=sigma)
    smooth_compare(ev_list1,ev_list2,reward_sum1,reward_sum2,name_map[to_eval],"Return",
                   [name1,name2],os.path.join(basepath,"return.svg"),sigma=sigma)
    smooth_compare(ev_list1,ev_list2,spill_list1,spill_list2,name_map[to_eval],"Spilled",
                   [name1,name2],os.path.join(basepath,"spilled.svg"),sigma=sigma)
    smooth_compare(ev_list1,ev_list2,glass_list1,glass_list2,name_map[to_eval],"fill-level (ml)",
                   [name1,name2],os.path.join(basepath,"fill_state.svg"),sigma=sigma)

def rotation_volume_analysis(policy,eval_env,save_path,render=False):
    eval_env.fixed_tsp = True
    eval_env.fixed_spill = True
    eval_env.time_step_punish = 1
    eval_env.spill_punish = 25
    eval_env.fixed_target_fill = True
    targ_fills = [120,150,180,210]
    action_lists = []
    rotation_lists = []
    volumes_lists = []
    for tf in targ_fills:
        eval_env.target_fill_state = tf
        state, done = eval_env.reset(use_gui=render), False
        action_list = []
        rotation_list = []
        volumes_list = []
        while not done:
            action = policy.select_action(state)
            if render:
                eval_env.render()
            volumes_list.append(eval_env.particle_locations["glass"])
            action_list.append(action[0])
            bottle_radians = R.from_matrix(env.bottle.rotation).as_euler("zyx")[0]
            rotation_list.append((bottle_radians/math.pi)*180)
            state, reward, done, _ = eval_env.step(action)
        action_lists.append(action_list)
        rotation_lists.append(rotation_list)
        volumes_lists.append(volumes_list)
    with open(save_path,"wb") as f:
        pickle.dump({"targ_fills":targ_fills,
                     "actions":action_lists,
                     "rotations":rotation_lists,
                     "volumes":volumes_lists},f)
    

def eval_1d(policy, eval_env, seed, basepath="plots/test", eval_episodes=10, to_eval="tsp", N=5, render=False, load=None):
    os.makedirs(basepath,exist_ok=True)
    name_map = {"tsp":"Time Step Punish",
                "spill":"Spill Punish",
                "targ":"Target fill level (ml)"}
    if load is None:
        policy.critic.eval()
        policy.actor.eval()
        eval_env.seed(seed + 100)
        eval_env.fixed_tsp = True
        eval_env.fixed_spill = True
        eval_env.fixed_target_fill = True
        eval_env.target_fill_state = eval_env.max_in_glass
        eval_env.time_step_punish = 1
        eval_env.spill_punish = 25
        all_q_val_lists = []
        b_list = []
        all_reward_lists = []
        max_angle_list = []
        all_action_list = []
        ev_list = []
        spill_list = []
        glass_list = []
        print("Evaluating")
        for i in trange(eval_episodes):
            state, done = eval_env.reset(use_gui=render), False
            if to_eval == "tsp":
                tsp = (eval_env.time_step_punish_range[0]+(eval_env.time_step_punish_range[1] -
                    eval_env.time_step_punish_range[0])/(eval_episodes-1) * i)
                ev_list.append(tsp)
                eval_env.time_step_punish = tsp
            elif to_eval == "spill":
                spill_punish = (eval_env.spill_range[0]+(eval_env.spill_range[1] -
                                eval_env.spill_range[0])/(eval_episodes-1) * i)
                eval_env.spill_punish = spill_punish
                ev_list.append(spill_punish)
            elif to_eval == "targ":
                target_fill = (eval_env.target_fill_range[0]+(eval_env.target_fill_range[1] -
                            eval_env.target_fill_range[0])/(eval_episodes-1) * i)
                eval_env.target_fill_state = target_fill
                print(target_fill)
                ev_list.append(target_fill)
            b = 0
            reward_list = []
            q_val_list = []
            action_list = []
            max_angle = 0
            while not done:
                b+=1
                action = policy.select_action(state)
                action_list.append(action)
                angle = state[0][0] if type(state)==tuple else state[0]
                max_angle = max(angle,max_angle)
                q_val_list.append(evalstuff(state,action,policy))
                state, reward, done, _ = eval_env.step(action)
                if render:
                    eval_env.render()
                reward_list.append(reward)
            all_q_val_lists.append(q_val_list)
            all_reward_lists.append(reward_list)
            all_action_list.append(action_list)
            spill_list.append(eval_env.particle_locations["spilled"])
            glass_list.append(eval_env.particle_locations["glass"])
            max_angle_list.append(max_angle)
            b_list.append(b)
        
        avg_reward = np.mean([np.sum(x) for x in all_reward_lists])
        with open(os.path.join(basepath,"data.pkl"),"wb") as f:
            to_save = [all_q_val_lists, b_list, all_reward_lists,
                       max_angle_list, all_action_list, ev_list, 
                       spill_list, glass_list, avg_reward]
            pickle.dump(to_save,f)
    else:
        with open(os.path.join(basepath,"data.pkl"),"rb") as f:
            all_q_val_lists, b_list, all_reward_lists, max_angle_list, all_action_list, ev_list, spill_list, glass_list, avg_reward = pickle.load(f)


    for i in range(len(max_angle_list)):
        radians = max_angle_list[i]*(math.pi-env.min_rotation)+env.min_rotation
        degrees = (radians*180)/math.pi
        max_angle_list[i] = degrees

    ev_list = np.array(ev_list)
    print(linregress(ev_list[ev_list>=100],np.array(b_list)[ev_list>=100]))
    #print(linregress(ev_list[ev_list>=0],np.array(max_angle_list)[ev_list>=0]))
    if to_eval=="targ":
        dev_list = (np.array(glass_list)-np.array(ev_list))
        #percent_list = (np.array(glass_list)-np.array(ev_list))/np.array(ev_list)
        #percent_list*=100
        #print(linregress(ev_list[ev_list>=340],np.abs(dev_list[ev_list>=340])))
        plot_mean(ev_list,dev_list,name_map[to_eval],"Deviation from target fill level (ml)","Deviation",
                  "Deviation from target fill level",os.path.join(basepath,f"{to_eval}_deviation.svg"),sigma=N)
        plot_mean(ev_list,np.abs(dev_list),name_map[to_eval],"Absolute deviation from target fill level (ml)","Deviation",
                  "Absolute deviation from target fill level",os.path.join(basepath,f"{to_eval}_abs_deviation.svg"),sigma=N)
    plot_mean(ev_list,b_list,name_map[to_eval],"Episode length","Episode length",
              "Episode lengths",os.path.join(basepath,f"{to_eval}_episode_length.svg"),sigma=N)
    plot_mean(ev_list,max_angle_list,name_map[to_eval],"Degrees","Degrees",
              f"Maximum angle of inclination",os.path.join(basepath,f"{to_eval}_angle.svg"),sigma=N)
    reward_sum = [np.sum(x) for x in all_reward_lists]
    plot_mean(ev_list,reward_sum,name_map[to_eval],"Return","total return","total return",
              os.path.join(basepath,f"{to_eval}_return.svg"),sigma=N)
    plot_action(all_action_list,os.path.join(basepath,"action.svg"))
    plot_mean(ev_list,spill_list,name_map[to_eval],"Spilled","num particles spilled",
              "Particles Spilled",os.path.join(basepath,f"{to_eval}_spilled.svg"),sigma=N)
    plot_mean(ev_list,glass_list,name_map[to_eval],"particles in glass","num particles in glass",
              "Final fill state",os.path.join(basepath,f"{to_eval}_fill.svg"),sigma=N)
    plot_q_compare(all_reward_lists,all_q_val_lists,args.discount,os.path.join(basepath,"q_compare.svg"))


    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Avg episode length {np.mean(b_list)}")
    print("---------------------------------------")
    eval_env.fixed_tsp = False
    eval_env.reset(use_gui=False)
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3_particles")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="water_pouring:Pouring-mdp-v0")          # OpenAI gym environment name
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
    parser.add_argument("--norm",type=str, default="layer")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--path",type=str, default="plots/test")
    parser.add_argument("--to_eval",type=str, default="tsp")
    parser.add_argument("--eval_episodes",type=int,default=100)
    parser.add_argument("--running_num",type=int, default=5)
    parser.add_argument("--load",type=str, default="")
    parser.add_argument("--load2",type=str, default="")
    parser.add_argument("--name1",type=str, default="default1")
    parser.add_argument("--name2",type=str, default="default2")
    parser.add_argument("--min_rot1",type=float, default=1.22)
    parser.add_argument("--min_rot2",type=float, default=1.22)
    parser.add_argument("--human_compare",action="store_true")
    parser.add_argument("--jerk_punish",type=float, default=0)
    parser.add_argument("--rot_vol_analysis",action="store_true")
    args = parser.parse_args()

    if args.load2!="":
        compare2(args.load,args.load2,args.name1,args.name2,args.min_rot1,args.min_rot2,basepath=args.path,sigma=args.running_num,to_eval=args.to_eval)
        exit()
    
    env_kwargs = {
        "policy_uncertainty":args.policy_uncertainty,
        "jerk_punish":args.jerk_punish
    }
    if args.human_compare:
        env_kwargs["scene_base"] = "scenes/smaller_scene.json"
    env = gym.make(args.env,**env_kwargs)
    if args.human_compare:
        env.max_in_glass = 215
        env.target_fill_range = [114,209]
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
        "policy_freq": int(args.policy_freq),
        "norm": None if args.norm=="" else args.norm
    }

    if args.load == "":
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
    else:
        policy = None


    if args.rot_vol_analysis:
        rotation_volume_analysis(policy,env,args.path,render=args.render)
    else:
        evaluations = eval_1d(policy, env, args.seed, basepath=args.path, to_eval=args.to_eval, render=args.render, N=args.running_num, eval_episodes=args.eval_episodes, load=None if args.load=="" else args.load)
    #eval_2d(policy,env,args.seed,args.path,render=args.render,sigma=args.running_num)