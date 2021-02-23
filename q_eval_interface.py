import torch
from TD3_particles import TD3
class q_eval_interface():
    def __init__(self,env,model_path,norm):
        kwargs = {
            "obs_space": env.observation_space,
            "action_space": env.action_space,
            "discount": 0.999,
            "tau": 0.005,
            "policy_freq": 2,
            "norm": norm
        }
        self.policy = TD3(**kwargs)
        self.policy.load(model_path)
    def eval_q(self,state,action):
        q_val = self.policy.eval_q(state,action)
        return (q_val[0][0]+q_val[1][0])/2