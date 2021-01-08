import numpy as np
import torch
import pickle
import os


class ReplayBuffer(object):
    def __init__(self, obs_space, action_space, max_size=int(2e5), load_folder=None):
        self.max_size = max_size
        self.store_np = ["state_features","state_particles","action",
                         "next_state_features","next_state_particles","reward",
                         "not_done"]
        self.store_pkl = ["ptr","size"]
        if load_folder is None:
            self.ptr = 0
            self.size = 0
            self.state_features = np.zeros((max_size,obs_space[0].shape[0]))
            self.state_particles = np.zeros((max_size, *obs_space[1].shape))
            self.action = np.zeros((max_size, action_space.shape[0]))
            self.next_state_features = np.zeros((max_size,obs_space[0].shape[0]))
            self.next_state_particles = np.zeros((max_size, *obs_space[1].shape))
            self.reward = np.zeros((max_size, 1))
            self.not_done = np.zeros((max_size, 1))
        else:
            self.load(load_folder)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self,folder):
        for attrib in self.store_pkl:
            with open(os.path.join(folder,attrib+".pkl"), "wb") as f:
                pickle.dump(self.__dict__[attrib],f,protocol=4)

        for attrib in self.store_np:
            with open(os.path.join(folder,attrib+".pkl"), "wb") as f:
                np.save(f,self.__dict__[attrib])
    
    def load(self,folder):
        for attrib in self.store_pkl:
            with open(os.path.join(folder,attrib+".pkl"), "rb") as f:
                self.__dict__[attrib] = pickle.load(f)
        for attrib in self.store_np:
            with open(os.path.join(folder,attrib+".pkl"), "rb") as f:
                self.__dict__[attrib] = np.load(f)

    def add(self, state, action, next_state, reward, done):
        self.state_features[self.ptr] = state[0]
        self.state_particles[self.ptr] = state[1]
        self.action[self.ptr] = action
        self.next_state_features[self.ptr] = next_state[0]
        self.next_state_particles[self.ptr] = next_state[1]
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state_features[ind]).to(self.device),
            torch.FloatTensor(self.state_particles[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state_features[ind]).to(self.device),
            torch.FloatTensor(self.next_state_particles[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

if __name__ == "__main__":
    env = gym.make("water_pouring:Pouring-mdp-full-v0")
    r = ReplayBuffer(env.observation_space, env.action_space)
    r.save("test.pkl")