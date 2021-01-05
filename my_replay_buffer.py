import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, obs_space, action_space, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state_features = np.zeros((max_size,obs_space[0].shape[0]))
        self.state_particles = np.zeros((max_size, *obs_space[1].shape))
        self.action = np.zeros((max_size, action_space.shape[0]))
        self.next_state_features = np.zeros((max_size,obs_space[0].shape[0]))
        self.next_state_particles = np.zeros((max_size, *obs_space[1].shape))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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