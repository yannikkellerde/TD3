import matplotlib.pyplot as plt
import numpy as np

class OrnsteinUhlenbeckActionNoise:
    # Source for this class: https://github.com/navneet-nmk/pytorch-rl/blob/8329234822dcb977931c72db691eabf5b635788c/Utils/random_process.py#L63

    def __init__(self, action_dim, mu = 0, theta = 0.1, sigma = 0.1):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


if __name__ == '__main__':
    n = OrnsteinUhlenbeckActionNoise(1)
    b = []
    for i in range(100):
        b.append(n.sample())
    plt.plot(b)
    plt.show()