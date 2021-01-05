import numpy as np
import matplotlib.pyplot as plt 

b = np.load("results/TD3_water_pouring:Pouring-mdp-full-v0_0.npy")[4:]
plt.plot(np.array(range(len(b)))*1000,b)
plt.show()
