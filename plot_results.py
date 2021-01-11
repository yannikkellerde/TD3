import numpy as np
import matplotlib.pyplot as plt 

b = np.load("results/TD3_water_pouring:Pouring-mdp-v0_100.npy")[4:]
plt.plot(np.array(range(len(b)))*100,b)
plt.show()
