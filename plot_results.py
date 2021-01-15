import numpy as np
import matplotlib.pyplot as plt 
import sys

b = np.load(sys.argv[1])
plt.plot(np.array(range(len(b)))*100,b)
plt.show()
