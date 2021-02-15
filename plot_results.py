import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt 
import os,sys

def res_file_plot():
    b = np.load(sys.argv[1])
    plt.plot(np.array(range(len(b)))*100,b)
    plt.show()

def from_folder(folder):
    max_timesteps = 2500000
    files = os.listdir(folder)
    files = list(filter(lambda x:"final" not in x and "175" not in x, files))
    files.sort(key=lambda x:int(x.split("ep-")[1].split("_ev")[0]))
    ev = [int(x.split("ev-")[1].split("-q")[0]) for x in files]
    print(files[ev.index(max(ev))])
    q = [int(x.split("q-")[1]) for x in files]
    x = [i*max_timesteps/len(files) for i,x in enumerate(files)]
    ev = gaussian_filter1d(ev,1)
    q = gaussian_filter1d(q,1)
    plt.yscale('symlog')
    plt.plot(ev,label="Policy performance")
    plt.plot(q,label="Average Q value")
    plt.xlabel("Training step")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from_folder(sys.argv[1])