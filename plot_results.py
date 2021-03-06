import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt 
import os,sys

def res_file_plot():
    b = np.load(sys.argv[1])
    plt.plot(np.array(range(len(b)))*100,b)
    plt.show()

def from_folder(folder,policy_performance=True,average_q=True,label="Policy Performance",show=False,color=None):
    max_timesteps = 1e6
    files = os.listdir(folder)
    files = list(filter(lambda x:"final" not in x and "175" not in x, files))
    files.sort(key=lambda x:int(x.split("ep-")[1].split("_")[0]))
    ev = [int(x.split("ev-")[1].split("_")[0]) for x in files]
    print(np.mean(ev))
    print(files[ev.index(max(ev))])
    q = [int(x.split("q-")[1]) for x in files]
    t = [int(x.split("_t-")[1].split("_")[0]) for x in files]
    #x = [i*max_timesteps/len(files) for i,x in enumerate(files)]
    #ev = gaussian_filter1d(ev,1)
    #q = gaussian_filter1d(q,1)
    #plt.yscale('symlog')
    if policy_performance:
        plt.plot(t,ev,label=label,color=color)
        plt.ylabel("Total return")
    else:
        plt.ylabel("Q-value")
    if average_q:
        plt.plot(t,q,label=label,color=color)
    plt.xlabel("Training step")
    plt.xlim(0,max_timesteps)
    if show:
        plt.legend()
        plt.show()

def compare_folders(f1,f2,path):
    os.makedirs(path,exist_ok=True)
    fig = plt.figure(figsize=(10,4))
    from_folder(f1,True,False,color="darkgreen")
    from_folder(f2,True,False,color="#6600cc")
    plt.savefig(os.path.join(path,"performance.svg"))
    plt.cla()
    fig = plt.figure(figsize=(10,4))
    from_folder(f1,False,True,"Particles",color="darkgreen")
    from_folder(f2,False,True,"Features",color="#6600cc")
    plt.legend()
    plt.savefig(os.path.join(path,"q-value.svg"))

if __name__ == "__main__":
    from_folder(sys.argv[1],show=True)
    #compare_folders(*sys.argv[1:])
