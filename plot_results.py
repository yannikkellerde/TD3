import numpy as np
import matplotlib.pyplot as plt 
import os,sys

def res_file_plot():
    b = np.load(sys.argv[1])
    plt.plot(np.array(range(len(b)))*100,b)
    plt.show()

def from_folder(folder):
    files = os.listdir(folder)
    files = list(filter(lambda x:"final" not in x and "175" not in x, files))
    files.sort(key=lambda x:int(x.split("ep-")[1].split("_ev")[0]))
    ev = [int(x.split("ev-")[1].split("-q")[0]) for x in files]
    print(files[ev.index(max(ev))])
    q = [int(x.split("q-")[1]) for x in files]
    plt.yscale('symlog')
    plt.plot(ev,label="total return")
    plt.plot(q,label="avg q")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from_folder(sys.argv[1])