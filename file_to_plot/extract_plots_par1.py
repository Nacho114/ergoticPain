import os
import numpy as np
import matplotlib.pyplot as plt


rootdir = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_part1_2017-12-18_20:05:42'

npy_files = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        curr_file = os.path.join(subdir, file)
        if 'energy_record' in curr_file and '.npy' in curr_file:
            energy_record = np.load(curr_file)
            npy_files.append(energy_record)

beta_values = [0.6, 0.8, 1, 1.2]
N_values = 40
nb_runs_values = 20


nb_folders = float(len(npy_files)) / len(beta_values)

#for n in range(nb_folders):
for idx, beta in enumerate(beta_values):
    plt.plot(npy_files[idx*4])


plt.legend(labels=beta_values, title="beta");
plt.xlabel("iterations");
plt.ylabel("energy");
plt.title("Energy at {}th Step for Different beta".format(1))
plt.show()
