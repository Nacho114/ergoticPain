import os
import numpy as np
import matplotlib.pyplot as plt

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

root_folder_name = os.getcwd() + '/combined_grid_search_Schedule_10vs1/'
mkdir(root_folder_name)

# Schedule 10

#21:09:21 (schedule = 10) and 21:05:52 (schedule = 1)]


rootdir1 = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_final/grid_search_2017-12-17_21:09:21'
rootdir2 = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_final/grid_search_2017-12-17_21:05:52'

N = 100
roots = [rootdir1, rootdir2]
joint_npy_files = []

for root in roots:
    npy_files = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            curr_file = os.path.join(subdir, file)
            if '.npy' in curr_file:
                energy_record = np.load(curr_file)
                file_name = os.path.basename(curr_file)
                npy_files.append((file_name, energy_record))
    joint_npy_files.append(npy_files)

labels_data = ['10', '1']
colors = ['b', 'r']

for i in range(len(joint_npy_files[0])):
    for idx in range(len(roots)):
        plt.plot(joint_npy_files[idx][i][1], colors[idx])
        plt.legend(labels=labels_data, title="Schedule");
        file_name  = joint_npy_files[idx][i][0].rsplit( ".", 1 )[ 0 ]
        title = file_name.replace("_", " ")
        title_prefix = 'N = {}, '.format(N)
        if hasNumbers(title):
            value = [int(s) for s in title.split() if s.isdigit()][0]
            value = round(float(value) / N, 1)
            plt.title(title_prefix + 'alpha = {}'.format(value))
            plt.xlabel("iterations");
            plt.ylabel("energy");
        else:
            plt.title(title_prefix + title)
            plt.xlabel("alpha");


    plt.savefig(root_folder_name + file_name)
    plt.close()
