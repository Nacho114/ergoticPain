import os
import numpy as np
import matplotlib.pyplot as plt

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

root_folder_name = os.getcwd() + '/plot_images/'
mkdir(root_folder_name)

#22/16/19 init_beta = 0.7
#21/46/54 init_beta = 0.4
#21/21/19 init_beta = 0.3
#21/09/21 init_beta = 0.1
rootdir1 = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_final/grid_search_2017-12-17_22:16:19'
rootdir2 = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_final/grid_search_2017-12-17_21:46:54'
rootdir3 = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_final/grid_search_2017-12-17_21:21:19'
rootdir4 = '/Users/aleman/gdrive/epfl/master/ma1/markovChains/ergoticPain/grid_search_final/grid_search_2017-12-17_21:09:21'

N = 100
roots = [rootdir1, rootdir2, rootdir3, rootdir4]
joint_npy_files = []

for root in roots:
    npy_files = []
    for subdir, dirs, files in os.walk(root):
        for file in sorted(files):
            curr_file = os.path.join(subdir, file)
            if '.npy' in curr_file and 'avg_' not in curr_file:
                energy_record = np.load(curr_file)
                file_name = os.path.basename(curr_file)
                npy_files.append((file_name, energy_record))
    joint_npy_files.append(npy_files)

labels_data = ['0.7', '0.4', '0.3', '0.1']
colors = ['b', 'y', 'g', 'r']

height = 5
width = 2
figs, axes = plt.subplots(nrows=height, ncols=width, figsize=(24,18));

for i in range(len(joint_npy_files[0]) - 2):
    row = int(i / width)
    col = i % width
    for idx in range(len(roots)):
        axes[row, col].plot(joint_npy_files[idx][i][1], colors[idx])
        axes[row, col].legend(labels=labels_data, title="Initial beta");
        file_name  = joint_npy_files[idx][i][0].rsplit( ".", 1 )[ 0 ]
        title = file_name.replace("_", " ")
        title_prefix = 'N = {}, '.format(N)
        if hasNumbers(title):
            value = [int(s) for s in title.split() if s.isdigit()][0]
            value = round(float(value) / N, 1)
            axes[row, col].set_title('Plot {}.  '.format(i + 1) + title_prefix + 'alpha = {}'.format(value))
            axes[row, col].set_xlabel("iterations");
            axes[row, col].set_ylabel("energy");
        else:
            axes[row, col].set_title(title_prefix + title)
            axes[row, col].set_xlabel("alpha");

figs.tight_layout()
plt.savefig(root_folder_name + "N={}.png".format(N), bbox_inches="tight")
