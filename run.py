import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import datetime

from multiprocessing import Pool


'''
Returns the sum of 2 vectors of not necesseraly the same length by appending 0's to the shorter vector.
'''
def sum_two_vec_pad(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

'''
returns the energy for an estimated weight vector omega, with data (X, Y)
'''
def energy(w, X, Y):
    Y_est = np.sign(np.dot(X, w))
    return 0.5 * np.sum((Y - Y_est)**2)

def accept_prob(wp, w, beta, X, Y):
    return min(1, np.exp(-beta*(energy(wp, X, Y) - energy(w, X, Y))))

def accept_prob_with_energy(wp, w, beta, X, Y):
    next_energy = energy(wp, X, Y)
    return (min(1, np.exp(-beta*(next_energy - energy(w, X, Y)))), next_energy)

def accept_prob_with_energy_using_energy(wp, prev_energy, beta, X, Y):
    next_energy = energy(wp, X, Y)
    return (min(1.0, np.exp(-beta*(next_energy - prev_energy))), next_energy)

def overlap(wp, w):
    return 1.0 / (w.shape[0]) * np.dot(w, wp)

def delta_energy_fast(w, wp, idx, X, Y):
    Xw = np.dot(X, w)
    w_tmp = np.zeros(w.shape[0])
    w_tmp[idx] = wp[idx] - w[idx]
    Xwp = Xw + np.dot(X, w_tmp)
    Y_est = np.sign(Xw)
    Y_estp = np.sign(Xwp)
    return 0.5 * np.sum((Y - Y_estp)**2),  0.5*np.sum((Y - Y_est)**2)

def accept_prob_fast(wp, w, beta, X, Y, idx):
    e_next, e = delta_energy_fast(w, wp, idx, X, Y)
    return min(1, np.exp(-beta*(e_next - e))), e_next

import time
import datetime

def get_timestamp():
    ts = time.time() 
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    return timestamp

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def metropolis_fastest(nb_iter, schedule, beta, beta_pace, X, Y, epsilon=0):

    N = X.shape[1]
    w = 2 * np.random.random_integers(0, 1, N) - 1

    current_energy = energy(w, X, Y)

    ctr = 0
    while (current_energy > epsilon and ctr < nb_iter):

        index_rand = np.random.randint(0, N)
        wp = np.copy(w)
        wp[index_rand] = -1 * wp[index_rand]

        accept_probability, next_energy = accept_prob_with_energy_using_energy(wp, current_energy, beta, X, Y)
        if np.random.uniform() < accept_probability:
            # accept the move, update the weights and the current energy
            w = wp
            current_energy = next_energy

        if ctr % schedule == 0:
            beta = beta * beta_pace


        ctr += 1
        #print("At iteration ", ctr)
    return w


############
# Import .mat files

file_name = 'data.mat'
var = scipy.io.loadmat(file_name)

y = var['y']
X = var['X']
M = int(var['M'])
N = int(var['N'])
M_test = int(var['M_test'])
X_test = var['X_test']

print('Dimensions:')
print('N = {}'.format(N) + '; M = {}'.format(M))

print('Running simmultion...')

############



###########

beta_values = [0.2, 0.3, 0.33, 0.36, 0.4, 0.43]
pace_values = [1.0002, 1.001]
schedule_values = [1, 3, 6, 10]
N_values = [780]
nb_runs_values = [1]
nb_iter = 4000

# Parameters
'''
beta = 0.4
beta_pace = 1.002
schedule = 2
'''

problems = map(lambda x,y:(x,y), N_values, nb_runs_values)
problem_list = list(problems)
alpha_values = np.linspace(0.5, 5, 10)
#alpha_values = [20]
#alpha_values = np.append(alpha_values, [7.0, 10.0])
M_values = {}

for n in N_values:
    M_values[n] = 9982

grid_points=[]
for prob in problem_list:
    gp_N = prob[0]
    gp_nb_runs = prob[1]
    gp_M = M_values[gp_N]
    for gp_beta in beta_values:
        for gp_pace in pace_values:
            for gp_schedule in schedule_values:
                parameter = {}
                parameter['N'] = gp_N
                parameter['nb_runs'] = gp_nb_runs
                parameter['beta'] = gp_beta
                parameter['pace'] = gp_pace
                parameter['M'] = gp_M
                parameter['schedule'] = gp_schedule
                grid_points.append(parameter)
#print(grid_points)
#

# Simulation
'''
w_est = metropolis_fastest(nb_iter, schedule, beta, beta_pace, X, y, epsilon=1e-3)
energy_of_est = energy(w_est, X, y)
y_est = np.dot(X, w_est)

print(energy_of_est)

# Store results
name = 'answer_ErgoticPain' + str(get_timestamp) + '.mat'
scipy.io.savemat(name, mdict={'w': w_est, 'E': energy_of_est, 'ytest': y_est})
'''

def run_simul(inp):
    thread_idx = inp[0]
    parameter_dict = inp[1]
    print('Running thread: ', thread_idx)
    
    N = parameter_dict['N']
    beta = parameter_dict['beta']
    beta_pace = parameter_dict['pace']
    M_values = parameter_dict['M']
    schedule = parameter_dict['schedule']

    w_est = metropolis_fastest(nb_iter, schedule, beta, beta_pace, X, y, epsilon=1e-3)
    energy_of_est = energy(w_est, X, y)
    y_est = np.dot(X, w_est)

    print(energy_of_est, " with thread_idx ", thread_idx)

    # Store results
    name = 'answer_ErgoticPain' + "thread_" + str(thread_idx) + "_ " + str(get_timestamp) + '.mat'
    scipy.io.savemat(name, mdict={'w': w_est, 'E': energy_of_est, 'ytest': y_est})

num_cores = 48

pool = Pool(num_cores)
print(len(grid_points))
ans = pool.map_async(run_simul, [(i, grid_pt) for i, grid_pt in enumerate(grid_points)])
pool.close()
pool.join()