import numpy as np


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
        print("At iteration ", ctr)
    return w


############
# Import .mat files

file_name = '../data.mat'
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

# Parameters

beta = 0.4
beta_pace = 1.002
nb_iter = 10000
schedule = 2

# Simulation

w_est = metropolis_fastest(nb_iter, schedule, beta, beta_pace, X, y, epsilon=1e-3)
energy_of_est = energy(w_est, X, y)
y_est = np.dot(X, w_est)

print(energy_of_est)

# Store results
name = 'answer_ErgoticPain' + str(get_timestamp) + '.mat'
scipy.io.savemat(name, mdict={'w': w_est, 'E': energy_of_est, 'ytest': y_est})

