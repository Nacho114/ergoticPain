import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from auxiliary_fun import *

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

    return w


############
# Import .mat files

file_name = 'randomData.mat'
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

beta = 0.7
beta_pace = 1
nb_iter = 10000
schedule = 10

# Simulation

w_est = metropolis_fastest(nb_iter, schedule, beta, beta_pace, X, Y, epsilon=0)
energy_of_est = energy(w_est, X, Y)
y_est = np.dot(X, w_est)

print(energy_of_est)

# Store results
scipy.io.savemat('answer_ErgoticPain.mat', mdict={'w': w_est, 'E': energy_of_est, 'ytest': y_est})
