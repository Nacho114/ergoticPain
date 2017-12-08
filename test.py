import numpy as np
import matplotlib.pyplot as plt

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

def overlap(wp, w):
    return 1.0 / (w.shape[0]) * np.dot(w, wp)

###

### Part 1

###


################ 1.

def metropolis(w_init, beta, X, Y, epsilon=0):

    N = w_init.shape[0]
    w = np.copy(w_init)
    wp = np.copy(w)

    energy_record = np.array([])
    energy_record = np.append(energy_record, energy(w_init, X, Y))

    while (energy(w, X, Y) > epsilon):

        index_rand = np.random.randint(0, N)
        wp = np.copy(w)
        wp[index_rand] = -1 * wp[index_rand]

        if np.random.uniform() < accept_prob(wp, w, beta, X, Y):
            w = np.copy(wp)

        energy_record= np.append(energy_record, energy(w, X, Y))

    return w, energy_record

def metropolis_mult(nb_runs, beta, X, Y, epsilon=0):
    N = X.shape[1]
    energy_record_acc = np.zeros(N)

    for k in range(0, nb_runs):
        w_init =  2 * np.random.random_integers(0, 1, N) - 1
        _, energy_record = metropolis(w_init, beta, X, Y)
        energy_record_acc = sum_two_vec_pad(energy_record_acc, energy_record)


    return energy_record_acc / nb_runs



################

################ 2.

def metropolis2(w_init, beta, X, Y, T=10):

    N = w_init.shape[0]
    M = X.shape[0]
    w = np.copy(w_init)
    wp = np.copy(w)

    for _ in range(0, T):

        index_rand = np.random.randint(0, N)
        wp = np.copy(w)
        wp[index_rand] = -1 * wp[index_rand]

        if np.random.uniform() < accept_prob(wp, w, beta, X, Y):
            w = np.copy(wp)

    energy_record = energy(w, X, Y)

    return (1.0/M) * energy_record

def metropolis_mult2(nb_runs, N, alpha_list, beta, T):

    normalized_energies_per_alpha = np.array([])

    for alpha in alpha_list:

        M = int(round(alpha * N))

        energy_record_acc = np.zeros(nb_runs)
        for i in range(nb_runs):

            w = 2 * np.random.random_integers(0, 1, N) - 1
            X = np.random.randn(M, N)
            Y = np.sign(np.dot(X, w))

            w_init = 2 * np.random.random_integers(0, 1, N) - 1

            energy_record_acc[i] = metropolis2(w_init, beta, X, Y, T)

        normalized_energies_per_alpha = np.append(normalized_energies_per_alpha, np.mean(energy_record_acc))

    return normalized_energies_per_alpha



################

# Tests for part 1.

# alpha and beta in [0.5, 5], for 'interesting' results as per paper.
N = 20
M = 100
alpha = M / N
beta = 1

w = 2 * np.random.random_integers(0, 1, N) - 1

X = np.random.randn(M, N)
Y = np.sign(np.dot(X, w))

w_init =  2 * np.random.random_integers(0, 1, N) - 1

#energy_record = metropolis_mult(100, beta, X, Y)


# Tests for part 2.

alpha_list = np.linspace(0.5,5,10)
T = 1000
nb_runs = 50
normalized_energy_record = metropolis_mult2(nb_runs, N, alpha_list, beta, T)



print(normalized_energy_record)

#plt.plot(alpha_list, normalized_energy_record)
#plt.show()



###

### Part 2.

###
