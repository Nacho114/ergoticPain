import numpy as np
import matplotlib.pyplot as plt

def energy(w, X, Y):
    Y_est = np.sign(np.dot(X, w))
    return 0.5 * np.sum((Y - Y_est)**2)

def accept_prob(wp, w, beta, X, Y):
    return min(1, np.exp(-beta*(energy(wp, X, Y) - energy(w, X, Y))))

def overlap(wp, w):
    return 1.0 / (w.shape[0]) * np.dot(w, wp)

def metropolis_sim_anneal(w_init, beta_init, beta_pace, X, Y, epsilon=0):
    M = X.shape[0]
    N = w_init.shape[0]
    w_est = np.copy(w_init)
    wp = np.copy(w_est)
    beta = beta_init
    energy_record = np.array([])
    energy_record = np.append(energy_record, energy(w_init, X, Y))
    print(energy(w_est, X, Y))
    while (energy(w_est, X, Y) / M > epsilon):

        index_rand = np.random.randint(0, N)
        wp = np.copy(w_est)
        wp[index_rand] = -1 * wp[index_rand]

        if np.random.uniform() < accept_prob(wp, w_est, beta, X, Y):
            w_est = np.copy(wp)

        beta = beta * beta_pace
        energy_record= np.append(energy_record, energy(w_est, X, Y))

    return w_est, energy_record


# Example - -

# Example for going through the alpha_list
# alpha_list = np.linspace(0.5,5,20)
#for alpha in alpha_list:
#    M = int(round(alpha * N))

N = 40
M =2000
alpha = M / N
beta_init = 0.01
beta_pace = 1.01
epsilon = 0.01

w = 2 * np.random.random_integers(0, 1, N) - 1
X = np.random.randn(M, N)
Y = np.sign(np.dot(X, w))
w_init = 2 * np.random.random_integers(0, 1, N) - 1


w_est, energy_record = metropolis_sim_anneal(w_init, beta_init, beta_pace, X, Y, epsilon)

plt.plot(energy_record)
plt.show()
