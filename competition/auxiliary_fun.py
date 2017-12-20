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
