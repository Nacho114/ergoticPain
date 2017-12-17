import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from multiprocessing import Pool

def energy(w, X, Y):
    Y_est = np.sign(np.dot(X, w))
    return 0.5 * np.sum((Y - Y_est)**2)

def accept_prob(wp, w, beta, X, Y):
    return min(1, np.exp(-beta*(energy(wp, X, Y) - energy(w, X, Y))))

def overlap(wp, w):
    return 1.0 / (w.shape[0]) * np.dot(w, wp)

def accept_prob_with_energy_using_energy(wp, prev_energy, beta, X, Y):
    next_energy = energy(wp, X, Y)
    return (min(1.0, np.exp(-beta*(next_energy - prev_energy))), next_energy)


def metropolis_sim_anneal_fastest(w_init, beta_init, beta_pace, X, Y, schedule = 1, epsilon=1e-7, max_iter=1000000):
    M = X.shape[0] # number of samples
    N = w_init.shape[0] # number of dimensions
    w_est = np.copy(w_init) 
    beta = beta_init

    energy_record = np.array([])
    current_energy = energy(w_est, X, Y)
    energy_record = np.append(energy_record, current_energy)
    ctr = 0

    while ((current_energy/M > epsilon) or ctr<max_iter):
        ctr +=1
        index_rand = np.random.randint(0, N)
        wp = np.copy(w_est)
        wp[index_rand] = -1 * wp[index_rand]

        accept_probability, next_energy = accept_prob_with_energy_using_energy(wp, current_energy, beta, X, Y)
        if np.random.uniform() < accept_probability:
            # accept the move, update the weights and the current energy
            w_est = wp 
            current_energy = next_energy
        if(ctr%schedule == 0):
            beta = beta * beta_pace

        energy_record= np.append(energy_record, current_energy)

    return w_est, energy_record, (1.0/M) * current_energy, ctr, beta

# Example - -

# Example for going through the alpha_list
# alpha_list = np.linspace(0.5,5,20)
#for alpha in alpha_list:
#    M = int(round(alpha * N))
'''
N = 40
M =2000
alpha = M / N
beta_init = 0.01
beta_pace = 1.01
#epsilon = 0.01
'''

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def conv_dict_to_str(parameter):
    parameter.pop('M')
    dict_str = str(parameter)
    dict_str = dict_str.replace(' ', '_')
    dict_str = dict_str.replace('\'', '')
    dict_str = dict_str.replace('{', '')
    dict_str = dict_str.replace('}', '')
    return dict_str

def run_metropolis_mult(parameter_dict):
    N = parameter_dict['N']
    nb_runs = parameter_dict['nb_runs']
    beta = parameter_dict['beta']
    pace = parameter_dict['pace']
    M_values = parameter_dict['M']
    schedule = parameter_dict['schedule']
    
    normalized_energies_per_alpha = np.array([])
    overlap_per_alpha = np.array([])
    avg_beta_last_per_alpha = np.array([])
    avg_iter_done_per_alpha = np.array([])
    
    alpha_list = M_values/float(N)
    root_folder = '/home/ssingh/markov_chain/ergoticPain/grid_search/'
    mkdir(root_folder)

    foldername = root_folder + conv_dict_to_str(parameter_dict) + '/'
    mkdir(foldername)
    print("bitch1")
    for M in M_values:
        print("bitch2", M)
        overlap_acc = np.zeros(nb_runs)
        normalized_energy_record_acc = np.zeros(nb_runs)
        iter_done_acc = np.zeros(nb_runs)
        beta_last_acc = np.zeros(nb_runs)
        energy_record_acc = np.zeros(N)
        for i in range(nb_runs):

            w = 2 * np.random.random_integers(0, 1, N) - 1
            X = np.random.randn(M, N)
            Y = np.sign(np.dot(X, w))

            w_init = 2 * np.random.random_integers(0, 1, N) - 1
            # the energy record is probably just the very last value
            # we should also save the averaged curves and then the averaged last value
            w_est, energy_record, normalized_energy_last, iter_done, beta_last = metropolis_sim_anneal_fastest(w_init, beta, pace, X, Y, schedule)
            
            energy_record_acc = sum_two_vec_pad(energy_record_acc, energy_record)

            normalized_energy_record_acc[i] = normalized_energy_last
            overlap_acc[i] = overlap(w_est, w)
            iter_done_acc[i] = iter_done
            beta_last_acc[i] = beta_last

        
        overlap_per_alpha = np.append(overlap_per_alpha, np.mean(overlap_acc))
        avg_beta_last_per_alpha = np.append(avg_beta_last_per_alpha, np.mean(beta_last_acc))
        avg_iter_done_per_alpha = np.append(avg_iter_done_per_alpha, np.mean(iter_done_acc))
        normalized_energies_per_alpha = np.append(normalized_energies_per_alpha, np.mean(normalized_energy_record_acc))
        
        #save the plot for accumulated energy record across all nb_runs
        plt.plot(energy_record_acc)
        plt.savefig(foldername + 'energy_record_acc_' + M + '.png')
        plt.close()
        np.save(foldername + 'energy_record_acc_' + M, energy_record_acc)
    

    #save the plot for each of the alpha's  
    plt.plot(alpha_list, normalized_energies_per_alpha)
    plt.savefig(foldername + 'normalized_energies_per_alpha.png')
    plt.close()
    plt.plot(alpha_list, overlap_per_alpha)
    plt.savefig(foldername + 'overlap_per_alpha.png')
    plt.close()

    # serialize the various values produced! 
    np.save(foldername + 'normalized_energies_per_alpha', normalized_energies_per_alpha)
    np.save(foldername + 'overlap_per_alpha', overlap_per_alpha)
    np.save(foldername + 'avg_beta_last_per_alpha', avg_beta_last_per_alpha)
    np.save(foldername + 'avg_iter_done_per_alpha', avg_iter_done_per_alpha)

    return normalized_energies_per_alpha, overlap_per_alpha, avg_beta_last_per_alpha, avg_iter_done_per_alpha


num_cores = 48
beta_values = [0.1, 0.3, 0.4, 0.7, 0.9]
pace_values = [1.0002, 1.001, 1.002]
schedule_values = [1, 10]
N_values = [40, 60, 75, 100]
nb_runs_values = [50, 40, 30, 20]

problems = map(lambda x,y:(x,y), N_values, nb_runs_values)
problem_list = list(problems)
alpha_values = np.linspace(0.5, 5, 10)
alpha_values = np.append(alpha_values, [7.0, 10.0])
M_values = {}
for n in N_values:
    M_values[n] = np.asarray(n*alpha_values, dtype=int)

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

pool = Pool(num_cores)
ans = pool.map_async(run_metropolis_mult, grid_points[:2])
ans.wait()
