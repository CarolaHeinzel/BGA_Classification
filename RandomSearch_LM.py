from scipy.stats import dirichlet, binom, norm
import numpy as np
import sys
import os
import math
import itertools

#  Linkage Model: MLE and bootstrap
script_dir = os.path.dirname(os.path.abspath(__file__)) 
module_path = os.path.join(script_dir) 
sys.path.append(module_path)

import Simulation_Linkage_diploid as sim
#%%
# Structure:
# 1) Simulate the data 
M = 50 # Number of rows
K = 3  # Number of states
# Create the true allele frequencies
p = sim.create_random_matrix(M, K)
print(p)
# Parameters
q = np.array([0.4, 0.1, 0.5])  # Initial probabilities for the first K-1 states
r = 0.5  # Parameter influencing transition probabilities
d_values = [0.1]*M  # Example time-dependent values for d_t


def compute_likelihood_single_chromosome(X, q, p, d, r):
    M = len(p[0])        # number of markers
    K = len(q)        # number of hidden states
    #print("p", p)
    # Initialize forward probabilities alpha
    alpha = np.zeros((M, K))
    # Initial step: alpha_1(k) = P(Z_1 = k) * P(X_1 | Z_1 = k)
    for k in range(K):
     #   print(p[k][0])
        emission_prob = q[k] * p[k][0] if X[0] == 1 else 1 - q[k] * p[k][0]
        alpha[0, k] = q[k] * emission_prob
    # Recursive step
    for m in range(1, M):
        for k in range(K):
            sum_prob = 0.0
            for k_prev in range(K):
                if k == k_prev:
                    trans_prob = np.exp(-d[m-1] * r) + (1 - np.exp(-d[m-1] * r)) * q[k_prev]
                else:
                    trans_prob = (1 - np.exp(-d[m-1] * r)) * q[k_prev]
                sum_prob += alpha[m - 1, k_prev] * trans_prob
            emission_prob = q[k] * p[k][m] if X[m] == 1 else 1 - q[k] * p[k][m]
            alpha[m, k] = sum_prob * emission_prob
    # Total likelihood = sum over final alphas
    return np.sum(alpha[-1])


def compute_likelihood_multiple_chromosome_diploid(X1, X2, q, p, d, r):

    l1 =  compute_likelihood_single_chromosome(X1, q, np.array(p).T, d, r)
    l2 =  compute_likelihood_single_chromosome(X2, q, np.array(p).T, d, r)
    res = l1 * l2
    return res

def random_search(X1, X2, p, d_t, K, n_iter=1000):

    best_log_likelihood = -np.inf
    best_q = None
    best_r = None
    
    for _ in range(n_iter):
        q = np.random.dirichlet(np.ones(K))
        r = np.random.uniform(0,6)

        current_ll = compute_likelihood_multiple_chromosome_diploid(
            X1, X2, q, p, d_t, r
        )
        if current_ll > best_log_likelihood:
            best_log_likelihood = current_ll
            best_q = q.copy()
            best_r = r
    
    return best_q, best_r, best_log_likelihood

def simulation(numRep, K, q, r, M, d_values, p):
    res = []
    res_r = []
    for i in range(numRep):
        print("i", i)
        x1, x2 = sim.simulate_markov_chain_diploid(K, q, r, M, d_values, p)
        est = random_search(x1, x2, p, d_values, K, n_iter=1000)
        res.append(est[0])
        res_r.append(est[1])
    return res, res_r
    
# Fix q and p and calculate different values for  x


# 2) Calculate the boostrap variance of the data


var_bootstrap = simulation(100, K, q, r, M, d_values, p)
