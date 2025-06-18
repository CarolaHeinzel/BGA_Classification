# Attention: here we only have haploid individuals, we have to extend it to diploid individual!!!

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
# Simulation according to the Linkage Model as mentioned in Falush (2003)
#%%

# Transition Matrices between the Z
def create_transition_matrix(K, q, r, d_t):
    """
    Create a transition matrix based on the initial probability vector and parameter r over time.

    Parameters:
    K (int): Number of Populations.
    q (numpy array): Ancestry of the individuals.
    r (float): Parameter influencing transition probabilities, interpretation: number of generations in the past at which the admixture happend.
    d_t (float): Distance between the markers.
    
    Returns:
    numpy array: A K x K transition matrix.
    """
    Q = np.zeros((K, K))

    # Define diagonal entries
    for k in range(K):
        Q[k, k] = sp.exp(-d_t * r) + (1 - sp.exp(-d_t * r)) * q[k]

    # Define non-diagonal entries
    for i in range(K):
        for j in range(K):
            if i != j:
                Q[i, j] = q[j] *(1 - sp.exp(-d_t * r)) 
    return Q

#d_t = 0
#Q = create_transition_matrix(K, q, r, d_t)
#print(Q)

def step(current_state1, K, q, r, d_t, p):
    """
    Perform one step of the Markov chain and update the current state based on the time-dependent transition matrix.

    Parameters:
    current_state (int): Current state of the Markov chain, i.e. population \in 0,..., K-1.
    K (int): Number of states i.e. number of populations.
    q (numpy array): as above.
    r (float): as above.
    d_t (float): as above.

    Returns:
    int: New current state after transition.
    """
    transition_matrix = create_transition_matrix(K, q, r, d_t)
    # Hidden State
    step1 = np.random.choice(range(K), p=transition_matrix[current_state1])
    # success probability
    theta1 = q[step1]*p[step1]

    x1 = np.random.binomial(1, theta1)
    return transition_matrix, step1, x1

def simulate_markov_chain(K, q, r, steps, d_values, p):
    """
    Simulate the Markov chain for a given steps time points.

    Parameters:
    K (int): as above
    q (numpy array): as above
    r (float): as above
    steps (int): The number of markers that are to simulate.
    d_values (list): as above
    p (numpy array): allele frequencies
    
    Returns:
    numpy array: Array of states visited.
    """
    pi = q 
    step1 =  np.random.choice(range(K),p = pi)  
    x0 = np.random.binomial(1,p[0,step1] * pi[step1])
    states = [int(x0)]
    for t in range(1, steps):
        Q, step1, x1 = step(step1, K, q, r, d_values[t], p[t])
        #print(current_state)
        states.append(x1)
    
    return states, Q

# Interpretation: allele frequencies
def create_random_matrix(M, k):
    """
    Create a matrix with M rows and k entries, filled with random values between 0 and 1.

    Parameters:
    M (int): Number of rows.
    k (int): Number of entries in each row.

    Returns:
    numpy.ndarray: A matrix with shape (M, k), filled with random values.
    """
    # Create a matrix filled with random values between 0 and 1
    matrix = np.random.rand(M, k)
    return matrix

# Parameters
M = 50  # Number of markers
K = 3  # Number of populations
# Create the true allele frequencies
p = create_random_matrix(M, K)
# Parameters
q = np.array([0.1, 0.1, 0.8])  # Ancestries
r = 1  
d_values = np.linspace(0.1, 1.0, M)  


states_visited, Q = simulate_markov_chain(K, q, r, M, d_values, p)

print("Simulated genetic data", states_visited)

plt.figure(figsize=(12, 6))
plt.plot(states_visited, marker='o')
plt.xlabel('Markers')
plt.ylabel('X')
plt.xticks(range(M))
plt.yticks(range(2))
plt.show()
