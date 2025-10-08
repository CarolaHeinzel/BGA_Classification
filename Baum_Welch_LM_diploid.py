import numpy as np
from   scipy.special import logsumexp
from   scipy.stats import multivariate_normal as mvn
from   scipy.optimize import minimize
import Simulation_Linkage_diploid as sim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import math


"""Code from https://gregorygundersen.com/blog/2020/11/28/hmms/#a1-expected-complete-log-likelihood
   adapted to the linkage model"""


def compute_trans_prob_log(log_q, r, d_values):
    """Note that trans_prob_log[l] is the transitionmatrix for the haploid case,
    so to calculate the actual transition probabilities, we have to take the product:
    log p(Z_{l+1}=(k_1, k_2)| Z_l=(j_1, j_2))= trans_prob_log[l][j_1, k_1] + trans_prob_log[l][j_2, k_2]"""
    K = len(log_q)

    trans_prob_log = []
    for l in range(len(d_values)):

        d_l = d_values[l]
        log_mat = np.zeros((K, K))

        # Define diagonal entries
        for k in range(K):
            log_mat[k, k] = logsumexp([-d_l*r, np.log(-math.expm1(-d_l*r))+log_q[k]])

        # Define non-diagonal entries
        for i in range(K):
            for j in range(K):
                if i != j:
                    log_mat[i, j] = log_q[j] +np.log(-math.expm1(-d_l*r))

        assert np.allclose(np.sum(np.exp(log_mat), axis=1), 1)
        trans_prob_log += [log_mat]

    #trans_prob_log = [np.log(sim.create_transition_matrix(K, q, r, d_l)) for d_l in d_values]
    #for l in range(len(d_values)):
        #assert np.allclose(np.sum(np.exp(trans_prob_log[l]), axis=1), 1)
    return trans_prob_log


def compute_log_emm_prob(X, log_q, p):
    N, D = X.shape
    K = len(log_q)
    #emm_prob = np.zeros((N, K, K))
    log_emm_prob = np.zeros((N, K, K))
    for l in range(N):
        for k_1 in range(K):
            for k_2 in range(K):
                #prob_1 = q[k_1] * p[l, k_1] if X[l][0] == 1 else 1 - q[k_1] * p[l, k_1]
                #prob_2 = q[k_2] * p[l, k_2] if X[l][1] == 1 else 1 - q[k_2] * p[l, k_2]
                #emm_prob[l, k_1, k_2] = prob_1 * prob_2
                log_prob_1 = log_q[k_1] + np.log(p[l, k_1]) if X[l][0] == 1 else np.log(-math.expm1(log_q[k_1] + np.log(p[l, k_1])))
                log_prob_2 = log_q[k_2] + np.log(p[l, k_2]) if X[l][1] == 1 else np.log(-math.expm1(log_q[k_2] + np.log(p[l, k_2])))
                log_emm_prob[l, k_1, k_2] = log_prob_1 + log_prob_2
    #log_emm_prob = np.log(emm_prob)
    return log_emm_prob


def Q(X, log_q, r, d_values, p, gamma, log_xi):
    """Function that is maximized in (r, q) in the M-step (Equation (17)) """
    N = X.shape[0]
    K = p.shape[1]

    log_emm_prob = compute_log_emm_prob(X, log_q, p)
    trans_prob_log = compute_trans_prob_log(log_q, r, d_values)

    xi = np.exp(log_xi)

    tmp = 0.0
    for k_1 in range(K):
        for k_2 in range(K):
            # initial state distributions
            tmp += gamma[0, k_1, k_2]*(log_q[k_1] + log_q[k_2])
            for l in range(N):
                # emission probabilities
                tmp += gamma[l, k_1, k_2]*log_emm_prob[l, k_1, k_2]
                for j_1 in range(K):
                    for j_2 in range(K):
                        if l < N-1:
                            # transition probabilities
                            tmp += xi[l, k_1, k_2, j_1, j_2] * (trans_prob_log[l][k_1, j_1] + trans_prob_log[l][k_2, j_2])
    return tmp


def baum_welch(X, K, n_iters, p, d_values, delta_q, delta_r):
    """EM for hidden Markov models, i.e. the Baum–Welch algorithm. Numerical
    instability handled by working in log space.
    """
    N = X.shape[0]

    # Initialize parameters \theta:
    #
    q = np.ones(K) / K
    assert np.isclose(q.sum(), 1)
    log_q = np.log(q)
    assert np.isclose(np.sum(np.exp(log_q)), 1)
    r = 1

    print("Start parameters: ", "q: ", q, "r: ", r)


    # The n-th row is log(alpha(z_n)).
    # The k-th column is value z_n takes.
    # So (nk)-th cell is alpha(z_n = k).
    log_alpha = np.zeros((N, K, K))
    log_beta  = np.zeros((N, K, K))

    for _ in range(n_iters):

        print("Iteration:" ,_+1)

        assert np.isclose(q.sum(), 1)

        # transition probabilities (list of (K × K)-matrices)
        trans_prob_log = compute_trans_prob_log(log_q, r, d_values)

        # Initialize emission probabilities (size N × K × K).
        log_emm_prob = compute_log_emm_prob(X, log_q, p)
        #print(log_emm_prob)


        # E-step (forward-backward algorithm).
        # ------------------------------------

        # Initialization Forward Pass (Equation (34))
        for k_1 in range(K):
            for k_2 in range(K):
                # \alpha(z_1)=p(z_1)p(x_1|z_1)
                log_alpha[0, k_1, k_2] = log_q[k_1] + log_q[k_2] + log_emm_prob[0, k_1, k_2]

        # Forward Pass (Equation (32))
        for l in range(1, N):
            for k_1 in range(K):
                for k_2 in range(K):
                    tmp = np.empty((K, K))
                    for j_1 in range(K):
                        for j_2 in range(K):
                            tmp[j_1, j_2] = log_alpha[l-1, j_1, j_2] + trans_prob_log[l-1][j_1, k_1] + trans_prob_log[l-1][j_2, k_2]
                    log_alpha[l, k_1, k_2] = logsumexp(tmp) + log_emm_prob[l, k_1, k_2]

        # Initialization Backward Pass
        log_beta[N-1] = 0  # log(1)

        # Backward Pass (Equation (33))
        for l in reversed(range(N-1)):
            for k_1 in range(K):
                for k_2 in range(K):
                    tmp = np.empty((K, K))
                    for j_1 in range(K):
                        for j_2 in range(K):
                            tmp[j_1, j_2] = (log_beta[l+1, j_1, j_2]
                                      + log_emm_prob[l+1, j_1, j_2]
                                      + trans_prob_log[l][k_1, j_1]
                                      + trans_prob_log[l][k_2, j_2])
                    log_beta[l, k_1, k_2] = logsumexp(tmp)

        # Compute first posterior moment, \gamma (size N × K × K).
        #
        # Equation (29)
        log_gamma = log_alpha + log_beta
        log_evidence = logsumexp(log_alpha[N-1]) # log(p(X))
        gamma = np.exp(log_gamma - log_evidence)
        #print(np.sum(gamma, axis=(1, 2)))
        assert np.allclose(np.sum(gamma, axis=(1, 2)), 1)

        # Compute second posterior moment, \xi (size N × K × K × K × K).
        # For the n-th sample, the (K × K × K × K) matrix A is defined such that
        # A_{k_1k_2j_1j_2} = E[z_{n} = (k_1, k_2), z_{n+1} = (j_1, j_2)].
        #
        # Equation (30)
        log_xi = np.empty((N - 1, K, K, K, K))
        for l in range(N - 1):
            for k_1 in range(K):
                for k_2 in range(K):
                    for j_1 in range(K):
                        for j_2 in range(K):
                            log_xi[l, k_1, k_2, j_1, j_2] = (
                                    log_alpha[l, k_1, k_2]
                                    + log_emm_prob[l + 1, j_1, j_2]
                                    + trans_prob_log[l][k_1, j_1]
                                    + trans_prob_log[l][k_2, j_2]
                                    + log_beta[l + 1, j_1, j_2]
                                    - log_evidence
                            )


        # M-step.
        # ------------------------------------

        # Maximization: minimize -Q with scipy.optimize.minimize
        #
        # function to minimize
        def opt(theta):
            q_opt = theta[:K]
            r_opt = theta[K]
            q_opt_log = np.log(q_opt)
            val = -Q(X, q_opt_log, r_opt, d_values, p, gamma, log_xi)
            #print("Q-value:", val, "with q:", q_opt, "r:", r_opt)
            return val

        theta0 = np.append(q, r)
        cons = {'type': 'eq', 'fun': lambda theta: np.sum(theta[:K]) - 1} # q_1,...,q_K must sum up to 1
        bounds = [(0, 1)] * K  # q_k must be between 0 and 1
        bounds.append((0, np.inf))  # r must be positive

        # keep the old parameters to see how much the parameters changed after minimization
        q_old = q
        r_old = r

        theta = minimize(opt,theta0, method='SLSQP', bounds=bounds, constraints=[cons])
        q = theta.x[:K]
        log_q = np.log(q)
        r = theta.x[K]

        print("Current parameters: ", "q: ", q, "r: ", r)

        if abs(r_old-r) < delta_r and np.max(np.abs(q_old-q)) < delta_q:
            break

        ############### Q plotten ####################

        # Gitter im Dreieck (q1+q2 <= 1)
        res = 15
        q1 = np.linspace(0, 0.5, res)
        q2 = np.linspace(0, 0.5, res)
        Q1, Q2 = np.meshgrid(q1, q2)
        mask = Q1 + Q2 <= 1
        Q1, Q2 = Q1[mask], Q2[mask]
        Q3 = 1 - Q1 - Q2

        # Plot vorbereiten
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Startwerte
        r0 = 0
        Z = np.array([opt([q1, q2, q3, r0]) for q1, q2, q3 in zip(Q1, Q2, Q3)])
        Z = np.array([Q(X, np.array([q1, q2, q3]), r0, d_values, p, gamma, log_xi) for q1, q2, q3 in zip(Q1, Q2, Q3)])


        surf = ax.plot_trisurf(Q1, Q2, Z, cmap="viridis", linewidth=0.2)
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")
        ax.set_zlabel("Q")

        # Update-Funktion für Animation
        def update(frame):
            ax.clear()
            r = frame / 10  # r läuft von 0 bis 1
            Z = np.array([Q(X, np.array([q1, q2, q3]), r, d_values, p, gamma, log_xi) for q1, q2, q3 in zip(Q1, Q2, Q3)])
            surf = ax.plot_trisurf(Q1, Q2, Z, cmap="viridis", linewidth=0.2)

            ax.set_title(f"r = {r:.2f}")
            ax.set_xlabel("q1")
            ax.set_ylabel("q2")
            ax.set_zlabel("Q")
            return surf,

        ani = animation.FuncAnimation(fig, update, frames=10, interval=200, blit=False)

        plt.show()
        ##################################################

    return q, r


def simulation():
    # Simulate the data
    M = 100 # Number of rows
    K = 3  # Number of states
    # Create the true allele frequencies
    p = sim.create_random_matrix(M, K)
    #print(p)
    # Parameters
    q = np.array([0.4, 0.1, 0.5])  # Initial probabilities for the first K-1 states
    r = 0.5  # Parameter influencing transition probabilities
    d_values = [(i+1) / (M - 1) for i in range(M - 1)]  # Test
    #d_values = [0.1]*(M-1) # Example time-dependent values for d_t
    x1, x2 = sim.simulate_markov_chain_diploid(K, q, r, M, d_values, p)
    return [x1, x2], K, p, d_values, q, r


if __name__ == "__main__":
    X, K, p, d_values, q_true, r_true = simulation()
    X = np.array(X).T

    # define when the iterations should stop
    delta_q, delta_r = 0.005, 0.005  # the optimization stops, if the change of parameters after maximization is smaller than delta

    q, r = baum_welch(X, K, 50, p, d_values, delta_q, delta_r)
    print("Parameters: ", "q: ", q, ", r: ", r)