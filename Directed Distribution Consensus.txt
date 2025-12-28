"""
Heavy-Tailed Voter Model on Directed Configuration Graphs

This script analyzes consensus dynamics for the voter model on directed networks
with heavy-tailed degree distributions. Networks are generated using the directed
configuration model (DCM), and empirical consensus times are compared with a
theoretical limiting distribution.

The code is intended for exploratory and research-oriented use.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import comb


def deg_sequence_perm_DCM(k, alpha, k_min, eulerian=False, out_regular=False):
    """
    Generate in- and out-degree sequences for a directed configuration model
    using a Pareto (heavy-tailed) distribution.
    """
    # Generate random in-degree sequence
    in_degrees = np.floor(np.random.pareto(alpha, k)) + k_min

    if eulerian:
        # Eulerian graphs require equal in- and out-degrees
        out_degrees = in_degrees

    elif out_regular:
        # Approximately regular out-degree sequence
        m = sum(in_degrees)
        d = m / k
        out_degrees = np.full(k, int(d))

        diff = m - sum(out_degrees)

        if diff != 0:
            adjustment_indices = np.random.choice(k, size=abs(int(diff)), replace=False)
            for idx in adjustment_indices:
                if diff > 0:
                    out_degrees[idx] += 1
                elif diff < 0 and out_degrees[idx] > 2:
                    out_degrees[idx] -= 1

    else:
        # Random permutation of in-degrees
        out_degrees = np.random.permutation(in_degrees)

    return in_degrees.astype(int), out_degrees.astype(int)


def deg_sequence_DCM(k, alpha, k_min_in, k_min_out):
    """
    Generate a valid degree sequence for the directed configuration model
    using Pareto-distributed in- and out-degrees.
    """
    while True:
        in_degrees = np.floor(np.random.pareto(alpha, k)) + k_min_in
        sum_in_degrees = sum(in_degrees)

        out_degrees = np.floor(np.random.pareto(alpha, k)) + k_min_out
        out_degrees[out_degrees < 1] = 1

        diff = sum_in_degrees - sum(out_degrees)

        while diff != 0:
            adjustment_indices = np.random.choice(k, size=int(abs(diff)), replace=True)
            for idx in adjustment_indices:
                if diff > 0:
                    out_degrees[idx] += 1
                elif diff < 0 and out_degrees[idx] > 2:
                    out_degrees[idx] -= 1
            diff = sum_in_degrees - sum(out_degrees)

        if sum(in_degrees) == sum(out_degrees):
            return in_degrees.astype(int), out_degrees.astype(int)


def Theta(n, deg_sequence_DCM):
    """
    Compute the theoretical scaling parameter theta for consensus time.
    """
    in_degrees = np.array(deg_sequence_DCM[0])
    out_degrees = np.array(deg_sequence_DCM[1])

    m = np.sum(in_degrees)
    delta = m / n
    beta = (1 / m) * np.sum(in_degrees ** 2)
    rho = (1 / m) * np.sum(in_degrees / out_degrees)
    gamma = (1 / m) * np.sum((in_degrees ** 2) / out_degrees)

    theta = (1 / 2) * delta / (
        ((gamma - rho) / (1 - rho) * (1 - (1 - np.sqrt(1 - rho)) / rho)) + beta - 1
    )
    return theta


def H(u):
    """
    Entropy-like function used in the theoretical analysis.
    """
    return -2 * ((1 - u) * np.log(1 - u) + u * np.log(u))


###################################################################################################
# CSV utilities
###################################################################################################

def read_csv_with_metadata(file_path):
    """
    Read a CSV file with metadata stored above the header.
    """
    metadata = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if "," in line:
            data_start = i
            break
        key_value = line.strip().split(":")
        if len(key_value) == 2:
            key, value = key_value
            metadata[key.strip()] = value.strip()

    data = pd.read_csv(file_path, skiprows=data_start)
    return metadata, data


def H(u):
    return -2 * ((1 - u) * np.log(1 - u) + u * np.log(u))


###################################################################################################
# Load empirical data
###################################################################################################

n = 3000
alpha = 1.5

file_path_all_cons = f'Alice_outputs/D_alpha={alpha}_all_consensus_times.csv'
data_all_cons = pd.read_csv(file_path_all_cons)

file_path_avg = f'Alice_outputs/D_alpha={alpha}_consensus_avg_data.csv'
metadata, data_avg = read_csv_with_metadata(file_path_avg)

file_path_theta = f'Alice_outputs/D_alpha={alpha}_all_theta_values.csv'
data_theta = pd.read_csv(file_path_theta)

# Extract metadata
n_min = int(metadata['# n_min'])
n_max = int(metadata['# n_max'])
number_of_iterations_n = int(metadata['# number_of_iterations_n'])
iteration_VM = int(metadata['# Iteration_VM'])
iteration_Graph = int(metadata['# Iteration_Graph'])
iteration_Degree_sequence = int(metadata['# Iteration_Degree_sequence'])
k_min_in = int(metadata['# k_min_in'])
k_min_out = int(metadata['# k_min_out'])
U = float(metadata['# U'])

# Extract data columns
n_values = data_avg.iloc[:, 0]
Average_Consensus_times = data_avg.iloc[:, 1]

All_Consensus_times = data_all_cons.groupby('n')['All_consensus_time'].apply(list).to_dict()
theta_values = data_theta.groupby('n')['Theta'].apply(list).to_dict()

theta_avg_values = [np.mean(theta_values[n]) for n in n_values]
theta_avg = theta_avg_values[-1]
Consensus_times_VM = All_Consensus_times[n]


###################################################################################################
# Theoretical distribution and ECDF comparison
###################################################################################################

ecdf_consensus = ECDF(np.array(Consensus_times_VM) * (1 / (theta_avg * n)))

num_samples = 5000

U_ber = np.random.binomial(1, U, num_samples)
A = np.random.geometric(1 - U, num_samples)
B = np.random.geometric(U, num_samples)

K = U_ber * A + (1 - U_ber) * B

X_samples = []

for k in K:
    k_values = np.arange(k + 1, k + 100)
    lambdas = np.array([comb(k_val, 2, exact=True) for k_val in k_values])
    Y_k = np.random.exponential(scale=1 / lambdas)
    X_samples.append(np.sum(Y_k))

X_samples = np.array(X_samples)

plt.hist(
    X_samples,
    bins='auto',
    density=True,
    alpha=0.6,
    edgecolor='black',
    label="Theoretical"
)

plt.hist(
    np.array(Consensus_times_VM) * (1 / (theta_avg * n)),
    bins='auto',
    density=True,
    alpha=0.6,
    edgecolor='black',
    label="Empirical"
)

plt.title(f"DCM - alpha = {alpha} - Consensus Density")
plt.grid()
plt.legend()
plt.show()
