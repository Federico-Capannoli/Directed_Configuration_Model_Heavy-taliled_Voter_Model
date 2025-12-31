"""
Directed Voter Model on Heavy-Tailed Networks
Consensus Times on the Largest Strongly Connected Component

This script studies consensus times of the voter model on directed networks
generated via the directed configuration model (DCM) with heavy-tailed degree
distributions. For each network size, the dynamics are restricted to the
largest strongly connected component (LSCC), and empirical consensus times
are compared with theoretical scaling predictions involving the parameter
theta and the entropy term H(U).
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def deg_sequence_perm_DCM(k, alpha, k_min, eulerian=False, out_regular=False):
    """
    Generate in- and out-degree sequences for a directed configuration model
    using a Pareto (heavy-tailed) distribution.
    """
    in_degrees = np.floor(np.random.pareto(alpha, k)) + k_min

    if eulerian:
        out_degrees = in_degrees

    elif out_regular:
        m = sum(in_degrees)
        d = m / k
        out_degrees = np.full(k, int(d))

        diff = m - sum(out_degrees)
        if diff != 0:
            adjustment_indices = np.random.choice(k, size=abs(int(diff)), replace=False)
            for idx in adjustment_indices:
                if diff > 0:
                    out_degrees[idx] += 1
                elif diff < 0 and out_degrees[idx] > 1:
                    out_degrees[idx] -= 1

    else:
        out_degrees = np.random.permutation(in_degrees)

    return in_degrees.astype(int), out_degrees.astype(int)


def Theta(n, deg_sequence_DCM):
    """
    Compute the theoretical scaling parameter theta from a degree sequence.
    """
    in_degrees = np.array(deg_sequence_DCM[0])
    out_degrees = np.array(deg_sequence_DCM[1])

    m = np.sum(in_degrees)
    delta = m / n
    beta = (1 / m) * np.sum(in_degrees ** 2)
    rho = (1 / m) * np.sum(in_degrees / out_degrees)
    gamma = (1 / m) * np.sum((in_degrees ** 2) / out_degrees)

    theta = (1 / 2) * delta / (
        ((gamma - rho) / (1 - rho) * (1 - np.sqrt(1 - rho)) / rho) + beta - 1
    )
    return theta


def Theta_from_graph(G_sub):
    """
    Compute theta directly from a subgraph (typically the LSCC),
    using its empirical in- and out-degree sequences.
    """
    in_degrees = np.array([d for _, d in G_sub.in_degree()])
    out_degrees = np.array([d for _, d in G_sub.out_degree()])

    m = in_degrees.sum()
    n_local = G_sub.number_of_nodes()

    if m == 0 or n_local == 0:
        return 0

    delta = m / n_local
    epsilon = 1e-10
    out_degrees_safe = np.where(out_degrees == 0, epsilon, out_degrees)

    beta = (1 / m) * np.sum(in_degrees ** 2)
    rho = (1 / m) * np.sum(in_degrees / out_degrees_safe)
    gamma = (1 / m) * np.sum((in_degrees ** 2) / out_degrees_safe)

    denom = ((gamma - rho) / (1 - rho) * (1 - np.sqrt(1 - rho)) / rho) + beta - 1
    if denom == 0:
        return 0

    theta = (1 / 2) * delta / denom
    return theta


def H(u):
    """
    Entropy-like function appearing in theoretical scaling predictions.
    """
    return -2 * ((1 - u) * np.log(1 - u) + u * np.log(u))


###################################################################################################
# Simulation parameters
###################################################################################################

alpha = 3
k_min = 1
U = 0.5

eulerian = False
out_regular = False

n_min = 100
n_max = 1000
number_of_iterations_n = 5

iteration_VM = 5
iteration_Graph = 5
iteration_Degree_sequence = 5

n_values = np.linspace(n_min, n_max, number_of_iterations_n).astype(int)


###################################################################################################
# Data storage
###################################################################################################

All_Consensus_times = {n: [] for n in n_values}
Consensus_times_degree_sequences = {n: [] for n in n_values}
Average_Consensus_times = []

theta_values = {n: [] for n in n_values}
G1_sizes = {n: [] for n in n_values}


###################################################################################################
# Main simulation loop
###################################################################################################

for n in n_values:

    Consensus_times_Degree_Sequence_average = []

    for Iter_deg_seq in range(iteration_Degree_sequence):

        Consensus_times_Graph_average = []
        degree_sequence = deg_sequence_perm_DCM(n, alpha, k_min, eulerian, out_regular)

        for Iter_graph in range(iteration_Graph):

            G = nx.directed_configuration_model(
                degree_sequence[0], degree_sequence[1]
            )

            largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
            G1 = G.subgraph(largest_scc_nodes).copy()
            n1 = G1.number_of_nodes()

            G1_sizes[n].append(n1)

            theta_val = Theta_from_graph(G1)
            theta_values[n].append(theta_val)

            Consensus_times_VM = []

            for Iter_VM in range(iteration_VM):

                node_colors = {
                    node: "red" if random.random() < U else "blue"
                    for node in G1.nodes()
                }

                discordant_edges = {
                    (u, v) for u, v in G1.edges()
                    if node_colors[u] != node_colors[v]
                }

                time = 0
                consensus = False
                nodes = list(G1.nodes())

                while not consensus:

                    u_candidates = [
                        node for node in nodes
                        if len(list(G1.successors(node))) > 0
                    ]

                    u = random.choice(u_candidates)
                    v = random.choice(list(G1.successors(u)))

                    if node_colors[u] != node_colors[v]:
                        node_colors[u] = node_colors[v]

                        for out_neighbor in G1.successors(u):
                            edge = (u, out_neighbor)
                            if node_colors[u] == node_colors[out_neighbor]:
                                discordant_edges.discard(edge)
                            else:
                                discordant_edges.add(edge)

                        for out_neighbor in G1.successors(v):
                            edge = (v, out_neighbor)
                            if node_colors[v] == node_colors[out_neighbor]:
                                discordant_edges.discard(edge)
                            else:
                                discordant_edges.add(edge)

                        for in_neighbor in G1.predecessors(u):
                            edge = (in_neighbor, u)
                            if node_colors[in_neighbor] == node_colors[u]:
                                discordant_edges.discard(edge)
                            else:
                                discordant_edges.add(edge)

                        for in_neighbor in G1.predecessors(v):
                            edge = (in_neighbor, v)
                            if node_colors[in_neighbor] == node_colors[v]:
                                discordant_edges.discard(edge)
                            else:
                                discordant_edges.add(edge)

                    time += 1

                    if len(discordant_edges) == 0:
                        consensus = True

                consensus_time_normalized = time / n1
                Consensus_times_VM.append(consensus_time_normalized)
                All_Consensus_times[n].append(consensus_time_normalized)

                print(
                    f'n = {n}/{n_max}, Deg_seq Iteration: {Iter_deg_seq + 1}/{iteration_Degree_sequence}, '
                    f'Graph Iteration: {Iter_graph + 1}/{iteration_Graph}, '
                    f'VM Iteration: {Iter_VM + 1}/{iteration_VM}, '
                    f'Time: {consensus_time_normalized}'
                )

            Consensus_times_Graph_average.append(
                np.mean(Consensus_times_VM)
            )

        Consensus_times_Degree_Sequence_average.append(
            np.mean(Consensus_times_Graph_average)
        )

        Consensus_times_degree_sequences[n].append(
            np.mean(Consensus_times_Graph_average)
        )

    Average_Consensus_times.append(
        np.mean(Consensus_times_Degree_Sequence_average)
    )


###################################################################################################
# Post-processing and visualization
###################################################################################################

theta_avg_values = [
    np.mean(theta_values[n]) if theta_values[n] else 0 for n in n_values
]

avg_G1_sizes = [
    np.mean(G1_sizes[n]) if G1_sizes[n] else 0 for n in n_values
]

theta_times_n = np.array(theta_avg_values) * np.array(avg_G1_sizes)

plt.figure(figsize=(10, 6))
plt.boxplot(
    [Consensus_times_degree_sequences[n] for n in n_values],
    positions=n_values,
    widths=0.1 * n_values
)
plt.xlabel('N')
plt.ylabel('Average Consensus Time')
plt.plot(n_values, Average_Consensus_times, 'o', label="Avg Consensus Time")
plt.plot(n_values, H(U) * theta_times_n, '-', color="green", label=r'$\theta N \times H(U)$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title(
    f'DCM with LSCC - Alpha = {alpha}; '
    f'VM iter: {iteration_VM}, Graph iter: {iteration_Graph}, '
    f'Deg_seq iter: {iteration_Degree_sequence}'
)
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(
    [Consensus_times_degree_sequences[n] for n in n_values],
    positions=n_values,
    widths=0.1 * n_values
)
plt.xlabel('N')
plt.ylabel('Average Consensus Time')
plt.plot(n_values, Average_Consensus_times, 'o', label="Avg Consensus Time")
plt.plot(n_values, H(U) * theta_times_n, '-', color="green", label=r'$\theta N \times H(U)$')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title(
    f'DCM with LSCC - Alpha = {alpha}; VM iter: {iteration_VM}, '
    f'Graph iter: {iteration_Graph}, Deg_seq iter: {iteration_Degree_sequence}'
)
plt.show()


###################################################################################################
# Output diagnostics
###################################################################################################

print("Theta values per n:")
print(theta_values)

print("Sizes of G1 (number of nodes) per n:")
print(G1_sizes)
