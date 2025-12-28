"""
Heavy-Tailed Voter Model on Directed Configuration Graphs
Temporal Evolution of Discordant Edges

This script simulates voter-model dynamics on a fixed directed network with
heavy-tailed degree distributions. Multiple realizations of the opinion
dynamics are run on the same graph, tracking the fraction of discordant edges
over time until consensus is reached.

The empirical trajectories are compared against the theoretical prediction
2U(1-U) * Phi.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def deg_sequence_perm_DCM(k, alpha, k_min, eulerian=False, out_regular=False, regular=False):
    """
    Generate in- and out-degree sequences for a directed configuration model
    using a heavy-tailed (Pareto) distribution or regular constructions.
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
                elif diff < 0 and out_degrees[idx] > 2:
                    out_degrees[idx] -= 1

    elif regular:
        out_degrees = np.full(k, k_min)
        in_degrees = out_degrees

    else:
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
    Compute the theoretical scaling parameter theta.
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


def Phi(n, deg_sequence_DCM):
    """
    Compute the theoretical coefficient Phi relating opinion density
    to the fraction of discordant edges.
    """
    in_degrees = np.array(deg_sequence_DCM[0])
    out_degrees = np.array(deg_sequence_DCM[1])

    m = np.sum(in_degrees)
    delta = m / n
    rho = (1 / m) * np.sum(in_degrees / out_degrees)

    Phi = 1 - ((1 - np.sqrt(1 - rho)) / (delta * rho))
    return Phi


def H(u):
    """
    Entropy-like function appearing in the theoretical analysis.
    """
    return -2 * ((1 - u) * np.log(1 - u) + u * np.log(u))


###################################################################################################
# Simulation parameters
###################################################################################################

alpha = 1
k_min = 5
U = 0.5

eulerian = False
out_regular = False
regular = False

n = 5000
Iter = 5
Connectivity_iter_max = 10


###################################################################################################
# Graph generation
###################################################################################################

total_fraction = {i: [] for i in range(Iter)}

degree_sequence = deg_sequence_perm_DCM(n, alpha, k_min, eulerian, out_regular, regular)
G = nx.directed_configuration_model(degree_sequence[0], degree_sequence[1])

Conn_iter = 0
while not nx.is_strongly_connected(G):
    G = nx.directed_configuration_model(degree_sequence[0], degree_sequence[1])
    if Conn_iter > Connectivity_iter_max:
        degree_sequence = deg_sequence_perm_DCM(n, alpha, k_min, eulerian, out_regular, regular)
    Conn_iter += 1

M = len(G.edges())


###################################################################################################
# Voter-model dynamics
###################################################################################################

for i in range(Iter):

    fraction_discordant_edges = []
    fraction_blue_vertices = []
    Consensus_times_VM = []

    node_colors = {i: "red" if random.random() < U else "blue" for i in range(n)}
    discordant_edges = {(u, v) for u, v in G.edges() if node_colors[u] != node_colors[v]}

    fraction_discordant_edges.append(len(discordant_edges) / M)

    time = 0
    consensus = False
    nodes = list(G.nodes())

    while not consensus:

        u = random.choice(nodes)
        v = random.choice(list(G.successors(u)))

        if node_colors[u] != node_colors[v]:
            node_colors[u] = node_colors[v]

            for out_neighbor in G.successors(u):
                edge = (u, out_neighbor)
                if node_colors[u] == node_colors[out_neighbor]:
                    discordant_edges.discard(edge)
                else:
                    discordant_edges.add(edge)

            for out_neighbor in G.successors(v):
                edge = (v, out_neighbor)
                if node_colors[v] == node_colors[out_neighbor]:
                    discordant_edges.discard(edge)
                else:
                    discordant_edges.add(edge)

            for in_neighbor in G.predecessors(u):
                edge = (in_neighbor, u)
                if node_colors[in_neighbor] == node_colors[u]:
                    discordant_edges.discard(edge)
                else:
                    discordant_edges.add(edge)

            for in_neighbor in G.predecessors(v):
                edge = (in_neighbor, v)
                if node_colors[in_neighbor] == node_colors[v]:
                    discordant_edges.discard(edge)
                else:
                    discordant_edges.add(edge)

        fraction_discordant_edges.append(len(discordant_edges) / M)
        time += 1

        if len(discordant_edges) == 0:
            consensus = True
            total_fraction[i].append(fraction_discordant_edges)


###################################################################################################
# Visualization
###################################################################################################

phi_value = 2 * U * (1 - U) * Phi(n, degree_sequence)

colors = [mcolors.to_rgba(cm.Oranges(i / Iter)) for i in range(Iter)]

plt.axhline(
    y=phi_value,
    color='red',
    linestyle='--',
    label=r'$2U(1-U)\varphi(\infty)$',
    zorder=3
)

for i in range(Iter):
    for run in total_fraction[i]:
        plt.plot(run, color=colors[i], zorder=2)

plt.xlabel('time')
plt.ylabel('fraction discordant edges')
plt.title(f'DCM - alpha = {alpha}')
plt.show()
