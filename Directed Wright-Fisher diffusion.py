"""
Heavy-Tailed Voter Model on Directed Configuration Graphs
Discordant Edges vs Opinion Density

This script simulates voter-model dynamics on directed networks with
heavy-tailed degree distributions generated via the directed configuration
model (DCM). It tracks the evolution of discordant edges and opinion density
until consensus is reached and compares empirical trajectories with a
theoretical parabola.
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
                elif diff < 0 and out_degrees[idx] > 2:
                    out_degrees[idx] -= 1

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
    Compute the theoretical coefficient relating discordant edges
    to opinion density.
    """
    in_degrees = np.array(deg_sequence_DCM[0])
    out_degrees = np.array(deg_sequence_DCM[1])

    m = np.sum(in_degrees)
    delta = m / n
    rho = (1 / m) * np.sum(in_degrees / out_degrees)

    Phi = 1 - (1 - np.sqrt(1 - rho)) / (delta * rho)
    return Phi


def H(u):
    """
    Entropy-like function appearing in theoretical analysis.
    """
    return -2 * ((1 - u) * np.log(1 - u) + u * np.log(u))


###################################################################################################
# Simulation parameters
###################################################################################################

alpha = 0.8
k_min = 5

eulerian = False
out_regular = False

n = 5000
Connectivity_iter_max = 10

U_values = [0.3, 0.5, 0.7, 0.9]


###################################################################################################
# Voter-model simulation
###################################################################################################

phi_values = []
total_discordant_edges = {U: [] for U in U_values}
total_blues = {U: [] for U in U_values}

for U in U_values:

    fraction_discordant_edges = []
    fraction_blue_vertices = []

    degree_sequence = deg_sequence_perm_DCM(n, alpha, k_min, eulerian, out_regular)
    G = nx.directed_configuration_model(degree_sequence[0], degree_sequence[1])

    Conn_iter = 0
    while not nx.is_strongly_connected(G):
        G = nx.directed_configuration_model(degree_sequence[0], degree_sequence[1])
        if Conn_iter > Connectivity_iter_max:
            degree_sequence = deg_sequence_perm_DCM(n, alpha, k_min, eulerian, out_regular)
        Conn_iter += 1

    M = len(G.edges())

    node_colors = {i: "red" if random.random() < U else "blue" for i in range(n)}
    discordant_edges = {(u, v) for u, v in G.edges() if node_colors[u] != node_colors[v]}

    fraction_discordant_edges.append(len(discordant_edges) / M)
    blues = sum(1 for color in node_colors.values() if color == "blue")
    fraction_blue_vertices.append(blues / n)

    time = 0
    consensus = False
    nodes = list(G.nodes())

    while not consensus:

        u = random.choice(nodes)
        v = random.choice(list(G.successors(u)))

        if node_colors[u] != node_colors[v]:
            if node_colors[v] == "blue" and node_colors[u] == "red":
                blues += 1
            else:
                blues -= 1

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

        fraction_blue_vertices.append(blues / n)
        fraction_discordant_edges.append(len(discordant_edges) / M)
        time += 1

        if len(discordant_edges) == 0:
            consensus = True
            phi_values.append(Phi(n, degree_sequence))
            total_discordant_edges[U].append(fraction_discordant_edges)
            total_blues[U].append(fraction_blue_vertices)


###################################################################################################
# Visualization
###################################################################################################

avg_phi = np.mean(phi_values)

max_len = max([max(map(len, total_discordant_edges[U])) for U in U_values])
parabola = [2 * avg_phi * x * (1 - x) for x in np.linspace(0, 1, max_len)]

colors = ['yellow', 'blue', 'green', 'orange']

plt.figure()
plt.plot(np.linspace(0, 1, max_len), parabola, color="red")

for i, U in enumerate(U_values):
    plt.scatter(
        total_blues[U],
        total_discordant_edges[U],
        color=colors[i],
        s=10
    )

plt.xlabel('fraction blue vertices')
plt.ylabel('fraction discordant edges')
plt.title(f'DCM - alpha = {alpha}')
plt.show()
