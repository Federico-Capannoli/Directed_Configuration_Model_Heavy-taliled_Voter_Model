# Heavy-Tailed Voter Model on Directed Networks

This repository contains Python simulations of **opinion dynamics and consensus formation** on **directed networks** with **heavy-tailed degree distributions**.  
The project explores how network topology (in particular, heavy-tailed in- and out-degree distributions) affects consensus time and opinion evolution.

The code is intended for **academic / educational use**, especially for students or researchers interested in:
- Complex networks
- Voter models and consensus dynamics
- Directed configuration models
- Heavy-tailed (Pareto-like) degree distributions

---

## 📂 Project Structure

The repository currently includes the following scripts:

- **`(D) Heavy-Tailed VM.py`**  
  Implements a voter model on directed networks generated using a heavy-tailed degree sequence.  
  Includes:
  - Directed configuration model construction
  - Heavy-tailed in-degree generation
  - Opinion update dynamics
  - Basic visualization and analysis

- **`(D) Annealed Heavy-Tailed VM.py`**  
  An annealed (mean-field–like) version of the heavy-tailed voter model, where network connections are randomized over time instead of being fixed.

- **`(D) Distribution Consensus.py`**  
  Studies consensus behavior by sampling from opinion distributions, focusing on how degree heterogeneity influences convergence and final states.

---

## Model Overview

The simulations are based on variants of the **voter model**, where:
- Each node represents an agent with a binary opinion.
- The network is **directed**, meaning influence is asymmetric.
- Degrees are drawn from **heavy-tailed (Pareto) distributions**, allowing for hubs and strong heterogeneity.
- Opinion updates follow stochastic rules based on neighbors’ states.

Some scripts explore:
- Eulerian vs non-Eulerian directed graphs
- Regular vs heterogeneous out-degree configurations
- Fixed (quenched) vs annealed network structures

---

## Requirements

The code is written in **Python 3** and uses standard scientific libraries:

- `numpy`
- `scipy`
- `networkx`
- `matplotlib`

You can install the dependencies with:

```bash
pip install numpy scipy networkx matplotlib
