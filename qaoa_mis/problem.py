"""Maximum Independent Set (MIS) as a penalized binary optimization problem.

Bitstrings are integers: vertex i is in the set when bit i of the index is 1.
This matches Qiskit's little-endian convention, where qubit i is bit i.

Cost to minimize:  C(x) = -sum_i x_i + penalty * sum_{(i,j) in E} x_i x_j

For penalty > 1 every minimizer is a maximum independent set: removing one
endpoint of a violated edge changes the cost by at most +1 - penalty < 0.
At penalty <= 1 infeasible sets can tie with or beat the true optimum.
"""

import itertools

import networkx as nx
import numpy as np


def bits(index, n):
    return [(index >> i) & 1 for i in range(n)]


def set_size(index):
    return index.bit_count()


def is_independent(graph, index):
    return not any((index >> i) & 1 and (index >> j) & 1 for i, j in graph.edges)


def cost(graph, index, penalty):
    violations = sum((index >> i) & 1 and (index >> j) & 1 for i, j in graph.edges)
    return -set_size(index) + penalty * violations


def cost_vector(graph, penalty):
    """C(x) for every bitstring, indexed by bitstring integer. Vectorized."""
    n = graph.number_of_nodes()
    idx = np.arange(2**n)
    x = (idx[:, None] >> np.arange(n)) & 1
    c = -x.sum(axis=1).astype(float)
    for i, j in graph.edges:
        c += penalty * (x[:, i] & x[:, j])
    return c


def feasibility_mask(graph):
    n = graph.number_of_nodes()
    idx = np.arange(2**n)
    x = (idx[:, None] >> np.arange(n)) & 1
    ok = np.ones(2**n, dtype=bool)
    for i, j in graph.edges:
        ok &= ~((x[:, i] & x[:, j]).astype(bool))
    return ok


def sizes_vector(n):
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)) & 1).sum(axis=1)


def brute_force_mis(graph):
    """Exact MIS size and all optimal bitstrings by enumeration (small n only)."""
    n = graph.number_of_nodes()
    best, winners = 0, []
    for index in range(2**n):
        if is_independent(graph, index):
            s = set_size(index)
            if s > best:
                best, winners = s, [index]
            elif s == best:
                winners.append(index)
    return best, winners


def ising(graph, penalty):
    """Map C(x) to  offset + sum_i h_i Z_i + sum_(i,j) J_ij Z_i Z_j  using x = (1 - z) / 2.

    Z has eigenvalue +1 on |0> and -1 on |1>, so x_i = 1 <-> qubit i in |1>.
    """
    h = {i: 0.5 - penalty * graph.degree(i) / 4 for i in graph.nodes}
    J = {(i, j): penalty / 4 for i, j in graph.edges}
    offset = -graph.number_of_nodes() / 2 + penalty * graph.number_of_edges() / 4
    return h, J, offset


def random_graph(n, edge_prob, seed):
    """Connected Erdos-Renyi graph with nodes labelled 0..n-1."""
    for attempt in itertools.count():
        g = nx.gnp_random_graph(n, edge_prob, seed=seed * 1000 + attempt)
        if nx.is_connected(g):
            return g
