import itertools

import networkx as nx
import numpy as np
import pytest

from qaoa_mis import problem


@pytest.fixture(params=[0, 1, 2])
def graph(request):
    return problem.random_graph(6, 0.5, seed=request.param)


def test_known_mis_sizes():
    assert problem.brute_force_mis(nx.path_graph(5))[0] == 3        # 0, 2, 4
    assert problem.brute_force_mis(nx.cycle_graph(6))[0] == 3
    assert problem.brute_force_mis(nx.complete_graph(5))[0] == 1
    assert problem.brute_force_mis(nx.star_graph(4))[0] == 4        # all leaves


def test_brute_force_matches_networkx(graph):
    # Independent sets of G are cliques of the complement.
    expected = max(len(c) for c in nx.find_cliques(nx.complement(graph)))
    assert problem.brute_force_mis(graph)[0] == expected


def test_cost_vector_matches_scalar_cost(graph):
    vec = problem.cost_vector(graph, penalty=2.5)
    for index in range(2 ** graph.number_of_nodes()):
        assert vec[index] == problem.cost(graph, index, 2.5)


def test_feasibility_mask_matches_scalar_check(graph):
    mask = problem.feasibility_mask(graph)
    for index in range(2 ** graph.number_of_nodes()):
        assert mask[index] == problem.is_independent(graph, index)


def test_ising_reproduces_cost_on_every_bitstring(graph):
    penalty = 1.7
    h, J, offset = problem.ising(graph, penalty)
    for index in range(2 ** graph.number_of_nodes()):
        z = [1 - 2 * b for b in problem.bits(index, graph.number_of_nodes())]
        energy = offset + sum(hi * z[i] for i, hi in h.items()) + sum(
            Jij * z[i] * z[j] for (i, j), Jij in J.items())
        assert energy == pytest.approx(problem.cost(graph, index, penalty))


@pytest.mark.parametrize("penalty", [1.01, 1.5, 3.0])
def test_minimizers_are_exactly_the_mis_when_penalty_above_one(graph, penalty):
    costs = problem.cost_vector(graph, penalty)
    minimizers = set(np.flatnonzero(np.isclose(costs, costs.min())))
    assert minimizers == set(problem.brute_force_mis(graph)[1])


def test_penalty_at_or_below_one_lets_infeasible_sets_compete():
    # On a single edge, {0, 1} costs -2 + penalty: a tie with {0} at penalty=1,
    # strictly better below it. This is why the sweep starts above 1.
    g = nx.Graph([(0, 1)])
    assert problem.cost(g, 0b11, 1.0) == problem.cost(g, 0b01, 1.0)
    assert problem.cost(g, 0b11, 0.5) < problem.cost(g, 0b01, 0.5)


def test_bit_order_is_little_endian():
    # Vertex 0 alone must be index 1, not index 2**(n-1), to match Qiskit.
    g = nx.path_graph(3)
    assert problem.bits(1, 3) == [1, 0, 0]
    assert problem.is_independent(g, 0b101)          # vertices 0 and 2
    assert not problem.is_independent(g, 0b011)      # vertices 0 and 1


def test_random_graph_is_connected_and_deterministic():
    for seed in range(5):
        g1, g2 = problem.random_graph(8, 0.3, seed), problem.random_graph(8, 0.3, seed)
        assert nx.is_connected(g1)
        assert sorted(g1.edges) == sorted(g2.edges)
        assert sorted(g1.nodes) == list(range(8))


def test_sizes_vector():
    assert list(problem.sizes_vector(2)) == [0, 1, 1, 2]
    assert all(problem.sizes_vector(4)[i] == bin(i).count("1") for i in range(16))


def test_edge_order_does_not_matter():
    edges = [(0, 1), (1, 2), (2, 3)]
    reference = problem.cost_vector(nx.path_graph(4), 2)
    for perm in itertools.permutations(edges):
        g = nx.Graph()
        g.add_nodes_from(range(4))
        g.add_edges_from(perm)
        assert np.array_equal(problem.cost_vector(g, 2), reference)
