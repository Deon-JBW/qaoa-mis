import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qaoa_mis import problem
from qaoa_mis.circuit import FastQAOA, qaoa_circuit, qiskit_probabilities
from qaoa_mis.optimize import optimize, score


@pytest.fixture
def graph():
    return problem.random_graph(5, 0.5, seed=3)


def test_cost_layer_is_the_cost_as_a_phase(graph):
    """With beta = 0 and no Hadamards the circuit is diag(exp(-i gamma C)), up to global phase.
    This pins down the Ising mapping, the RZ/RZZ factor of 2, and qubit order at once."""
    penalty, gamma = 2.0, 0.37
    qc, gammas, betas = qaoa_circuit(graph, penalty, p=1, prepare_plus=False)
    U = Operator(qc.assign_parameters({gammas[0]: gamma, betas[0]: 0.0})).data

    assert np.allclose(U, np.diag(np.diag(U)))
    phases = np.diag(U) / np.diag(U)[0]
    costs = problem.cost_vector(graph, penalty)
    assert np.allclose(phases, np.exp(-1j * gamma * (costs - costs[0])))


@pytest.mark.parametrize("p", [1, 2, 3])
def test_fast_simulator_matches_qiskit(graph, p):
    rng = np.random.default_rng(p)
    gammas, betas = rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)
    fast = FastQAOA(graph, 1.5).probabilities(gammas, betas)
    ref = qiskit_probabilities(graph, 1.5, gammas, betas)
    assert np.allclose(fast, ref, atol=1e-10)


def test_zero_angles_give_uniform_distribution(graph):
    probs = FastQAOA(graph, 2.0).probabilities([0.0], [0.0])
    assert np.allclose(probs, 1 / 2 ** graph.number_of_nodes())


def test_probabilities_are_normalized(graph):
    probs = FastQAOA(graph, 3.0).probabilities([0.4, 1.1], [0.9, 0.2])
    assert probs.sum() == pytest.approx(1.0)


def test_score_of_uniform_distribution(graph):
    r = score(graph, 2.0, [0.0], [0.0])
    n = graph.number_of_nodes()
    mask = problem.feasibility_mask(graph)
    assert r.p_feasible == pytest.approx(mask.sum() / 2**n)
    assert r.p_optimal == pytest.approx(len(problem.brute_force_mis(graph)[1]) / 2**n)


def test_optimizer_beats_uniform_and_is_reproducible(graph):
    uniform = score(graph, 2.0, [0.0], [0.0])
    r1 = optimize(graph, 2.0, p=1, restarts=4, seed=7)
    r2 = optimize(graph, 2.0, p=1, restarts=4, seed=7)
    assert r1.energy < uniform.energy
    assert r1.energy == r2.energy


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("penalty", [1.5, 5.0])
def test_warm_start_never_worse_than_previous_depth(seed, penalty):
    """Regression: the first full sweep got worse with depth in 6 of 180 cases.
    Repeating the last layer is not the same circuit as depth p-1, and COBYLA
    can finish worse than it starts; a zero layer plus keeping starts fixes both."""
    graph = problem.random_graph(6, 0.4, seed=seed)
    prev = optimize(graph, penalty, p=1, restarts=2, seed=seed)
    for p in (2, 3):
        r = optimize(graph, penalty, p=p, restarts=0, seed=seed, warm_start=(prev.gammas, prev.betas))
        assert r.energy <= prev.energy + 1e-9
        prev = r


def test_zero_layer_is_the_identity(graph):
    sim = FastQAOA(graph, 2.0)
    assert np.allclose(sim.state([0.8], [0.3]), sim.state([0.8, 0.0], [0.3, 0.0]))
