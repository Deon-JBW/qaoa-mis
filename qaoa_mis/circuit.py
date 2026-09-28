"""QAOA ansatz for MIS, as a Qiskit circuit and as a fast NumPy simulator.

The Qiskit circuit is the reference. The NumPy path exists because parameter
sweeps call the objective hundreds of thousands of times; tests check the two
agree to numerical precision.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from qaoa_mis.problem import cost_vector, ising


def qaoa_circuit(graph, penalty, p, prepare_plus=True):
    """|+>^n, then p rounds of exp(-i gamma H_C) exp(-i beta H_M) with H_M = sum X_i.

    prepare_plus=False omits the initial Hadamards, leaving just the QAOA unitary.
    """
    n = graph.number_of_nodes()
    h, J, _ = ising(graph, penalty)
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)

    qc = QuantumCircuit(n)
    if prepare_plus:
        qc.h(range(n))
    for layer in range(p):
        # RZ(t) = exp(-i t Z / 2), RZZ(t) = exp(-i t ZZ / 2), hence the factor of 2.
        for i, hi in h.items():
            qc.rz(2 * gammas[layer] * hi, i)
        for (i, j), Jij in J.items():
            qc.rzz(2 * gammas[layer] * Jij, i, j)
        for i in range(n):
            qc.rx(2 * betas[layer], i)
    return qc, gammas, betas


def qiskit_probabilities(graph, penalty, gammas, betas):
    qc, g_params, b_params = qaoa_circuit(graph, penalty, len(gammas))
    values = dict(zip(g_params, gammas, strict=True)) | dict(zip(b_params, betas, strict=True))
    bound = qc.assign_parameters(values)
    return Statevector(bound).probabilities()


class FastQAOA:
    """Statevector simulation using the diagonal cost directly.

    The cost layer is a diagonal phase, and the mixer is the same RX on every
    qubit, so neither needs a gate-by-gate simulation.
    """

    def __init__(self, graph, penalty):
        self.n = graph.number_of_nodes()
        self.costs = cost_vector(graph, penalty)

    def state(self, gammas, betas):
        n = self.n
        psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
        for gamma, beta in zip(gammas, betas, strict=True):
            psi = psi * np.exp(-1j * gamma * self.costs)
            c, s = np.cos(beta), -1j * np.sin(beta)
            psi = psi.reshape([2] * n)
            for axis in range(n):
                a0 = np.take(psi, 0, axis=axis)
                a1 = np.take(psi, 1, axis=axis)
                psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=axis)
            psi = psi.reshape(-1)
        return psi

    def probabilities(self, gammas, betas):
        return np.abs(self.state(gammas, betas)) ** 2

    def expectation(self, gammas, betas):
        return float(self.probabilities(gammas, betas) @ self.costs)
