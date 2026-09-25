"""Classical outer loop: tune QAOA angles, then score the output distribution."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from qaoa_mis.circuit import FastQAOA
from qaoa_mis.problem import brute_force_mis, feasibility_mask, sizes_vector


@dataclass
class Result:
    penalty: float
    p: int
    gammas: np.ndarray
    betas: np.ndarray
    energy: float
    p_feasible: float      # probability a sample is an independent set
    p_optimal: float       # probability a sample is a maximum independent set
    approx_ratio: float    # E[size if feasible else 0] / MIS size


def optimize(graph, penalty, p, restarts=8, seed=0, warm_start=None):
    """Minimize <C> with COBYLA from several starting points.

    warm_start: (gammas, betas) of length p-1, extended to depth p two ways:
      - append a zero layer. gamma = beta = 0 is the identity, so this start is
        exactly the depth p-1 state, and depth p can never score worse.
      - repeat the last layer. Not equivalent to depth p-1, but often a better
        place for the optimizer to start.
    """
    sim = FastQAOA(graph, penalty)
    rng = np.random.default_rng(seed)

    def objective(theta):
        return sim.expectation(theta[:p], theta[p:])

    starts = [np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
              for _ in range(restarts)]
    if warm_start is not None:
        g, b = warm_start
        starts.append(np.concatenate([np.append(g, 0.0), np.append(b, 0.0)]))
        starts.append(np.concatenate([np.append(g, g[-1]), np.append(b, b[-1])]))

    # COBYLA can end somewhere worse than where it started, so the starting
    # points themselves stay in the running. Together with the zero-layer
    # start, that is what makes the depth guarantee hold.
    candidates = [(objective(x0), x0) for x0 in starts]
    for x0 in starts:
        r = minimize(objective, x0, method="COBYLA", options={"maxiter": 500})
        candidates.append((r.fun, r.x))
    _, best = min(candidates, key=lambda c: c[0])
    gammas, betas = best[:p], best[p:]
    return score(graph, penalty, gammas, betas, sim)


def score(graph, penalty, gammas, betas, sim=None):
    sim = sim or FastQAOA(graph, penalty)
    probs = sim.probabilities(gammas, betas)
    feasible = feasibility_mask(graph)
    sizes = sizes_vector(graph.number_of_nodes())
    mis_size, _ = brute_force_mis(graph)
    return Result(
        penalty=penalty,
        p=len(gammas),
        gammas=np.asarray(gammas),
        betas=np.asarray(betas),
        energy=float(probs @ sim.costs),
        p_feasible=float(probs[feasible].sum()),
        p_optimal=float(probs[feasible & (sizes == mis_size)].sum()),
        approx_ratio=float((probs * sizes * feasible).sum() / mis_size),
    )
