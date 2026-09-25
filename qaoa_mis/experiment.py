"""Sweep penalty strength and circuit depth over random graphs.

    python -m qaoa_mis.experiment            # full sweep -> results/sweep.csv
    python -m qaoa_mis.experiment --quick    # small smoke run
"""

import argparse
import csv
import time
from pathlib import Path

import networkx as nx

from qaoa_mis.optimize import optimize, score
from qaoa_mis.problem import brute_force_mis, random_graph

FIELDS = ["graph", "n", "edges", "mis_size", "penalty", "p", "energy",
          "p_feasible", "p_optimal", "approx_ratio", "uniform_p_feasible", "greedy_ratio"]


def greedy_ratio(graph, mis_size):
    """Classical baseline: networkx's randomized maximal independent set, best of 20."""
    best = max(len(nx.maximal_independent_set(graph, seed=s)) for s in range(20))
    return best / mis_size


def run(n, edge_prob, graphs, penalties, depths, restarts, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for g_id in range(graphs):
            graph = random_graph(n, edge_prob, seed=g_id)
            mis_size, _ = brute_force_mis(graph)
            uniform = score(graph, 2.0, [0.0], [0.0]).p_feasible
            greedy = greedy_ratio(graph, mis_size)
            for penalty in penalties:
                prev = None
                for p in depths:
                    start = time.perf_counter()
                    r = optimize(graph, penalty, p, restarts=restarts, seed=g_id,
                                 warm_start=(prev.gammas, prev.betas) if prev else None)
                    prev = r
                    writer.writerow({
                        "graph": g_id, "n": n, "edges": graph.number_of_edges(), "mis_size": mis_size,
                        "penalty": penalty, "p": p, "energy": round(r.energy, 6),
                        "p_feasible": round(r.p_feasible, 6), "p_optimal": round(r.p_optimal, 6),
                        "approx_ratio": round(r.approx_ratio, 6),
                        "uniform_p_feasible": round(uniform, 6), "greedy_ratio": round(greedy, 6),
                    })
                    print(f"graph {g_id} penalty {penalty:>4} p {p}: feasible {r.p_feasible:.3f} "
                          f"optimal {r.p_optimal:.3f} ratio {r.approx_ratio:.3f} "
                          f"({time.perf_counter() - start:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/sweep.csv"))
    args = parser.parse_args()
    if args.quick:
        run(n=6, edge_prob=0.4, graphs=2, penalties=[1.0, 2.0], depths=[1, 2], restarts=2, out=args.out)
    else:
        run(n=8, edge_prob=0.4, graphs=10, penalties=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            depths=[1, 2, 3, 4], restarts=8, out=args.out)


if __name__ == "__main__":
    main()
