import csv

from qaoa_mis.experiment import FIELDS, run
from qaoa_mis.plots import main as plot


def test_sweep_writes_one_row_per_config_and_plots(tmp_path):
    out = tmp_path / "sweep.csv"
    run(n=4, edge_prob=0.5, graphs=2, penalties=[1.5, 3.0], depths=[1, 2], restarts=1, out=out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 2 * 2
    assert list(rows[0]) == FIELDS
    for r in rows:
        assert 0 <= float(r["p_optimal"]) <= float(r["p_feasible"]) <= 1 + 1e-9
        assert 0 <= float(r["approx_ratio"]) <= 1 + 1e-9

    plot(out, tmp_path / "figures")
    assert (tmp_path / "figures" / "penalty_sweep.png").stat().st_size > 0
