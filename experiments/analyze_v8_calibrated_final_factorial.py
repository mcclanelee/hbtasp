"""Validate, summarize, and plot the calibrated-priority final factorial."""

from __future__ import annotations

from experiments import analyze_v4_final_factorial as base
from experiments import analyze_v5_clustered_uncertainty as clustered
from experiments import plot_v5_final_dynamic_tradeoff as plots


def main():
    out = base.ROOT / "experiments/checkpoints/v8_calibrated_final_factorial"
    # The base analyzer accepts an explicit output directory.
    import sys
    old = sys.argv
    try:
        sys.argv = [old[0], "--out", str(out)]
        base.main()
    finally:
        sys.argv = old
    clustered.SOURCE = out / "cell_results.csv"
    clustered.OUT = out / "clustered_uncertainty.json"
    clustered.main()
    plots.OUT = out
    plots.DATA = out / "cell_results.csv"
    plots.main()


if __name__ == "__main__":
    main()
