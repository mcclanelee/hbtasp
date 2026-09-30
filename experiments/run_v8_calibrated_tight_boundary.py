"""Tight-period dynamic boundary aligned with the calibrated final pool."""

from experiments import run_v5_tight_period_dynamic_boundary as boundary


def main():
    root = boundary.ROOT
    boundary.POOL_PATH = root / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
    boundary.OUT = root / "experiments/checkpoints/v8_calibrated_tight_boundary"
    boundary.RESULT = boundary.OUT / "cell_results.csv"
    boundary.main()


if __name__ == "__main__":
    main()
