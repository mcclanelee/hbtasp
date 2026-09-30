"""R2.3 stratified replay aligned with the calibrated final factorial pool."""

from experiments import run_r2_3_multidefect_stratified as replay


def main():
    root = replay.ROOT
    replay.POOL_PATH = root / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
    replay.OUT = root / "experiments/checkpoints/v8_calibrated_multidefect_stratified"
    replay.main()


if __name__ == "__main__":
    main()
