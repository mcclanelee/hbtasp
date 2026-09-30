"""Authoritative 2x2 factorial using the held-out calibrated histogram pool."""

from __future__ import annotations

from experiments import run_v5_final_factorial as factorial


def main():
    out = factorial.ROOT / "experiments/checkpoints/v8_calibrated_final_factorial"
    out.mkdir(parents=True, exist_ok=True)
    pool_path = out / "calibrated_histogram_test_pool.json"
    if not pool_path.is_file():
        raise FileNotFoundError(
            "The frozen held-out calibrated priority pool is required: "
            f"{pool_path}"
        )
    factorial.POOL_PATH = pool_path
    factorial.OUT = out
    factorial.RESULT = out / "cell_results.csv"
    factorial.main()


if __name__ == "__main__":
    main()
