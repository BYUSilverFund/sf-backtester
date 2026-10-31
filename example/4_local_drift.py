"""Small end-to-end local entry point, including an independent legacy comparison."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sf_backtester import ExecutionConfig, optimize_and_simulate
from sf_quant.simulation import SimulationConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dates = [date(2024, 1, d) for d in (2, 3, 4)]
    keys = {"date": [d for d in dates for _ in range(2)], "barrid": ["A", "B"]*3}
    returns = pl.DataFrame({**keys, "return": [0., 0., -.5, 0., 1., 0.]})
    targets = pl.DataFrame({**keys, "weight": [.5]*6})
    # This fixture's decisions are known before the close, so explicit lag zero
    # is justified. Production close-observed signals default to lag one.
    result = optimize_and_simulate(targets, lambda data: data, returns,
        execution=ExecutionConfig(simulation=SimulationConfig(
            frequency="weekly", execution_lag_sessions=0)), output_path=args.output)
    legacy = (targets.join(returns, on=["date", "barrid"])
        .group_by("date").agg((pl.col("weight")*pl.col("return")).sum().alias("legacy_constant_weight_return")))
    comparison = result.daily.select("date", "return", "nav", "two_sided_turnover").join(legacy, on="date").sort("date")
    comparison.write_csv(args.output / "comparison.csv")
    assert np.isclose(result.daily["nav"][-1], 1.0, atol=1e-14)
    assert np.isclose(np.prod(1+comparison["legacy_constant_weight_return"].to_numpy()), 1.125, atol=1e-14)
    print(comparison.write_csv())
    print("Final wealth: constant-share simulation=1.0; legacy constant-weight arithmetic=1.125")


if __name__ == "__main__":
    main()
