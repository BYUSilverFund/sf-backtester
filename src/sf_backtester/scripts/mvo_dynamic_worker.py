"""MVO dynamic worker script for SLURM jobs.

This script is executed by each SLURM array task to process a single year.
"""

import argparse
import datetime as dt

import polars as pl
import sf_quant.data as sfd
import sf_quant.backtester as sfb
from sf_backtester.scripts.utils import get_constraints


def run_backtest_by_year(
    df: pl.LazyFrame,
    year: int,
    output_dir: str,
    n_cpus: int,
    constraints: list[str],
    active_weights: bool,
    target_active_risk: float,
    enforce_zero_beta: bool,
) -> None:
    """Run MVO backtest for a single year.

    Args:
        df: LazyFrame containing the data.
        year: Year to process.
        output_dir: Directory to write output.
        n_cpus: Number of CPUs for parallel execution.
        constraints: List of constraint names.
        active_weights: Active weights flag.
        target_active_risk: Target active risk.
        enforce_zero_beta: Whether to enforce zero active beta.
    """
    year_start = dt.date(year, 1, 1)
    year_end = dt.date(year, 12, 31)

    filtered = (
        df.filter(pl.col("date").is_between(year_start, year_end))
        .collect()
    )

    if "benchmark_weight" not in filtered.columns:
        benchmark_weights = (
            sfd.load_benchmark(start=year_start, end=year_end)
            .select(
                "date",
                "barrid",
                pl.col("weight").alias("benchmark_weight"),
            )
        )
        filtered = (
            filtered.join(benchmark_weights, on=["date", "barrid"], how="left")
            .with_columns(pl.col("benchmark_weight").fill_null(0.0))
        )

    print(f"Processing year {year}: {len(filtered)} rows")

    constraint_objects = get_constraints(constraints)

    weights = sfb.dynamic_backtest_parallel(
        data=filtered,
        constraints=constraint_objects,
        target_active_risk=target_active_risk,
        active_weights=active_weights,
        n_cpus=n_cpus,
        enforce_zero_beta=enforce_zero_beta,
    )

    output_path = f"{output_dir}/{year}.parquet"
    weights.write_parquet(output_path)
    print(f"Wrote weights to {output_path}")


def main() -> None:
    """Main entry point for the worker script."""
    parser = argparse.ArgumentParser(
        description="Run MVO backtest for a single year."
    )

    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to parquet file containing the data.",
    )
    parser.add_argument(
        "--target_active_risk",
        type=float,
        required=True,
        help="Target active risk.",
    )
    parser.add_argument(
        "--active_weights",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Active weights flag (pass 'true' or 'false').",
    )
    parser.add_argument(
        "--enforce_zero_beta",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Whether to enforce zero active beta (pass 'true' or 'false').",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output parquet file.",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        required=True,
        help="Number of CPUs for parallel execution.",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        required=True,
        help="List of constraint names (e.g., ZeroBeta ZeroInvestment).",
    )

    args = parser.parse_args()

    df = pl.scan_parquet(args.data_path)

    run_backtest_by_year(
        df=df,
        year=args.year,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        constraints=args.constraints,
        active_weights=args.active_weights,
        target_active_risk=args.target_active_risk,
        enforce_zero_beta=args.enforce_zero_beta,
    )


if __name__ == "__main__":
    main()
