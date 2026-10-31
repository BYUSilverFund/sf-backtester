from dataclasses import replace
from datetime import date
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import polars as pl
import pytest
from sf_backtester import (
    BacktestConfig,
    BacktestDynamicConfig,
    BacktestDynamicRunner,
    BacktestRunner,
    ExecutionConfig,
    SlurmConfig,
    execute_targets,
    optimize_and_simulate,
)
from sf_backtester.execution import finalize_run
from sf_quant.simulation import SimulationConfig, simulate_portfolio

D = [date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]


def panel(dates, values, name="weight"):
    return pl.DataFrame(
        {
            "date": [d for d in dates for _ in range(2)],
            "barrid": ["A", "B"] * len(dates),
            name: np.asarray(values, dtype=float).ravel(),
        }
    )


def setup(tmp_path, dynamic=False):
    data = panel(D[:2], [[0.02, 0], [0.02, 0]], "alpha")
    returns = panel(D, [[0, 0], [0.1, 0], [0, 0.2], [0, 0]], "return")
    bm = panel(D, [[0.5, 0.5]] * 4)
    returns.write_parquet(tmp_path / "returns.parquet")
    bm.write_parquet(tmp_path / "benchmark.parquet")
    execution = ExecutionConfig(
        returns_path=str(tmp_path / "returns.parquet"),
        benchmark_path=str(tmp_path / "benchmark.parquet"),
    )
    kwargs = dict(
        signal_name="test",
        data_path=str(tmp_path / "input.parquet"),
        project_root=str(tmp_path),
        byu_email="test@example.org",
        constraints=["ZeroInvestment"],
        slurm=SlurmConfig(),
        execution=execution,
    )
    if dynamic:
        return (
            data,
            returns,
            bm,
            BacktestDynamicConfig(
                **kwargs, target_active_risk=0.05, active_weights=True
            ),
        )
    return data, returns, bm, BacktestConfig(**kwargs, gamma=1)


@pytest.mark.parametrize("dynamic", [False, True])
def test_runner_submits_afterok_and_finalizes_exact_local_replay(
    tmp_path, monkeypatch, dynamic
):
    data, returns, bm, cfg = setup(tmp_path, dynamic)
    calls = []

    def submit(script, dry_run=False, dependency=None):
        calls.append((script, dependency))
        return CompletedProcess([], 0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr("sf_backtester.slurm.submit_job", submit)
    runner = BacktestDynamicRunner(cfg) if dynamic else BacktestRunner(cfg)
    job = runner.submit(data=data)
    assert job.state == "submitted"
    assert job.optimization_job_id == "101" and job.simulation_job_id == "102"
    assert calls[0][1] is None and calls[1][1] == "101"
    assert "simulate_worker" in calls[1][0]
    assert "set -euo pipefail" in calls[0][0]
    root = Path(job.run_dir)
    targets = (
        panel(D[:2], [[0.1, -0.1]] * 2) if dynamic else panel(D[:2], [[0.6, 0.4]] * 2)
    )
    for year in (2023, 2024):
        targets.filter(pl.col("date").dt.year() == year).write_parquet(
            root / "targets" / f"{year}.parquet"
        )
    result = finalize_run(root / "manifest.json")
    expected = simulate_portfolio(
        targets,
        returns,
        benchmark_weights=bm,
        config=replace(
            cfg.execution.simulation, weight_type="active" if dynamic else "total"
        ),
    )
    assert result.daily.equals(expected.daily)
    assert result.holdings.equals(expected.holdings)
    assert result.trades.equals(expected.trades)
    assert job.status()["state"] == "complete"
    assert (root / "realized/daily.parquet").is_file()
    # Separate submissions never reuse annual files, even with identical parameters.
    second = runner.submit(data=data)
    assert second.run_dir != job.run_dir


def test_finalizer_rejects_partial_years_and_can_recover(tmp_path, monkeypatch):
    data, _, _, cfg = setup(tmp_path)
    monkeypatch.setattr("sf_backtester.slurm.submit_job", lambda *a, **k: None)
    job = BacktestRunner(cfg).submit(data=data, dry_run=True)
    root = Path(job.run_dir)
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize_run(root / "manifest.json")
    assert job.status()["state"] == "failed"
    for year, d in zip((2023, 2024), D[:2]):
        panel([d], [[0.6, 0.4]]).write_parquet(root / "targets" / f"{year}.parquet")
    finalize_run(root / "manifest.json")
    assert job.status()["state"] == "complete"


def test_yaml_round_trip_and_legacy_optout(tmp_path):
    _, _, _, cfg = setup(tmp_path)
    cfg.execution.simulation = SimulationConfig(
        frequency="monthly", transaction_cost_bps=7
    )
    cfg.to_yaml(tmp_path / "config.yaml")
    reloaded = BacktestConfig.from_yaml(tmp_path / "config.yaml")
    assert reloaded.execution.simulation == cfg.execution.simulation
    cfg.execution = None
    cfg.to_yaml(tmp_path / "config.yaml")
    assert BacktestConfig.from_yaml(tmp_path / "config.yaml").execution is None


def test_actual_optimizer_composition_matches_independent_expectation():
    from sf_quant.optimizer.constraints import FullInvestment, LongOnly
    from sf_quant.optimizer.optimizers import mve_optimizer

    data = panel(D[:1], [[0.02, 0]], "alpha")
    returns = panel(D[:3], [[0, 0], [0.1, 0], [0, 0.2]], "return")

    def optimizer(frame):
        weights = mve_optimizer(
            ids=["A", "B"],
            alphas=np.array([0.02, 0]),
            factor_exposures=np.zeros((2, 1)),
            factor_covariance=np.eye(1),
            specific_risk=np.array([0.1, 0.1]),
            gamma=1,
            constraints=[FullInvestment(), LongOnly()],
        )
        return weights.with_columns(pl.lit(D[0]).alias("date"))

    execution = ExecutionConfig(simulation=SimulationConfig(execution_lag_sessions=0))
    high = optimize_and_simulate(data, optimizer, returns, execution=execution)
    low = execute_targets(optimizer(data), returns, execution=execution)
    np.testing.assert_allclose(high.daily["nav"], [1, 1.06, 1.14], atol=1e-8)
    assert high.daily.equals(low.daily)


def test_empty_submission_rejected(tmp_path):
    data, _, _, cfg = setup(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        BacktestRunner(cfg).submit(data=data.head(0))


def test_sbatch_uses_afterok_argument(monkeypatch):
    from sf_backtester.slurm import submit_job

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, "123\n", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    submit_job("#!/bin/bash\ntrue\n", dependency="123")
    assert "--dependency=afterok:123" in calls[0]


def test_finalizer_refuses_mutated_input_snapshot(tmp_path, monkeypatch):
    data, _, _, cfg = setup(tmp_path)
    monkeypatch.setattr("sf_backtester.slurm.submit_job", lambda *a, **k: None)
    job = BacktestRunner(cfg).submit(data=data, dry_run=True)
    root = Path(job.run_dir)
    for year, d in zip((2023, 2024), D[:2]):
        panel([d], [[0.6, 0.4]]).write_parquet(root / "targets" / f"{year}.parquet")
    data.with_columns(pl.col("alpha") + 1).write_parquet(root / "input.parquet")
    with pytest.raises(RuntimeError, match="changed"):
        finalize_run(root / "manifest.json")
