"""Chronological execution of saved targets and isolated SLURM run manifests."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import polars as pl
from sf_quant.simulation import SimulationConfig, SimulationResult, simulate_portfolio


@dataclass
class ExecutionConfig:
    """Decimal end-dated input files, or percent asset data from sf_quant.data.

    Database return interval semantics must be verified on the cluster before a
    historical production run. Files must already contain unshifted decimal returns.
    None disables integration only when explicitly set on the runner config.
    """

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    returns_path: str | None = None
    benchmark_path: str | None = None
    sessions_path: str | None = None
    valuation_end: str | None = None

    def __post_init__(self):
        if isinstance(self.simulation, dict):
            self.simulation = SimulationConfig(**self.simulation)


@dataclass(frozen=True)
class Submission:
    optimization_job_id: str | None
    simulation_job_id: str | None
    run_dir: str
    state: str = "submitted"

    def status(self) -> dict:
        """File-based completion, plus live scheduler state when available."""
        import shutil
        import subprocess

        path = Path(self.run_dir) / "status.json"
        status = (
            json.loads(path.read_text()) if path.exists() else {"state": self.state}
        )
        if shutil.which("sacct") and self.optimization_job_id:
            result = subprocess.run(
                [
                    "sacct",
                    "-j",
                    self.optimization_job_id,
                    "--noheader",
                    "--parsable2",
                    "--format=JobID,State,ExitCode",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            status["scheduler"] = result.stdout
            if status.get("state") != "complete" and any(
                x in result.stdout
                for x in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")
            ):
                status["state"] = "optimization_failed"
        return status


def execute_targets(
    targets,
    asset_returns,
    *,
    benchmark_weights=None,
    execution: ExecutionConfig | None = None,
    output_path=None,
) -> SimulationResult:
    """High-level local replay: no Ray, SLURM, database, or optimization required."""
    execution = execution or ExecutionConfig()
    sessions = (
        pl.read_parquet(execution.sessions_path)["date"].to_list()
        if execution.sessions_path
        else None
    )
    return simulate_portfolio(
        targets,
        asset_returns,
        benchmark_weights=benchmark_weights,
        sessions=sessions,
        config=execution.simulation,
        output_path=output_path,
    )


def optimize_and_simulate(
    data,
    optimizer,
    asset_returns,
    *,
    benchmark_weights=None,
    execution=None,
    output_path=None,
) -> SimulationResult:
    """Local composition; optimizer(data) must return complete target snapshots."""
    return execute_targets(
        optimizer(data),
        asset_returns,
        benchmark_weights=benchmark_weights,
        execution=execution,
        output_path=output_path,
    )


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_run(manifest_path: str | Path) -> SimulationResult:
    """Run only after successful workers; also validates every expected decision."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    if (root / "realized").exists():
        raise FileExistsError(
            "This run already has realized outputs; use a new replay directory"
        )
    manifest = json.loads(manifest_path.read_text())
    _write_json(root / "status.json", {"state": "simulating"})
    try:
        execution = ExecutionConfig(**manifest["execution"])
        target_files = [
            root / "targets" / f"{year}.parquet" for year in manifest["years"]
        ]
        if not all(p.is_file() for p in target_files):
            raise RuntimeError("Annual optimization outputs are incomplete")
        targets = pl.read_parquet(target_files).sort("date", "barrid")
        expected = (
            pl.read_parquet(root / "input.parquet", columns=["date"])["date"].unique().sort().to_list()
        )
        if targets["date"].unique().sort().to_list() != expected:
            raise RuntimeError("Target dates do not match this run's input snapshot")
        if (
            _file_sha256(root / "input.parquet")
            != manifest["input_sha256"]
        ):
            raise RuntimeError("Input snapshot changed after submission")
        start, last = targets["date"].min(), targets["date"].max()
        end = (
            date.fromisoformat(execution.valuation_end)
            if execution.valuation_end
            else last + timedelta(days=14)
        )
        if execution.returns_path:
            returns = pl.read_parquet(execution.returns_path).filter(
                pl.col("date").is_between(start, end)
            )
        else:
            from sf_quant.data import load_assets

            returns = load_assets(
                start=start,
                end=end,
                in_universe=False,
                columns=["date", "barrid", "return"],
            ).with_columns(pl.col("return") / 100)
        if execution.valuation_end is None:
            tail = (
                returns.filter(pl.col("date") > last)["date"].unique().sort().to_list()
            )
            needed = execution.simulation.execution_lag_sessions + 1
            if len(tail) < needed:
                raise ValueError(
                    "Insufficient trailing sessions to realize the final decision"
                )
            returns = returns.filter(pl.col("date") <= tail[needed - 1])
        if execution.benchmark_path:
            benchmark = pl.read_parquet(execution.benchmark_path)
        else:
            from sf_quant.data import load_benchmark

            benchmark = load_benchmark(start=start, end=returns["date"].max())
        benchmark = benchmark.filter(
            pl.col("date").is_between(start, returns["date"].max())
        )
        result = execute_targets(
            targets,
            returns,
            benchmark_weights=benchmark,
            execution=execution,
            output_path=root / "realized",
        )
        _write_json(
            root / "status.json", {"state": "complete", "sessions": result.daily.height}
        )
        return result
    except Exception as exc:
        _write_json(root / "status.json", {"state": "failed", "error": str(exc)})
        raise


def submit_integrated(
    config, data, years, *, dynamic: bool, dry_run: bool = False
) -> Submission:
    """Fresh run ID prevents mixing annual targets or overwriting old results."""
    from sf_backtester.slurm import (
        generate_sbatch_script,
        generate_sbatch_script_dynamic,
        submit_job,
    )

    if not years:
        raise ValueError("Cannot submit an empty backtest")
    execution = config.execution
    sim = replace(
        execution.simulation,
        weight_type=(
            ("active" if config.active_weights else "total")
            if dynamic
            else execution.simulation.weight_type
        ),
    )
    execution = replace(execution, simulation=sim)
    root = Path(config.output_dir).resolve() / "runs" / uuid4().hex
    root.mkdir(parents=True)
    (root / "targets").mkdir()
    logs = root / "logs"
    logs.mkdir()
    data.write_parquet(root / "input.parquet")
    manifest = {
        "years": years,
        "configuration": asdict(config),
        "execution": asdict(execution),
        "input_sha256": _file_sha256(root / "input.parquet"),
        "packages": {name: version(name) for name in ("sf-quant", "sf-backtester", "polars", "numpy")},
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "status.json", {"state": "prepared"})
    isolated = replace(
        config,
        data_path=str(root / "input.parquet"),
        output_dir=str(root / "targets"),
        logs_dir=str(logs),
    )
    generate = generate_sbatch_script_dynamic if dynamic else generate_sbatch_script
    optimize_script = generate(isolated, years)
    finalize_script = f"""#!/bin/bash
#SBATCH --job-name={config.signal_name}_simulation
#SBATCH --output={logs}/simulation_%j.out
#SBATCH --error={logs}/simulation_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem={config.slurm.mem}
#SBATCH --time={config.slurm.time}
set -euo pipefail
source {shlex.quote(str(Path(config.project_root) / ".venv/bin/activate"))}
python -m sf_backtester.scripts.simulate_worker --manifest {shlex.quote(str(root / "manifest.json"))}
"""
    (root / "optimize.sh").write_text(optimize_script)
    (root / "simulate.sh").write_text(finalize_script)
    try:
        result = submit_job(optimize_script, dry_run=dry_run)
    except Exception as exc:
        _write_json(root / "status.json", {"state": "optimization_submission_failed", "error": str(exc)})
        raise
    if dry_run:
        submit_job(finalize_script, dry_run=True)
        _write_json(root / "status.json", {"state": "dry_run"})
        print(f"Dry run saved at {root}; no jobs submitted.")
        return Submission(None, None, str(root), "dry_run")
    job = result.stdout.strip().split(";")[0]
    try:
        final = submit_job(finalize_script, dependency=job)
    except Exception as exc:
        _write_json(
            root / "status.json",
            {
                "state": "finalizer_submission_failed",
                "optimization_job_id": job,
                "error": str(exc),
            },
        )
        raise
    final_job = final.stdout.strip().split(";")[0]
    _write_json(
        root / "submission.json",
        {"optimization_job_id": job, "simulation_job_id": final_job},
    )
    _write_json(
        root / "status.json",
        {
            "state": "submitted",
            "optimization_job_id": job,
            "simulation_job_id": final_job,
        },
    )
    print(
        f"Submitted optimization {job}; simulation {final_job} waits for afterok. Backtest is not complete."
    )
    print(f"Run directory: {root}")
    return Submission(job, final_job, str(root))
