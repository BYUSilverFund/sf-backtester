# Integrated drift-aware backtests

Both `BacktestRunner.submit()` and `BacktestDynamicRunner.submit()` now default to
optimization followed by chronological simulation. Optimizer objectives and
Ray per-date optimization remain unchanged.

```python
from sf_backtester import ExecutionConfig
from sf_quant.simulation import SimulationConfig

config.execution = ExecutionConfig(simulation=SimulationConfig(
    frequency="monthly", execution_lag_sessions=1, allow_short=False))
submission = runner.submit()  # Submission means queued, NOT completed
print(submission.run_dir)
print(submission.status())
```

Dynamic runners derive active/total weight type from `config.active_weights`.
Fixed-gamma runners require `simulation.weight_type="active"` explicitly for an
active sleeve; otherwise they execute the saved values as total portfolio weights.
Set `allow_short=True` explicitly for shorts/borrowing under the simplified model.
See sf-quant/DRIFT_AWARE.md for accounting and financing limits.

Equivalent YAML (nested within existing Backtest[Dynamic]Config):

```yaml
execution:
  simulation:
    frequency: weekly
    rebalance_day: first
    execution_lag_sessions: 1
    weight_type: active
    allow_short: false
    transaction_cost_bps: 0
  returns_path: null
  benchmark_path: null
  sessions_path: null
  valuation_end: null
```

Optional returns/benchmark/sessions paths are Parquet files; returns must be
DECIMAL end-dated total returns. Without files, the finalizer loads sf_quant asset
data WITHOUT universe filtering, divides asset percent returns by 100, and loads
full daily benchmark compositions. The vendor's end-date semantics must still be
verified on the cluster. These repos contain no source ingestion code or local
production database, and their legacy return calculations disagree about timing.
If valuation_end is omitted, include enough trailing sessions to realize the last
decision (lag+1 sessions). An explicit valuation_end defines a deliberate horizon.

Every new integrated run is isolated:

```
weights/<signal>/<gamma-or-risk>/runs/<unique-id>/
  manifest.json       # configuration and input SHA256
  input.parquet       # immutable alpha snapshot
  targets/<year>.parquet
  optimize.sh
  simulate.sh
  submission.json    # array and finalizer job IDs
  status.json
  logs/
  realized/{daily,holdings,trades}.parquet
  realized/config.json
```

Existing `weights/<signal>/<risk>/<year>.parquet` files are PRESERVED and reusable;
new run targets are deliberately namespaced to prevent mixing configurations.
Consumers of newly submitted jobs must use Submission.run_dir/targets or realized,
not a broad recursive glob over all runs. This is an intentional output migration.
Optimization workers write annual files atomically. The simulator is submitted
with `--dependency=afterok:<array_job_id>` and validates every expected target date
and the alpha snapshot hash before it runs. Shells fail fast. Failed/cancelled
optimization arrays never unlock simulation. Live SLURM execution is not tested
on Windows; subprocess arguments and finalizer execution are covered locally.

`sf_backtester status RUN_DIR` reads completion/failure and sacct where available.
After a simulation failure caused by corrected data/environment, use
`sf_backtester finalize RUN_DIR` to reuse targets. Existing realized outputs are
never overwritten. For failed optimization tasks, requeue the failed SLURM array
tasks against the SAME snapshot (or submit a fresh isolated run); after successful
recovery finalize explicitly if the original dependency became unsatisfiable.
Do not edit the immutable input or merge targets from unrelated runs.

For explicit optimization-only compatibility set `execution=None` / YAML
`execution: null`. Legacy target-return APIs remain available separately.

Local, database-free entry points:

```python
from sf_backtester import execute_targets, optimize_and_simulate
result = execute_targets(targets, returns, benchmark_weights=benchmark,
                         execution=execution)
# Or optimize(data) -> targets, then exactly the same execution:
result = optimize_and_simulate(data, optimize, returns,
                              benchmark_weights=benchmark, execution=execution)
```

Run the deterministic example with a NEW output directory:
`python example/4_local_drift.py --output /tmp/drift-example-unique`.
Test from a shared workspace: `python -m pytest sf-quant/tests sf-backtester/tests -q`.
Install editable packages into the SAME project venv activated in generated SLURM
scripts; checking out source alone does not update a pre-existing pip installation.
