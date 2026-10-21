# sf-backtester

SLURM-based parallel backtesting for quantitative finance. Distributes MVO optimization across compute nodes, processing one year per task.

## Installation

```bash
pip install sf-backtester
```

## Usage

### CLI

```bash
# Run backtest
sf_backtester run config.yml

# Run dynamic backtest
sf_backtester run-dynamic config.yml

# Preview sbatch script without submitting
sf_backtester run config.yml --dry-run
```

### Python API

```python
from sf_backtester import BacktestRunner, BacktestConfig, SlurmConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

config = BacktestConfig(
    signal_name="momentum",
    gamma=50,
    data_path="/path/to/alphas.parquet",
    project_root="/path/to/project",
    byu_email="you@byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
    slurm=slurm_config,
)

runner = BacktestRunner(config)

# Submit to SLURM
runner.submit()
```

Or load from YAML:

```python
from sf_backtester import BacktestRunner

runner = BacktestRunner.from_yaml("config.yml")

runner.submit()
```

You can also pass a DataFrame directly:

```python
from sf_backtester import BacktestRunner
import polars as pl

runner = BacktestRunner.from_yaml("config.yml")

data = pl.read_parquet("alphas.parquet")

runner.submit(data=data)
```

## Configuration

### Standard backtest (YAML)

```yaml
signal_name: momentum
gamma: 500
data_path: /path/to/alphas.parquet
project_root: /path/to/project
byu_email: you@byu.edu

constraints:
  - ZeroBeta
  - ZeroInvestment

slurm:
  n_cpus: 8
  mem: 32G
  time: "03:00:00"
  mail_type: BEGIN,END,FAIL
  max_concurrent_jobs: 31
```

### Dynamic backtest (YAML)

The dynamic backtest solves in active weight space and calibrates each date to the requested annualized
tracking error. It requires benchmark weights and, when `enforce_zero_beta` is true, predicted betas in the
input data. Long-only bounds and zero-sum active weights are enforced by the optimizer.

```yaml
signal_name: momentum
target_active_risk: 0.05
active_weights: true
enforce_zero_beta: true
data_path: /path/to/alphas.parquet
project_root: /path/to/project
byu_email: you@byu.edu

constraints:
  - ZeroBeta
  - ZeroInvestment

slurm:
  n_cpus: 8
  mem: 32G
  time: "06:00:00"
  mail_type: BEGIN,END,FAIL
  max_concurrent_jobs: 31
```

### Available constraints

- `ZeroBeta`
- `ZeroInvestment`
- `UnitBeta`
- `FullInvestment`
- `LongOnly`
- `NoBuyingOnMargin`

## Data format

Input parquet must have columns:
- `date`: Date column
- `barrid`: Asset identifier  
- `alpha`: Alpha signal values
- `predicted_beta`: Predicted beta values
- `benchmark_weight`: Benchmark weight (if doing total portfolio)

Output is one parquet per year in `output_dir/{year}.parquet` containing portfolio weights.

## Publishing
1. Bump the version

```bash
uv version v*.*.*
```

2. Add changes (it can be just the version change)

```bash
git add .
git commit -m "Bumped version."
```

3. Tag the branch

```bash
git tag v*.*.*
```

4. Push to origin

```bash
git push --tags
```

