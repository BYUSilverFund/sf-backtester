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

The dynamic backtest maximizes expected alpha in active weight space subject to an annualized
tracking-error cap and the exact constraints supplied. The cap need not bind when other constraints
limit the portfolio. Specify every intended mandate explicitly; for a long-only portfolio with
zero active investment and beta, use the following configuration:

```yaml
signal_name: momentum
target_active_risk: 0.05
active_weights: true
data_path: /path/to/alphas.parquet
project_root: /path/to/project
byu_email: you@byu.edu

constraints:
  - ZeroInvestment
  - LongOnly
  - ZeroBeta

slurm:
  n_cpus: 8
  mem: 32G
  time: "06:00:00"
  mail_type: BEGIN,END,FAIL
  max_concurrent_jobs: 31
```

Set `active_weights: true` to save active weights `h`, or `false` to save total weights `h + b`.
This changes the output representation, not the optimization constraints. Beta neutrality is
controlled exclusively by including `ZeroBeta` in `constraints`; the former boolean beta flag has
been removed from the configuration and worker interfaces.

### Constraint specifications

Both standard and dynamic configurations accept constraint-name strings, parameterized mappings,
or a mixture of the two. For example, an active extension with normalized benchmark weights can use:

```yaml
constraints:
  - ZeroInvestment
  - name: GrossLeverage
    max_leverage: 1.6
```

The leverage limit must be a finite, positive real number; booleans are invalid. The name-only entry
`GrossLeverage` uses its default limit of `1.0`. With net total investment of 1, a gross cap of 1.6
permits up to 130% long and 30% short exposure; the cap alone does not set net investment or require
the portfolio to reach those exposures.

| Constraint | Rule |
|---|---|
| `ZeroInvestment` | Active/direct weights sum to zero |
| `ZeroBeta` | Active/direct beta exposure equals zero |
| `FullInvestment` | Total weights sum to one |
| `LongOnly` | Total weights are nonnegative |
| `UnitBeta` | Total beta exposure equals one |
| `NoBuyingOnMargin` | Sum of positive total weights is at most one |
| `GrossLeverage` | Sum of absolute total weights is at most `max_leverage` |

In the dynamic optimizer, total-space constraints apply to `h + b` when benchmark weights `b` are
provided and to `h` otherwise. Active-space constraints always apply to `h`. The standard optimizer
applies these rules directly to its optimized weights without benchmark context.

An empty list, `constraints: []`, adds no mandate constraints. In a dynamic backtest, only the risk
cap remains: dollar neutrality, beta neutrality, full investment, and long-only holdings are not
implied. This can produce net exposure and short positions, and the risk cap need not bind. In a
standard backtest, an empty list retains the mean-variance objective and its `gamma` risk penalty.

The same helper validates and constructs constraints before submission and on each worker.
Malformed specifications, unknown names, unsupported parameters, and invalid limits fail before
job preparation or submission. Configurations are validated again before job preparation so edits
made after construction are also checked.

### Worker command-line arguments

Generated SLURM scripts send the specifications as one shell-quoted JSON argument. Both worker
modules accept `--constraints-json` or the legacy names-only `--constraints`, but not both:

```bash
# Parameterized specifications (append other required worker arguments)
--constraints-json '["ZeroInvestment", {"name": "GrossLeverage", "max_leverage": 1.6}]'

# Existing names-only form
--constraints ZeroInvestment ZeroBeta

# Explicitly empty constraints, using either form
--constraints-json '[]'
--constraints
```

JSON may not contain NaN or infinities. For manual shell commands, preserve the single quotes
around the JSON payload so it reaches the worker as one argument.

## Data format

Input parquet must have `date`, `barrid` (asset identifier), and `alpha` columns.
Include `predicted_beta` when using `ZeroBeta` or `UnitBeta`.

For dynamic backtests, include `benchmark_weight` to define the benchmark on each asset/date.
If the column is absent, the worker loads benchmark weights automatically, joins on `date` and
`barrid`, and fills unmatched weights with zero. Loading a benchmark does not add any constraints.
Constraints use weights aligned to the input universe, so ensure that universe and its benchmark
weights match the intended mandate; for example, `ZeroInvestment` produces net total investment
of one only when the aligned benchmark weights sum to one.

For standalone long-short research that applies gross or long-exposure limits directly to `h`,
provide an explicit `benchmark_weight` column of zeros. Otherwise automatic benchmark loading
will make total-space limits apply to `h + b`. With an empty constraint list the benchmark still
loads when absent, but it does not anchor the solution; it only affects total-weight output.

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

