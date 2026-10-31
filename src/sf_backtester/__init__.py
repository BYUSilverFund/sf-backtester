"""sf-backtester: SLURM-based parallel backtesting for quantitative finance."""

from sf_backtester.config import BacktestConfig, BacktestDynamicConfig, SlurmConfig
from sf_backtester.runner import BacktestRunner, BacktestDynamicRunner
from sf_backtester.execution import ExecutionConfig, Submission, execute_targets, optimize_and_simulate

__version__ = "0.2.0"
__all__ = ["BacktestConfig", "BacktestDynamicConfig", "BacktestRunner", "BacktestDynamicRunner", "SlurmConfig"]
__all__ += ["ExecutionConfig", "Submission", "execute_targets", "optimize_and_simulate"]
