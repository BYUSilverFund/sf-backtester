from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig, ExecutionConfig
from sf_quant.simulation import SimulationConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

backtest_config = BacktestConfig(
    signal_name='momentum',
    data_path='momentum_alphas.parquet',
    gamma=50,
    execution=ExecutionConfig(simulation=SimulationConfig(weight_type="active", allow_short=True)),
    project_root="/home/amh1124/Projects/sf-backtester",
    byu_email="amh1124@byu.edu",
    constraints=[
        "ZeroBeta",
        "ZeroInvestment"
    ],
    slurm=slurm_config,
)

backtest_runner = BacktestRunner(backtest_config)
backtest_runner.submit()
