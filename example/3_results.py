"""Read REALIZED results after Submission.status() reports complete.

Replace RUN_DIR with the run directory returned by runner.submit().
Targets remain in RUN_DIR/targets; holdings and returns are in RUN_DIR/realized.
"""
from pathlib import Path
import polars as pl
import sf_quant.performance as sfp

RUN_DIR = Path("weights/momentum/0.05/runs/REPLACE_WITH_RUN_ID")
daily = pl.read_parquet(RUN_DIR / "realized/daily.parquet")
returns = daily.select("date", "return")
print(sfp.generate_returns_summary_table(returns))
print(sfp.get_trade_turnover_stats(daily))
sfp.generate_returns_chart(returns, title="Drift-aware Momentum")
