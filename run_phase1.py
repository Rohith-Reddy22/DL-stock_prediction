import sys
sys.path.insert(0, ".")

from configs.data_config import DataConfig
from data.pipeline import run_phase1

config = DataConfig(
    ticker="AAPL",
    start_date="2018-01-01",
    end_date="2020-06-04",  # ← change to this
)

result = run_phase1(config)

print("\nPhase 1 complete.")
print(f"  Train : {len(result['train_df'])} rows")
print(f"  Val   : {len(result['val_df'])} rows")
print(f"  Test  : {len(result['test_df'])} rows")
print(f"  Class weights: {result['class_weights']}")