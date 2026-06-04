from pathlib import Path
import pandas as pd


"""
This script aligns prediction files from different model implementations to a
common target-date convention. The different date labels arise from
implementation choices (not errors in the forecasting models). 

LSTM prediction files already use target-date alignment, while the remaining models
require a one-period date shift. After processing, all prediction files share 
a consistent and correct date index, enabling direct comparison and evaluation across models.
"""

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "src" / "predictions" / "higher_oss"
OUTPUT_DIR = PROJECT_ROOT / "src" / "predictions" / "final_aligned_main_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def is_lstm_file(path: Path) -> bool:
    return "LSTM" in path.stem.upper()

# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------
csv_files = sorted(INPUT_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}\n")

for csv_path in csv_files:
    print(f"Processing: {csv_path.name}")

    df = pd.read_csv(csv_path)

    if "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    else:
        raise ValueError(f"{csv_path.name}: no 'Date' or 'date' column found.")

    df[date_col] = pd.to_datetime(df[date_col])

    if is_lstm_file(csv_path):
        # LSTM already uses target-date alignment
        # Drop first row to align with shifted non-LSTM files
        if len(df) < 2:
            print(f"  Skipping {csv_path.name}: fewer than 2 rows.")
            continue
        df = df.iloc[1:].copy()

    else:
        # Non-LSTM: relabel each row with the next trading date
        df[date_col] = df[date_col].shift(-1)

        # Drop last row only, since shifted date becomes NA there
        df = df.iloc[:-1].copy()

    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    out_path = OUTPUT_DIR / csv_path.name
    df.to_csv(out_path, index=False)

    print(f"  Saved aligned file to: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df[date_col].min().date()} -> {df[date_col].max().date()}\n")

print("Done.")