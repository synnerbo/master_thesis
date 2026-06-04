from dataclasses import dataclass
from functools import cached_property

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  
EURUSD_PATH = ROOT / "data" / "RV_IV_combo_1.csv"

TRAIN_VALIDATION_SPLIT = "2014-02-05"
VALIDATION_TEST_SPLIT = "2016-02-05"

@dataclass
class DataColumnsSplit:
    X: np.ndarray                 
    col_names: list[str]        
    y: np.ndarray                
    dates: List[pd.Timestamp]     

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": self.dates,
                **{f: self.X[:, i] for i, f in enumerate(self.col_names)},
                "log_return": self.y,
            }
        ).set_index("date")


@dataclass
class TrainValTestData:
    train: DataColumnsSplit
    validation: DataColumnsSplit
    test: DataColumnsSplit


def get_timeseries_ml_splits(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    date_col: str = "date",
    target_col: str = "log_return",
    train_validation_split: str = TRAIN_VALIDATION_SPLIT,
    validation_test_split: str = VALIDATION_TEST_SPLIT,
    
) -> TrainValTestData:
    if df is None or len(df) == 0:
        raise ValueError("Pass a non-empty cleaned DataFrame.")

    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing date_col='{date_col}'. Found: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target_col='{target_col}'. Found: {df.columns.tolist()}")
    

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True) # drop rows with missing features or target

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[target_col].to_numpy(dtype=np.float32)
    d_all = pd.to_datetime(df[date_col]).tolist()

    d_series = pd.to_datetime(d_all)
    train_cut = pd.to_datetime(train_validation_split)
    val_cut = pd.to_datetime(validation_test_split)

    train_mask = d_series < train_cut
    val_mask   = (d_series >= train_cut) & (d_series < val_cut)
    test_mask  = d_series >= val_cut

    def pack(mask) -> DataColumnsSplit:
        idx = np.where(mask)[0]
        return DataColumnsSplit(
            X=X_all[idx],
            col_names=feature_cols,
            y=y_all[idx],
            dates=[d_series[i] for i in idx],
        )

    return TrainValTestData(
        train=pack(train_mask),
        validation=pack(val_mask),
        test=pack(test_mask),
    )

def quick_split_check(data, n=3):
    def show(name, part):
        dates = list(map(pd.to_datetime, part.dates))
        n_samples = len(dates)
        d0 = dates[0].date() if n_samples else "—"
        d1 = dates[-1].date() if n_samples else "—"
        print(f"\n[{name}] samples={n_samples}  range={d0} → {d1}")
        if n_samples:
            preview = pd.DataFrame(part.X[:n, :min(part.X.shape[1], 5)],
                       columns=part.col_names[:min(part.X.shape[1], 5)])
            preview["date"] = [d.date() for d in dates[:n]]
            preview["y"] = part.y[:n]
            print(preview)
    show("Train", data.train)
    show("Validation", data.validation)
    show("Test", data.test)

