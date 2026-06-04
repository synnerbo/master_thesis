

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
import numpy as np
import pandas as pd

TRAIN_VALIDATION_SPLIT = "2014-02-05"
VALIDATION_TEST_SPLIT = "2016-02-05"



@dataclass
class LabelledDataSet:
    X: np.ndarray # shape (n_samples, sequence_length, n_features)
    col_names: list[str] # names of the features in the last dimension of X (n_features)
    y: np.ndarray # target vaariable 
    y_dates: list[pd.Timestamp] # dates corresponding to the target variable

    def filter_by_dates(
            self, from_date: pd.Timestamp | None, to_date: pd.Timestamp | None
    ) -> "LabelledDataSet":
        mask = np.ones(len(self.y_dates), dtype=bool)
        y_dates = pd.to_datetime(self.y_dates).normalize()
        if from_date is not None:
            from_ts = pd.Timestamp(from_date).normalize()
            mask &= y_dates >= from_ts
        if to_date is not None:
            to_ts = pd.Timestamp(to_date).normalize()
            mask &= y_dates <= to_ts
        return LabelledDataSet(
            X=self.X[mask],
            col_names=self.col_names,
            y=self.y[mask],
            y_dates=[d for d, m in zip(self.y_dates, mask) if m],
        )
    
    @cached_property
    def dates(self) -> np.ndarray:
        return np.array(self.y_dates)

    

@dataclass
class ProcessedData:
    train: LabelledDataSet
    validation: LabelledDataSet
    test: LabelledDataSet

    def _full(self) -> LabelledDataSet:
        return LabelledDataSet(
            X=np.concatenate([self.train.X, self.validation.X, self.test.X], axis=0),
            col_names=self.train.col_names,
            y=np.concatenate([self.train.y, self.validation.y, self.test.y], axis=0),
            y_dates=list(self.train.y_dates) + list(self.validation.y_dates) + list(self.test.y_dates),
        )

    def get_training_set_for_date(self, prediction_date: pd.Timestamp) -> LabelledDataSet:
        full = self._full()
        return full.filter_by_dates(None, prediction_date - pd.Timedelta(days=1))

    def get_test_set_for_date(self, prediction_date: pd.Timestamp, to_date=None) -> LabelledDataSet:
        full = self._full()
        return full.filter_by_dates(prediction_date, to_date)

def _make_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    lookback_days: int
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    date_list: list[pd.Timestamp] = []

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df[date_col]).tolist()

    for t in range(lookback_days, len(df)):
        # do not include the current day t in the features, only the past lookback_days days up to t-1
        # takes the lookback rows for all the chosen feature columns, resulting in a 2D array of shape (lookback_days, n_features)
        X_list.append(feats[t-lookback_days : t, :]) # shape (lookback_days, n_features)
        # features from past 30 days to predict the target variable at day t
        y_list.append(float(y[t])) 
        date_list.append(dates[t])

    X_array = np.asarray(X_list, dtype=np.float32) 
    y_array = np.asarray(y_list, dtype=np.float32)
    return X_array, y_array, date_list

def get_lstm_train_test_new(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback_days: int = 30,
    *,
    date_col: str = "date",
    target_col: str = "log_return",
) -> ProcessedData:
    """
    Prepare data for LSTM.

    If transform_to_logvar=True (recommended):
      - variance-like features (RV + other realized measures in decimal variance) -> log(var + eps)
      - IV features that are DAILY VOL in decimal -> log(vol^2 + eps)  (i.e. log daily variance)
      - signed features (can be negative) -> sign(x)*log1p(|x|)
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    needed = [date_col, target_col] + feature_cols
    for col in needed:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' is missing from the DataFrame. Available columns: {df.columns.tolist()}"
            )

    if len(df) <= lookback_days:
        raise ValueError(
            f"DataFrame has {len(df)} rows, which is not enough for lookback_days={lookback_days}."
        )

    # Build sequences
    X, y, y_dates = _make_lstm_sequences(df, feature_cols, target_col, date_col, lookback_days)

    dates = pd.to_datetime(y_dates)

    # define split dates
    train_val_split = pd.Timestamp(TRAIN_VALIDATION_SPLIT)
    val_test_split = pd.Timestamp(VALIDATION_TEST_SPLIT)

    train_mask = dates < train_val_split
    val_mask = (dates >= train_val_split) & (dates < val_test_split)
    test_mask = dates >= val_test_split

    train = LabelledDataSet(
        X=X[train_mask],
        col_names=feature_cols,
        y=y[train_mask],
        y_dates=list(dates[train_mask])
    )

    validation = LabelledDataSet(
        X=X[val_mask],
        col_names=feature_cols,
        y=y[val_mask],
        y_dates=list(dates[val_mask])
    )

    test = LabelledDataSet(
        X=X[test_mask],
        col_names=feature_cols,
        y=y[test_mask],
        y_dates=list(dates[test_mask])
    )

    return ProcessedData(train=train, validation=validation, test=test)
    