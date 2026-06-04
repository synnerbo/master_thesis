import json  
import os
from pathlib import Path
import sys

import numpy as np  
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import early_stopping

SRC_DIR = Path(__file__).resolve().parents[4] 
sys.path.insert(0, str(SRC_DIR))

from src.data_handling.treebased_ML_processing import get_timeseries_ml_splits
from src.settings.feature_combination import get_feature_cols

import argparse, json
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--params", type=str, default=None, help="Path to JSON with best_params")
    return p.parse_args()

def load_params(params_path: str | None) -> dict:
    if not params_path:
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)

EURUSD_DATA_PATH = SRC_DIR / "data" / "RV_IV_combo_1.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[3]  
OUT_DIR = PROJECT_ROOT / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 2000
args = parse_args() 
VERSION = args.version
BEST_PARAMS = load_params(args.params)
if not args.params:
    print("No --params provided. Skipping run.")
    sys.exit(0)

if not BEST_PARAMS:
    print(f"Params file {args.params} is empty or invalid. Skipping run.")
    sys.exit(0)

EXPLAIN = True
SHAP_START = pd.Timestamp("2018-12-03")
SHAP_END = pd.Timestamp("2022-12-29")

EXPLAIN_DIR = (
    PROJECT_ROOT
    / "save_models"
    / "save_shap"
    / "lightgbm"
    / f"lgbm_shap_{VERSION}_EURUSD"
)
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

SAVE_FI_HISTORY = True
FI_OUTDIR = (
    PROJECT_ROOT
    / "save_models"
    / "builtIn_FeatureImportance"
    / "lightgbm"
    / f"lgbm_feature_importance_{VERSION}_EURUSD"
)
FI_OUTDIR.mkdir(parents=True, exist_ok=True)

FI_TARGET_ALPHAS = {"0.990", "0.975", "0.950", "0.050", "0.025", "0.010"}
SAVE_MODEL_ALPHAS = {"0.990", "0.975", "0.950", "0.050", "0.025", "0.010"}

TRAINED_DIR = PROJECT_ROOT / "trained_secondRun"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
# ===============================================================


def ensure_non_crossing_unified(df: pd.DataFrame) -> pd.DataFrame:
    quantile_cols = [col for col in df.columns if col.startswith("Quantile_")]

    def get_alpha(col_name):
        return float(col_name.split("_")[-1])

    quantile_columns_sorted = sorted(quantile_cols, key=get_alpha)

    for idx in df.index:
        for i in range(len(quantile_columns_sorted) - 1):
            col_current = quantile_columns_sorted[i]
            col_next = quantile_columns_sorted[i + 1]
            val_current = df.at[idx, col_current]
            val_next = df.at[idx, col_next]

            if val_current > val_next:
                df.at[idx, col_next] = val_current + 0.0001

    return df


def combine_processed_data_into_df(window_size=WINDOW_SIZE):
    df = pd.read_csv(EURUSD_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feature_cols = get_feature_cols(VERSION)

    data = get_timeseries_ml_splits(
        df=df,
        feature_cols=feature_cols,
        date_col="date",
        target_col="log_return",
    )

    X_train = data.train.X
    y_train = data.train.y
    X_val = data.validation.X
    y_val = data.validation.y
    X_test = data.test.X
    y_test = data.test.y
    dates_train = data.train.dates
    dates_val = data.validation.dates
    dates_test = data.test.dates

    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train["Date"] = pd.to_datetime(dates_train)
    df_train["Target"] = y_train
    print("Train set shape:", df_train.shape)

    df_val = pd.DataFrame(X_val, columns=feature_cols)
    df_val["Date"] = pd.to_datetime(dates_val)
    df_val["Target"] = y_val
    print("Validation set shape:", df_val.shape)

    df_train = pd.concat([df_train, df_val], axis=0)
    print("df_train shape after merging with df_val:", df_train.shape)

    df_train = df_train.tail(window_size)
    print("df_train shape after keeping only last window_size rows:", df_train.shape)

    print("unique dates in df_train:", df_train["Date"].nunique())
    print("First date in df_train:", df_train["Date"].min())
    print("Last date in df_train:", df_train["Date"].max())

    df_test = pd.DataFrame(X_test, columns=feature_cols)
    df_test["Date"] = pd.to_datetime(dates_test)
    df_test["Target"] = y_test
    print("Test set shape:", df_test.shape)

    df_big = pd.concat([df_train, df_test], axis=0)
    print("df_big shape after merging df_train and df_test:", df_big.shape)

    df_big = df_big.sort_values(by="Date").reset_index(drop=True)
    print("df_big shape after sorting by Date:", df_big.shape)

    return df_big, feature_cols

def train_and_predict_lgb(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    quantile_alpha,
    label=None,
    return_model: bool = False,
):
    """Train and predict using LightGBM for a specific quantile."""
    should_save_model = quantile_alpha in SAVE_MODEL_ALPHAS
    fname = TRAINED_DIR / f"lgbm_{VERSION}_{label}_{quantile_alpha}_{WINDOW_SIZE}.txt"

    # Load only for quantiles you actually save
    if should_save_model and fname.exists():
        print(f"Model {fname} already exists. Loading model...")
        booster = lgb.Booster(model_file=str(fname))
        preds = booster.predict(X_test)

        if return_model:
            class _BoosterWrapper:
                def __init__(self, booster):
                    self.booster_ = booster

            return preds, _BoosterWrapper(booster)

        return preds

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile_alpha,
        metric="quantile",
        boosting_type="gbdt",
        random_state=72,
        verbose=-1,
        **BEST_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Save only selected quantiles
    if should_save_model:
        model.booster_.save_model(str(fname))

    preds = model.predict(X_test)
    return (preds, model) if return_model else preds
# ============================================================


ES_QUANTILES = [0.01, 0.025, 0.05, 0.165, 0.835, 0.95, 0.975, 0.99]
p = 5

quantiles = []
for es in ES_QUANTILES:
    if es < 0.5:
        new_quantiles = [es - (es * i / p) for i in range(p)]
    else:
        new_quantiles = [es + ((1 - es) * i / p) for i in range(p)]
    quantiles.extend(new_quantiles)

quantiles = [f"{q:.3f}" for q in quantiles]


def run_quantile_regression_rolling_window(
    df_big: pd.DataFrame,
    feature_cols: list,
    window_size: int = WINDOW_SIZE,
    horizon: int = 1,
    step: int = 1,
    quantiles=quantiles,
):
    unique_dates = df_big["Date"].sort_values().unique()

    predictions_list = []
    feature_importance_records = []  

    for i in tqdm(
        range(window_size, len(unique_dates) - horizon + 1, step),
        desc="Rolling Window Progress",
    ):
        train_strart_index = i - window_size
        train_end_index = i
        train_dates_range = unique_dates[train_strart_index:train_end_index]

        test_date_index = i + horizon - 1
        if test_date_index >= len(unique_dates):
            break
        test_date_val = unique_dates[test_date_index]

        df_window = df_big[df_big["Date"].isin(train_dates_range)].copy()
        if len(df_window) < 2:
            continue

        df_window.sort_values(by="Date", inplace=True)
        split_index = int(len(df_window) * 0.8)
        df_train = df_window.iloc[:split_index]
        df_val = df_window.iloc[split_index:]

        df_test = df_big[df_big["Date"] == test_date_val].copy()
        if df_test.empty:
            continue

        X_train = df_train[feature_cols]
        y_train = df_train["Target"].values
        X_val = df_val[feature_cols]
        y_val = df_val["Target"].values
        X_test = df_test[feature_cols]
        y_test = df_test["Target"].values
        test_dates = df_test["Date"].values

        # ============================================================
        if EXPLAIN and (SHAP_START <= test_date_val <= SHAP_END):
            ddir = EXPLAIN_DIR / test_date_val.strftime("%Y-%m-%d")
            ddir.mkdir(parents=True, exist_ok=True)

            X_train.to_parquet(ddir / "X_train.parquet", index=False)
            X_val.to_parquet(ddir / "X_val.parquet", index=False)
            X_test.assign(Date=test_dates).to_parquet(ddir / "X_test.parquet", index=False)

            meta = {
                "feature_cols": feature_cols,
                "date": test_date_val.strftime("%Y-%m-%d"),
                "version": VERSION,
                "window_size": window_size,
                "horizon": horizon,
                "step": step,
                "es_p": 5,
            }
            with open(ddir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        # ============================================================

        pred_quantiles = {}

        for alpha_str in tqdm(quantiles, desc="Quantiles Progress", leave=False):
            if SAVE_FI_HISTORY:
                y_pred, fitted_model = train_and_predict_lgb(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    quantile_alpha=alpha_str,
                    label=test_date_val.strftime("%Y-%m-%d"),
                    return_model=True,
                )
            else:
                y_pred = train_and_predict_lgb(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    quantile_alpha=alpha_str,
                    label=test_date_val.strftime("%Y-%m-%d"),
                    return_model=False,
                )
                fitted_model = None

            pred_quantiles[alpha_str] = y_pred

            # ============================================================
        
            if SAVE_FI_HISTORY and fitted_model is not None and alpha_str in FI_TARGET_ALPHAS:
                booster = fitted_model.booster_
                gains = booster.feature_importance(importance_type="gain")
                names = booster.feature_name()

                imp_df = pd.DataFrame({"Feature": names, "importance": gains})
                # align to your feature_cols order
                imp_df = imp_df.set_index("Feature").reindex(feature_cols).fillna(0).reset_index()

                imp_df["share"] = imp_df["importance"] / (imp_df["importance"].sum() + 1e-12)
                imp_df["Date"] = pd.Timestamp(test_date_val)
                imp_df["Alpha"] = alpha_str
                imp_df["Model"] = VERSION

                feature_importance_records.append(imp_df)
            # ============================================================

        for row_index in range(len(df_test)):
            row_dict = {
                "Date": test_dates[row_index],
                "TrueY": y_test[row_index],
            }
            for alpha_str in quantiles:
                row_dict[f"Quantile_{alpha_str}"] = pred_quantiles[alpha_str][row_index]
            predictions_list.append(row_dict)

    # ============================================================
    if SAVE_FI_HISTORY and feature_importance_records:
        fi_df = pd.concat(feature_importance_records, ignore_index=True)
        fi_out = FI_OUTDIR / f"feature_importance_lgbm_{VERSION}_EURUSD_{WINDOW_SIZE}ws.csv"
        fi_df.to_csv(fi_out, index=False)
        print(f"Saved feature importance history to {fi_out}")
    # ============================================================

    df_predictions = pd.DataFrame(predictions_list)
    quantile_cols = [c for c in df_predictions.columns if c.startswith("Quantile_")]
    final_cols = ["Date", "TrueY"] + sorted(quantile_cols)
    df_predictions = df_predictions[final_cols].sort_values(by="Date").reset_index(drop=True)
    return df_predictions


def main_global_rolling_preds():
    df_big, feature_cols = combine_processed_data_into_df(window_size=WINDOW_SIZE)
    print("DF_big shape:", df_big.shape)

    df_predictions = run_quantile_regression_rolling_window(
        df_big=df_big,
        feature_cols=feature_cols,
        window_size=WINDOW_SIZE,
        horizon=1,
        step=1,
        quantiles=quantiles,
    )
    print("Prediction shape:")
    print(df_predictions.shape)

    df_no_crossing = ensure_non_crossing_unified(df_predictions)
    print("Prediction shape after ensuring non-crossing quantiles:")
    print(df_no_crossing.shape)
    return df_no_crossing


def estimate_es_from_predictions(
    df_predictions: pd.DataFrame,
    es_alpha_list: list = ES_QUANTILES,
    p: int = 5,
) -> pd.DataFrame:
    df_out = df_predictions.copy()

    for es_alpha in es_alpha_list:
        if es_alpha < 0.5:
            alpha_subs = [es_alpha - (es_alpha * i / p) for i in range(p)]
        else:
            alpha_subs = [es_alpha + ((1 - es_alpha) * i / p) for i in range(p)]

        alpha_subs_3dec = [f"{q:.3f}" for q in alpha_subs]
        sub_quantile_cols = [f"Quantile_{q}" for q in alpha_subs_3dec]

        existing_cols = [col for col in sub_quantile_cols if col in df_out.columns]
        if not existing_cols:
            print(f"WARNING: No columns found for ES alpha {es_alpha}. Skipping.")
            continue

        df_out[f"ES_{es_alpha:.3f}"] = df_out[existing_cols].mean(axis=1)

    return df_out


final_df = main_global_rolling_preds()
es_df = estimate_es_from_predictions(final_df, es_alpha_list=ES_QUANTILES, p=5)

outfile = OUT_DIR / f"lgbm_predictions_{VERSION}_EURUSD_{WINDOW_SIZE}ws_IVVol.csv"
es_df.to_csv(outfile, index=False)
print(f"Saved predictions to {outfile}")