# %%
import os
import sys
import json
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from pathlib import Path
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--db", type=str, default=None)  
args = parser.parse_args()

VERSION = args.version
N_TRIALS = args.trials

REPO_ROOT = Path(__file__).resolve().parents[2]  
SRC_DIR    = REPO_ROOT / "src"                    
sys.path.insert(0, str(SRC_DIR))

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "optuna"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# =============================================================================
# 1. Load Data
# =============================================================================
SRC_DIR = Path(__file__).resolve().parents[2]  # -> .../src
sys.path.insert(0, str(SRC_DIR))
from src.data_handling.treebased_ML_processing import get_timeseries_ml_splits
from src.settings.feature_combination import get_feature_cols

EURUSD_DATA_PATH = SRC_DIR / "data" / "RV_IV_combo_1.csv"



df = pd.read_csv(EURUSD_DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
# get feature colums based on VERSION
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


X_train = pd.DataFrame(X_train, columns=feature_cols)
X_val = pd.DataFrame(X_val, columns=feature_cols)

# Set correct types
# Make sure everything is float
X_train = X_train.astype("float64")
X_val = X_val.astype("float64")

print("\n=== Feature configuration ===")
print(f"VERSION: {VERSION}")
print(f"Number of features: {len(feature_cols)}")
print("Feature columns (in order):")
for i, col in enumerate(feature_cols):
    print(f"{i:02d}: {col}")
print("=============================\n")

print("\n========== TRAIN DATA VALIDATION ==========")


print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("y_train mean/std:", float(np.mean(y_train)), float(np.std(y_train)))
print("y_val mean/std:", float(np.mean(y_val)), float(np.std(y_val)))
print("Feature columns used:", list(X_train.columns))

print("===========================================\n")


print("Data loaded.")

CONFIDENCE_LEVELS = [0.90, 0.95, 0.98]
lower_quantiles = [float(np.round((1 - cl) / 2, 5)) for cl in CONFIDENCE_LEVELS]
upper_quantiles = [float(np.round(1 - lq, 5)) for lq in lower_quantiles]
n_es_quantiles = 5
lower_es_quantiles = [
    float(np.round(small_q, 5))
    for q in lower_quantiles
    for small_q in np.linspace(0.0, q, n_es_quantiles + 1)[1:]
]
upper_es_quantiles = [float(np.round(1.0 - q, 5)) for q in lower_es_quantiles]

all_quantiles = sorted(
    set(lower_quantiles + upper_quantiles + lower_es_quantiles + upper_es_quantiles)
)
print(f"Quantiles used for tuning ({len(all_quantiles)}): {all_quantiles}")


# =============================================================================
# 2. Define Quantile Loss
# =============================================================================
def quantile_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.maximum(alpha * residual, (alpha - 1) * residual).mean()


# =============================================================================
# 3. Define Optuna Objective
# =============================================================================
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15, log=True),
        "depth": trial.suggest_int("depth", 2, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.5, 3.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 2.0),
        "border_count": trial.suggest_int("border_count", 128, 256),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 5
        ),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "random_seed": 72,
        "early_stopping_rounds": 30,
        "verbose": False,
    }
 
 
    print(f"Parameters: {json.dumps(params, indent=2)}")

    quantile_losses = {}

    for alpha in all_quantiles:
        print(f"Training for quantile: {alpha:.5f}")

        model = CatBoostRegressor(
            **params,
            loss_function=f"Quantile:alpha={alpha}",
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )

        preds = model.predict(X_val)
        loss = quantile_loss(y_val, preds, alpha)
        quantile_losses[alpha] = loss

    for alpha, loss in quantile_losses.items():
        trial.set_user_attr(f"QL_{alpha:.5f}", loss)

    avg_loss = np.mean(list(quantile_losses.values()))
    return avg_loss



# =============================================================================
# 5. Run Optuna Study
# =============================================================================


from optuna.exceptions import DuplicatedStudyError

if args.db is None:
    storage_url = f"sqlite:///{(OUT_DIR / 'optuna.db').as_posix()}"
else:
    storage_url = f"sqlite:///{Path(args.db).as_posix()}"

study_name = f"catboost_tuning_{VERSION}_IV_Vol_longerTestPeriod_run2"

try:
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
    )
except DuplicatedStudyError:
    print(f"[SKIP] Study already exists: {study_name}")
    sys.exit(0)

study.optimize(objective, n_trials=N_TRIALS)
# =============================================================================
# 6. Save Results
# =============================================================================
print("Best parameters:")
print(study.best_params)

best_params_df = pd.DataFrame([study.best_params])
best_params_df.to_csv(OUT_DIR / f"CatBoost_EURUSD_ES_hyperparams_{VERSION}.csv", index=False)

print("Done.")