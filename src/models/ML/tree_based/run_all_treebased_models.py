import subprocess
import sys
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.settings.feature_combination import VERSION_TO_FLAGS

OPTUNA_DIR = REPO_ROOT / "src" / "hyperparameter_tuning" / "optuna" / "optuna_IVVol_RVVar_longerTestPeriod_run2"              
TMP_JSON_DIR = OPTUNA_DIR / "_params_json"  
TMP_JSON_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SCRIPTS = {
    "lightgbm": REPO_ROOT / "src" / "models" / "ML" / "tree_based" / "lightGBM_model.py",
    "catboost": REPO_ROOT / "src" / "models" / "ML" / "tree_based" / "catboost_model.py",
    "xgboost":  REPO_ROOT / "src" / "models" / "ML" / "tree_based" / "xgboost_model.py",
}

CSV_NAME = {
    "lightgbm": lambda v: OPTUNA_DIR / f"LightGBM_EURUSD_ES_hyperparams_{v}.csv",
    "catboost": lambda v: OPTUNA_DIR / f"CatBoost_EURUSD_ES_hyperparams_{v}.csv",
    "xgboost":  lambda v: OPTUNA_DIR / f"XGBoost_EURUSD_ES_hyperparams_{v}.csv", 
}

MODELS = ["xgboost"]  # ["lightgbm", "catboost", "xgboost"]


def csv_to_json(csv_path: Path, out_json: Path) -> bool:
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    if df.empty:
        return False
    params = df.iloc[0].to_dict()

    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}

    out_json.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return True


def run_one(model_key: str, version: str, params_json: Path):
    cmd = [
        sys.executable, str(MODEL_SCRIPTS[model_key]),
        "--version", version,
        "--params", str(params_json),
    ]
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    versions = list(VERSION_TO_FLAGS.keys())

    for model_key in MODELS:
        for version in versions:
            csv_path = CSV_NAME[model_key](version)

            params_json = TMP_JSON_DIR / f"{model_key}_{version}.json"
            ok = csv_to_json(csv_path, params_json)

            if not ok:
                print(f"[SKIP] Missing/empty params CSV: {csv_path}")
                continue

            # Load params for printing
            with open(params_json, "r", encoding="utf-8") as f:
                params_dict = json.load(f)

            print("\n" + "-" * 80)
            print(f"MODEL   : {model_key}")
            print(f"VERSION : {version}")
            print("HYPERPARAMETERS:")
            for k, v in params_dict.items():
                print(f"  {k}: {v}")
            print("-" * 80)

            run_one(model_key, version, params_json)


if __name__ == "__main__":
    main()