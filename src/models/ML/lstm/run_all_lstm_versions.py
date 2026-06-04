from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC_DIR))

from src.settings.feature_combination import VERSION_TO_FLAGS

LSTM_SCRIPT = SRC_DIR / "src" / "models" / "ML" / "lstm" / "lstm_qreg.py"
PARAMS_DIR = SRC_DIR / "src" / "hyperparameter_tuning" / "lstm" / "optuna" / "lstm"

#VERSIONS = list(VERSION_TO_FLAGS.keys())
VERSIONS = ["IV_RV_D", "IV_RV_W", "IV_SLOPE_CURVE_D", "IV_SLOPE_CURVE_W"] 


def main():
    print("Running LSTM with best tuned hyperparameters for:")
    for v in VERSIONS:
        print(f"  - {v}")
    print()

    for version in VERSIONS:
        params_path = PARAMS_DIR / f"LSTM_QREG_EURUSD_hyperparams_{version}.json"

        if not params_path.exists():
            print(f"Skipping {version}: params file not found -> {params_path}")
            continue

        print("=" * 80)
        print(f"Running version: {version}")
        print(f"Params file: {params_path}")
        print("=" * 80)

        cmd = [
            sys.executable,
            str(LSTM_SCRIPT),
            "--version", version,
            "--params", str(params_path),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"FAILED for version: {version}")
            continue

        print(f"Finished version: {version}\n")

    print("All requested versions processed.")


if __name__ == "__main__":
    main()