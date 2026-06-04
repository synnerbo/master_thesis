# run_all_lstm_optuna_versions.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Project root:
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.settings.feature_combination import VERSION_TO_FLAGS

OPTUNA_SCRIPT = PROJECT_ROOT / "src" / "hyperparameter_tuning" / "lstm" / "lstm_optuna.py"

# Number of trials per version
N_TRIALS = 100

# shared database path
DB_PATH = PROJECT_ROOT / "src" / "hyperparameter_tuning" / "lstm" / "optuna" / "lstm" / "optuna_lstm_qreg.db"


def main():
    #versions = list(VERSION_TO_FLAGS.keys())
    versions = ["IV_RV_D", "IV_RV_W", "IV_SLOPE_CURVE_D", "IV_SLOPE_CURVE_W"]  

    print("Will run Optuna for these versions:")
    for v in versions:
        print(f"  - {v}")
    print()

    for version in versions:
        print("=" * 80)
        print(f"Running version: {version}")
        print("=" * 80)

        cmd = [
            sys.executable,
            str(OPTUNA_SCRIPT),
            "--version",
            version,
            "--trials",
            str(N_TRIALS),
            "--db",
            str(DB_PATH),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nFAILED for version: {version}")
            print("Stopping script.")
            sys.exit(result.returncode)

        print(f"\nFinished version: {version}\n")

    print("All versions completed.")


if __name__ == "__main__":
    main()