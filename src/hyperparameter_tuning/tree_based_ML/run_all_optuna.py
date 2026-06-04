import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tio4900_master_thesis
sys.path.insert(0, str(REPO_ROOT))

from src.settings.feature_combination import VERSION_TO_FLAGS


HERE = Path(__file__).resolve().parent
(HERE / "optuna").mkdir(exist_ok=True)

SCRIPTS = [
    HERE / "lightGBM_optuna.py",
    HERE / "catboost_optuna.py",
    HERE / "xgboost_optuna.py",
]

def run_script(script_path: Path, version: str, trials: int):
    # DB per model (one file per script)
    db_path = HERE / "optuna" / f"db_{script_path.stem}.db"

    cmd = [
        sys.executable, str(script_path),
        "--version", version,
        "--trials", str(trials),
        "--db", str(db_path),
    ]
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)

def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    versions = list(VERSION_TO_FLAGS.keys())

    for v in versions:
        for script in SCRIPTS:
            run_script(script, v, trials)

if __name__ == "__main__":
    main()

"""
# Running the code
# Write in terminal: 

python src/hyperparameter_tuning/run_all_optuna.py 100

# (or replace 100 with the desired number of trials for each tuning run)
"""