"""Environment check for the SPARK project, run it from the project root."""
import importlib
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# import name : package name used by pip
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "xgboost": "xgboost",
    "streamlit": "streamlit",
    "google.protobuf": "protobuf",
    "ipykernel": "ipykernel",
}
OPTIONAL = {"imblearn": "imbalanced-learn"}

BUNDLES = ["Decision_Tree", "Random_Forest", "KNN", "Logistic_regression", "SVM", "XGBoost"]
TABLES = ["v2_baselines.csv", "v2_fresh_folds.csv", "v2_holdout_results.csv",
          "v2_portuguese_results.csv", "v2_six_models_full_metrics.csv"]


def check_packages(packages):
    missing = []
    for module, pip_name in packages.items():
        try:
            importlib.import_module(module)
            try:
                version = metadata.version(pip_name)
            except metadata.PackageNotFoundError:
                version = "version unknown"
            print(f"  ok       {pip_name:<18} {version}")
        except Exception as error:
            print(f"  MISSING  {pip_name:<18} {type(error).__name__}: {error}")
            missing.append(pip_name)
    return missing


def check_files():
    expected = ["app/app.py", "src/spark_core.py",
                "data/raw/student-mat.csv", "data/raw/student-por.csv",
                "data/processed/v2_math_dev.csv",
                "data/processed/v2_math_holdout_do_not_open.csv"]
    expected += [f"models/v2_bundle_{name}.joblib" for name in BUNDLES]
    expected += [f"outputs/tables/{name}" for name in TABLES]

    absent = [path for path in expected if not (ROOT / path).exists()]
    for path in expected:
        print(f"  {'ok     ' if path not in absent else 'MISSING'}  {path}")
    return absent


def check_scoring():
    try:
        import spark_core
    except Exception as error:
        print(f"  FAILED   importing spark_core: {type(error).__name__}: {error}")
        return

    print("  ok       spark_core imports")
    path = ROOT / "models" / "v2_bundle_Random_Forest.joblib"
    if not path.exists():
        print("  skipped  scoring test, Random Forest bundle not found")
        return

    import pandas as pd
    bundle = spark_core.load_bundle(path)
    sample = pd.DataFrame({"G1": [11, 6], "G2": [9, np.nan]})
    print(spark_core.score_students(sample, bundle).to_string())


def main():
    print("Python in use:", sys.executable)
    if ".venv" not in sys.executable or "OneDrive" in sys.executable:
        print("  WARNING  this is not the project's .venv, activate it first")

    print("\nRequired packages")
    missing = check_packages(REQUIRED)
    print("\nOptional packages, only needed for the version 1 notebooks")
    check_packages(OPTIONAL)
    print("\nProject files")
    absent = check_files()
    print("\nScoring test, two invented students")
    check_scoring()

    print("\nSummary")
    if missing:
        print("  Install missing packages with:")
        print("  python -m pip install " + " ".join(missing))
    if absent:
        print(f"  {len(absent)} project files are missing, see the list above")
    if not missing and not absent:
        print("  Everything needed is in place")


if __name__ == "__main__":
    main()
    