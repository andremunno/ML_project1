from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
SUBMISSION_PATH = ROOT / "data" / "submissions" / "submission.csv"
RESULTS_DIR = ROOT / "experiments" / "results"

# --- Reproducibility ---
SEED = 38