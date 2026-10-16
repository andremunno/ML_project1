from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
METADATA_PATH = ROOT / "data" / "metadata" / "variables_BRFSS_2015.csv"
SUBMISSION_PATH = ROOT / "data" / "submissions" / "submission.csv"
RESULTS_DIR = ROOT / "experiments" / "results"

# --- Reproducibility ---
SEED = 38