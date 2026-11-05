"""Central configuration: paths, random seed, label set, experiment sizes.

Everything that influences results lives here so the pipeline is reproducible
from a single place.
"""
from pathlib import Path

# Global random seed. Used for data generation, the train/test split,
# cross-validation folds, model initialisation and bootstrap resampling.
SEED = 42

# Project layout -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DATASET_CSV = DATA_DIR / "cve_synthetic.csv"

# Severity labels in ordinal order (CVSS v3.x qualitative rating scale).
LABELS = ["Low", "Medium", "High", "Critical"]

# Dataset / experiment sizes -----------------------------------------------
N_SAMPLES = 8000          # synthetic vulnerability records to generate
TEST_SIZE = 0.20          # held-out fraction (stratified)
CV_FOLDS = 5              # folds for hyper-parameter search on the train set
N_BOOTSTRAP = 1000        # bootstrap resamples for test-set confidence intervals
LABEL_NOISE = 0.05        # prob. that one hidden CVSS metric is re-scored ("analyst disagreement")
