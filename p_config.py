"""Shared settings. Values follow PLAN.md (version 1)."""
import os

DATA_ROOT = r"D:\New research fog work\kaggle_fog\Competition Dataset"   # local copy of the FoG detection contest data
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
G = 9.80665
RAW_FS = {"tdcsfog": 128, "defog": 100}
DATASETS = ["tdcsfog", "defog"]
LAB = ["StartHesitation", "Turn", "Walking"]

# primary pipeline
FS = 64
WIN_S = 6.0
OVERLAP = 0.5
MARGIN_S = 2.0
MIN_WINDOWS = 3
AC_GATE = 0.3          # autocorrelation step-peak threshold (fixed in the plan)
LR_C = 1.0
N_PERM = 1000
N_BOOT = 1000
SEED = 42

# variants built in the same pass (exploratory / sensitivity)
VARIANTS = {
    "primary": dict(fs=64, win=6.0, margin=2.0),
    "margin0": dict(fs=64, win=6.0, margin=0.0),
    "margin5": dict(fs=64, win=6.0, margin=5.0),
    "win10": dict(fs=64, win=10.0, margin=2.0),
    "fs32": dict(fs=32, win=6.0, margin=2.0),
    "fs16": dict(fs=16, win=6.0, margin=2.0),
}

FEATURES = ["logP_loc", "logP_frz", "log_ratio", "rms_V", "rms_ML", "rms_AP", "f_step", "step_peak_rel",
            "HR_V", "HR_AP", "HR_ML", "step_reg", "stride_reg", "symmetry", "log_jerk"]
AMPLITUDE_FEATURES = ["logP_loc", "logP_frz", "rms_V", "rms_ML", "rms_AP", "log_jerk"]
BASELINE_FEATURE = "logP_loc"


def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p
