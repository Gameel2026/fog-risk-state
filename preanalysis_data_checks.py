"""Pre-analysis data checks (no medication-state result is computed).
1) units: mean acceleration magnitude per dataset (expect about 1 g or about 9.81 m/s^2)
2) DeFOG Valid / Task columns: share of samples, and where FoG annotations fall relative to them
3) number of FoG-free 6-s windows (50% overlap, 2-s margin) per recording, by dataset and medication state
Run from anywhere:  python preanalysis_data_checks.py
"""
import os
import glob
import numpy as np
import pandas as pd

# local folder holding the FoG detection contest data (Zenodo doi:10.5281/zenodo.12565292)
ROOT = r"D:\New research fog work\kaggle_fog\Competition Dataset"
OUT = os.path.dirname(os.path.abspath(__file__))
FS = {"tdcsfog": 128, "defog": 100}
WIN_S, STEP_S, MARGIN_S = 6.0, 3.0, 2.0


def windows(ok, fs):
    w, s = int(WIN_S * fs), int(STEP_S * fs)
    if len(ok) < w:
        return 0
    c = np.concatenate([[0], np.cumsum(~ok)])
    starts = np.arange(0, len(ok) - w + 1, s)
    return int(((c[starts + w] - c[starts]) == 0).sum())


def main():
    rows, unit_rows, vt = [], [], []
    for ds in ["tdcsfog", "defog"]:
        meta = pd.read_csv(os.path.join(ROOT, ds + "_metadata.csv")).set_index("Id")
        files = sorted(glob.glob(os.path.join(ROOT, "train", ds, "*.csv")))
        fs = FS[ds]
        for i, f in enumerate(files):
            rid = os.path.basename(f)[:-4]
            if rid not in meta.index:
                continue
            d = pd.read_csv(f)
            acc = d[["AccV", "AccML", "AccAP"]].to_numpy(float)
            mag = np.linalg.norm(acc, axis=1)
            unit_rows.append(dict(dataset=ds, rec=rid, mean_mag=float(np.mean(mag)), median_mag=float(np.median(mag))))
            fog = (d[["StartHesitation", "Turn", "Walking"]].sum(axis=1) > 0).to_numpy()
            if ds == "defog":
                valid, task = d["Valid"].astype(bool).to_numpy(), d["Task"].astype(bool).to_numpy()
                vt.append(dict(rec=rid, n=len(d), valid=valid.mean(), task=task.mean(), valid_and_task=(valid & task).mean(),
                               fog_samples=int(fog.sum()), fog_in_valid=int((fog & valid).sum()), fog_in_task=int((fog & task).sum()),
                               fog_outside_valid_task=int((fog & ~(valid & task)).sum())))
                keep = valid & task
            else:
                keep = np.ones(len(d), bool)
            m = int(MARGIN_S * fs)
            near = fog.copy()
            if fog.any():
                k = np.convolve(fog.astype(int), np.ones(2 * m + 1, int), mode="same") > 0
                near = k
            ok = keep & ~near
            rows.append(dict(dataset=ds, rec=rid, subject=meta.loc[rid, "Subject"], visit=meta.loc[rid, "Visit"],
                             medication=meta.loc[rid, "Medication"], minutes=len(d) / fs / 60,
                             fog_pct=100 * fog.mean(), free_windows=windows(ok, fs)))
            if i % 50 == 0:
                print(ds, i, "/", len(files), flush=True)
    R = pd.DataFrame(rows)
    U = pd.DataFrame(unit_rows)
    R.to_csv(os.path.join(OUT, "preanalysis_windows_per_recording.csv"), index=False)
    U.to_csv(os.path.join(OUT, "preanalysis_units.csv"), index=False)
    if vt:
        pd.DataFrame(vt).to_csv(os.path.join(OUT, "preanalysis_defog_valid_task.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n1) UNITS: mean acceleration magnitude per dataset")
    print(U.groupby("dataset")[["mean_mag", "median_mag"]].describe().round(3).to_string())
    if vt:
        V = pd.DataFrame(vt)
        print("\n2) DEFOG Valid/Task: mean share of samples, and FoG samples inside/outside")
        print(V[["valid", "task", "valid_and_task"]].mean().round(3).to_string())
        print("FoG samples total %d, in Valid %d, in Task %d, outside Valid&Task %d" % (
            V.fog_samples.sum(), V.fog_in_valid.sum(), V.fog_in_task.sum(), V.fog_outside_valid_task.sum()))
    print("\n3) FoG-free 6-s windows per recording (2-s margin)")
    print(R.groupby(["dataset", "medication"]).free_windows.describe().round(1).to_string())
    print(R.groupby(["dataset", "medication"]).fog_pct.describe().round(1).to_string())
    print("recordings with < 3 free windows:")
    print(R[R.free_windows < 3].groupby(["dataset", "medication"]).size().to_string())
    paired = []
    for ds, g in R[R.free_windows >= 3].groupby("dataset"):
        p = g.groupby(["subject", "visit"]).medication.nunique()
        paired.append((ds, int(p[p == 2].reset_index().subject.nunique())))
    print("patients still paired (ON and OFF in the same visit) after excluding recordings with < 3 windows:", paired)


if __name__ == "__main__":
    main()
