"""Label-free check of the frozen gait gate (added 4 November 2026, after review comments).
Recomputes the Otsu threshold without each patient in turn, and from each source cohort alone,
and counts how many primary windows would change their gate decision. No medication label is read.
Output: results/gate_loo_check.json
"""
import os
import json
import numpy as np
import pandas as pd
import p_config as C
from p2_freeze_gate import otsu


def main():
    W = pd.read_pickle(os.path.join(C.OUT, "windows_primary.pkl"))
    R = pd.read_csv(os.path.join(C.OUT, "recordings.csv"), dtype={"subject": str}, usecols=["rec", "dataset", "subject"])
    W = W.merge(R, on="rec")
    assert "medication" not in W.columns
    thr0 = json.load(open(os.path.join(C.OUT, "frozen_gate.json")))["logP_loc_threshold"]
    x, ac = W["logP_loc"].to_numpy(), W["gate_ac"].to_numpy()
    pid = (W["dataset"] + ":" + W["subject"]).to_numpy()
    base = (x > thr0) & (ac > C.AC_GATE)
    thr, changed = [], 0
    for p in np.unique(pid):
        t = otsu(x[pid != p])
        thr.append(t)
        m = pid == p
        changed += int((base[m] != ((x[m] > t) & (ac[m] > C.AC_GATE))).sum())
    out = {"frozen_threshold": thr0, "leave_one_patient_out_threshold_min": float(min(thr)),
           "leave_one_patient_out_threshold_max": float(max(thr)), "windows_total": int(len(W)),
           "windows_changed_leave_one_patient_out": changed}
    for src in C.DATASETS:
        t = otsu(x[(W["dataset"] == src).to_numpy()])
        m = (W["dataset"] != src).to_numpy()
        out["source_only_%s_threshold" % src] = t
        out["windows_changed_in_target_when_source_is_%s" % src] = int((base[m] != ((x[m] > t) & (ac[m] > C.AC_GATE))).sum())
        out["windows_in_target_when_source_is_%s" % src] = int(m.sum())
    json.dump(out, open(os.path.join(C.OUT, "gate_loo_check.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
