"""Amendment 02, step 1: build freeze-free windows and features for the DeFOG recordings in train/notype.
Same rules as p1_build_windows.py (primary variant only). Label-free: no medication information is read.
Outputs: results/windows_notype.pkl and results/recordings_notype.csv (and their sha256 in the log).
"""
import os
import hashlib
import numpy as np
import pandas as pd
import p_config as C
import p1_build_windows as P1

COLS = ["Time", "AccV", "AccML", "AccAP", "Event", "Valid", "Task"]


def flag(s):
    return s.astype(str).str.strip().str.lower().eq("true").to_numpy()


def main():
    meta = pd.read_csv(os.path.join(C.DATA_ROOT, "defog_metadata.csv"), usecols=["Id"], dtype=str)
    used = set(pd.read_csv(os.path.join(C.OUT, "recordings.csv"), usecols=["rec"], dtype=str)["rec"])
    ids = [i for i in meta["Id"].astype(str) if i not in used]
    tasks = pd.read_csv(os.path.join(C.DATA_ROOT, "tasks.csv"))
    p = C.VARIANTS["primary"]
    raw = C.RAW_FS["defog"]
    rows, recs = [], []
    for k, rid in enumerate(ids):
        f = os.path.join(C.DATA_ROOT, "train", "notype", rid + ".csv")
        if not os.path.exists(f):
            print("missing file", rid)
            continue
        d = pd.read_csv(f)
        assert list(d.columns) == COLS, (rid, list(d.columns))
        ev = d["Event"].to_numpy()
        assert set(np.unique(ev)).issubset({0, 1}), (rid, np.unique(ev))
        n = len(d)
        t = tasks[tasks["Id"].astype(str) == rid]
        if len(t):
            assert t["End"].max() <= n / raw + 1.0, (rid, t["End"].max(), n / raw)
        acc = d[["AccV", "AccML", "AccAP"]].to_numpy(float)
        fog = ev > 0
        keep = flag(d["Valid"]) & flag(d["Task"])
        x = P1.resample(acc, raw, p["fs"]).astype(np.float64)
        idx = np.minimum(np.round(np.arange(len(x)) * raw / p["fs"]).astype(int), n - 1)
        ws = P1.windows_for(x, keep[idx], fog[idx], p["fs"], p["win"], p["margin"])
        for r in ws:
            r["rec"] = rid
        rows.extend(ws)
        recs.append(dict(rec=rid, minutes=n / raw / 60.0, scored_minutes=keep.sum() / raw / 60.0,
                         fog_pct=100.0 * (fog & keep).sum() / max(keep.sum(), 1), n_windows=len(ws)))
        print("recording", k + 1, "/", len(ids), rid, "windows", len(ws), flush=True)
    W = pd.DataFrame(rows)
    R = pd.DataFrame(recs)
    pw = os.path.join(C.OUT, "windows_notype.pkl")
    pr = os.path.join(C.OUT, "recordings_notype.csv")
    W.to_pickle(pw)
    R.to_csv(pr, index=False)
    print("recordings", len(R), "windows", len(W))
    for q in (pw, pr):
        print("sha256 of %s: %s" % (os.path.basename(q), hashlib.sha256(open(q, "rb").read()).hexdigest()))


if __name__ == "__main__":
    main()
