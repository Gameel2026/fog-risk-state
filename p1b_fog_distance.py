"""Amendment 01, step 1: distance of every primary window to annotated FoG (label-free).
Rebuilds the 64-Hz FoG and scored-sample masks exactly as p1_build_windows.py and measures, for each window
of windows_primary.pkl, the distance in seconds to the nearest FoG sample, to the next FoG onset,
since the last FoG offset, and to the nearest unscored sample. No medication information is read.
Output: results/fog_distance_primary.csv (and its sha256 in the log).
"""
import os
import hashlib
import numpy as np
import pandas as pd
import p_config as C
import p1_build_windows as P1

FS = C.VARIANTS["primary"]["fs"]
W_N = int(round(C.VARIANTS["primary"]["win"] * FS))
STEP = int(round(C.VARIANTS["primary"]["win"] * (1 - C.OVERLAP) * FS))


def masks(rid, ds, split, f, sol):
    if split == "train":
        cols = C.LAB + (["Valid", "Task"] if ds == "defog" else [])
        d = pd.read_csv(f, usecols=cols)
        n = len(d)
        fog = d[C.LAB].to_numpy().max(axis=1) > 0
        keep = (d["Valid"].astype(bool).to_numpy() & d["Task"].astype(bool).to_numpy()) if ds == "defog" else np.ones(n, bool)
    else:
        n = len(pd.read_csv(f, usecols=["AccV"]))
        s = sol[rid]
        ok = (s[:, 0] >= 0) & (s[:, 0] < n)
        fog = np.zeros(n, bool)
        ign = np.ones(n, bool)
        fog[s[ok, 0].astype(int)] = s[ok, 1].astype(bool)
        ign[s[ok, 0].astype(int)] = s[ok, 2].astype(bool)
        keep = ~ign
    raw = C.RAW_FS[ds]
    n64 = int(np.ceil(n * FS / raw))  # same length as resample_poly output
    idx = np.minimum(np.round(np.arange(n64) * raw / FS).astype(int), n - 1)
    return fog[idx], keep[idx]


def dist_to(points, s, e):
    """Distance (samples) from window [s, e) to the nearest point outside it; inf if none."""
    if len(points) == 0:
        return np.full(len(s), np.inf)
    i = np.searchsorted(points, s)
    before = np.where(i > 0, s - points[np.maximum(i - 1, 0)], np.inf)
    j = np.searchsorted(points, e)
    after = np.where(j < len(points), points[np.minimum(j, len(points) - 1)] - (e - 1), np.inf)
    inside = (j - i) > 0
    return np.where(inside, 0.0, np.minimum(before, after))


def main():
    W = pd.read_pickle(os.path.join(C.OUT, "windows_primary.pkl"))[["rec", "win_index", "t_start_s"]]
    assert "medication" not in W.columns
    meta, recs = P1.recordings()
    need = set(W["rec"])
    recs = [r for r in recs if r[0] in need]
    sol = P1.load_solution() if any(sp != "train" for _, _, sp, _ in recs) else {}
    rows = []
    for k, (rid, ds, split, f) in enumerate(recs):
        if split != "train" and rid not in sol:
            continue
        fog, keep = masks(rid, ds, split, f, sol)
        g = W[W["rec"] == rid]
        s = g["win_index"].to_numpy() * STEP
        e = s + W_N
        assert np.all(np.abs(g["t_start_s"].to_numpy() - s / FS) < 1e-9)
        fi = np.flatnonzero(fog)
        f_int = fog.astype(int)
        on = np.flatnonzero(np.diff(np.r_[0, f_int]) == 1)
        off = np.flatnonzero(np.diff(np.r_[f_int, 0]) == -1) + 1  # first sample after an episode
        j = np.searchsorted(on, e)
        nxt = np.where(j < len(on), (on[np.minimum(j, max(len(on) - 1, 0))] - e) if len(on) else np.inf, np.inf)
        i = np.searchsorted(off, s, side="right")
        last = np.where(i > 0, (s - off[np.maximum(i - 1, 0)]) if len(off) else np.inf, np.inf)
        rows.append(pd.DataFrame({
            "rec": rid, "win_index": g["win_index"].to_numpy(),
            "fog_dist_s": dist_to(fi, s, e) / FS,
            "t_to_next_onset_s": np.asarray(nxt, float) / FS,
            "t_since_last_offset_s": np.asarray(last, float) / FS,
            "unscored_dist_s": dist_to(np.flatnonzero(~keep), s, e) / FS}))
        if k % 100 == 0:
            print("recording", k + 1, "/", len(recs), flush=True)
    D = pd.concat(rows, ignore_index=True)
    p = os.path.join(C.OUT, "fog_distance_primary.csv")
    D.to_csv(p, index=False)
    print("windows", len(D), "of", len(W))
    print(D.describe().T)
    print("sha256 of fog_distance_primary.csv:", hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
