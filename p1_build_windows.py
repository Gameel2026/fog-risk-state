"""Step 1: build freeze-free walking windows and their features for every recording.
Medication labels are attached as metadata but are not used here.
Output: results/windows_<variant>.pkl and results/recordings.csv
"""
import os
import glob
from math import gcd
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import p_config as C
from p_features import window_features


def load_solution():
    path = os.path.join(C.DATA_ROOT, "official solution.csv")
    head = pd.read_csv(path, nrows=5)
    labs = [c for c in C.LAB if c in head.columns]
    usage = "Usage" if "Usage" in head.columns else None
    idcol = "Id" if "Id" in head.columns else head.columns[0]
    sol = {}
    for k, ch in enumerate(pd.read_csv(path, chunksize=2000000)):
        parts = ch[idcol].astype(str).str.rsplit("_", n=1)
        ser = parts.str[0].to_numpy()
        t = parts.str[1].astype(int).to_numpy()
        fog = ch[labs].to_numpy().max(axis=1) > 0
        ign = ch[usage].astype(str).str.lower().eq("ignored").to_numpy() if usage else np.zeros(len(ch), bool)
        df = pd.DataFrame({"s": ser, "t": t, "fog": fog, "ign": ign})
        for s, g in df.groupby("s"):
            sol.setdefault(s, []).append(g[["t", "fog", "ign"]].to_numpy())
        print("solution chunk", k + 1, flush=True)
    return {s: np.concatenate(v) for s, v in sol.items()}


def recordings():
    meta = []
    for ds in C.DATASETS:
        m = pd.read_csv(os.path.join(C.DATA_ROOT, ds + "_metadata.csv"))
        m["dataset"] = ds
        meta.append(m)
    meta = pd.concat(meta).set_index("Id")
    out = []
    for split in ["train", "public", "private"]:
        for ds in C.DATASETS:
            for f in sorted(glob.glob(os.path.join(C.DATA_ROOT, split, ds, "*.csv"))):
                rid = os.path.splitext(os.path.basename(f))[0]
                if rid in meta.index:
                    out.append((rid, ds, split, f))
    return meta, out


def resample(x, raw, target):
    g = gcd(raw, target)
    return resample_poly(x, target // g, raw // g, axis=0)


def margin_mask(fog, fs, margin_s):
    if margin_s <= 0 or not fog.any():
        return fog.copy()
    m = int(round(margin_s * fs))
    return np.convolve(fog.astype(int), np.ones(2 * m + 1, int), mode="same") > 0


def windows_for(acc, keep, fog, fs, win_s, margin_s):
    w = int(round(win_s * fs))
    step = int(round(win_s * (1 - C.OVERLAP) * fs))
    ok = keep & ~margin_mask(fog, fs, margin_s)
    if len(ok) < w:
        return []
    c = np.concatenate([[0], np.cumsum(~ok)])
    rows = []
    for k, s in enumerate(range(0, len(ok) - w + 1, step)):
        if c[s + w] - c[s] == 0:
            feat = window_features(acc[s:s + w], fs)
            feat["t_start_s"] = s / fs
            feat["win_index"] = k
            rows.append(feat)
    return rows


def main():
    out = C.ensure(C.OUT)
    meta, recs = recordings()
    need_sol = any(sp != "train" for _, _, sp, _ in recs)
    sol = load_solution() if need_sol else {}
    fss = sorted({v["fs"] for v in C.VARIANTS.values()})
    win_rows = {v: [] for v in C.VARIANTS}
    rec_rows = []
    for i, (rid, ds, split, f) in enumerate(recs):
        d = pd.read_csv(f)
        acc = d[["AccV", "AccML", "AccAP"]].to_numpy(float)
        if ds == "tdcsfog":
            acc = acc / C.G
        n = len(d)
        if split == "train":
            fog = (d[C.LAB].to_numpy().max(axis=1) > 0)
            if ds == "defog":
                keep = d["Valid"].astype(bool).to_numpy() & d["Task"].astype(bool).to_numpy()
            else:
                keep = np.ones(n, bool)
        else:
            if rid not in sol:
                continue
            s = sol[rid]
            ok = (s[:, 0] >= 0) & (s[:, 0] < n)
            fog = np.zeros(n, bool)
            ign = np.ones(n, bool)
            fog[s[ok, 0].astype(int)] = s[ok, 1].astype(bool)
            ign[s[ok, 0].astype(int)] = s[ok, 2].astype(bool)
            keep = ~ign
        raw = C.RAW_FS[ds]
        res = {}
        for fs in fss:
            x = resample(acc, raw, fs).astype(np.float64)
            idx = np.minimum(np.round(np.arange(len(x)) * raw / fs).astype(int), n - 1)
            res[fs] = (x, keep[idx], fog[idx])
        info = dict(rec=rid, dataset=ds, split=split, subject=str(meta.loc[rid, "Subject"]),
                    visit=int(meta.loc[rid, "Visit"]), medication=str(meta.loc[rid, "Medication"]).lower(),
                    test=int(meta.loc[rid, "Test"]) if "Test" in meta.columns and pd.notna(meta.loc[rid].get("Test", np.nan)) else -1,
                    minutes=n / raw / 60.0, scored_minutes=keep.sum() / raw / 60.0,
                    fog_pct=100.0 * (fog & keep).sum() / max(keep.sum(), 1))
        for v, p in C.VARIANTS.items():
            x, k, fg = res[p["fs"]]
            rows = windows_for(x, k, fg, p["fs"], p["win"], p["margin"])
            for r in rows:
                r.update(rec=rid)
            win_rows[v].extend(rows)
            if v == "primary":
                info["n_windows"] = len(rows)
        rec_rows.append(info)
        if i % 50 == 0:
            print("recording", i + 1, "/", len(recs), flush=True)
    R = pd.DataFrame(rec_rows)
    R.to_csv(os.path.join(out, "recordings.csv"), index=False)
    for v, rows in win_rows.items():
        W = pd.DataFrame(rows)
        W.to_pickle(os.path.join(out, "windows_%s.pkl" % v))
        print(v, "windows", len(W))
    print(R.groupby(["dataset", "split"]).size())


if __name__ == "__main__":
    main()
