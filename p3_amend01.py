"""Amendment 01: post hoc exploratory analyses (see AMENDMENT_01.md).
A  task-matched comparison of the original model
B  personal-baseline model (visit-centred, state-balanced), within-patient evaluation only
C  windows away from FoG (descriptive)
D  structural reference for recording-level LOPO AUC
Inputs : results/windows_primary.pkl, recordings.csv, frozen_gate.json, pred_tdcsfog.csv, pred_defog.csv,
         results.json, fog_distance_primary.csv, <DATA_ROOT>/tasks.csv
Outputs: results/results_amend01.json, results/amend01_patient_values.csv, results/fig_amend01.png
Run    : python p3_amend01.py --dryrun   (labels shuffled within patient-visit, seed 77; for checking only)
         python p3_amend01.py            (the single real run)
"""
import os
import sys
import json
import hashlib
import platform
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import roc_auc_score
import p_config as C
import p3_analysis as P

DRY = "--dryrun" in sys.argv
TASKS = os.path.join(C.DATA_ROOT, "tasks.csv")
FOGD = os.path.join(C.OUT, "fog_distance_primary.csv")
EXCL = ("MB", "Rest")          # MiniBest items and rest periods (home)
MIN_STRAT_WIN = 2              # windows per state in a home stratum
MIN_REC_WIN = 3                # windows per recording
FALLBACK_MIN_PATIENTS = 20     # home: merge cued/uncued and the two hotspots below this
FOG_DIST = (30.0, 60.0)        # s; 30 main, 60 sensitivity
C_GATE = {"defog": 15, "tdcsfog": 20}
C2_GATE, C2_PRE, C2_FAR, C2_MIN = 15, 30.0, 60.0, 2
N_BOOT = 1000
SEED_BOOT, SEED_DRY = 42, 77
TOL = 1e-6


def log(*a):
    P.log(*a)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


# ----------------------------------------------------------------------------- tasks and strata

def assign_task(W):
    """Exact task name if the 6-s window lies wholly inside one interval of tasks.csv, else None."""
    T = pd.read_csv(TASKS)
    T["Id"] = T["Id"].astype(str)
    groups = dict(tuple(T.groupby("Id")))
    out = np.array([None] * len(W), dtype=object)
    n_multi = 0
    ts = W["t_start_s"].to_numpy()
    for rec, idx in W.groupby("rec").indices.items():
        if rec not in groups:
            continue
        t = groups[rec]
        s = ts[idx][:, None]
        inside = (t["Begin"].to_numpy()[None, :] <= s) & (t["End"].to_numpy()[None, :] >= s + C.WIN_S)
        n_multi += int((inside.sum(1) > 1).sum())
        has = inside.any(1)
        out[idx[has]] = t["Task"].to_numpy()[inside[has].argmax(1)]
    return out, n_multi


def home_clean(W):
    """DeFOG training recordings, task assigned, MiniBest/Rest removed, >= 3 windows per recording."""
    W = W[(W["dataset"] == "defog") & (W["split"] == "train")].copy()
    W["task"], n_multi = assign_task(W)
    n_all = len(W)
    W = W[W["task"].notna()]
    n_task = len(W)
    W = W[~W["task"].astype(str).str.startswith(EXCL)]
    n = W.groupby("rec").size()
    W = W[W["rec"].isin(n[n >= MIN_REC_WIN].index)]
    info = {"windows_train": n_all, "windows_inside_a_task": n_task, "windows_after_MB_Rest_and_min3": int(len(W)),
            "windows_in_two_tasks": n_multi}
    return W, info


def paired_pvs(df):
    n = df.groupby("pv")["y"].nunique()
    return n[n == 2].index


def stratum_cells(df, ds, collapse=False):
    """One row per (pid, pv, stratum) with OFF (1) and ON (0) scores = median window logit."""
    df = df[df["pv"].isin(paired_pvs(df))].copy()
    if ds == "tdcsfog":
        df["s"] = df["test"].astype(int)
        r = df.groupby(["pid", "pv", "s", "y", "rec"])["z"].median().reset_index()
        return r.groupby(["pid", "pv", "s", "y"])["z"].mean().unstack("y").dropna()
    s = df["task"].astype(str)
    if collapse:
        s = s.str.replace("-C$", "", regex=True).str.replace("Hotspot2", "Hotspot1")
    df["s"] = s
    n = df.groupby(["pid", "pv", "s", "y"]).size().unstack("y").fillna(0)
    cell = df.groupby(["pid", "pv", "s", "y"])["z"].median().unstack("y")
    if 0 not in n.columns or 1 not in n.columns:
        return cell.iloc[0:0]
    ok = (n[0] >= MIN_STRAT_WIN) & (n[1] >= MIN_STRAT_WIN)
    return cell[ok.reindex(cell.index).fillna(False)].dropna()


def home_cells(df):
    cell = stratum_cells(df, "defog")
    used = "exact tasks"
    if cell.index.get_level_values("pid").nunique() < FALLBACK_MIN_PATIENTS:
        cell = stratum_cells(df, "defog", collapse=True)
        used = "fallback: cued merged with uncued, Hotspot1 merged with Hotspot2"
    return cell, used


def matched_diff(cell):
    return (cell[1] - cell[0]).groupby(level=["pid", "pv"]).mean().groupby(level="pid").mean()


def pairwise_acc(cell):
    a = (cell[1] > cell[0]).astype(float) + 0.5 * (cell[1] == cell[0])
    return a.groupby(level="pid").mean()


def unmatched_diff(df):
    """OFF - ON of recording scores (median window logit) averaged within pv and state, mean over pvs."""
    df = df[df["pv"].isin(paired_pvs(df))]
    r = df.groupby(["pid", "pv", "y", "rec"])["z"].median().groupby(level=["pid", "pv", "y"]).mean().unstack("y").dropna()
    if not len(r):
        return pd.Series(dtype=float)
    return (r[1] - r[0]).groupby(level="pid").mean()


# ----------------------------------------------------------------------------- statistics

def boot(values, fn, rng):
    v = np.asarray(values, float)
    if len(v) < 3:
        return [None, None]
    out = [fn(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def describe(d, rng, test=False):
    d = d.dropna()
    res = {"n_patients": int(len(d))}
    if not len(d):
        return res
    res.update(mean=float(d.mean()), median=float(d.median()), positive=int((d > 0).sum()),
               zeros=int((d == 0).sum()), median_ci95=boot(d, np.median, rng),
               share_positive_ci95=boot(d, lambda x: float(np.mean(x > 0)), rng))
    if len(d) >= 5 and d.std(ddof=1) > 0:
        res["dz"] = float(d.mean() / d.std(ddof=1))
        p = float(stats.wilcoxon(d, alternative="greater").pvalue)
        res["wilcoxon_p_one_sided" if test else "nominal_p_one_sided"] = p
    return res


def describe_acc(a, rng):
    a = a.dropna()
    if not len(a):
        return {"n_patients": 0}
    return {"n_patients": int(len(a)), "mean": float(a.mean()), "mean_ci95": boot(a, np.mean, rng)}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def structural_auc(Wd):
    """Intercept-only LOPO: each patient scored by the OFF share of the other patients (no sensor feature)."""
    share = Wd.groupby("pid")["y"].mean()
    n = len(share)
    r = Wd.drop_duplicates("rec")
    return float(roc_auc_score(r["y"], r["pid"].map((share.sum() - share) / (n - 1))))


# ----------------------------------------------------------------------------- model B

def center_pv(X, rec, pv):
    df = pd.DataFrame(X)
    cols = list(range(X.shape[1]))
    df["rec"] = rec
    df["pv"] = pv
    c = df.groupby(["pv", "rec"])[cols].median().groupby(level="pv").mean()
    return X - c.reindex(pv).to_numpy()


def balanced_weights(W):
    k = W.groupby(["pv", "y"])["rec"].transform("size").to_numpy(float)
    npv = W.groupby("pid")["pv"].transform("nunique").to_numpy(float)
    s = 1.0 / (k * 2.0 * npv)
    return s * len(s) / s.sum()


def lopo_personal(Wd):
    Wd = Wd[Wd["pv"].isin(paired_pvs(Wd))].reset_index(drop=True)
    X = Wd[C.FEATURES].to_numpy(float)
    y = Wd["y"].to_numpy()
    pid = Wd["pid"].to_numpy()
    rec, pv = Wd["rec"].to_numpy(), Wd["pv"].to_numpy()
    z = np.empty(len(Wd))
    for g in np.unique(pid):
        te = pid == g
        tr = ~te
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xc = center_pv(np.where(np.isfinite(X), X, med), rec, pv)
        prep = P.Prep(Xc[tr])
        w = P.fit_lr(prep(Xc[tr]), y[tr].astype(float), balanced_weights(Wd[tr]))
        z[te] = prep(Xc[te]) @ w[:-1] + w[-1]
    return Wd.assign(z=z)


# ----------------------------------------------------------------------------- C

def c_far(df, dist, ds, rng):
    out = {}
    for thr in FOG_DIST:
        g = df[df["fog_dist_s"] >= thr]
        n = g.groupby("rec").size()
        g = g[g["rec"].isin(n[n >= MIN_REC_WIN].index)]
        d = unmatched_diff(g)
        key = "far_%ds" % int(thr)
        out[key] = {"n_patients": int(len(d)), "windows": int(len(g)), "feasible": bool(len(d) >= C_GATE[ds])}
        if out[key]["feasible"]:
            out[key].update(describe(d, rng))
    return out


def c2_pre_vs_far(df, ds, rng):
    by = ["pid", "y", "rec"] + (["task"] if ds == "defog" else [])
    df = df.assign(kind=np.where(df["t_to_next_onset_s"] < C2_PRE, "pre",
                                 np.where(df["fog_dist_s"] >= C2_FAR, "far", "other")))
    df = df[df["kind"] != "other"]
    rows = []
    for key, g in df.groupby(by):
        a, b = g[g["kind"] == "pre"]["z"], g[g["kind"] == "far"]["z"]
        if len(a) >= C2_MIN and len(b) >= C2_MIN:
            rows.append(dict(pid=key[0], y=key[1], diff=float(a.median() - b.median())))
    R = pd.DataFrame(rows)
    out = {}
    for y, lab in [(0, "on"), (1, "off")]:
        d = R[R["y"] == y].groupby("pid")["diff"].mean() if len(R) else pd.Series(dtype=float)
        out[lab] = {"n_patients": int(len(d)), "feasible": bool(len(d) >= C2_GATE)}
        if out[lab]["feasible"]:
            out[lab].update(describe(d, rng))
    return out


# ----------------------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(SEED_BOOT)
    R = pd.read_csv(os.path.join(C.OUT, "recordings.csv"), dtype={"subject": str})
    gate = json.load(open(os.path.join(C.OUT, "frozen_gate.json")))
    W, _ = P.load_variant("primary", gate, R)
    if DRY:
        W["y"] = P.swap_within_pv(W, np.random.default_rng(SEED_DRY))
    res = {"amendment": "AMENDMENT_01.md", "dry_run": DRY,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__},
           "inputs_sha256": {"tasks.csv": sha(TASKS), "fog_distance_primary.csv": sha(FOGD),
                             "frozen_gate.json": sha(os.path.join(C.OUT, "frozen_gate.json"))},
           "parameters": {"exclude_prefixes": list(EXCL), "min_windows_per_state_in_home_stratum": MIN_STRAT_WIN,
                          "min_windows_per_recording": MIN_REC_WIN, "fallback_min_patients": FALLBACK_MIN_PATIENTS,
                          "fog_distance_s": list(FOG_DIST), "c_gates": C_GATE, "c2": [C2_GATE, C2_PRE, C2_FAR, C2_MIN],
                          "n_boot": N_BOOT, "seeds": [SEED_BOOT, SEED_DRY], "lr_C": C.LR_C}}

    # ---- A: regenerate original LOPO logits and check reproduction
    Z = {}
    for ds in C.DATASETS:
        Wd = W[W["dataset"] == ds].reset_index(drop=True)
        z = P.lopo_logits(P.arrays(Wd, C.FEATURES))
        Z[ds] = Wd.assign(z=z)
        if not DRY:
            old = pd.read_csv(os.path.join(C.OUT, "pred_%s.csv" % ds)).set_index("rec")["prob"]
            new = P.rec_table(Wd, z).set_index("rec")["prob"]
            dev = float(np.max(np.abs(new.reindex(old.index) - old)))
            res.setdefault("reproduction_max_abs_diff", {})[ds] = dev
            if not np.isfinite(dev) or dev > TOL:
                res["stopped"] = "reproduction failed for %s (max diff %g)" % (ds, dev)
                json.dump(res, open(os.path.join(C.OUT, "results_amend01.json"), "w"), indent=2)
                raise SystemExit(res["stopped"])
    log("A logits ready")

    lab = Z["tdcsfog"]
    home_all = Z["defog"]
    home, hinfo = home_clean(home_all)
    res["home_window_flow"] = hinfo
    pv_vals = []

    A = {}
    cell_lab = stratum_cells(lab, "tdcsfog")
    cell_home, used = home_cells(home)
    res["home_strata_rule"] = used
    for name, cell in [("lab", cell_lab), ("home", cell_home)]:
        d = matched_diff(cell)
        A[name] = {"cells": int(len(cell)), "patient_visits": int(cell.index.get_level_values("pv").nunique()),
                   "matched_difference": describe(d, rng, test=True),
                   "pairwise_accuracy": describe_acc(pairwise_acc(cell), rng)}
        pv_vals.append(pd.DataFrame({"analysis": "A_" + name, "pid": d.index, "value": d.values}))
    A["home_decomposition_unmatched"] = {
        "all_recordings": describe(unmatched_diff(home_all), rng),
        "train_recordings": describe(unmatched_diff(home_all[home_all["split"] == "train"]), rng),
        "train_without_MiniBest_Rest": describe(unmatched_diff(home), rng),
        "test_recordings_MiniBest_not_removable": describe(unmatched_diff(home_all[home_all["split"] != "train"]), rng)}
    A["lab_unmatched"] = describe(unmatched_diff(lab), rng)
    res["A"] = A
    log("A done")

    # ---- B: personal-baseline model
    B = {}
    for name, df, ds in [("lab", lab, "tdcsfog"), ("home", home, "defog")]:
        Wb = lopo_personal(df.drop(columns=["z"]))
        cell = stratum_cells(Wb, ds) if ds == "tdcsfog" else home_cells(Wb)[0]
        d = matched_diff(cell)
        B[name] = {"patients": int(Wb["pid"].nunique()), "patient_visits": int(Wb["pv"].nunique()),
                   "cells": int(len(cell)), "matched_difference": describe(d, rng, test=True),
                   "unmatched_difference": describe(unmatched_diff(Wb), rng),
                   "pairwise_accuracy": describe_acc(pairwise_acc(cell), rng)}
        pv_vals.append(pd.DataFrame({"analysis": "B_" + name, "pid": d.index, "value": d.values}))
    res["B"] = B
    log("B done")

    # ---- Holm family
    fam = [("A_lab", A["lab"]), ("A_home", A["home"]), ("B_lab", B["lab"]), ("B_home", B["home"])]
    ps = [x["matched_difference"].get("wilcoxon_p_one_sided", 1.0) for _, x in fam]
    adj = holm(ps)
    res["holm_family"] = {k: {"p": float(p), "p_holm": float(a), "observed": bool(a < 0.05)}
                          for (k, _), p, a in zip(fam, ps, adj)}
    log("Holm", res["holm_family"])

    # ---- C: windows away from FoG
    if os.path.exists(FOGD):
        Dst = pd.read_csv(FOGD)
        Cres = {}
        for name, df, ds in [("lab", lab, "tdcsfog"), ("home", home, "defog")]:
            g = df.merge(Dst, on=["rec", "win_index"], how="inner")
            Cres[name] = {"far_from_fog": c_far(g, Dst, ds, rng), "pre_onset_vs_far": c2_pre_vs_far(g, ds, rng)}
        res["C"] = Cres
    else:
        res["C"] = {"error": "fog_distance_primary.csv not found"}
    log("C done")

    # ---- D: structural reference
    orig = json.load(open(os.path.join(C.OUT, "results.json")))
    D = {}
    for ds in C.DATASETS:
        Wd = Z[ds]
        D[ds] = {"lopo_auc": orig["primary"]["lopo_" + ds]["auc"],
                 "permutation_null_mean": orig["hypotheses"]["H2a"]["per_dataset"][ds]["null_mean_auc"],
                 "intercept_only_auc_all": structural_auc(Wd),
                 "intercept_only_auc_paired": structural_auc(Wd[Wd["pv"].isin(paired_pvs(Wd))])}
    D["pooled"] = {"lopo_auc": orig["primary"]["lopo_pooled"]["auc"], "intercept_only_auc_all": structural_auc(W)}
    res["D"] = D

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    tag = "_dryrun" if DRY else ""
    json.dump(clean(res), open(os.path.join(C.OUT, "results_amend01%s.json" % tag), "w"), indent=2)
    pd.concat(pv_vals).to_csv(os.path.join(C.OUT, "amend01_patient_values%s.csv" % tag), index=False)
    log("saved results_amend01%s.json" % tag)
    print(json.dumps(clean({"home_window_flow": hinfo, "home_strata_rule": used, "holm_family": res["holm_family"],
                            "A_cells": {k: A[k]["cells"] for k in ["lab", "home"]},
                            "B_patients": {k: B[k]["patients"] for k in ["lab", "home"]}}), indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        V = pd.concat(pv_vals)
        names = ["A_lab", "A_home", "B_lab", "B_home"]
        fig, ax = plt.subplots(figsize=(5, 3))
        for i, n in enumerate(names):
            v = V[V["analysis"] == n]["value"].to_numpy()
            ax.plot(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.15, 0.15, len(v)), v, "o", ms=2.5, alpha=0.6)
            if len(v):
                ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, "k-", lw=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["A lab", "A home", "B lab", "B home"])
        ax.set_ylabel("Matched OFF - ON logit")
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT, "fig_amend01%s.png" % tag), dpi=300)
        plt.close(fig)
    except Exception as e:
        log("figure error", repr(e))


if __name__ == "__main__":
    main()
