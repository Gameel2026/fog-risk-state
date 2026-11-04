"""Amendment 02: replication on unseen DeFOG visits (see AMENDMENT_02.md).
Primary : original home leave-one-patient-out model of each subject, applied unchanged to the new visit.
S1      : laboratory-trained transfer model (all tDCS-FoG windows).
S2      : Amendment 01 personal-baseline model (DeFOG training recordings of other patients, MiniBest/Rest removed).
Descriptive: deployment model, unmatched difference, pairwise accuracy, the 8 excluded ON recordings.
Run    : python p3_amend02.py --dryrun   (labels swapped within subject-visit, seed 77; prints counts only)
         python p3_amend02.py            (the single real run)
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
import p_config as C
import p3_analysis as P
import p3_amend01 as A

DRY = "--dryrun" in sys.argv
FALLBACK_MIN_PATIENTS = 15
MIN_PATIENTS_FOR_TEST = 10
SEED_BOOT, SEED_DRY = 42, 77
TOL = 1e-6


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def load_new(gate):
    W = pd.read_pickle(os.path.join(C.OUT, "windows_notype.pkl"))
    W = W[(W["logP_loc"] > gate["logP_loc_threshold"]) & (W["gate_ac"] > C.AC_GATE)].copy()
    n = W.groupby("rec").size()
    W = W[W["rec"].isin(n[n >= C.MIN_WINDOWS].index)]
    meta = pd.read_csv(os.path.join(C.DATA_ROOT, "defog_metadata.csv"), dtype={"Id": str, "Subject": str})
    meta["Id"] = meta["Id"].astype(str)
    meta = meta.rename(columns={"Id": "rec", "Subject": "subject", "Visit": "visit", "Medication": "medication"})
    W = W.merge(meta[["rec", "subject", "visit", "medication"]], on="rec")
    W["subject"] = W["subject"].astype(str)
    W["medication"] = W["medication"].astype(str).str.lower()
    W["y"] = (W["medication"] == "off").astype(int)
    W["dataset"] = "defog"
    W["split"] = "notype"
    W["test"] = -1
    W["fog_pct"] = np.nan
    W["pid"] = "defog:" + W["subject"]
    W["pv"] = W["pid"] + ":v" + W["visit"].astype(int).astype(str)
    return W.sort_values(["rec", "t_start_s"]).reset_index(drop=True)


def clean_tasks(W):
    W = W.copy()
    W["task"], n_multi = A.assign_task(W)
    W = W[W["task"].notna()]
    W = W[~W["task"].astype(str).str.startswith(A.EXCL)]
    n = W.groupby("rec").size()
    return W[W["rec"].isin(n[n >= A.MIN_REC_WIN].index)].reset_index(drop=True), n_multi


def cells_of(df):
    cell = A.stratum_cells(df, "defog")
    rule = "exact tasks"
    if cell.index.get_level_values("pid").nunique() < FALLBACK_MIN_PATIENTS:
        cell = A.stratum_cells(df, "defog", collapse=True)
        rule = "fallback: cued merged with uncued, Hotspot1 merged with Hotspot2"
    return cell, rule


def summary(df, rng, test):
    cell, rule = cells_of(df)
    d = A.matched_diff(cell)
    return {"strata_rule": rule, "cells": int(len(cell)),
            "matched_difference": A.describe(d, rng, test=test),
            "pairwise_accuracy": A.describe_acc(A.pairwise_acc(cell), rng),
            "unmatched_difference_after_MiniBest_Rest_removal": A.describe(A.unmatched_diff(df), rng)}, d


def main():
    rng = np.random.default_rng(SEED_BOOT)
    R = pd.read_csv(os.path.join(C.OUT, "recordings.csv"), dtype={"subject": str})
    gpath = os.path.join(C.OUT, "frozen_gate.json")
    gate = json.load(open(gpath))
    Wn = load_new(gate)
    used_pv = set("defog:" + R.loc[R["dataset"] == "defog", "subject"] + ":v" + R.loc[R["dataset"] == "defog", "visit"].astype(str))
    newpv = Wn[~Wn["pv"].isin(used_pv)]
    both = newpv.groupby("pv")["y"].nunique()
    rep_pv = both[both == 2].index
    Rep = Wn[Wn["pv"].isin(rep_pv)].reset_index(drop=True)
    Excl = Wn[~Wn["pv"].isin(rep_pv)].reset_index(drop=True)
    if DRY:
        Rep["y"] = P.swap_within_pv(Rep, np.random.default_rng(SEED_DRY))
    RepC, n_multi = clean_tasks(Rep)
    res = {"amendment": "AMENDMENT_02.md", "dry_run": DRY,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
           "inputs_sha256": {"windows_notype.pkl": sha(os.path.join(C.OUT, "windows_notype.pkl")),
                             "recordings_notype.csv": sha(os.path.join(C.OUT, "recordings_notype.csv")),
                             "tasks.csv": sha(A.TASKS), "frozen_gate.json": sha(gpath)},
           "counts": {"new_recordings_after_gate": int(Wn["rec"].nunique()),
                      "replication_subject_visits": int(len(rep_pv)),
                      "replication_recordings": int(Rep["rec"].nunique()),
                      "replication_windows_gated": int(len(Rep)),
                      "replication_windows_after_MiniBest_Rest": int(len(RepC)),
                      "windows_in_two_tasks": int(n_multi),
                      "excluded_recordings": int(Excl["rec"].nunique())}}

    # ---- original data
    W, _ = P.load_variant("primary", gate, R)
    Wd = W[W["dataset"] == "defog"].reset_index(drop=True)
    Wt = W[W["dataset"] == "tdcsfog"].reset_index(drop=True)
    F = C.FEATURES
    X, y, pid = Wd[F].to_numpy(float), Wd["y"].to_numpy(), Wd["pid"].to_numpy()

    # ---- primary: original home LOPO fold model of each subject
    z_rep = np.full(len(RepC), np.nan)
    z_excl = np.full(len(Excl), np.nan)
    old = pd.read_csv(os.path.join(C.OUT, "pred_defog.csv")).set_index("rec")["prob"]
    repro = []
    for g in sorted(set(RepC["pid"]) | set(Excl["pid"])):
        tr = pid != g
        if tr.all():
            res.setdefault("skipped_no_original_windows", []).append(g)
            continue
        mo, mr, me = (pid == g), (RepC["pid"] == g).to_numpy(), (Excl["pid"] == g).to_numpy()
        Xte = np.vstack([X[mo], RepC.loc[mr, F].to_numpy(float), Excl.loc[me, F].to_numpy(float)])
        zte, _, _ = P.train_predict(X[tr], y[tr], pid[tr], Xte)
        k1, k2 = mo.sum(), mo.sum() + mr.sum()
        prob = P.rec_table(Wd[mo].reset_index(drop=True), zte[:k1]).set_index("rec")["prob"]
        repro.append(float(np.max(np.abs(prob - old.reindex(prob.index)))))
        z_rep[mr] = zte[k1:k2]
        z_excl[me] = zte[k2:]
    dev = max(repro) if repro else np.nan
    res["reproduction_max_abs_diff"] = dev
    if not DRY and (not np.isfinite(dev) or dev > TOL):
        res["stopped"] = "reproduction failed (max diff %g)" % dev
        json.dump(res, open(os.path.join(C.OUT, "results_amend02.json"), "w"), indent=2, default=float)
        raise SystemExit(res["stopped"])

    # ---- S1: laboratory-trained transfer model
    z_s1, _, _ = P.train_predict(Wt[F].to_numpy(float), Wt["y"].to_numpy(), Wt["pid"].to_numpy(), RepC[F].to_numpy(float))

    # ---- S2: personal baseline trained on DeFOG training recordings of other patients
    home, _ = A.home_clean(Wd)
    home = home[home["pv"].isin(A.paired_pvs(home))].reset_index(drop=True)
    z_s2 = np.full(len(RepC), np.nan)
    for g in sorted(set(RepC["pid"])):
        Tr = home[home["pid"] != g].reset_index(drop=True)
        Xtr = Tr[F].to_numpy(float)
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xc = A.center_pv(np.where(np.isfinite(Xtr), Xtr, med), Tr["rec"].to_numpy(), Tr["pv"].to_numpy())
        prep = P.Prep(Xc)
        w = P.fit_lr(prep(Xc), Tr["y"].to_numpy().astype(float), A.balanced_weights(Tr))
        m = (RepC["pid"] == g).to_numpy()
        Xn = RepC.loc[m, F].to_numpy(float)
        Xn = A.center_pv(np.where(np.isfinite(Xn), Xn, med), RepC.loc[m, "rec"].to_numpy(), RepC.loc[m, "pv"].to_numpy())
        z_s2[m] = prep(Xn) @ w[:-1] + w[-1]

    # ---- deployment model (descriptive; it saw these subjects' earlier visits)
    M = json.load(open(os.path.join(C.OUT, "deploy_model.json")))
    Fn = RepC[F].to_numpy(float)
    Fn = np.where(np.isfinite(Fn), Fn, np.array(M["median"]))
    z_dep = ((Fn - np.array(M["mean"])) / np.array(M["sd"])) @ np.array(M["coef"]) + M["intercept"]

    models = [("primary", z_rep), ("S1_lab_trained", z_s1), ("S2_personal_baseline", z_s2), ("deployment_descriptive", z_dep)]
    vals = []
    if DRY:
        out = {}
        for name, z in models:
            cell, rule = cells_of(RepC.assign(z=z))
            out[name] = {"patients_with_cells": int(cell.index.get_level_values("pid").nunique()), "cells": int(len(cell)),
                         "strata_rule": rule, "all_scores_finite": bool(np.isfinite(z).all())}
        res["dry_run_counts"] = out
        n_per = RepC.groupby(["pid", "task", "y"]).size()
        res["windows_per_stratum_state_median"] = float(n_per.median()) if len(n_per) else None
        json.dump(res, open(os.path.join(C.OUT, "results_amend02_dryrun.json"), "w"), indent=2, default=float)
        print(json.dumps(res, indent=2, default=float))
        return

    seq, alive = {}, True
    for name, z in models:
        s, d = summary(RepC.assign(z=z), rng, test=(name != "deployment_descriptive"))
        if name != "deployment_descriptive":
            md = s["matched_difference"]
            p = md.get("wilcoxon_p_one_sided", np.nan)
            enough = md["n_patients"] >= MIN_PATIENTS_FOR_TEST
            s["test_run"] = bool(enough)
            s["confirmatory"] = bool(alive)
            s["significant"] = bool(enough and np.isfinite(p) and p < 0.05)
            if not s["significant"]:
                alive = False
            if enough and len(d):
                s["wilcoxon_method"] = "exact" if len(d) <= 25 and (d == 0).sum() == 0 else "approximate"
        seq[name] = s
        vals.append(pd.DataFrame({"model": name, "pid": d.index, "value": d.values}))
    res["models"] = seq

    # ---- the excluded ON recordings paired with their earlier OFF recordings (descriptive)
    rows = []
    po = pd.read_csv(os.path.join(C.OUT, "pred_defog.csv"))
    for rec, g in clean_tasks(Excl.assign(z=z_excl))[0].groupby("rec"):
        if not np.isfinite(g["z"]).all():
            continue
        pv = g["pv"].iloc[0]
        off = po[(po["pv"] == pv) & (po["y"] == 1)]["prob"]
        if len(off):
            rows.append(dict(rec=rec, pv=pv, new_on_prob=float(np.median(P.sigmoid(g["z"]))), earlier_off_prob=float(off.mean())))
    E = pd.DataFrame(rows)
    res["excluded_on_recordings"] = {"n": int(len(E)), "off_minus_on": E.assign(d=E["earlier_off_prob"] - E["new_on_prob"])["d"].round(4).tolist() if len(E) else []}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    json.dump(clean(res), open(os.path.join(C.OUT, "results_amend02.json"), "w"), indent=2)
    pd.concat(vals).to_csv(os.path.join(C.OUT, "amend02_patient_values.csv"), index=False)
    print(json.dumps(clean({"counts": res["counts"], "reproduction": dev,
                            "primary": seq["primary"]["matched_difference"], "sequence": {k: (v.get("significant"), v.get("confirmatory")) for k, v in seq.items()}}), indent=2))


if __name__ == "__main__":
    main()
