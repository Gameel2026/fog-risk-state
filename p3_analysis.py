"""Step 3: medication-state analysis as fixed in PLAN.md (version 1).
Inputs : results/windows_<variant>.pkl, results/recordings.csv, results/frozen_gate.json, subjects.csv
Outputs: results/results.json (single source of every number), results/pred_*.csv, results/dz_features.csv,
         results/fig_*.png
Run    : python p3_analysis.py            (all analyses)
         python p3_analysis.py --quick    (100 permutations / bootstraps, for a dry run only)
"""
import os
import sys
import json
import time
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import p_config as C

warnings.filterwarnings("ignore", category=RuntimeWarning)
QUICK = "--quick" in sys.argv
N_PERM = 100 if QUICK else C.N_PERM
N_BOOT = 100 if QUICK else C.N_BOOT
try:
    from joblib import Parallel, delayed
    N_JOBS = max(1, (os.cpu_count() or 2) - 1)
except ImportError:
    Parallel = None
    N_JOBS = 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data

def otsu(x, bins=256):
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def load_variant(v, gate, R):
    """Apply the gait gate and the minimum-window rule. Medication labels are attached only after gating."""
    p = C.VARIANTS[v]
    W = pd.read_pickle(os.path.join(C.OUT, "windows_%s.pkl" % v))
    prim = C.VARIANTS["primary"]
    if p["fs"] == prim["fs"] and p["win"] == prim["win"]:
        thr, rule = gate["logP_loc_threshold"], "frozen"
    else:  # band power scales with window length and rate: same label-free rule, recomputed
        thr, rule = otsu(W["logP_loc"].to_numpy()), "same Otsu rule recomputed for this rate/window"
    W = W[(W["logP_loc"] > thr) & (W["gate_ac"] > C.AC_GATE)].copy()
    cnt = W.groupby("rec").size()
    W = W[W["rec"].isin(cnt[cnt >= C.MIN_WINDOWS].index)]
    W = W.merge(R[["rec", "dataset", "subject", "visit", "medication", "test", "fog_pct", "split"]], on="rec")
    W["y"] = (W["medication"] == "off").astype(int)
    W["pid"] = W["dataset"] + ":" + W["subject"].astype(str)
    W["pv"] = W["pid"] + ":v" + W["visit"].astype(str)
    W = W.sort_values(["rec", "t_start_s"]).reset_index(drop=True)
    return W, {"threshold": thr, "rule": rule}


def cohort(W, R):
    out = {}
    for ds in C.DATASETS:
        r_all = R[R["dataset"] == ds]
        w = W[W["dataset"] == ds]
        recs = w.drop_duplicates("rec")
        both = recs.groupby("pv")["y"].nunique()
        paired_pv = both[both == 2].index
        out[ds] = {"recordings_total": int(len(r_all)), "patients_total": int(r_all["subject"].nunique()),
                   "recordings_kept": int(len(recs)), "recordings_excluded": int(len(r_all) - len(recs)),
                   "patients_kept": int(recs["pid"].nunique()),
                   "patients_paired": int(recs[recs["pv"].isin(paired_pv)]["pid"].nunique()),
                   "windows_kept": int(len(w)),
                   "windows_per_recording_median": float(w.groupby("rec").size().median()) if len(w) else 0.0}
    return out


# ----------------------------------------------------------------------------- model

def fit_lr(X, y, s, c=C.LR_C, w0=None, iters=60, tol=1e-7):
    """L2 logistic regression, same objective as scikit-learn (0.5|w|^2 + C*sum s_i*loss_i, intercept free).
    Newton steps with backtracking."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    pen = np.ones(d + 1)
    pen[-1] = 0.0
    w = np.zeros(d + 1) if w0 is None else w0.copy()

    def obj(w):
        z = Xa @ w
        return 0.5 * np.sum(pen * w * w) + c * np.sum(s * (np.logaddexp(0, z) - y * z))

    f = obj(w)
    for _ in range(iters):
        z = Xa @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = pen * w + c * (Xa.T @ (s * (p - y)))
        H = np.diag(pen + 1e-10) + c * (Xa * (s * p * (1 - p))[:, None]).T @ Xa
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:
            fn = obj(w - t * step)
            if fn <= f + 1e-12:
                break
            t /= 2
        w = w - t * step
        done = abs(f - fn) < tol * max(1.0, abs(f)) or np.max(np.abs(t * step)) < tol
        f = fn
        if done:
            break
    return w


class Prep:
    """Median imputation and standardisation fitted on training windows only."""

    def __init__(self, X):
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med)
        self.mu = Xi.mean(axis=0)
        self.sd = Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0

    def __call__(self, X):
        return (np.where(np.isfinite(X), X, self.med) - self.mu) / self.sd


def patient_weights(pid):
    """Each patient contributes equally; weights scaled to mean 1 so C keeps its usual meaning."""
    _, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
    s = 1.0 / cnt[inv]
    return s * len(s) / s.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_predict(Xtr, ytr, ptr, Xte, model="lr", w0=None):
    prep = Prep(Xtr)
    A, B = prep(Xtr), prep(Xte)
    s = patient_weights(ptr)
    if model == "lr":
        w = fit_lr(A, ytr.astype(float), s, w0=w0)
        return B @ w[:-1] + w[-1], A @ w[:-1] + w[-1], w
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=C.SEED).fit(A, ytr, sample_weight=s)
    pte = np.clip(m.predict_proba(B)[:, 1], 1e-6, 1 - 1e-6)
    ptr_ = np.clip(m.predict_proba(A)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(pte / (1 - pte)), np.log(ptr_ / (1 - ptr_)), None


def rec_prob(rec_codes, z, agg="median", n_rec=None):
    """Window logits -> recording probability (median of window probabilities, or mean logit)."""
    df = pd.DataFrame({"r": rec_codes, "z": z})
    if agg == "median":
        return df.assign(p=sigmoid(df["z"])).groupby("r")["p"].median()
    return sigmoid(df.groupby("r")["z"].mean())


# ----------------------------------------------------------------------------- arrays & folds

def arrays(W, feats, y=None):
    return dict(X=W[feats].to_numpy(float), y=(W["y"].to_numpy() if y is None else y),
                pid=W["pid"].to_numpy(), rec=pd.factorize(W["rec"])[0])


def lopo_logits(a, model="lr", warm=None, return_w=False):
    z = np.empty(len(a["y"]))
    ws, thr80 = {}, {}
    for g in np.unique(a["pid"]):
        te = a["pid"] == g
        tr = ~te
        zte, ztr, w = train_predict(a["X"][tr], a["y"][tr], a["pid"][tr], a["X"][te], model,
                                    None if warm is None else warm.get(g))
        z[te] = zte
        ws[g] = w
        if not return_w:
            continue
        # operating point fixed on training recordings: 80% specificity among ON recordings
        pr = rec_prob(a["rec"][tr], ztr)
        yr = pd.Series(a["y"][tr]).groupby(a["rec"][tr]).first()
        thr80[g] = float(np.quantile(pr[yr == 0], 0.8)) if (yr == 0).any() else 0.5
    return (z, ws, thr80) if return_w else z


def rec_table(W, z, agg="median"):
    recs = W.drop_duplicates("rec").set_index("rec")[["dataset", "pid", "pv", "y", "fog_pct", "test"]]
    p = pd.DataFrame({"rec": W["rec"].to_numpy(), "z": z})
    if agg == "median":
        pr = p.assign(q=sigmoid(p["z"])).groupby("rec")["q"].median()
    else:
        pr = sigmoid(p.groupby("rec")["z"].mean())
    rt = recs.join(pr.rename("prob"))
    rt["n_win"] = W.groupby("rec").size()
    return rt.reset_index()


# ----------------------------------------------------------------------------- statistics

def auc(rt):
    return float(roc_auc_score(rt["y"], rt["prob"])) if rt["y"].nunique() == 2 else np.nan


def patient_diff(rt):
    """OFF - ON per patient: mean over recordings within state and visit, then mean over visits."""
    v = rt.groupby(["pid", "pv", "y"])["prob"].mean().unstack("y")
    if 0 not in v.columns or 1 not in v.columns:
        return pd.Series(dtype=float), v
    v = v.dropna()
    d = (v[1] - v[0]).groupby(level="pid").mean()
    return d, v.groupby(level="pid").mean()


def boot_ci(values_fn, pids, rng, n=N_BOOT):
    u = np.unique(pids)
    out = []
    for _ in range(n):
        out.append(values_fn(rng.choice(u, len(u), replace=True)))
    out = np.array(out, float)
    out = out[np.isfinite(out)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if len(out) else [np.nan, np.nan]


def resample_rt(rt, chosen):
    g = dict(tuple(rt.groupby("pid")))
    parts = []
    for k, pid in enumerate(chosen):
        x = g[pid].copy()
        x["pid"] = pid + "#%d" % k
        x["pv"] = x["pv"] + "#%d" % k
        parts.append(x)
    return pd.concat(parts)


def concordance_split(rt):
    """Within-patient-visit concordance versus AUC computed only over pairs from different patients."""
    off, on = rt[rt["y"] == 1], rt[rt["y"] == 0]
    if not len(off) or not len(on):
        return np.nan, np.nan
    po, pn = off["prob"].to_numpy()[:, None], on["prob"].to_numpy()[None, :]
    sc = (po > pn) + 0.5 * (po == pn)
    same_pv = off["pv"].to_numpy()[:, None] == on["pv"].to_numpy()[None, :]
    diff_p = off["pid"].to_numpy()[:, None] != on["pid"].to_numpy()[None, :]
    return (float(sc[same_pv].mean()) if same_pv.any() else np.nan,
            float(sc[diff_p].mean()) if diff_p.any() else np.nan)


def summarize(rt, rng, with_ci=True):
    d, _ = patient_diff(rt)
    within, between = concordance_split(rt)
    res = {"n_recordings": int(len(rt)), "n_patients": int(rt["pid"].nunique()), "auc": auc(rt),
           "n_paired_patients": int(len(d)),
           "median_diff_off_minus_on": float(d.median()) if len(d) else np.nan,
           "patients_off_gt_on": int((d > 0).sum()), "concordance_within_patient": within,
           "auc_between_patients": between}
    if len(d) >= 5:
        w = stats.wilcoxon(d, alternative="greater")
        res["wilcoxon_p_one_sided"] = float(w.pvalue)
        res["dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
    if with_ci:
        res["auc_ci95"] = boot_ci(lambda ch: auc(resample_rt(rt, ch)), rt["pid"].to_numpy(), rng)
        if len(d) >= 5:
            dd = d.to_numpy()
            res["median_diff_ci95"] = boot_ci(lambda ch: np.median(d.loc[ch]), d.index.to_numpy(), rng)
            res["concordance_ci95"] = boot_ci(lambda ch: float((d.loc[ch] > 0).mean()), d.index.to_numpy(), rng)
            del dd
    return res


def swap_within_pv(W, rng):
    """Swap ON/OFF labels at random within each patient-visit that holds both states."""
    pv = W["pv"].to_numpy()
    both = W.groupby("pv")["y"].nunique()
    u = both[both == 2].index.to_numpy()
    flip = dict(zip(u, rng.random(len(u)) < 0.5))
    f = np.array([flip.get(x, False) for x in pv])
    return np.where(f, 1 - W["y"].to_numpy(), W["y"].to_numpy())


def _perm_lopo(W_small, feats, seed, warm):
    rng = np.random.default_rng(seed)
    y = swap_within_pv(W_small, rng)
    a = arrays(W_small, feats, y)
    z = lopo_logits(a, warm=warm)
    rt = rec_table(W_small.assign(y=y), z)
    return auc(rt)


def perm_test_lopo(W, feats, obs, seeds, warm):
    cols = list(dict.fromkeys(feats + ["rec", "pid", "pv", "y", "dataset", "fog_pct", "test"]))
    Ws = W[cols]
    if Parallel is not None and N_JOBS > 1:
        null = Parallel(n_jobs=N_JOBS)(delayed(_perm_lopo)(Ws, feats, s, warm) for s in seeds)
    else:
        null = [_perm_lopo(Ws, feats, s, warm) for s in seeds]
    null = np.array(null, float)
    return float((1 + np.sum(null >= obs)) / (1 + len(null))), float(np.nanmean(null))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


# ----------------------------------------------------------------------------- analyses

def run_config(W, feats, rng, model="lr", agg="median", with_ci=True, tag=""):
    """Within-dataset LOPO, pooled LOPO and cross-dataset transfer for one window set and feature set."""
    out, rts, extra = {}, {}, {}
    for ds in C.DATASETS:
        Wd = W[W["dataset"] == ds].reset_index(drop=True)
        a = arrays(Wd, feats)
        z, ws, thr = lopo_logits(a, model, return_w=True)
        rt = rec_table(Wd, z, agg)
        rts[ds] = rt
        out["lopo_" + ds] = summarize(rt, rng, with_ci)
        extra[ds] = dict(W=Wd, z=z, ws=ws, thr=thr)
    a = arrays(W, feats)
    z, ws, thr = lopo_logits(a, model, return_w=True)
    rt = rec_table(W, z, agg)
    rts["pooled"] = rt
    out["lopo_pooled"] = summarize(rt, rng, with_ci)
    extra["pooled"] = dict(W=W, z=z, ws=ws, thr=thr)
    for src, dst in [("tdcsfog", "defog"), ("defog", "tdcsfog")]:
        Ws, Wt = W[W["dataset"] == src], W[W["dataset"] == dst].reset_index(drop=True)
        zt, _, _ = train_predict(Ws[feats].to_numpy(float), Ws["y"].to_numpy(), Ws["pid"].to_numpy(),
                                 Wt[feats].to_numpy(float), model)
        rt = rec_table(Wt, zt, agg)
        rts["cross_%s_to_%s" % (src, dst)] = rt
        out["cross_%s_to_%s" % (src, dst)] = summarize(rt, rng, with_ci)
    return out, rts, extra


def sens_at_80spec(extra):
    W, z, thr = extra["W"], extra["z"], extra["thr"]
    rt = rec_table(W, z)
    rt["thr"] = rt["pid"].map(thr)
    pos = rt["prob"] > rt["thr"]
    return {"sensitivity": float(pos[rt["y"] == 1].mean()), "specificity": float((~pos)[rt["y"] == 0].mean()),
            "threshold_median": float(rt["thr"].median())}


def random_split_control(W, feats, rng, k=5, repeats=5):
    out = {}
    for ds in C.DATASETS:
        Wd = W[W["dataset"] == ds].reset_index(drop=True)
        recs = Wd["rec"].unique()
        aucs = []
        for r in range(repeats):
            perm = rng.permutation(recs)
            fold = {x: i % k for i, x in enumerate(perm)}
            f = Wd["rec"].map(fold).to_numpy()
            z = np.empty(len(Wd))
            X, y, p = Wd[feats].to_numpy(float), Wd["y"].to_numpy(), Wd["pid"].to_numpy()
            for i in range(k):
                te = f == i
                z[te], _, _ = train_predict(X[~te], y[~te], p[~te], X[te])
            aucs.append(auc(rec_table(Wd, z)))
        out[ds] = {"auc_random_recording_split_mean": float(np.mean(aucs)), "repeats": repeats, "folds": k}
    return out


def order_checks(extra_by_ds):
    out = {}
    # tDCS-FoG: recordings of one state within a visit carry a test number
    rt = rec_table(extra_by_ds["tdcsfog"]["W"], extra_by_ds["tdcsfog"]["z"])
    rhos = []
    for (pv, y), g in rt[rt["test"] >= 0].groupby(["pv", "y"]):
        if len(g) >= 3 and g["test"].nunique() >= 3:
            rhos.append(stats.spearmanr(g["test"], g["prob"]).correlation)
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    out["tdcsfog_test_order"] = {"n_patient_visit_states": int(len(rhos)),
                                 "median_rho": float(np.median(rhos)) if len(rhos) else np.nan,
                                 "wilcoxon_p_two_sided": float(stats.wilcoxon(rhos).pvalue) if len(rhos) >= 5 else np.nan}
    # within-recording time trend of window probability, per dataset and state
    for ds, ex in extra_by_ds.items():
        if ds == "pooled":
            continue
        W = ex["W"].assign(q=sigmoid(ex["z"]))
        for y, lab in [(0, "on"), (1, "off")]:
            r = []
            for _, g in W[W["y"] == y].groupby("rec"):
                if len(g) >= 5:
                    r.append(stats.spearmanr(g["t_start_s"], g["q"]).correlation)
            r = np.array([x for x in r if np.isfinite(x)])
            out["%s_within_recording_trend_%s" % (ds, lab)] = {
                "n_recordings": int(len(r)), "median_rho": float(np.median(r)) if len(r) else np.nan,
                "wilcoxon_p_two_sided": float(stats.wilcoxon(r).pvalue) if len(r) >= 5 else np.nan}
    return out


def window_count_curve(extra_by_ds, ks=(1, 2, 4, 8, None)):
    out = {}
    for ds in C.DATASETS:
        W = extra_by_ds[ds]["W"].assign(z=extra_by_ds[ds]["z"])
        for k in ks:
            Wk = W if k is None else W.groupby("rec").head(k)
            out["%s_k_%s" % (ds, "all" if k is None else k)] = auc(rec_table(Wk.reset_index(drop=True),
                                                                            Wk["z"].to_numpy()))
    return out


def feature_dz(W, rng):
    rows = []
    for ds in C.DATASETS:
        rec = W[W["dataset"] == ds].groupby(["rec", "pid", "pv", "y"])[C.FEATURES].mean().reset_index()
        for f in C.FEATURES:
            v = rec.groupby(["pid", "pv", "y"])[f].mean().unstack("y").dropna()
            d = (v[1] - v[0]).groupby(level="pid").mean().dropna()
            if len(d) < 5 or d.std(ddof=1) == 0:
                continue
            dz = d.mean() / d.std(ddof=1)
            ci = boot_ci(lambda ch: d.loc[ch].mean() / d.loc[ch].std(ddof=1), d.index.to_numpy(), rng)
            rows.append(dict(dataset=ds, feature=f, n=len(d), dz=float(dz), dz_lo=ci[0], dz_hi=ci[1],
                             p=float(stats.wilcoxon(d).pvalue)))
    D = pd.DataFrame(rows)
    if len(D):
        D["q_bh"] = np.nan
        for ds in D["dataset"].unique():
            m = D["dataset"] == ds
            D.loc[m, "q_bh"] = bh(D.loc[m, "p"])
    return D


def temporal_fog_link(extra_defog):
    """DeFOG: OFF probability from freeze-free windows in the first half of the scored time
    versus FoG burden in the second half, across patients within each medication state."""
    import p1_build_windows as P1
    meta, recs = P1.recordings()
    recs = [r for r in recs if r[1] == "defog"]
    sol = P1.load_solution() if any(sp != "train" for _, _, sp, _ in recs) else {}
    W = extra_defog["W"].assign(q=sigmoid(extra_defog["z"]))
    rows = []
    for rid, ds, split, f in recs:
        if rid not in set(W["rec"]):
            continue
        if split == "train":
            d = pd.read_csv(f, usecols=["Valid", "Task"] + C.LAB)
            keep = d["Valid"].astype(bool).to_numpy() & d["Task"].astype(bool).to_numpy()
            fog = d[C.LAB].to_numpy().max(axis=1) > 0
        else:
            if rid not in sol:
                continue
            n = int(sol[rid][:, 0].max()) + 1
            fog, keep = np.zeros(n, bool), np.zeros(n, bool)
            fog[sol[rid][:, 0].astype(int)] = sol[rid][:, 1].astype(bool)
            keep[sol[rid][:, 0].astype(int)] = ~sol[rid][:, 2].astype(bool)
        idx = np.flatnonzero(keep)
        if len(idx) < 10:
            continue
        mid = idx[len(idx) // 2]
        second = keep.copy()
        second[:mid] = False
        burden = 100.0 * (fog & second).sum() / max(second.sum(), 1)
        g = W[W["rec"] == rid]
        g1 = g[g["t_start_s"] + C.WIN_S <= mid / C.RAW_FS["defog"]]
        if len(g1) < 1:
            continue
        rows.append(dict(rec=rid, pid=g["pid"].iloc[0], y=int(g["y"].iloc[0]),
                         prob_first_half=float(g1["q"].median()), fog_pct_second_half=burden))
    T = pd.DataFrame(rows)
    out = {}
    for y, lab in [(0, "on"), (1, "off")]:
        t = T[T["y"] == y].groupby("pid")[["prob_first_half", "fog_pct_second_half"]].mean() if len(T) else T
        if len(t) >= 5:
            r = stats.spearmanr(t["prob_first_half"], t["fog_pct_second_half"])
            out[lab] = {"n_patients": int(len(t)), "spearman_rho": float(r.correlation),
                        "p_one_sided_positive": float(r.pvalue / 2 if r.correlation > 0 else 1 - r.pvalue / 2)}
    return out, T


def updrs_link(d_pooled):
    S = pd.read_csv(os.path.join(C.DATA_ROOT, "subjects.csv"))
    S["dU"] = S["UPDRSIII_Off"] - S["UPDRSIII_On"]
    u = S.groupby("Subject")["dU"].mean().dropna()
    d = d_pooled.copy()
    d.index = [i.split(":", 1)[1] for i in d.index]
    j = pd.concat([d.rename("dp"), u.rename("dU")], axis=1, join="inner").dropna()
    if len(j) < 5:
        return {"n_patients": int(len(j))}
    r = stats.spearmanr(j["dp"], j["dU"])
    return {"n_patients": int(len(j)), "spearman_rho": float(r.correlation),
            "p_one_sided": float(r.pvalue / 2 if r.correlation > 0 else 1 - r.pvalue / 2)}


# ----------------------------------------------------------------------------- figures

def figures(rts, D, wc, rate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    for i, ds in enumerate(C.DATASETS):
        _, v = patient_diff(rts[ds])
        for _, r in v.iterrows():
            ax[i].plot([0, 1], [r[0], r[1]], color="0.6" if r[1] <= r[0] else "C3", lw=0.8)
        ax[i].set_xticks([0, 1], ["ON", "OFF"])
        ax[i].set_title({"tdcsfog": "Laboratory (tDCS-FoG)", "defog": "Home (DeFOG)"}[ds], fontsize=9)
        ax[i].set_ylabel("Estimated OFF probability" if i == 0 else "")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, "fig_paired.png"), dpi=300)
    plt.close(fig)
    if len(D):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        feats = list(dict.fromkeys(D["feature"]))
        for j, ds in enumerate(C.DATASETS):
            d = D[D["dataset"] == ds].set_index("feature").reindex(feats)
            yy = np.arange(len(feats)) + (j - 0.5) * 0.3
            ax.errorbar(d["dz"], yy, xerr=[d["dz"] - d["dz_lo"], d["dz_hi"] - d["dz"]], fmt="o", ms=3,
                        label={"tdcsfog": "Laboratory", "defog": "Home"}[ds])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(np.arange(len(feats)), feats, fontsize=7)
        ax.set_xlabel("Within-patient effect size dz (OFF - ON)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT, "fig_dz_forest.png"), dpi=300)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ks = ["1", "2", "4", "8", "all"]
    for ds in C.DATASETS:
        ax[0].plot(ks, [wc["%s_k_%s" % (ds, k)] for k in ks], "o-", label=ds)
        ax[1].plot(["64", "32", "16"], [rate[ds][r] for r in ["64", "32", "16"]], "o-", label=ds)
    ax[0].set_xlabel("Windows per recording")
    ax[1].set_xlabel("Sampling rate (Hz)")
    for a in ax:
        a.set_ylabel("Recording AUC")
        a.axhline(0.5, color="k", lw=0.5, ls=":")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, "fig_resources.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    R = pd.read_csv(os.path.join(C.OUT, "recordings.csv"), dtype={"subject": str})
    gpath = os.path.join(C.OUT, "frozen_gate.json")
    gate = json.load(open(gpath))
    res = {"plan": "PLAN.md version 1", "quick_dry_run": QUICK, "n_perm": N_PERM, "n_boot": N_BOOT,
           "frozen_gate_sha256": hashlib.sha256(open(gpath, "rb").read()).hexdigest(), "gate": gate}

    W, g = load_variant("primary", gate, R)
    res["cohort"] = cohort(W, R)
    log("primary windows", len(W), res["cohort"])

    # ---- primary configuration
    prim, rts, extra = run_config(W, C.FEATURES, rng)
    res["primary"] = prim
    for k, rt in rts.items():
        rt.to_csv(os.path.join(C.OUT, "pred_%s.csv" % k), index=False)
    log("primary done", {k: round(v["auc"], 3) for k, v in prim.items()})

    # ---- confirmatory chain (fixed sequence, one-sided alpha 0.05)
    hyp = {}
    hyp["H1"] = {"test": "Wilcoxon signed-rank, pooled LOPO, per-patient OFF-ON", "p": prim["lopo_pooled"].get("wilcoxon_p_one_sided", np.nan)}
    h2a = {}
    for ds in C.DATASETS:
        ex = extra[ds]
        pval, null_mean = perm_test_lopo(ex["W"], C.FEATURES, prim["lopo_" + ds]["auc"],
                                         range(1000 + 0, 1000 + N_PERM), ex["ws"])
        h2a[ds] = {"auc": prim["lopo_" + ds]["auc"], "p_perm": pval, "null_mean_auc": null_mean}
        log("H2a", ds, h2a[ds])
    hyp["H2a"] = {"test": "LOPO AUC > 0.5 in each dataset, within patient-visit label permutation",
                  "per_dataset": h2a, "p": max(v["p_perm"] for v in h2a.values())}
    h2b = {}
    for src, dst in [("tdcsfog", "defog"), ("defog", "tdcsfog")]:
        key = "cross_%s_to_%s" % (src, dst)
        rt = rts[key]
        obs = auc(rt)
        Wt = W[W["dataset"] == dst]
        recy = rt.set_index("rec")
        null = []
        prng = np.random.default_rng(C.SEED + 7)
        for _ in range(N_PERM):
            y = swap_within_pv(Wt.drop_duplicates("rec").reset_index(drop=True), prng)
            null.append(roc_auc_score(y, recy.loc[Wt.drop_duplicates("rec")["rec"], "prob"]))
        h2b[key] = {"auc": obs, "p_perm": float((1 + np.sum(np.array(null) >= obs)) / (1 + N_PERM))}
    hyp["H2b"] = {"test": "cross-dataset AUC > 0.5 in each direction, label permutation in the target data",
                  "per_direction": h2b, "p": max(v["p_perm"] for v in h2b.values())}
    d_pooled, _ = patient_diff(rts["pooled"])
    hyp["H4"] = dict(test="Spearman, per-patient OFF-ON probability vs UPDRS-III OFF-ON, one-sided", **updrs_link(d_pooled))
    hyp["H4"]["p"] = hyp["H4"].get("p_one_sided", np.nan)
    alive = True
    for h in ["H1", "H2a", "H2b", "H4"]:
        hyp[h]["confirmatory"] = alive
        hyp[h]["supported"] = bool(alive and np.isfinite(hyp[h]["p"]) and hyp[h]["p"] < 0.05)
        if not hyp[h]["supported"]:
            alive = False
    res["hypotheses"] = hyp
    # per-dataset replication of H1 (within-dataset LOPO predictions pooled over both datasets as a sensitivity)
    both = pd.concat([rts["tdcsfog"], rts["defog"]])
    d2, _ = patient_diff(both)
    res["H1_sensitivity_within_dataset_predictions_pooled"] = {
        "n": int(len(d2)), "p_one_sided": float(stats.wilcoxon(d2, alternative="greater").pvalue) if len(d2) >= 5 else np.nan,
        "patients_off_gt_on": int((d2 > 0).sum())}
    log("hypotheses", {k: (v["p"], v["supported"]) for k, v in hyp.items()})

    # ---- controls
    ctl = {}
    ctl["operating_point_80spec"] = {k: sens_at_80spec(extra[k]) for k in ["tdcsfog", "defog", "pooled"]}
    ctl["random_split"] = random_split_control(W, C.FEATURES, rng)
    sh = W.assign(y=swap_within_pv(W, np.random.default_rng(C.SEED + 1)))
    ctl["label_shuffle_within_patient_visit"] = {
        ds: auc(rec_table(sh[sh["dataset"] == ds].reset_index(drop=True),
                          lopo_logits(arrays(sh[sh["dataset"] == ds].reset_index(drop=True), C.FEATURES))))
        for ds in C.DATASETS}
    ctl["permutation_null_mean_auc"] = {ds: h2a[ds]["null_mean_auc"] for ds in C.DATASETS}
    ctl["order"] = order_checks(extra)
    base, _, _ = run_config(W, [C.BASELINE_FEATURE], rng, with_ci=False)
    ctl["baseline_single_feature_logP_loc"] = base
    for v in ["margin0", "margin5"]:
        Wv, gv = load_variant(v, gate, R)
        o, _, _ = run_config(Wv, C.FEATURES, rng, with_ci=False)
        ctl["residual_fog_" + v] = dict(gate=gv, cohort=cohort(Wv, R), **o)
    ff = {}
    for k in ["tdcsfog", "defog", "pooled"]:
        rt = rts[k]
        ff[k] = summarize(rt[rt["fog_pct"] == 0], rng, with_ci=False)
    ctl["fog_free_recordings_only"] = ff
    res["controls"] = ctl
    log("controls done")

    # ---- exploratory
    ex = {}
    rate = {ds: {"64": prim["lopo_" + ds]["auc"]} for ds in C.DATASETS}
    for v, key in [("fs32", "32"), ("fs16", "16"), ("win10", None)]:
        Wv, gv = load_variant(v, gate, R)
        o, _, _ = run_config(Wv, C.FEATURES, rng, with_ci=False)
        ex[v] = dict(gate=gv, cohort=cohort(Wv, R), **o)
        if key:
            for ds in C.DATASETS:
                rate[ds][key] = o["lopo_" + ds]["auc"]
    ex["sampling_rate_auc"] = rate
    wc = window_count_curve(extra)
    ex["window_count_auc"] = wc
    noamp = [f for f in C.FEATURES if f not in C.AMPLITUDE_FEATURES]
    ex["no_amplitude_features"], _, _ = run_config(W, noamp, rng, with_ci=False)
    ex["mean_logit_aggregation"], _, _ = run_config(W, C.FEATURES, rng, agg="mean_logit", with_ci=False)
    ex["gradient_boosting"], _, _ = run_config(W, C.FEATURES, rng, model="gbt", with_ci=False)
    D = feature_dz(W, rng)
    D.to_csv(os.path.join(C.OUT, "dz_features.csv"), index=False)
    ex["feature_dz"] = D.to_dict(orient="records")
    ex["lr_coefficients_pooled_full_data"] = None
    a = arrays(W, C.FEATURES)
    _, _, w = train_predict(a["X"], a["y"], a["pid"], a["X"][:1])
    ex["lr_coefficients_pooled_full_data"] = dict(zip(C.FEATURES + ["intercept"], map(float, w)))
    try:
        tl, T = temporal_fog_link(extra["defog"])
        T.to_csv(os.path.join(C.OUT, "temporal_fog_link.csv"), index=False)
        ex["temporal_fog_link_defog"] = tl
    except Exception as e:  # exploratory; never blocks the main results
        ex["temporal_fog_link_defog"] = {"error": repr(e)}
    res["exploratory"] = ex
    log("exploratory done")

    figures(rts, D, wc, rate)
    res["runtime_min"] = round((time.time() - t0) / 60, 1)

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

    with open(os.path.join(C.OUT, "results.json"), "w") as fh:
        json.dump(clean(res), fh, indent=2)
    log("saved results.json; runtime", res["runtime_min"], "min")
    print(json.dumps(clean({"cohort": res["cohort"], "hypotheses": hyp}), indent=2))


if __name__ == "__main__":
    main()
