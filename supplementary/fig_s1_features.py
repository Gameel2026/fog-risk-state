import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8})
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results") + os.sep
LAB_C, HOME_C = "#2a78d6", "#eb6834"
d = pd.read_csv(R + "dz_features.csv")
coef = json.load(open(R + "results.json"))["exploratory"]["lr_coefficients_pooled_full_data"]
groups = [("Amplitude", [("rms_V", "RMS V"), ("rms_AP", "RMS AP"), ("rms_ML", "RMS ML"), ("log_jerk", "Log jerk")]),
          ("Spectral power", [("logP_loc", "Log power 0.5-3 Hz"), ("logP_frz", "Log power 3-8 Hz"), ("log_ratio", "Log power ratio")]),
          ("Rhythm", [("f_step", "Step frequency"), ("step_peak_rel", "Step peak strength")]),
          ("Harmonics", [("HR_V", "Harmonic ratio V"), ("HR_AP", "Harmonic ratio AP"), ("HR_ML", "Harmonic ratio ML")]),
          ("Regularity", [("step_reg", "Step regularity"), ("stride_reg", "Stride regularity"), ("symmetry", "Step symmetry")])]
rows, ylab, yv, y = [], [], [], 0
gy = []
for g, feats in groups:
    gy.append((y, g)); y += 0.7
    for f, name in feats:
        L = d[(d.dataset == "tdcsfog") & (d.feature == f)].iloc[0]; H = d[(d.dataset == "defog") & (d.feature == f)].iloc[0]
        rows.append(dict(group=g, feature=f, label=name, lab_dz=L.dz, lab_lo=L.dz_lo, lab_hi=L.dz_hi, lab_q=L.q_bh,
                         home_dz=H.dz, home_lo=H.dz_lo, home_hi=H.dz_hi, home_q=H.q_bh, coef=coef[f], y=y))
        y += 1
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "feature_table.csv"), index=False)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.45, 3.3), gridspec_kw=dict(width_ratios=[3.0, 1.1], wspace=0.12), sharey=True)
for _, r in T.iterrows():
    for dz, lo, hi, q, c, off in [(r.lab_dz, r.lab_lo, r.lab_hi, r.lab_q, LAB_C, -0.17), (r.home_dz, r.home_lo, r.home_hi, r.home_q, HOME_C, 0.17)]:
        ax.plot([lo, hi], [r.y + off] * 2, color=c, lw=0.9)
        ax.plot(dz, r.y + off, "o", ms=3.6, mec=c, mfc=c if q < 0.05 else "white", mew=0.9)
    ax2.barh(r.y, r.coef, height=0.55, color="0.45")
ax.axvline(0, color="0.3", lw=0.6); ax2.axvline(0, color="0.3", lw=0.6)
for yy, g in gy:
    ax.text(-1.02, yy, g, fontsize=7, fontweight="bold", va="center")
ax.set_yticks(T.y); ax.set_yticklabels(T.label, fontsize=6.8); ax.invert_yaxis()
ax.set_xlim(-1.05, 1.0); ax.set_xlabel("Within-patient effect dz (OFF - ON)", fontsize=7.2)
ax2.set_xlabel("Model weight", fontsize=7.2); ax2.set_xlim(-0.55, 0.45)
for a in (ax, ax2):
    a.tick_params(labelsize=6.8, length=2); a.spines[["top", "right"]].set_visible(False)
ax.plot([], [], "o", mec=LAB_C, mfc=LAB_C, ms=3.6, label="Laboratory"); ax.plot([], [], "o", mec=HOME_C, mfc=HOME_C, ms=3.6, label="Home")
ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), fontsize=6.5, frameon=False, handletextpad=0.2)
fig.subplots_adjust(left=0.30, right=0.98, top=0.98, bottom=0.11); ax.set_xticks([-1, -0.5, 0, 0.5]); ax2.set_xticks([-0.4, 0, 0.4]); fig.savefig(os.path.join(HERE, "fig_s1_features.png"), dpi=600)
print(T[["label","lab_dz","lab_q","home_dz","home_q","coef"]].round(3).to_string())
