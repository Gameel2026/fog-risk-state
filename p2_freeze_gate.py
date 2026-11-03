"""Step 2: fix the gait-gate threshold on the pooled primary windows WITHOUT medication labels.
Rule (fixed in advance): Otsu threshold on log 0.5-3 Hz band power, pooled over both datasets,
combined with the planned autocorrelation step-peak threshold of 0.3.
Output: results/frozen_gate.json and its sha256 (commit both before step 3).
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
import p_config as C


def otsu(x, bins=256):
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(c[np.argmax(between)])


def main():
    W = pd.read_pickle(os.path.join(C.OUT, "windows_primary.pkl"))
    assert "medication" not in W.columns
    thr = otsu(W["logP_loc"].to_numpy())
    kept = (W["logP_loc"] > thr) & (W["gate_ac"] > C.AC_GATE)
    gate = {"rule": "Otsu on pooled log 0.5-3 Hz power (primary windows, no labels) and step autocorrelation > 0.3",
            "logP_loc_threshold": thr, "ac_threshold": C.AC_GATE,
            "windows_total": int(len(W)), "windows_kept": int(kept.sum())}
    p = os.path.join(C.OUT, "frozen_gate.json")
    with open(p, "w") as fh:
        json.dump(gate, fh, indent=2)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(json.dumps(gate, indent=2))
    print("sha256 of frozen_gate.json:", h)


if __name__ == "__main__":
    main()
