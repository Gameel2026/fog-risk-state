"""Window features from a single lower-back accelerometer (columns: V, ML, AP in g)."""
import numpy as np


def _acf(x):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / a[0] if a[0] > 0 else np.zeros(n)


def harmonic_ratio(x, fs, f_stride, n_harm=10, inverted=False):
    n = len(x)
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    amps = []
    for k in range(1, n_harm + 1):
        fk = k * f_stride
        if fk >= fs / 2:
            break
        amps.append(X[np.argmin(np.abs(f - fk))])
    if len(amps) < 2:
        return np.nan
    amps = np.array(amps)
    even, odd = amps[1::2].sum(), amps[0::2].sum()
    if inverted:
        even, odd = odd, even
    return float(even / odd) if odd > 0 else np.nan


def window_features(w, fs):
    """w: (n, 3) array V, ML, AP in g. Returns dict of features plus gate quantities."""
    n = len(w)
    x = w - w.mean(axis=0)
    hann = np.hanning(n)[:, None]
    P = np.abs(np.fft.rfft(x * hann, axis=0)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    Ps = P.sum(axis=1)
    loc = (f >= 0.5) & (f < 3.0)
    frz = (f >= 3.0) & (f <= min(8.0, fs / 2 - 1e-9))
    eps = 1e-12
    p_loc, p_frz = Ps[loc].sum() + eps, Ps[frz].sum() + eps
    # dominant step frequency from the vertical axis in 0.5-3 Hz
    Pv = P[:, 0]
    k = np.argmax(np.where(loc, Pv, -1))
    f_step = float(f[k])
    step_peak_rel = float(Pv[k] / (Pv[(f >= 0.5) & (f <= min(8.0, fs / 2 - 1e-9))].sum() + eps))
    # autocorrelation regularity on the vertical axis
    ac = _acf(w[:, 0])
    lo, hi = int(0.3 * fs), min(int(1.2 * fs), n - 1)
    if hi > lo + 1:
        lag1 = lo + int(np.argmax(ac[lo:hi]))
        step_reg = float(ac[lag1])
        l2lo, l2hi = int(1.6 * lag1), min(int(2.4 * lag1) + 1, n - 1)
        stride_reg = float(ac[l2lo + int(np.argmax(ac[l2lo:l2hi]))]) if l2hi > l2lo + 1 else np.nan
    else:
        step_reg, stride_reg = np.nan, np.nan
    symmetry = float(step_reg / stride_reg) if stride_reg and np.isfinite(stride_reg) and stride_reg > 0 else np.nan
    f_stride = f_step / 2.0 if f_step > 0 else np.nan
    if np.isfinite(f_stride) and f_stride > 0:
        hr_v = harmonic_ratio(w[:, 0], fs, f_stride)
        hr_ap = harmonic_ratio(w[:, 2], fs, f_stride)
        hr_ml = harmonic_ratio(w[:, 1], fs, f_stride, inverted=True)
    else:
        hr_v = hr_ap = hr_ml = np.nan
    jerk = np.diff(w, axis=0) * fs
    return {
        "logP_loc": float(np.log(p_loc)), "logP_frz": float(np.log(p_frz)), "log_ratio": float(np.log(p_frz / p_loc)),
        "rms_V": float(np.sqrt(np.mean(x[:, 0] ** 2))), "rms_ML": float(np.sqrt(np.mean(x[:, 1] ** 2))),
        "rms_AP": float(np.sqrt(np.mean(x[:, 2] ** 2))),
        "f_step": f_step, "step_peak_rel": step_peak_rel,
        "HR_V": hr_v, "HR_AP": hr_ap, "HR_ML": hr_ml,
        "step_reg": step_reg, "stride_reg": stride_reg, "symmetry": symmetry,
        "log_jerk": float(np.log(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))) + eps)),
        "gate_ac": step_reg,
    }
