"""Run the Cortex-M4F build of the OFF-state estimator (flash.bin) in the Unicorn ARM emulator.
1) agreement with Python on every primary window of every recording: features, gait gate, logit,
   and the recording score (median logit of gated windows, recordings with >= 3 gated windows)
2) executed instructions, approximate cycles, memory and estimated energy per window
Run from the mcu folder inside paper3\\code:  python run_fog_mcu.py      (needs: pip install unicorn capstone)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from unicorn import Uc, UcError, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_CODE, UC_HOOK_MEM_READ
from unicorn import arm_const as A

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import p_config as C
import p1_build_windows as P1

FLASH, RAM, IN, OUT, GATE = 0x00000000, 0x20000000, 0x60000000, 0x70000000, 0x78000000
STACK_TOP = RAM + 0x10000
STOP = 0x0003FF00
N = 384
NOUT = 17
CLOCK_HZ = 64e6
WINDOW_PERIOD_S = 3.0          # one 6-s window every 3 s (50% overlap)
RUN_UA_PER_MHZ = 52.0          # nRF52840 CPU running from flash, cache enabled (datasheet value used in paper 2)
IDLE_UA = 3.16                 # nRF52840 System ON idle with RTC (datasheet value used in paper 2)
BATCH = 64
FLASH_WS = (0, 1, 2)


def page(n):
    return max(0x1000, (n + 0xFFF) & ~0xFFF)


def symbols():
    s = {}
    for line in open(os.path.join(HERE, "symbols.txt")):
        p = line.split()
        if len(p) == 3:
            s[p[2]] = int(p[0], 16)
    return s


def new_cpu(mode_name):
    if mode_name == "cortex-m4":
        mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        try:
            mu.ctl_set_cpu_model(A.UC_CPU_ARM_CORTEX_M4)
        except Exception:
            pass
    else:
        mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        mu.reg_write(A.UC_ARM_REG_C1_C0_2, mu.reg_read(A.UC_ARM_REG_C1_C0_2) | (0xF << 20))
        mu.reg_write(A.UC_ARM_REG_FPEXC, 0x40000000)
    mu.mem_map(FLASH, 0x40000)
    mu.mem_write(FLASH, open(os.path.join(HERE, "flash.bin"), "rb").read())
    mu.mem_write(STOP, b"\x00\xbf\x00\xbf")
    mu.mem_map(RAM, 0x10000)
    return mu


def emulate(mode_name, entry, X, hook=None):
    """X: float32 (n, 384, 3). Returns out (n, 17) float32 and gate (n,) bool."""
    mu = new_cpu(mode_name)
    n = len(X)
    xb = np.ascontiguousarray(X, np.float32).tobytes()
    mu.mem_map(IN, page(len(xb)))
    mu.mem_write(IN, xb)
    mu.mem_map(OUT, page(4 * NOUT * n))
    mu.mem_map(GATE, page(n))
    mu.reg_write(A.UC_ARM_REG_SP, STACK_TOP)
    mu.reg_write(A.UC_ARM_REG_LR, STOP | 1)
    for r, v in ((A.UC_ARM_REG_R0, IN), (A.UC_ARM_REG_R1, n), (A.UC_ARM_REG_R2, OUT), (A.UC_ARM_REG_R3, GATE)):
        mu.reg_write(r, v)
    hs = []
    if hook:
        hs.append(mu.hook_add(UC_HOOK_CODE, hook[0]))
        hs.append(mu.hook_add(UC_HOOK_MEM_READ, hook[1], begin=FLASH, end=FLASH + 0x3FFFF))
    mu.emu_start(entry | 1, STOP)
    for h in hs:
        mu.hook_del(h)
    out = np.frombuffer(bytes(mu.mem_read(OUT, 4 * NOUT * n)), np.float32).reshape(n, NOUT)
    g = np.frombuffer(bytes(mu.mem_read(GATE, n)), np.uint8).astype(bool)
    return out, g


def cycle_fn():
    try:
        from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_MCLASS
        md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
    except Exception:
        return None
    w = {"vldr": 2, "vstr": 2, "vmla": 3, "vmls": 3, "vfma": 3, "vfms": 3, "vnmla": 3, "vnmls": 3,
         "vdiv": 14, "vsqrt": 14, "ldr": 2, "ldrb": 2, "ldrh": 2, "ldrsh": 2, "sdiv": 6, "udiv": 6}

    def cyc(code, addr):
        for ins in md.disasm(code, addr, 1):
            m = ins.mnemonic.split(".")[0]
            if m in ("push", "pop", "vpush", "vpop", "ldm", "stm", "vldmia", "vstmia", "ldmia", "stmia"):
                return 1 + max(1, ins.op_str.count(",") + 1)
            if m.startswith("b") and m not in ("bic", "bics", "bfi", "bfc"):
                return 2
            return w.get(m, 1)
        return 1
    return cyc


def count(mode_name, entry, X):
    cyc = cycle_fn()
    cache, tot = {}, {"ins": 0, "cyc": 0, "fetch": 0.0, "fread": 0}

    def on_code(uc, addr, size, _):
        tot["ins"] += 1
        tot["fetch"] += size / 4.0
        if cyc is not None:
            c = cache.get(addr)
            if c is None:
                c = cache[addr] = cyc(bytes(uc.mem_read(addr, size)), addr)
            tot["cyc"] += c

    def on_read(uc, access, addr, size, value, _):
        tot["fread"] += 1
    emulate(mode_name, entry, X, (on_code, on_read))
    return tot


def main():
    S = symbols()
    M = json.load(open(os.path.join(C.OUT, "deploy_model.json")))
    W = pd.read_pickle(os.path.join(C.OUT, "windows_primary.pkl"))
    meta, recs = P1.recordings()
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    rows, scores = [], []
    mode_used = None
    feats = C.FEATURES
    med, mu_, sd, coef = (np.array(M[k], float) for k in ("median", "mean", "sd", "coef"))
    byrec = dict(tuple(W.groupby("rec")))
    for i, (rid, ds, split, f) in enumerate(recs):
        if rid not in byrec:
            continue
        g = byrec[rid].sort_values("win_index")
        d = pd.read_csv(f, usecols=["AccV", "AccML", "AccAP"])
        acc = d.to_numpy(float)
        if ds == "tdcsfog":
            acc = acc / C.G
        x = P1.resample(acc, C.RAW_FS[ds], C.FS).astype(np.float64)
        starts = g["win_index"].to_numpy() * (N // 2)
        X = np.stack([x[s:s + N] for s in starts]).astype(np.float32)
        if mode_used is None:
            for mode_name in ("cortex-m4", "thumb2-vfp-fallback"):
                try:
                    emulate(mode_name, S["run"], X[:1])
                    mode_used = mode_name
                    break
                except UcError as e:
                    say("emulator mode %s failed: %s" % (mode_name, e))
            if mode_used is None:
                say("RESULT: emulation failed")
                sys.exit(1)
        out = np.zeros((len(X), NOUT), np.float32)
        gate = np.zeros(len(X), bool)
        for b in range(0, len(X), BATCH):
            out[b:b + BATCH], gate[b:b + BATCH] = emulate(mode_used, S["run"], X[b:b + BATCH])
        F = g[feats].to_numpy(float)
        Fi = np.where(np.isfinite(F), F, med)
        zp = ((Fi - mu_) / sd) @ coef + M["intercept"]
        gp = ((g["logP_loc"] > M["gate_logP_loc"]) & (g["gate_ac"] > M["gate_ac"])).to_numpy()
        err = np.abs(out[:, :15].astype(float) - F)
        nanmatch = np.isfinite(out[:, :15]) == np.isfinite(F)
        row = dict(rec=rid, dataset=ds, split=split, windows=len(X), gate_agree=int((gate == gp).sum()),
                   nan_pattern_agree=int(nanmatch.all(1).sum()), max_abs_logit=float(np.max(np.abs(out[:, 16] - zp))))
        for j, k in enumerate(feats):
            e = err[:, j][np.isfinite(err[:, j])]
            row["maxerr_" + k] = float(e.max()) if len(e) else 0.0
        rows.append(row)
        zc, zpy = out[gate, 16].astype(float), zp[gp]
        scores.append(dict(rec=rid, dataset=ds, kept_c=int(gate.sum() >= C.MIN_WINDOWS), kept_py=int(gp.sum() >= C.MIN_WINDOWS),
                           score_c=float(np.median(1 / (1 + np.exp(-zc)))) if len(zc) else np.nan,
                           score_py=float(np.median(1 / (1 + np.exp(-zpy)))) if len(zpy) else np.nan))
        if i % 50 == 0:
            print("recording", i + 1, "/", len(recs), flush=True)

    df = pd.DataFrame(rows)
    sc = pd.DataFrame(scores)
    df.to_csv(os.path.join(HERE, "mcu_agreement.csv"), index=False)
    sc.to_csv(os.path.join(HERE, "mcu_recording_scores.csv"), index=False)
    say("emulated core: %s" % mode_used)
    for ds, g in df.groupby("dataset"):
        s = sc[sc["dataset"] == ds]
        both = s[(s["kept_c"] == 1) & (s["kept_py"] == 1)]
        say("%-8s recordings %4d windows %6d gate agreement %.4f%% NaN-pattern agreement %.4f%% max|logit diff| %.1e" % (
            ds, len(g), g.windows.sum(), 100 * g.gate_agree.sum() / g.windows.sum(),
            100 * g.nan_pattern_agree.sum() / g.windows.sum(), g.max_abs_logit.max()))
        say("         retained recordings identical: %s (C %d, Python %d); max |recording score diff| %.1e" % (
            bool((s["kept_c"] == s["kept_py"]).all()), s["kept_c"].sum(), s["kept_py"].sum(),
            float(np.max(np.abs(both["score_c"] - both["score_py"]))) if len(both) else float("nan")))
    say("max abs feature error over all windows: " + ", ".join(
        "%s %.1e" % (k, df["maxerr_" + k].max()) for k in feats))

    # cost per window: difference between 9 windows and 1 window removes the entry overhead
    rid, ds, split, f = [r for r in recs if r[1] == "defog" and r[0] in byrec][0]
    g = byrec[rid].sort_values("win_index")
    d = pd.read_csv(f, usecols=["AccV", "AccML", "AccAP"]).to_numpy(float)
    x = P1.resample(d, C.RAW_FS[ds], C.FS)
    X = np.stack([x[s:s + N] for s in g["win_index"].to_numpy()[:9] * (N // 2)]).astype(np.float32)
    one = count(mode_used, S["run"], X[:1])
    many = count(mode_used, S["run"], X)
    k = 8.0
    ins = (many["ins"] - one["ins"]) / k
    cyc = (many["cyc"] - one["cyc"]) / k
    fetch = (many["fetch"] - one["fetch"]) / k
    fread = (many["fread"] - one["fread"]) / k
    say("per 6-s window (one window every %.0f s):" % WINDOW_PERIOD_S)
    say("  executed instructions %.0f, 32-bit instruction fetches %.0f, data reads from flash %.0f" % (ins, fetch, fread))
    say("  lower bound (1 cycle per instruction): %.2f ms at 64 MHz" % (ins / CLOCK_HZ * 1e3))
    if cyc > 0:
        say("  approx. core cycles (Cortex-M4 TRM timings): %.0f" % cyc)
        for ws in FLASH_WS:
            c = cyc + ws * (fetch + fread)
            ms = c / CLOCK_HZ * 1e3
            ua = RUN_UA_PER_MHZ * CLOCK_HZ / 1e6 * (ms / 1e3) / WINDOW_PERIOD_S
            say("  flash wait states %d, no cache (upper bound): %.0f cycles = %.2f ms, CPU duty %.3f%%, "
                "estimated average CPU current %.2f uA above %.2f uA idle" % (ws, c, ms, 100 * ms / 1e3 / WINDOW_PERIOD_S, ua, IDLE_UA))
    for l in open(os.path.join(HERE, "size.txt")):
        if any(t in l for t in (".text", ".rodata", ".bss", ".data")):
            say("  " + l.strip())
    for l in open(os.path.join(HERE, "fog_state.su")):
        say("  stack " + l.strip())
    open(os.path.join(HERE, "mcu_results.txt"), "w").write("\n".join(lines) + "\n")
    print("saved mcu_results.txt, mcu_agreement.csv and mcu_recording_scores.csv")


if __name__ == "__main__":
    main()
