/* OFF-state estimator for one 6-s window of lower-back acceleration (16 Hz, 96 samples, axes V, ML, AP in g).
 * Bare-metal C for a Cortex-M4F (single-precision FPU), no libc.
 * Reproduces p_features.window_features(), the frozen gait gate and the frozen pooled logistic model.
 * Entry point:  int fog_window(const float *x, float *out)
 *   x   : 96 x 3 floats, row-major (V, ML, AP per sample)
 *   out : 15 features (C.FEATURES order), gate_ac, logit  -> 17 floats
 *   returns 1 if the window passes the gait gate, else 0
 */
#include <stdint.h>
#include "fog_params.h"

#define N 96
#define NAX 3
#define EPS 1e-12f

static float ln_f(float x) {                 /* natural log, |rel err| < 1e-7 for x > 0 */
  union { float f; uint32_t u; } v = { x };
  int e = (int)((v.u >> 23) & 0xFF) - 127;
  v.u = (v.u & 0x007FFFFF) | 0x3F800000;
  float m = v.f;
  if (m > 1.41421356f) { m *= 0.5f; e += 1; }
  float t = (m - 1.0f) / (m + 1.0f), t2 = t * t;
  float s = t * (2.0f + t2 * (0.6666667f + t2 * (0.4f + t2 * (0.2857143f + t2 * 0.2222222f))));
  return s + (float)e * 0.69314718f;
}

static float sqrt_f(float x) { return __builtin_sqrtf(x); }   /* VSQRT.F32 with -fno-math-errno */

static float NAN_F(void) { union { uint32_t u; float f; } v = { 0x7FC00000u }; return v.f; }
static int is_nan(float x) { return x != x; }

static float xc[NAX][N];        /* mean-removed axes */
static float xw[NAX][N];        /* mean-removed and Hann-windowed axes */
static float ac[N];             /* autocorrelation of V, lags 0..AC_MAX */

/* power |X_k|^2 of the real DFT bin k of one windowed axis (direct sum with the cosine table) */
static float bin_power(const float *v, int k) {
  float re = 0.0f, im = 0.0f;
  uint32_t idx = 0, sdx = N - N / 4;          /* sin(2*pi*i/N) = cos(2*pi*(i - N/4)/N) */
  for (int n = 0; n < N; n++) {
    re += v[n] * COS_T[idx];
    im -= v[n] * COS_T[sdx];
    idx += (uint32_t)k;
    if (idx >= N) idx -= N;
    sdx += (uint32_t)k;
    if (sdx >= N) sdx -= N;
  }
  return re * re + im * im;
}

static float harmonic_ratio(int axis, int kstep, int inverted) {
  float even = 0.0f, odd = 0.0f;
  int count = 0;
  for (int j = 1; j <= 10; j++) {
    int b = HR_IDX[kstep][j - 1];
    if (b < 0) break;
    float a = sqrt_f(bin_power(xw[axis], b));
    if (j % 2 == 0) even += a; else odd += a;
    count++;
  }
  if (count < 2) return NAN_F();
  if (inverted) { float t = even; even = odd; odd = t; }
  return odd > 0.0f ? even / odd : NAN_F();
}

int fog_window(const float *x, float *out) {
  float rms[NAX];
  for (int a = 0; a < NAX; a++) {
    float m = 0.0f;
    for (int n = 0; n < N; n++) m += x[n * NAX + a];
    m *= (1.0f / N);
    float ss = 0.0f;
    for (int n = 0; n < N; n++) {
      float d = x[n * NAX + a] - m;
      xc[a][n] = d;
      xw[a][n] = d * HANN[n];
      ss += d * d;
    }
    rms[a] = sqrt_f(ss * (1.0f / N));
  }

  /* band powers (sum over axes) and the vertical spectrum peak */
  float p_loc = 0.0f, p_frz = 0.0f, pv_band = 0.0f, pv_max = -1.0f;
  int kmax = K_LOC0;
  for (int k = K_LOC0; k <= K_FRZ1; k++) {
    float pv = bin_power(xw[0], k);
    float ps = pv + bin_power(xw[1], k) + bin_power(xw[2], k);
    if (k <= K_LOC1) {
      p_loc += ps;
      if (pv > pv_max) { pv_max = pv; kmax = k; }
    } else {
      p_frz += ps;
    }
    pv_band += pv;
  }
  p_loc += EPS;
  p_frz += EPS;
  float f_step = (float)kmax * (16.0f / N);
  float step_peak_rel = pv_max / (pv_band + EPS);

  /* linear autocorrelation of the vertical axis (mean removed), normalised by lag 0 */
  for (int L = 0; L <= AC_MAX; L++) {
    float s = 0.0f;
    for (int n = 0; n + L < N; n++) s += xc[0][n] * xc[0][n + L];
    ac[L] = s;
  }
  float a0 = ac[0];
  for (int L = 0; L <= AC_MAX; L++) ac[L] = a0 > 0.0f ? ac[L] / a0 : 0.0f;

  int lag1 = AC_LO;
  for (int L = AC_LO + 1; L < AC_HI; L++) if (ac[L] > ac[lag1]) lag1 = L;
  float step_reg = ac[lag1];
  int l2lo = L2LO[lag1], l2hi = L2HI[lag1];
  float stride_reg = NAN_F();
  if (l2hi > l2lo + 1) {
    int b = l2lo;
    for (int L = l2lo + 1; L < l2hi; L++) if (ac[L] > ac[b]) b = L;
    stride_reg = ac[b];
  }
  float symmetry = (!is_nan(stride_reg) && stride_reg > 0.0f) ? step_reg / stride_reg : NAN_F();

  float hr_v = harmonic_ratio(0, kmax, 0);
  float hr_ap = harmonic_ratio(2, kmax, 0);
  float hr_ml = harmonic_ratio(1, kmax, 1);

  float js = 0.0f;
  for (int n = 0; n + 1 < N; n++) {
    for (int a = 0; a < NAX; a++) {
      float d = (x[(n + 1) * NAX + a] - x[n * NAX + a]) * 16.0f;
      js += d * d;
    }
  }
  float log_jerk = ln_f(sqrt_f(js / (float)(N - 1)) + EPS);

  float f[NFEAT];
  f[0] = ln_f(p_loc);
  f[1] = ln_f(p_frz);
  f[2] = ln_f(p_frz / p_loc);
  f[3] = rms[0]; f[4] = rms[1]; f[5] = rms[2];
  f[6] = f_step; f[7] = step_peak_rel;
  f[8] = hr_v; f[9] = hr_ap; f[10] = hr_ml;
  f[11] = step_reg; f[12] = stride_reg; f[13] = symmetry;
  f[14] = log_jerk;

  float z = W_B;
  for (int j = 0; j < NFEAT; j++) {
    out[j] = f[j];
    float v = is_nan(f[j]) ? W_MED[j] : f[j];
    z += W_COEF[j] * (v - W_MU[j]) / W_SD[j];
  }
  out[NFEAT] = step_reg;
  out[NFEAT + 1] = z;
  return (f[0] > GATE_LOGP && step_reg > GATE_AC) ? 1 : 0;
}

/* emulator entry: process n windows stored back to back; out has 17 floats per window,
 * gate flags written to g. Returns n. */
int run(const float *x, int n, float *out, uint8_t *g) {
  for (int i = 0; i < n; i++) g[i] = (uint8_t)fog_window(x + i * N * NAX, out + i * (NFEAT + 2));
  return n;
}
