# File: led_optimization.py
# Path: ./led_optimization.py
#
# Usage:
#  1) put your SPD CSV at ./data/Problem2_LED_SPD.csv
#     CSV columns: Wavelength, Blue, Green, Red, Warm White, Cold White
#     (accepts Chinese headers "波长" etc. - script will try to match common names)
#  2) optional: put melanopic response CSV at ./data/melanopic_response.csv
#     columns: Wavelength, Melanopic (values aligned to 380-780)
#  3) pip install -r requirements (see below)
#  4) run: python led_optimization.py
#
# Dependencies (example):
# pip install numpy pandas scipy matplotlib colour-science

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# Try importing colour; if missing provide friendly error and fallback hints
try:
    import colour
    from colour import SDS_ILLUMINANTS
except Exception as e:
    colour = None
    print("Warning: 'colour' library not available or import failed. "
          "Install with `pip install colour-science` for accurate colorimetric calculations.")
    print("Error:", e)

# --------------------------
# User-editable parameters
# --------------------------
INPUT_CSV = "data/Problem2_LED_SPD.csv"
MEL_RESPONSE_CSV = "data/melanopic_response.csv"   # optional fallback
OUT_DIR = "out"
WAVELENGTH_MIN = 380
WAVELENGTH_MAX = 780
WAVELENGTH_STEP = 1
WAVELENGTHS = np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + 1, WAVELENGTH_STEP)

# Constraints & definitions
DAY_CCT_MIN, DAY_CCT_MAX = 5500.0, 6500.0   # 6000 +/- 500 -> 5500-6500
NIGHT_CCT_MIN, NIGHT_CCT_MAX = 2500.0, 3500.0
DAY_RG_MIN, DAY_RG_MAX = 95.0, 105.0
DAY_RF_MIN = 88.0
NIGHT_RF_MIN = 80.0

# Optimization settings
DE_POPSIZE = 40  # differential evolution population size
DE_MAXITER = 200

# Create out dir
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------
# Utilities: load SPD CSV
# --------------------------
def load_spd_table(csv_path):
    """
    Expect a table with first column wavelength and 5 columns for channels:
    Blue, Green, Red, Warm White, Cold White (names flexible).
    Returns wavelengths array and dict of channel_name -> values (same length).
    """
    df = pd.read_csv(csv_path)
    # normalize column names
    cols = [c.strip().lower() for c in df.columns]
    # find wavelength column
    wl_idx = None
    for i, c in enumerate(cols):
        if "wave" in c or "wl" in c or "波长" in c or "wavelength" in c:
            wl_idx = i
            break
    if wl_idx is None:
        raise ValueError("Cannot find wavelength column in input CSV. Column names: " + ", ".join(df.columns))
    wl = df.iloc[:, wl_idx].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    # find channels: try to match keywords
    channel_map = {}
    for i, c in enumerate(cols):
        if i == wl_idx:
            continue
        # detect basic names
        if any(k in c for k in ["blue", "b 蓝"]):
            channel_map["Blue"] = df.iloc[:, i].to_numpy(dtype=float)
        elif any(k in c for k in ["green", "g 绿"]):
            channel_map["Green"] = df.iloc[:, i].to_numpy(dtype=float)
        elif any(k in c for k in ["red", "r 红"]):
            channel_map["Red"] = df.iloc[:, i].to_numpy(dtype=float)
        elif "warm" in c or "ww" in c or "warm white" in c or "暖" in c:
            channel_map["Warm White"] = df.iloc[:, i].to_numpy(dtype=float)
        elif "cold" in c or "cw" in c or "cold white" in c or "冷" in c:
            channel_map["Cold White"] = df.iloc[:, i].to_numpy(dtype=float)
        else:
            # fallback: include any remaining as extra channels but warn
            channel_map[df.columns[i]] = df.iloc[:, i].to_numpy(dtype=float)

    # check we have at least 5 channels
    if len(channel_map) < 5:
        print("Warning: detected channels:", list(channel_map.keys()))
        print("Expecting 5 channels (Blue, Green, Red, Warm White, Cold White). If names differ, please ensure CSV headers.")
    return wl, channel_map


def resample_to_common_grid(wl_src, spd_src, wl_target=WAVELENGTHS):
    """Interpolate source SPD (wl_src, spd_src) to wl_target grid."""
    return np.interp(wl_target, wl_src, spd_src, left=0.0, right=0.0)


# --------------------------
# Colourmetric helpers
# --------------------------
def sd_to_XYZ(wl, spd):
    if colour is None:
        raise RuntimeError("colour library required for accurate XYZ calculations.")
    # make spectral distribution
    sd = colour.SpectralDistribution(dict(zip(wl, spd)), name="mix")
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    # interpolate to match cmfs shape (spectral domain)
    sd = sd.interpolate(cmfs.shape)  # 这里去掉 .range
    XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs)
    return np.asarray(XYZ)



def xyz_to_xy(XYZ):
    X, Y, Z = XYZ
    denom = X + Y + Z
    if denom == 0:
        return (0.0, 0.0)
    return (X / denom, Y / denom)


def xy_to_CCT_Duv(xy):
    """Compute CCT and Duv using colour library where possible.
    Returns (CCT_in_K, Duv)."""
    if colour is None:
        raise RuntimeError("colour library required for accurate CCT/Duv calculation.")

    try:
        # 使用colour 0.4.4版本支持的接口计算CCT
        CCT = colour.xy_to_CCT_CIE_D(xy)
    except Exception:
        # 兜底值，防止失败
        CCT = 0.0

    try:
        # 计算u'v'
        uv = colour.xy_to_uv(xy)
        # 计算对应黑体辐射的xy，注意函数名也可能有差异，尝试这个：
        xy_bb = colour.temperature.CCT_to_xy(CCT)
        uv_bb = colour.xy_to_uv(xy_bb)
        # 计算欧氏距离作为Duv近似
        Duv = np.linalg.norm(np.array(uv) - np.array(uv_bb))
    except Exception:
        Duv = 0.0

    return float(CCT), float(Duv)


# --------------------------
# Melanopic response (CIE S-026)
# --------------------------
def load_melanopic_response(wl_target=WAVELENGTHS):
    """Try to obtain melanopic action spectrum. Prefer colour's built-in; else load from CSV; else return None."""
    # 1) try colour (if it exposes photoreceptor sensitivity)
    if colour is not None:
        try:
            # Many versions of colour expose photoreceptor sensitivity functions under SDS
            if 'CIE 2018 MSC' in colour.sd:
                pass
        except Exception:
            pass
    # 2) try loading file
    if os.path.exists(MEL_RESPONSE_CSV):
        df = pd.read_csv(MEL_RESPONSE_CSV)
        # Expect columns Wavelength, Melanopic
        wl_src = df.iloc[:, 0].to_numpy(dtype=float)
        mel_src = df.iloc[:, 1].to_numpy(dtype=float)
        mel_resampled = resample_to_common_grid(wl_src, mel_src, wl_target)
        return mel_resampled
    # 3) no data -> return None
    return None


def compute_melanopic_photopic_ratio(wl, spd, melanopic_response):
    """
    Compute melanopic (Emel) and photopic (Ev) and return ratio Emel/Ev.
    melanopic_response: array aligned to wl (same shape) OR None -> fallback using approximation.
    """
    # photopic luminous efficacy function V(lambda)
    if colour is None:
        raise RuntimeError("colour library recommended for photopic luminous efficiency.")
    V = colour.SDS_V['Photopic']  # returns spectral distribution
    # ensure V on same wavelength grid
    V_vals = np.array([V.get(w, 0.0) for w in wl])  # may be slow; alternative: interpolate V
    # better: create spectral distribution for V and sample:
    V_sd = colour.SpectralDistribution(dict(zip(range(len(wl)), V_vals)))  # placeholder (not used further)
    # But simplest: use colour.sd_to_XYZ to compute photopic response via Y (since Y is photopic luminance)
    # Compute photopic illuminance (relative):
    # Convert spd -> XYZ then Y is photopic luminous response (relative)
    XYZ = sd_to_XYZ(wl, spd)
    Ev = float(XYZ[1])  # Y channel proportional to photopic response
    # melanopic:
    if melanopic_response is not None:
        Emel = np.trapz(spd * melanopic_response, wl)
        # To put on similar scale as Ev, we can compute ratio Emel / Ev (Ev maybe in different units); hence mel-DER defined as:
        if Ev == 0:
            return 0.0
        return float(Emel / Ev)
    else:
        # fallback: approximate melanopic with a weighted short-wave sensitivity
        # This is crude: approximate melanopic by weighting near 480 nm more
        weights = np.exp(-0.5 * ((wl - 480.0) / 30.0) ** 2)
        Emel = np.trapz(spd * weights, wl)
        if Ev == 0:
            return 0.0
        return float(Emel / Ev)


# --------------------------
# TM-30 (Rf, Rg) placeholder
# --------------------------
def compute_tm30_metrics(wl, spd):
    """
    Attempt to compute TM-30 Rf and Rg using 'colour' if available.
    If not available, return None for Rf/Rg and save SPD for offline TM-30.
    """
    if colour is None:
        return None, None
    try:
        # colour.characterisation.tm30 exists in some versions; usage may vary.
        # We attempt a common usage pattern; if it fails, fallback to None.
        sd = colour.SpectralDistribution(dict(zip(wl, spd)))
        # some colour versions require a domain and illuminant reference
        result = colour.characterisation.tm301(sd)  # may raise
        # result is typically a dict with 'Rf' 'Rg' keys
        Rf = result.get('Rf', None)
        Rg = result.get('Rg', None)
        return Rf, Rg
    except Exception:
        # attempt alternate API
        try:
            sd = colour.SpectralDistribution(dict(zip(wl, spd)))
            result = colour.characterisation.tm30(sd)
            Rf = result.get('Rf', None)
            Rg = result.get('Rg', None)
            return Rf, Rg
        except Exception:
            # fallback: save SPD to file for external TM-30 calculation
            out_spd_path = os.path.join(OUT_DIR, "mix_spd_for_tm30.csv")
            pd.DataFrame({"Wavelength": wl, "SPD": spd}).to_csv(out_spd_path, index=False)
            print(f"TM-30 calculation not available in this environment. Saved SPD to {out_spd_path} for offline TM-30 analysis.")
            return None, None


# --------------------------
# Aggregate metrics computation
# --------------------------
def compute_all_metrics(wl, spd, melanopic_response=None, reference_d65=None):
    """
    Returns a dict with keys: CCT, Duv, Rf, Rg, melDER
    melDER defined as (Emel/Ev) / (Emel/Ev for D65) if reference_d65 provided; otherwise raw Emel/Ev.
    """
    xyz = sd_to_XYZ(wl, spd)
    xy = xyz_to_xy(xyz)
    CCT, Duv = xy_to_CCT_Duv(xy)
    Rf, Rg = compute_tm30_metrics(wl, spd)
    mel_ratio = compute_melanopic_photopic_ratio(wl, spd, melanopic_response)
    # compute reference D65 ratio if possible
    mel_der = mel_ratio
    if reference_d65 is not None:
        ref_spd = reference_d65
        ref_mel = compute_melanopic_photopic_ratio(wl, ref_spd, melanopic_response)
        if ref_mel != 0:
            mel_der = mel_ratio / ref_mel
    return {"CCT": CCT, "Duv": Duv, "Rf": Rf, "Rg": Rg, "mel-DER": mel_der, "xy": xy}


# --------------------------
# Mix SPD utility
# --------------------------
def mix_spds(spd_list, weights):
    """
    spd_list: list of arrays each length len(WAVELENGTHS)
    weights: array-like of same length, assumed non-negative and sum to 1
    returns mix array
    """
    arr = np.vstack(spd_list)  # shape (n_channels, n_wl)
    mix = np.dot(weights, arr)
    return mix


# --------------------------
# Feasibility pre-scan (random sampling)
# --------------------------
def feasibility_scan(spd_list, melanopic_response=None, reference_d65=None, n_samples=2000):
    """
    Randomly sample weight vectors on simplex and compute metrics to check whether constraints feasible.
    Returns a small report (min/max ranges observed) and sample of feasible points.
    """
    n = len(spd_list)
    rng = np.random.default_rng(seed=0)
    feasible_samples = []
    observed = []
    for _ in range(n_samples):
        # sample on simplex via Dirichlet
        w = rng.dirichlet(np.ones(n))
        mix = mix_spds(spd_list, w)
        metrics = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)
        observed.append(metrics)
        # check day constraints
        ok_day = (DAY_CCT_MIN <= metrics["CCT"] <= DAY_CCT_MAX) and \
                 (metrics["Rg"] is not None and DAY_RG_MIN <= metrics["Rg"] <= DAY_RG_MAX) and \
                 (metrics["Rf"] is not None and metrics["Rf"] >= DAY_RF_MIN)
        ok_night = (NIGHT_CCT_MIN <= metrics["CCT"] <= NIGHT_CCT_MAX) and \
                   (metrics["Rf"] is not None and metrics["Rf"] >= NIGHT_RF_MIN)
        feasible_samples.append({"w": w, "metrics": metrics, "ok_day": ok_day, "ok_night": ok_night})
    return feasible_samples


# --------------------------
# Optimization routines
# --------------------------
def optimize_day(spd_list, melanopic_response=None, reference_d65=None):
    """
    Day optimization: maximize Rf subject to constraints:
      CCT in [DAY_CCT_MIN, DAY_CCT_MAX], Rg in [DAY_RG_MIN, DAY_RG_MAX], Rf >= DAY_RF_MIN
    Use SLSQP with equality sum(w)-1 and bounds [0,1].
    If TM-30 unavailable (Rf/Rg None), optimization will fallback to penalized objective on CCT only.
    """
    n = len(spd_list)
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n

    def constraints_sum(w):
        return np.sum(w) - 1.0

    def objective_negRf(w):
        mix = mix_spds(spd_list, w / np.sum(w))
        metrics = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)
        Rf = metrics.get("Rf")
        if Rf is None:
            # no Rf available -> encourage CCT close to 6000 as proxy
            cct = metrics.get("CCT", 6000)
            # penalize distance to ideal 6000
            return abs(cct - 6000.0)
        return -float(Rf)

    cons = [{'type': 'eq', 'fun': constraints_sum}]

    # add inequality constraints for CCT, Rg, Rf when available (wrapped with try)
    def cct_low(w):
        mix = mix_spds(spd_list, w / np.sum(w))
        return compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)["CCT"] - DAY_CCT_MIN

    def cct_high(w):
        mix = mix_spds(spd_list, w / np.sum(w))
        return DAY_CCT_MAX - compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)["CCT"]

    cons.append({'type': 'ineq', 'fun': cct_low})
    cons.append({'type': 'ineq', 'fun': cct_high})

    # Rg and Rf constraints will be soft if not computable; we add them defensively
    def rg_low(w):
        mix = mix_spds(spd_list, w / np.sum(w))
        val = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65).get("Rg")
        if val is None:
            return -1.0  # allow (ineffective)
        return val - DAY_RG_MIN

    def rg_high(w):
        mix = mix_spds(spd_list, w / np.sum(w))
        val = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65).get("Rg")
        if val is None:
            return 1.0
        return DAY_RG_MAX - val

    def rf_low(w):
        mix = mix_spds(spd_list, w / np.sum(w))
        val = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65).get("Rf")
        if val is None:
            return 1.0
        return val - DAY_RF_MIN

    cons.append({'type': 'ineq', 'fun': rg_low})
    cons.append({'type': 'ineq', 'fun': rg_high})
    cons.append({'type': 'ineq', 'fun': rf_low})

    res = minimize(objective_negRf, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'ftol': 1e-6, 'maxiter': 500, 'disp': True})
    if not res.success:
        print("Day optimization did not converge (message):", res.message)
    w_opt = np.clip(res.x, 0.0, 1.0)
    if np.sum(w_opt) == 0:
        w_opt = np.ones_like(w_opt) / len(w_opt)
    else:
        w_opt = w_opt / np.sum(w_opt)
    mix = mix_spds(spd_list, w_opt)
    metrics = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)
    return w_opt, mix, metrics, res


def optimize_night(spd_list, melanopic_response=None, reference_d65=None):
    """
    Night optimization: minimize mel-DER subject to CCT in [NIGHT_CCT_MIN,NIGHT_CCT_MAX] and Rf >= NIGHT_RF_MIN.
    Use differential evolution with simplex constraint via normalization in objective and penalty for constraints.
    """
    n = len(spd_list)

    def obj_penalized(x):
        # x is unconstrained in [0,1] by bounds; normalize
        x = np.clip(x, 0.0, 1.0)
        if np.sum(x) == 0:
            w = np.ones(n) / n
        else:
            w = x / np.sum(x)
        mix = mix_spds(spd_list, w)
        metrics = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)
        mel = metrics.get("mel-DER", None)
        if mel is None:
            mel = 0.0
        penalty = 0.0
        cct = metrics.get("CCT", 3000)
        rf = metrics.get("Rf", None)
        # penalize CCT out of range
        if cct < NIGHT_CCT_MIN:
            penalty += (NIGHT_CCT_MIN - cct) * 0.1
        if cct > NIGHT_CCT_MAX:
            penalty += (cct - NIGHT_CCT_MAX) * 0.1
        # penalize Rf < min
        if rf is not None and rf < NIGHT_RF_MIN:
            penalty += (NIGHT_RF_MIN - rf) * 0.5
        # objective is mel + penalty
        return float(mel + penalty)

    bounds = [(0.0, 1.0)] * n
    res = differential_evolution(obj_penalized, bounds, maxiter=DE_MAXITER, popsize=DE_POPSIZE, disp=True)
    x = np.clip(res.x, 0.0, 1.0)
    if np.sum(x) == 0:
        w = np.ones(n) / n
    else:
        w = x / np.sum(x)
    mix = mix_spds(spd_list, w)
    metrics = compute_all_metrics(WAVELENGTHS, mix, melanopic_response, reference_d65)
    return w, mix, metrics, res


# --------------------------
# Main run
# --------------------------
def main():
    # 1) load data
    if not os.path.exists(INPUT_CSV):
        print(f"Input SPD CSV not found at {INPUT_CSV}. Please place your Problem2 file there.")
        sys.exit(1)
    wl_src, channels = load_spd_table(INPUT_CSV)
    # check expected channel order and fill missing with zeros if needed
    required_keys = ["Blue", "Green", "Red", "Warm White", "Cold White"]
    spd_list = []
    chan_names = []
    for key in required_keys:
        if key in channels:
            spd_list.append(resample_to_common_grid(wl_src, channels[key], WAVELENGTHS))
            chan_names.append(key)
        else:
            # if missing, add zeros and warn
            spd_list.append(np.zeros_like(WAVELENGTHS, dtype=float))
            chan_names.append(key)
            print(f"Warning: channel {key} not found in CSV; using zeros for this channel.")

    # 2) try load melanopic response and D65 reference
    melanopic_response = load_melanopic_response(WAVELENGTHS)
    if melanopic_response is None:
        print("Melanopic response not found. For accurate mel-DER, provide CIE S-026 melanopic action spectrum "
              f"in {MEL_RESPONSE_CSV}. Using crude approximation instead (not recommended for final results).")

    # reference D65 SPD - try to get from colour if available
    reference_d65 = None
    if colour is not None:
        try:
            d65 = colour.SDS_ILLUMINANTS['D65']
            # sample D65 to our grid
            d65_vals = np.array([d65.get(w, 0.0) for w in WAVELENGTHS])
            reference_d65 = d65_vals
        except Exception:
            reference_d65 = None

    # 3) feasibility scan (optional but useful)
    print("Running feasibility scan (random sampling) to gauge constraint space (this may take a few seconds)...")
    samples = feasibility_scan(spd_list, melanopic_response, reference_d65, n_samples=800)
    ok_day_count = sum(1 for s in samples if s["ok_day"])
    ok_night_count = sum(1 for s in samples if s["ok_night"])
    print(f"Feasibility scan results: {ok_day_count} samples satisfy day constraints; {ok_night_count} satisfy night constraints (out of {len(samples)} samples).")

    # 4) optimize day
    print("Starting day optimization (maximize Rf with constraints)...")
    w_day, mix_day, metrics_day, res_day = optimize_day(spd_list, melanopic_response, reference_d65)
    print("Day metrics:", metrics_day)
    print("Day weights:", dict(zip(chan_names, w_day.round(4))))

    # save results
    df_mix_day = pd.DataFrame({"Wavelength": WAVELENGTHS, "SPD": mix_day})
    df_mix_day.to_csv(os.path.join(OUT_DIR, "mix_day_spd.csv"), index=False)

    # 5) optimize night
    print("Starting night optimization (minimize mel-DER with constraints)...")
    w_night, mix_night, metrics_night, res_night = optimize_night(spd_list, melanopic_response, reference_d65)
    print("Night metrics:", metrics_night)
    print("Night weights:", dict(zip(chan_names, w_night.round(4))))
    df_mix_night = pd.DataFrame({"Wavelength": WAVELENGTHS, "SPD": mix_night})
    df_mix_night.to_csv(os.path.join(OUT_DIR, "mix_night_spd.csv"), index=False)

    # 6) save weights and metrics
    df_weights = pd.DataFrame([w_day, w_night], columns=chan_names, index=["Day_opt", "Night_opt"])
    df_weights.to_csv(os.path.join(OUT_DIR, "optimal_weights.csv"))
    df_metrics = pd.DataFrame([metrics_day, metrics_night], index=["Day_opt", "Night_opt"])
    df_metrics.to_csv(os.path.join(OUT_DIR, "optimal_metrics.csv"))

    # 7) plot outputs
    plt.figure(figsize=(8, 4))
    for i, spd in enumerate(spd_list):
        plt.plot(WAVELENGTHS, spd / (np.max(spd) + 1e-12), label=f"{chan_names[i]} (norm)")
    plt.plot(WAVELENGTHS, mix_day / (np.max(mix_day) + 1e-12), label="Day mix (norm)", linewidth=2, color='black')
    plt.plot(WAVELENGTHS, mix_night / (np.max(mix_night) + 1e-12), label="Night mix (norm)", linewidth=2, color='magenta')
    plt.xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative SPD (norm)")
    plt.legend()
    plt.title("Channel SPDs and Day/Night Mixes (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spds_and_mixes.png"), dpi=200)
    plt.close()

    # weight bars
    plt.figure(figsize=(6, 3))
    x = np.arange(len(chan_names))
    plt.bar(x - 0.15, w_day, width=0.3, label='Day')
    plt.bar(x + 0.15, w_night, width=0.3, label='Night')
    plt.xticks(x, chan_names, rotation=20)
    plt.ylabel("Weight (sum=1)")
    plt.legend()
    plt.title("Optimized Channel Weights")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "weights.png"), dpi=200)
    plt.close()

    print("Optimization finished. Outputs saved in", OUT_DIR)
    print("Files: mix_day_spd.csv, mix_night_spd.csv, optimal_weights.csv, optimal_metrics.csv, spds_and_mixes.png, weights.png")
    # if TM-30 couldn't compute, inform user
    if metrics_day.get("Rf") is None or metrics_night.get("Rf") is None:
        print("Note: TM-30 (Rf/Rg) not computed in this environment. If you need precise TM-30 results, install a colour version with TM-30 support or compute TM-30 separately using the saved SPD files in the out/ directory.")


if __name__ == "__main__":
    main()
