"""
Reproducible analysis for the manuscript
"A Two-Decade Decline in Daytime Land Surface Temperature over Ajmer, Semi-Arid Northwest India".

Usage:  python src/ajmer_lst_analysis.py [data_dir] [results_dir]
        (defaults: ./data and ./results)

Inputs (relative to data_dir), all individual CSV files:
  MOD11A2/MOD11A2.csv, MYD11A2/MYD11A2.csv       ORNL DAAC pixel subsets (17x17 x 1 km; LST, QC, view time/angle, clear-sky)
  MOD11A2/statistics_LST_{Day,Night}_1km.csv     ORNL DAAC scene statistics (QA-filtered); same for MYD11A2/
  VNP21A2/statistics_LST_Day_1KM.csv             VIIRS scene statistics
  MCD12Q1/MCD12Q1_LC_Type1_2019_1km.csv          land cover, 2019
  MCD12Q2/MCD12Q2.csv                            phenology, 2001-2019
  ERA5Land/hourly/era5land_ajmer_YYYY.csv        ERA5-Land hourly skt, t2m at the UTC hours bracketing each overpass
  ERA5Land/era5land_daily_ajmer.csv              ERA5-Land daily aggregates

Outputs: results/summary.txt plus one CSV per table / intermediate series.
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstlib import mk_p, sen, fmt, month_anom, annual, detrend, partial_r, ols, read_pixels, popcount, interp_hourly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
LON = 74.64105            # window centre, deg E
UTC_OFFSET_H = LON / 15.  # local solar time -> UTC
Y0, Y1 = 2001, 2021       # complete calendar years for all trends
FIRST_FULL_YEAR = {"Terra": 2001, "Aqua": 2003}  # Aqua begins 4 Jul 2002; its partial 2002 is excluded everywhere
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# ---------------------------------------------------------------- MODIS inputs
lc = open(os.path.join(DATA_DIR, "MCD12Q1", "MCD12Q1_LC_Type1_2019_1km.csv")).read().strip().split(",")[6:]
lc = pd.Series(lc).astype(int).value_counts().sort_index()
log("MCD12Q1 2019 LC_Type1 counts in window (12=cropland, 13=urban, 16=barren):", {int(k): int(v) for k, v in lc.items()}, "total", int(lc.sum()))
PLAT = {"Terra": ("MOD11A2", "MOD11A2"), "Aqua": ("MYD11A2", "MYD11A2")}
scenes = {}
for plat, (prod, fn) in PLAT.items():
    folder = os.path.join(DATA_DIR, prod)
    px = read_pixels(os.path.join(folder, f"{fn}.csv"))
    log(f"{plat}: {len(px['LST_Day_1km'])} composites, {px['LST_Day_1km'].index.min().date()} to {px['LST_Day_1km'].index.max().date()}")
    for tod in ["Day", "Night"]:
        st = pd.read_csv(os.path.join(folder, f"statistics_LST_{tod}_1km.csv"), parse_dates=["date"]).set_index("date")
        lst_raw = px[f"LST_{tod}_1km"]; valid = lst_raw > 0
        vt = px[f"{tod}_view_time"].where(valid & (px[f"{tod}_view_time"] != 255)) * 0.1          # h, local solar
        va = (px[f"{tod}_view_angl"].where(valid & (px[f"{tod}_view_angl"] != 255)) - 65).abs()   # deg, |zenith|
        cs = popcount(px[f"Clear_sky_{tod.lower()}s"].fillna(0)).astype(float)                    # clear days / 8
        df = pd.DataFrame({"lst": st["mean"], "qa_pct": st["per_cent_pixels_pass_qa"],
                           "vtime": vt.mean(axis=1), "vang": va.mean(axis=1), "clear": cs.mean(axis=1)})
        df = df[(df.lst >= 250) & (df.lst <= 340)]
        if plat == "Aqua":
            df = df[df.index.year >= FIRST_FULL_YEAR["Aqua"]]   # drop partial 2002 from all Aqua analyses
        df["anom"] = month_anom(df, "lst")
        scenes[(plat, tod)] = df
        log(f"  {tod}: {len(df)} valid scenes; LST {df.lst.min():.1f}-{df.lst.max():.1f} K; "
            f"mean overpass {df.vtime.mean():.2f} h; mean |VZA| {df.vang.mean():.1f} deg; QA {df.qa_pct.mean():.1f}%")

# ---------------------------------------------------------------- ERA5-Land matched to overpass
hr = pd.concat(pd.read_csv(f, parse_dates=["datetime_utc"]) for f in sorted(glob.glob(os.path.join(DATA_DIR, "ERA5Land", "hourly", "era5land_ajmer_*.csv"))))
hr = hr.drop_duplicates("datetime_utc").set_index("datetime_utc").sort_index()

def era5_at(times_utc, var):
    return interp_hourly(hr.index.values, hr[var].values, times_utc)

for key, df in scenes.items():
    for var in ["skt", "t2m"]:
        vals = []
        for d, lt in zip(df.index, df.vtime):
            if np.isnan(lt): vals.append(np.nan); continue
            days = pd.date_range(d, periods=8, freq="D")
            tu = days + pd.to_timedelta(lt - UTC_OFFSET_H, unit="h")
            vals.append(np.nanmean(era5_at(tu.values, var)))
        df[f"e_{var}"] = vals
        df[f"e_{var}_anom"] = month_anom(df, f"e_{var}")

# ---------------------------------------------------------------- 3.1 / Table 1
log("\n== Table 1: trends 2001-2021, K/decade ==")
T1 = []
for (plat, tod), df in scenes.items():
    row = {"series": f"{plat} {tod.lower()}time"}
    for lab, col in [("MODIS", "anom"), ("ERA5 skt", "e_skt_anom"), ("ERA5 t2m", "e_t2m_anom")]:
        r = sen(annual(df[col].dropna())); row[lab] = fmt(r); row.update({f"{lab}_{k}": v for k, v in r.items()})
    T1.append(row); log(row["series"], "|", row["MODIS"], "|", row["ERA5 skt"], "|", row["ERA5 t2m"])
pd.DataFrame(T1).to_csv(os.path.join(OUT, "table1_trends.csv"), index=False)

# ---------------------------------------------------------------- 3.2 geometry & sampling
log("\n== 3.2 geometry and sampling controls (daytime) ==")
G = []
for plat in ["Terra", "Aqua"]:
    df = scenes[(plat, "Day")].dropna(subset=["vtime", "vang"]).copy()
    df["vt_anom"] = month_anom(df, "vtime"); df["va_dm"] = df.vang - df.vang.mean()
    dt = sen(annual(df.vtime * 60)); da = sen(annual(df.vang))
    s_t = stats.linregress(df.vt_anom, df.anom).slope; s_a = stats.linregress(df.va_dm, df.anom).slope
    imp_t = dt["slope"]/60*s_t; imp_a = da["slope"]*s_a
    b = np.linalg.lstsq(np.column_stack([np.ones(len(df)), df.vt_anom, df.va_dm]), df.anom, rcond=None)[0]
    df["resid"] = df.anom - b[1]*df.vt_anom - b[2]*df.va_dm
    before = sen(annual(df.anom)); after = sen(annual(df.resid))
    g = dict(platform=plat, time_drift_min_dec=dt["slope"], time_drift_p=dt["p"], sens_K_per_h=s_t, implied_time=imp_t,
             vza_drift_deg_dec=da["slope"], vza_drift_p=da["p"], sens_K_per_deg=s_a, implied_vza=imp_a,
             total_implied=imp_t+imp_a, trend_before=before["slope"], trend_after=after["slope"], p_after=after["p"])
    G.append(g); log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()})
pd.DataFrame(G).to_csv(os.path.join(OUT, "geometry_controls.csv"), index=False)
for (plat, tod), df in scenes.items():
    c = sen(annual(df.clear)); q = sen(annual(df.qa_pct))
    log(f"{plat} {tod}: clear-sky count trend {fmt(c)}; QA% mean {df.qa_pct.mean():.1f}, trend {fmt(q)}")

# ---------------------------------------------------------------- VIIRS
log("\n== VIIRS VNP21A2 day ==")
v = pd.read_csv(os.path.join(DATA_DIR, "VNP21A2", "statistics_LST_Day_1KM.csv"), parse_dates=["date"]).set_index("date")
log(f"VNP21A2: {len(v)} composites {v.index.min().date()} to {v.index.max().date()}")
v = v[(v["mean"] >= 250) & (v["mean"] <= 340)][["mean"]].rename(columns={"mean": "lst"}); v["anom"] = month_anom(v, "lst")
log("VIIRS 2012-2021:", fmt(sen(annual(v.anom, 2012, 2021))))
for plat in ["Terra", "Aqua"]:
    df = scenes[(plat, "Day")]; a = df[df.index.year >= 2012].copy(); a["anom"] = month_anom(a, "lst")
    log(f"{plat} day 2012-2021:", fmt(sen(annual(a.anom, 2012, 2021))))
j = v.join(scenes[("Aqua", "Day")][["lst"]], rsuffix="_aqua", how="inner"); j["d"] = j.lst - j.lst_aqua
log("VIIRS-Aqua bias:", f"mean {j.d.mean():+.2f} K, trend", fmt(sen(annual(j.d, 2012, 2021))))

# ---------------------------------------------------------------- Table 2
log("\n== Table 2: scene-level MODIS vs ERA5-Land skt ==")
T2 = []
for (plat, tod), df in scenes.items():
    d = df.dropna(subset=["e_skt"]); diff = d.lst - d.e_skt
    T2.append(dict(series=f"{plat} {tod.lower()}time", n=len(d), bias=diff.mean(), sd=diff.std(), r=np.corrcoef(d.lst, d.e_skt)[0, 1]))
    log(T2[-1])
pd.DataFrame(T2).to_csv(os.path.join(OUT, "table2_scene_comparison.csv"), index=False)

# ---------------------------------------------------------------- step 2003-2012 vs 2013-2021
log("\n== Step change, month-anomalies 2003-2012 vs 2013-2021 (Welch) ==")
for plat in ["Terra", "Aqua"]:
    df = scenes[(plat, "Day")]
    for lab, col in [("MODIS", "anom"), ("ERA5 skt", "e_skt_anom")]:
        a = df.loc[(df.index.year >= 2003) & (df.index.year <= 2012), col].dropna()
        b = df.loc[(df.index.year >= 2013) & (df.index.year <= 2021), col].dropna()
        t, p = stats.ttest_ind(b, a, equal_var=False)
        log(f"{plat} {lab}: step {b.mean()-a.mean():+.2f} K, p={p:.2e}")

# ---------------------------------------------------------------- month-resolved
log("\n== Month-resolved daytime trends, K/decade ==")
M = []
for plat in ["Terra", "Aqua"]:
    df = scenes[(plat, "Day")]
    for m in range(1, 13):
        sub = df[df.index.month == m]
        r1 = sen(annual(sub.anom)); r2 = sen(annual(sub.e_skt_anom.dropna()))
        M.append(dict(platform=plat, month=m, modis=r1["slope"], modis_p=r1["p"], era5=r2["slope"], era5_p=r2["p"]))
Mdf = pd.DataFrame(M); Mdf.to_csv(os.path.join(OUT, "month_resolved_trends.csv"), index=False)
log(Mdf.round(3).to_string(index=False))

# ---------------------------------------------------------------- phenology
log("\n== Table 3: MCD12Q2 Cycle 1 phenometrics 2001-2019 ==")
ph_raw = pd.read_csv(os.path.join(DATA_DIR, "MCD12Q2", "MCD12Q2.csv"), header=None, low_memory=False)
ph_raw = ph_raw.dropna(subset=[2]).copy()
ph_raw["year"] = ph_raw[2].str[1:5].astype(int)
PH = {}
for metric in ["Greenup", "MidGreenup", "Maturity", "Peak", "Senescence", "MidGreendown", "Dormancy"]:
    g = ph_raw[ph_raw[5] == f"{metric}.Num_Modes_01"].set_index("year").iloc[:, 5:].apply(pd.to_numeric, errors="coerce")
    g = g.where(g != 32767)
    doy = g.sub([(pd.Timestamp(f"{y}-01-01") - pd.Timestamp("1970-01-01")).days - 1 for y in g.index], axis=0)
    PH[metric] = doy.mean(axis=1)
PH = pd.DataFrame(PH).loc[2001:2019]
T3 = []
for m in PH:
    r = sen(PH[m]); T3.append(dict(metric=m, **r)); log(m, fmt(r, " d/dec"))
pd.DataFrame(T3).to_csv(os.path.join(OUT, "table3_phenology.csv"), index=False)
PH.to_csv(os.path.join(OUT, "phenology_annual.csv"))

# ---------------------------------------------------------------- ERA5 daily hydroclimate
log("\n== ERA5-Land hydroclimate 2001-2021 ==")
dd = pd.read_csv(os.path.join(DATA_DIR, "ERA5Land", "era5land_daily_ajmer.csv"), parse_dates=["date"]).set_index("date")
dd["P_mm"] = dd.precip * 1000
yrs = range(Y0, Y1 + 1)
def seas(col, months, how):
    x = dd[dd.index.month.isin(months)]; g = x.groupby(x.index.year)[col]
    return (g.sum() if how == "sum" else g.mean()).loc[Y0:Y1]
SOND = [9, 10, 11, 12]; JJAS = [6, 7, 8, 9]
hyd = pd.DataFrame({"P_SOND": seas("P_mm", SOND, "sum"), "P_JJAS": seas("P_mm", JJAS, "sum"),
                    "SM1_SOND": seas("swvl1", SOND, "mean"), "transp_SOND": seas("transp", SOND, "mean"),
                    "evap_SOND": seas("evap", SOND, "mean")})
# Monsoon withdrawal proxy: day of year by which 95% of June-October rainfall has accumulated
wd = {}
for y in yrs:
    x = dd.loc[f"{y}-06-01":f"{y}-10-31", "P_mm"].cumsum(); wd[y] = x.index[np.argmax(x.values >= 0.95*x.values[-1])].dayofyear
hyd["withdrawal_doy"] = pd.Series(wd)
for c, u in [("P_SOND", " mm/dec"), ("P_JJAS", " mm/dec"), ("SM1_SOND", " m3m-3/dec"), ("withdrawal_doy", " d/dec"),
             ("transp_SOND", ""), ("evap_SOND", "")]:
    r = sen(hyd[c]); log(c, f"slope {r['slope']:+.3g}{u} p={r['p']:.3f}")
lai_oct = dd[dd.index.month == 10].groupby(dd[dd.index.month == 10].index.year).lai.mean()
log(f"October LAI: mean {lai_oct.mean():.3f}, SD across years {lai_oct.std():.2e}")
hyd.to_csv(os.path.join(OUT, "era5_hydroclimate_annual.csv"))

# ---------------------------------------------------------------- 3.6 attribution (2001-2019)
log("\n== Attribution, 2001-2019 ==")
def seas_anom(plat, col, months):
    df = scenes[(plat, "Day")]; x = df[df.index.month.isin(months)][col].dropna()
    return x.groupby(x.index.year).mean().loc[2001:2019]
for plat in ["Terra", "Aqua"]:
    A = pd.DataFrame({"L": seas_anom(plat, "anom", SOND), "Le": seas_anom(plat, "e_skt_anom", SOND),
                      "Lm": seas_anom(plat, "anom", [5, 6, 7]), "MGD": PH.MidGreendown,
                      "SM": hyd.SM1_SOND.loc[2001:2019], "PJ": hyd.P_JJAS.loc[2001:2019]}).dropna()
    L, Le, Lm, MGD, SM, PJ = (A[c].values for c in ["L", "Le", "Lm", "MGD", "SM", "PJ"])
    log(f"{plat}: years {A.index.min()}-{A.index.max()} (n={len(A)})")
    r, p = stats.pearsonr(L, MGD); rd, pd_ = stats.pearsonr(detrend(L), detrend(MGD))
    log(f"  r(SOND LST, MidGreendown) = {r:.3f} (p={p:.1e}); detrended {rd:.3f} (p={pd_:.1e})")
    re_, pe = stats.pearsonr(detrend(Le), detrend(MGD)); log(f"  ERA5 skt detrended r = {re_:.3f} (p={pe:.4f})")
    rm, pm = stats.pearsonr(detrend(Lm), detrend(MGD)); log(f"  May-Jul falsification detrended r = {rm:+.3f} (p={pm:.2f})")
    rs, ps = stats.pearsonr(L, SM); log(f"  r(SOND LST, SOND SM1) = {rs:.3f} (p={ps:.1e})")
    log(f"  r(MidGreendown, SM1) = {stats.pearsonr(MGD, SM)[0]:.3f}")
    pr1 = partial_r(L, MGD, SM); pr2 = partial_r(L, SM, MGD)
    log(f"  partial r(LST,MGD|SM) = {pr1[0]:.3f} (p={pr1[1]:.3f}); partial r(LST,SM|MGD) = {pr2[0]:.3f} (p={pr2[1]:.3f})")
    r2a = ols(L, [MGD])[0]; r2b = ols(L, [SM])[0]; r2j, adj, b, pv = ols(L, [MGD, SM]); r2k, adjk, bk, pk = ols(L, [MGD, SM, PJ])
    log(f"  R2 MGD {r2a:.2f}; R2 SM {r2b:.2f}; joint R2 {r2j:.2f} adj {adj:.2f} (p MGD {pv[1]:.3f}, p SM {pv[2]:.3f}); "
        f"+JJAS rain p = {pk[3]:.2f}")

open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG))

# ---------------------------------------------------------------- clear-sky sampling as a control
log("\n== Clear-sky count as an additional control (daytime) ==")
for plat in ["Terra", "Aqua"]:
    df = scenes[(plat, "Day")].dropna(subset=["vtime", "vang", "clear"]).copy()
    df["cs_anom"] = month_anom(df, "clear"); df["vt_anom"] = month_anom(df, "vtime"); df["va_dm"] = df.vang - df.vang.mean()
    sens = stats.linregress(df.cs_anom, df.anom)
    drift = sen(annual(df.clear))
    X = np.column_stack([np.ones(len(df)), df.vt_anom, df.va_dm, df.cs_anom])
    b = np.linalg.lstsq(X, df.anom, rcond=None)[0]
    res = df.anom - X[:, 1:] @ b[1:]
    log(f"{plat}: clear-day drift {drift['slope']:+.2f} d/composite/dec (p={drift['p']:.3f}); "
        f"sensitivity {sens.slope:+.3f} K per clear day (p={sens.pvalue:.3f}); implied {drift['slope']*sens.slope:+.3f} K/dec; "
        f"trend after removing time+VZA+clear-sky: {fmt(sen(annual(res)))}")
open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(LOG))

# ---------------------------------------------------------------- save scene-level tables for figures
for (plat, tod), df in scenes.items():
    df.to_csv(os.path.join(OUT, f"scenes_{plat}_{tod}.csv"), index_label="date")
v.to_csv(os.path.join(OUT, "scenes_VIIRS_Day.csv"), index_label="date")
