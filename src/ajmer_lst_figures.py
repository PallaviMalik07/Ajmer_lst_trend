"""
Figures 1-5 for the Ajmer LST manuscript. Run after ajmer_lst_analysis.py.
Usage: python src/ajmer_lst_figures.py [results_dir]
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DPI = 400
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "font.family": "DejaVu Sans"})
Y0, Y1 = 2001, 2021
SC = {(p, t): pd.read_csv(os.path.join(R, f"scenes_{p}_{t}.csv"), parse_dates=["date"], index_col="date")
      for p in ["Terra", "Aqua"] for t in ["Day", "Night"]}
C = {"Terra": "#1f4e79", "Aqua": "#b03a2e", "ERA5": "#e67e22"}

def mk_p(y):
    y = np.asarray(y, float); n = len(y)
    s = sum(np.sign(y[j]-y[i]) for i in range(n) for j in range(i+1, n))
    _, c = np.unique(y, return_counts=True)
    v = (n*(n-1)*(2*n+5) - sum(t*(t-1)*(2*t+5) for t in c))/18.
    z = 0. if s == 0 else (s-np.sign(s))/np.sqrt(v); return 2*(1-stats.norm.cdf(abs(z)))
def annual(s, y0=Y0, y1=Y1):
    s = s.dropna(); s = s[(s.index.year >= y0) & (s.index.year <= y1)]; return s.groupby(s.index.year).mean()
def month_anom(s):
    return s - s.groupby(s.index.month).transform("mean")
def sen_line(ax, a, **kw):
    sl, ic, lo, hi = stats.theilslopes(a.values, a.index.values.astype(float))
    ax.plot(a.index, ic + sl*a.index.values, "--", **kw); return sl*10, mk_p(a.values)
def pfmt(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
def star(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else " (n.s.)"
def save(fig, name):
    fig.savefig(os.path.join(R, name), dpi=DPI, bbox_inches="tight"); plt.close(fig); print("saved", name)

# ---------------- Figure 1
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
for i, tod in enumerate(["Day", "Night"]):
    for j, plat in enumerate(["Terra", "Aqua"]):
        ax = axs[i, j]; df = SC[(plat, tod)]
        ax.plot(df.index, df.lst, color=C[plat], lw=0.4, alpha=0.6)
        ax.set_ylabel("8-day LST (K)", color=C[plat])
        ax2 = ax.twinx(); a = annual(df.anom); a.index = pd.to_datetime(a.index.astype(str) + "-07-01")
        ax2.plot(a.index, a.values, "ko-", ms=2.5, lw=1)
        yrs = annual(df.anom)
        sl, ic, lo, hi = stats.theilslopes(yrs.values, yrs.index.values.astype(float))
        ax2.plot(a.index, ic + sl*yrs.index.values, "k--", lw=1)
        ax2.set_ylabel("Annual anomaly (K)")
        ax.set_title(f"{plat} {tod.lower()}time: {sl*10:+.2f} K dec$^{{-1}}$ ({pfmt(mk_p(yrs.values))})")
fig.tight_layout(); save(fig, "fig1_lst_timeseries.png")

# ---------------- Figure 2
M = pd.read_csv(os.path.join(R, "month_resolved_trends.csv"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
x = np.arange(1, 13); w = 0.38
for ax, plat in zip(axs, ["Terra", "Aqua"]):
    m = M[M.platform == plat]
    ax.axvspan(8.5, 12.5, color="0.9", zorder=0)
    ax.bar(x - w/2, m.modis, w, color=C[plat], label="MODIS LST")
    ax.bar(x + w/2, m.era5, w, color=C["ERA5"], label="ERA5-Land skin T")
    for xi, v, p in zip(x - w/2, m.modis, m.modis_p):
        if p < .05: ax.text(xi, v - 0.15, "*", ha="center", va="top", fontsize=9)
    for xi, v, p in zip(x + w/2, m.era5, m.era5_p):
        if p < .05: ax.text(xi, v - 0.15, "*", ha="center", va="top", fontsize=9, color=C["ERA5"])
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_title(f"{plat} sampling"); ax.set_ylim(-3.8, 2.4)
axs[0].set_ylabel("Daytime trend (K dec$^{-1}$)"); axs[0].legend(loc="lower left", frameon=False)
fig.tight_layout(); save(fig, "fig2_month_resolved.png")

# ---------------- Figure 3 (Terra drift panels + before/after controls, both platforms)
def controls(plat):
    df = SC[(plat, "Day")].dropna(subset=["vtime", "vang", "clear"]).copy()
    vt = month_anom(df.vtime); va = df.vang - df.vang.mean(); cs = month_anom(df.clear)
    out = {"Uncorrected": annual(df.anom)}
    for lab, cols in [("Time\n+ VZA", [vt, va]), ("Time + VZA\n+ clear sky", [vt, va, cs])]:
        X = np.column_stack([np.ones(len(df))] + [c.values for c in cols])
        b = np.linalg.lstsq(X, df.anom.values, rcond=None)[0]
        out[lab] = annual(pd.Series(df.anom.values - X[:, 1:] @ b[1:], index=df.index))
    res = {}
    for k, a in out.items():
        sl, ic, lo, hi = stats.theilslopes(a.values, a.index.values.astype(float)); res[k] = (sl*10, lo*10, hi*10)
    return res
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
df = SC[("Terra", "Day")]
for ax, col, lab, sc in [(axs[0], "vtime", "Overpass time (local solar, h)", 1), (axs[1], "vang", "|View zenith angle| (deg)", 1)]:
    a = annual(df[col]); ax.plot(a.index, a.values, "o-", color=C["Terra"], ms=3, lw=1)
    sl, p = sen_line(ax, a, color="k", lw=1)
    u = "min" if col == "vtime" else "deg"; sl = sl*60 if col == "vtime" else sl
    ax.set_title(f"Terra: {sl:+.2f} {u} dec$^{{-1}}$ (p={p:.2f})"); ax.set_ylabel(lab)
ax = axs[2]; labels = None
for k, plat in enumerate(["Terra", "Aqua"]):
    r = controls(plat); labels = list(r)
    for i, lab in enumerate(labels):
        s, lo, hi = r[lab]; xi = i + (k - 0.5)*0.35
        ax.bar(xi, s, 0.33, color=C[plat], alpha=[1, .7, .45][i], label=plat if i == 0 else None)
        ax.errorbar(xi, s, yerr=[[s-lo], [hi-s]], color="k", lw=0.8, capsize=2)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=6); ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Daytime trend (K dec$^{-1}$)"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "fig3_artifact_controls.png")

# ---------------- Figure 4
PH = pd.read_csv(os.path.join(R, "phenology_annual.csv"), index_col=0)
H = pd.read_csv(os.path.join(R, "era5_hydroclimate_annual.csv"), index_col=0)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axs[0]
for m, c in [("MidGreenup", "#2e7d32"), ("MidGreendown", "#8d6e63"), ("Dormancy", "#5d4037")]:
    ax.plot(PH.index, PH[m], "o-", ms=2.5, lw=1, color=c, label=m); sen_line(ax, PH[m], color=c, lw=0.8)
ax.set_ylabel("Day of year"); ax.set_ylim(150, 370); ax.set_xticks([2002, 2010, 2018]); ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.40), handlelength=1.0, fontsize=5.5, labelspacing=0.2); ax.set_title("MCD12Q2 Cycle 1")
ax = axs[1]; s = SC[("Terra", "Day")].anom; s = s[s.index.month.isin([9, 10, 11, 12])]
L = s.groupby(s.index.year).mean().loc[2001:2019]
ax.scatter(PH.MidGreendown, L, s=12, color=C["Terra"]); b = np.polyfit(PH.MidGreendown, L, 1)
xx = np.linspace(PH.MidGreendown.min(), PH.MidGreendown.max(), 10); ax.plot(xx, np.polyval(b, xx), "k-", lw=1)
r = stats.pearsonr(PH.MidGreendown, L)[0]
ax.set_xlabel("MidGreendown (DOY)"); ax.set_ylabel("Sep–Dec Terra day anomaly (K)"); ax.set_title(f"r = {r:.2f} (n = {len(L)})")
ax = axs[2]; hh = H.loc[Y0:Y1]
ax.bar(hh.index, hh.P_JJAS, color="#90caf9"); ax.set_ylabel("Jun–Sep rainfall (mm)")
ax2 = ax.twinx(); ax2.plot(hh.index, hh.SM1_SOND, "o-", color="#0d47a1", ms=2.5, lw=1)
ax2.set_ylabel("Sep–Dec soil moisture L1 (m$^3$ m$^{-3}$)"); ax.set_title("ERA5-Land")
fig.tight_layout(); save(fig, "fig4_phenology_moisture.png")

# ---------------- Figure 5
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axs[0]; d = SC[("Terra", "Day")].dropna(subset=["e_skt"])
ax.scatter(d.e_skt, d.lst, s=2, alpha=0.4, color=C["Terra"]); lim = [d[["e_skt", "lst"]].min().min()-1, d[["e_skt", "lst"]].max().max()+1]
ax.plot(lim, lim, "k-", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ERA5-Land skin T (K)"); ax.set_ylabel("MODIS Terra day LST (K)")
ax.set_title(f"r = {np.corrcoef(d.e_skt, d.lst)[0,1]:.3f}, n = {len(d)}")
ax = axs[1]
for lab, s, c in [("Terra", SC[("Terra", "Day")].anom, C["Terra"]), ("Aqua", SC[("Aqua", "Day")].anom, C["Aqua"]),
                  ("ERA5 (Terra)", SC[("Terra", "Day")].e_skt_anom, C["ERA5"])]:
    a = annual(s); ax.plot(a.index, a.values, "o-", ms=2.5, lw=1, color=c, label=lab)
ax.axvline(2012.5, color="k", ls=":", lw=1); ax.axhline(0, color="0.5", lw=0.5)
ax.set_ylabel("Annual daytime anomaly (K)"); ax.legend(frameon=False)
ax = axs[2]; k = 0
for plat in ["Terra", "Aqua"]:
    for lab, col, c in [("MODIS", "anom", C[plat]), ("ERA5", "e_skt_anom", C["ERA5"])]:
        s = SC[(plat, "Day")][col].dropna()
        a = s[(s.index.year >= 2003) & (s.index.year <= 2012)]; b = s[(s.index.year >= 2013) & (s.index.year <= 2021)]
        st = b.mean() - a.mean(); se = np.sqrt(a.var()/len(a) + b.var()/len(b))
        ax.bar(k, st, color=c); ax.errorbar(k, st, yerr=se, color="k", capsize=2, lw=0.8); k += 1
ax.set_xticks(range(4)); ax.set_xticklabels(["Terra\nMODIS", "Terra\nERA5", "Aqua\nMODIS", "Aqua\nERA5"], fontsize=6.5)
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Step 2013–21 minus 2003–12 (K)")
fig.tight_layout(); save(fig, "fig5_era5_step.png")
