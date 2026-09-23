"""Statistical and I/O helpers used by ajmer_lst_analysis.py (importable for testing)."""
import numpy as np, pandas as pd
from scipy import stats

Y0, Y1 = 2001, 2021

# ---------------------------------------------------------------- statistics
def mk_p(y):
    """Mann-Kendall two-sided p (normal approximation, continuity-corrected, tie-corrected)."""
    y = np.asarray(y, float); n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n) for j in range(i + 1, n))
    _, c = np.unique(y, return_counts=True)
    v = (n*(n-1)*(2*n+5) - sum(t*(t-1)*(2*t+5) for t in c)) / 18.
    z = 0. if s == 0 else (s - np.sign(s)) / np.sqrt(v)
    return 2 * (1 - stats.norm.cdf(abs(z)))

def sen(series, per=10.):
    """Theil-Sen slope with 95% CI, scaled per decade; MK p."""
    s = series.dropna()
    sl, ic, lo, hi = stats.theilslopes(s.values, s.index.values.astype(float), 0.95)
    return dict(slope=sl*per, lo=lo*per, hi=hi*per, p=mk_p(s.values), n=len(s))

def fmt(d, u=""):
    return f"{d['slope']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{u} p={d['p']:.3f} (n={d['n']})"

def month_anom(df, col):
    """Anomaly from calendar-month climatology (climatology from all rows given)."""
    clim = df.groupby(df.index.month)[col].transform("mean")
    return df[col] - clim

def annual(an, y0=Y0, y1=Y1):
    a = an[(an.index.year >= y0) & (an.index.year <= y1)]
    return a.groupby(a.index.year).mean()

def detrend(x):
    t = np.arange(len(x)); b = np.polyfit(t, x, 1); return x - np.polyval(b, t)

def partial_r(x, y, z):
    """r(x,y | z) and p (df = n-3)."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = np.corrcoef(rx, ry)[0, 1]; n = len(x); t = r*np.sqrt((n-3)/(1-r**2))
    return r, 2*stats.t.sf(abs(t), n-3)

def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X)); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X@b; n, k = X.shape; ss = res@res; sst = ((y-y.mean())**2).sum()
    r2 = 1 - ss/sst; adj = 1 - (1-r2)*(n-1)/(n-k)
    se = np.sqrt(np.diag(ss/(n-k)*np.linalg.inv(X.T@X))); p = 2*stats.t.sf(abs(b/se), n-k)
    return r2, adj, b, p


def read_pixels(path):
    raw = pd.read_csv(path, header=None, low_memory=False)
    raw = raw.assign(date=pd.to_datetime(raw[2].str[1:], format="%Y%j"))
    out = {}
    for band, g in raw.groupby(5):
        out[band] = g.set_index("date").iloc[:, 5:5+289].apply(pd.to_numeric, errors="coerce").sort_index()
    return out

def popcount(a):
    v = a.fillna(0).values.astype(np.uint8)
    return pd.DataFrame(np.unpackbits(v[..., None], axis=-1).sum(-1), index=a.index)


def interp_hourly(index_utc, values, times_utc):
    """Linear interpolation of an hourly series to arbitrary UTC times.
    Returns NaN where the bracketing samples are not exactly 1 h apart."""
    ht = np.asarray(index_utc, dtype="datetime64[s]").astype(np.int64)
    t = np.asarray(times_utc, dtype="datetime64[s]").astype(np.int64)
    i = np.clip(np.searchsorted(ht, t), 1, len(ht) - 1)
    t0, t1 = ht[i-1], ht[i]; v0, v1 = np.asarray(values)[i-1], np.asarray(values)[i]
    out = v0 + (t - t0) / (t1 - t0) * (v1 - v0)
    out[(t1 - t0) != 3600] = np.nan
    return out
