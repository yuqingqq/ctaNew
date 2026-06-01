"""Deep inspection of the per-sym Ridge model. Why does it produce 77% compressed
predictions when features only compressed 14-40%?

Decomposes pred = sum(coef * z_feature) into:
  1) RidgeCV alpha chosen per sym (model regularization)
  2) Per-sym coefficient magnitudes (are they all small? extreme?)
  3) After-preproc input magnitudes H1 vs H2 (do z-inputs compress?)
  4) Per-feature contribution to total pred (which features drive the magnitude?)
  5) Compare pred dispersion explained: how much from input compression vs coef cancellation
"""
import pandas as pd, numpy as np, pickle, importlib.util, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
ARTIFACT = REPO/"live/models/convexity_portable.pkl"

print("loading artifact + panel...")
with open(ARTIFACT, "rb") as f: art = pickle.load(f)
feat_cols = art["feat_cols"]; models = art["models"]; sstats = art["sstats"]; hstats = art["hstats"]
print(f"  artifact: {len(models)} per-sym Ridge models, {len(feat_cols)} features")

p = pd.read_parquet(PANEL, columns=["symbol","open_time"]+feat_cols)
p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
p = p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

# ============ 1) RidgeCV alphas chosen per sym ============
print("\n=== 1) RidgeCV alpha chosen per sym ===")
alphas = pd.Series({sym: m.alpha_ for sym, m in models.items()})
print(f"  syms: {len(alphas)}")
print(f"  alpha distribution: {alphas.value_counts().sort_index().to_dict()}")
print(f"  → if mostly large alphas (10/100): heavy regularization → small coefs → small preds")

# ============ 2) Per-sym coefficient magnitudes ============
print("\n=== 2) Per-sym coefficient magnitudes ===")
coefs = pd.DataFrame({sym: m.coef_ for sym, m in models.items()}, index=feat_cols).T
print(f"  mean |coef| across syms, per feature (top 5 / bottom 5):")
mean_abs = coefs.abs().mean().sort_values(ascending=False)
print(f"  TOP 5: {mean_abs.head(5).round(4).to_dict()}")
print(f"  BOT 5: {mean_abs.tail(5).round(4).to_dict()}")
print(f"  per-sym coef magnitude (sum of |coef| per sym):")
sym_mag = coefs.abs().sum(axis=1)
print(f"    median {sym_mag.median():.3f}  p10 {sym_mag.quantile(0.1):.3f}  p90 {sym_mag.quantile(0.9):.3f}  max {sym_mag.max():.3f}")
print(f"  → if many syms have small total |coef|: model is over-regularized")

# ============ 3) After-preproc input magnitudes ============
print("\n=== 3) After-preproc input magnitudes (per-sym z-input dispersion) H1 vs H2 ===")
# apply preproc per sym, compute mean |z| for H1 and H2
def mean_abs_z(window):
    out = []
    for sym, g in window.groupby("symbol"):
        if sym not in models: continue
        Xv = x6.apply_preproc(g, feat_cols, sstats[sym], hstats[sym])
        out.append(pd.Series(np.abs(Xv).mean(axis=0), index=feat_cols, name=sym))
    return pd.DataFrame(out)
h1 = p[(p["open_time"]>=H1[0])&(p["open_time"]<H1[1])]
h2 = p[(p["open_time"]>=H2[0])&(p["open_time"]<H2[1])]
z1 = mean_abs_z(h1); z2 = mean_abs_z(h2)
print(f"  mean |z-input| per feature (averaged across syms):")
print(f"  {'feature':<28} {'H1 |z|':>10} {'H2 |z|':>10} {'ratio H2/H1':>14}")
for f in feat_cols:
    a,b = z1[f].mean(), z2[f].mean()
    print(f"  {f:<28} {a:>10.3f} {b:>10.3f} {(b/a if a else float('nan')):>14.3f}")
print(f"  → if z-inputs are smaller in H2: frozen training stats made inputs flatter")

# ============ 4) Per-feature contribution to total pred ============
print("\n=== 4) Per-feature pred contribution (coef × |z|) H1 vs H2 ===")
print(f"  {'feature':<28} {'H1 |contrib|':>12} {'H2 |contrib|':>12} {'ratio':>10}")
contrib_h1 = z1.mean() * coefs.abs().mean()
contrib_h2 = z2.mean() * coefs.abs().mean()
for f in feat_cols:
    a,b = contrib_h1.get(f, np.nan), contrib_h2.get(f, np.nan)
    print(f"  {f:<28} {a:>12.4f} {b:>12.4f} {(b/a if a else float('nan')):>10.3f}")
print(f"  total |pred contribution| H1: {contrib_h1.sum():.3f}")
print(f"  total |pred contribution| H2: {contrib_h2.sum():.3f}")
print(f"  → biggest contributors that compressed: candidates to investigate")

# ============ 5) actual pred dispersion via artifact (sanity check) ============
print("\n=== 5) sanity: artifact prediction dispersion H1 vs H2 ===")
def pred_disp(window, label):
    preds_list = []
    for ot, g in window.groupby("open_time"):
        ps = []
        for sym in g["symbol"].unique():
            if sym not in models: continue
            row = g[g["symbol"]==sym]
            Xv = x6.apply_preproc(row, feat_cols, sstats[sym], hstats[sym])
            pv = models[sym].predict(Xv)
            ps.append((sym, pv[0]))
        if len(ps)>=10:
            preds = np.array([x[1] for x in ps])
            preds_list.append((ot, preds.std(), preds.min(), preds.max()))
    df = pd.DataFrame(preds_list, columns=["open_time","disp","min","max"])
    print(f"  {label}: mean pred_disp={df['disp'].mean():.3f}  mean range={(df['max']-df['min']).mean():.3f}  n={len(df)}")
    return df
_ = pred_disp(h1, "H1")
_ = pred_disp(h2, "H2")
