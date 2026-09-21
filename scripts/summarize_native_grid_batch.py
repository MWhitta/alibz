"""Summarise batch native-grid recovery results (one or more CSVs)."""
import sys
import numpy as np
import pandas as pd

frames = []
for p in sys.argv[1:]:
    df = pd.read_csv(p)
    df["source"] = p.split("/")[-1].replace("results_", "").replace(".csv", "")
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
ok = df[df.ok == 1].copy()
err = df[df.ok == 0]
print(f"files: {df.file.nunique()}  rows: {len(df)}  error rows: {len(err)}")
if len(err):
    print("errors by message:")
    print(err.error.str.slice(0, 90).value_counts().head(8).to_string())
ok["exact"] = ok["exact"].astype(int)
print("\nper segment:")
g = ok.groupby("segment")
print(pd.DataFrame({
    "n": g.size(),
    "exact_frac": g.exact.mean().round(4),
    "relres_p50": g.relres.median().map("{:.1e}".format),
    "relres_p90": g.relres.quantile(0.9).map("{:.1e}".format),
    "relres_max": g.relres.max().map("{:.1e}".format),
    "delta_px_p5": g.delta_px.quantile(0.05).round(2),
    "delta_px_p95": g.delta_px.quantile(0.95).round(2),
    "stretch_ppm_p5": g.stretch_ppm.quantile(0.05).round(0),
    "stretch_ppm_p95": g.stretch_ppm.quantile(0.95).round(0),
    "win_used_min": g.windows_used.min(),
    "sec_p50": g.seconds.median().round(1),
}).to_string())
print("\nper source x segment exact fraction:")
print(ok.pivot_table(index="source", columns="segment", values="exact", aggfunc=["mean", "count"]).round(3).to_string())
bad = ok[ok.exact == 0].sort_values("relres", ascending=False)
print(f"\nnon-exact segments: {len(bad)} of {len(ok)}")
if len(bad):
    print("worst 15:")
    print(bad[["source", "campaign", "file", "segment", "relres", "delta_px", "stretch_ppm", "windows_used", "max_window_relres", "on_model_lo", "on_model_hi"]].head(15).to_string(index=False, max_colwidth=40))
    print("\nnon-exact by campaign:")
    print(bad.groupby(["source", "campaign"]).size().sort_values(ascending=False).head(12).to_string())
    print("\nrelres distribution of non-exact: ", np.percentile(bad.relres, [10, 50, 90]).round(8))
# consistency of the corrections across a campaign (same session -> similar shift?)
if "campaign" not in ok.columns: ok["campaign"] = ok.file.str.slice(0, 12)
print("\ncorrection spread within campaign (NIR delta_px std, top campaigns by count):")
nir = ok[ok.segment == "NIR"]
cs = nir.groupby(["source", "campaign"]).agg(n=("delta_px", "size"), delta_mean=("delta_px", "mean"), delta_std=("delta_px", "std"), stretch_mean=("stretch_ppm", "mean")).sort_values("n", ascending=False)
print(cs.head(12).round(3).to_string())
