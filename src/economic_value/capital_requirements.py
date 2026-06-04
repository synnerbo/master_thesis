import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
 
 
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
 
 
PATH_IV = "./src/predictions/final_aligned/egarch_IV.csv"
 
POSITION_NOK_M = 500
MODEL_MULTIPLIER = 1.5
ROLLING_WINDOW = 60
 
EVAL_START = "2022-01-01"
 
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# plot colours 
C_DAILY = "#BFC3C8" 
C_IMA   = "#3A4750"   
C_BIND  = "#1F77B4"  
C_FLAT  = "#7A7A7A"  
C_SAVE  = "#59A14F"   
C_EXTRA = "#5E7287"   
 
 
def style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
 
def load_forecasts(path):
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else "date"
    df = df.set_index(pd.to_datetime(df[date_col])).drop(columns=[date_col]).sort_index()
    df["ES_1d"] = df["ES_0.025"].abs()
    df["ES_10d"] = df["ES_1d"] * np.sqrt(10)
    return df
 
# main logic 
def compute_ima_capital(df, m, roll, position):
    # Most recent observation: ES forecast for day t 
    es_recent = df["ES_10d"]
    # 60-day rolling average scaled by multiplier 
    es_avg_scaled = df["ES_10d"].rolling(roll, min_periods=roll).mean() * m
    # Capital charge: max of recent observation and scaled rolling average
    capital = np.maximum(es_recent, es_avg_scaled) * position
    return capital, es_recent * position, es_avg_scaled * position
 
 
def save_figure(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}.{ext}"))
    plt.close(fig)
 
# plotting code 
def plot_capital_requirements(cap_iv, es_recent, es_avg_scaled, cap_flat):
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    fig.patch.set_facecolor("white")
    style_ax(ax)
 
    # Which days are set by the most-recent observation rather than the scaled rolling average 
    binding = es_recent.values >= es_avg_scaled.values
 
    ax.fill_between(cap_iv.index, cap_iv, cap_flat,
                    where=(cap_iv < cap_flat), color=C_SAVE, alpha=0.16,
                    zorder=0, label="Capital saving region")
    ax.fill_between(cap_iv.index, cap_iv, cap_flat,
                    where=(cap_iv >= cap_flat), color=C_EXTRA, alpha=0.20,
                    zorder=0, label="Additional capital charge")
 
    ax.plot(es_recent.index, es_recent, color=C_DAILY, lw=0.9, ls="-",
            alpha=0.85, zorder=1,
            label="Most recent ES$_t$ (EGARCH-IV forecasts)")
    
    ax.plot(cap_iv.index, [cap_flat] * len(cap_iv), color=C_FLAT, lw=1.3,
            ls="-.", zorder=2, label="Flat-rate requirement (8%)")
 
    ax.plot(cap_iv.index, cap_iv, color=C_IMA, lw=1.8, ls="-", alpha=1.0,
            zorder=4, solid_capstyle="round",
            label="IMA capital charge ($m\\times$60-day avg)")
 
    cap_binding = np.where(binding, cap_iv.values, np.nan)
    ax.plot(cap_iv.index, cap_binding, color=C_BIND, lw=2.2, zorder=5,
            solid_capstyle="round", marker="o", markersize=1.5,
            markeredgecolor="none", label="IMA set by daily ES")
 
    ax.set_ylabel("Capital requirement (NOK million)")
    ax.set_xlim(cap_iv.index.min() - pd.Timedelta(days=10),
                cap_iv.index.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
 
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              frameon=False, fontsize=11)
 
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, top=0.96)
    save_figure(fig, "fig_capital_requirements_mar3341_with_IMA_checks")
 
 
def main():
    iv = load_forecasts(PATH_IV)
    cap_iv_full, es_recent_full, es_avg_scaled_full = compute_ima_capital(
        iv, MODEL_MULTIPLIER, ROLLING_WINDOW, POSITION_NOK_M)
 
    cap_iv = cap_iv_full[cap_iv_full.index >= EVAL_START]
    es_recent = es_recent_full[es_recent_full.index >= EVAL_START]
    es_avg_scaled = es_avg_scaled_full[es_avg_scaled_full.index >= EVAL_START]
 
    assert np.allclose(
        cap_iv.values,
        np.maximum(es_recent.values, es_avg_scaled.values)
    ), "Capital charge does not equal max of recent ES and scaled rolling average"
 
    cap_flat = 0.08 * POSITION_NOK_M
    plot_capital_requirements(cap_iv, es_recent, es_avg_scaled, cap_flat)
 
 
if __name__ == "__main__":
    main()
 