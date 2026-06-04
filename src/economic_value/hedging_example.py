import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
 
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})
 
# ------------------ CONFIG ------------------
MODEL_1_FILE = "./src/predictions/combined_lstm_preds_final/LSTM_IV_combined_ES_withIVRV.csv"
#MODEL_1_FILE = "./src/predictions/final_aligned/egarch_IV.csv"
#MODEL_2_FILE = "./src/predictions/final_aligned/egarch_RV.csv"
#MODEL_2_FILE = "./src/predictions/final_aligned/egarch_RV.csv"
MODEL_2_FILE = "./src/predictions/combined_lstm_preds_final/LSTM_RV_combined_ES_final.csv"
 
#MODEL_1_FILE = "./src/predictions/final_aligned/LSTM_IV.csv"
#MODEL_2_FILE = "./src/predictions/final_aligned/LSTM_RV.csv"
 
MODEL_1_LABEL = "LSTM_IV"
MODEL_2_LABEL = "LSTM_RV"
 
# file containing EUR/USD close prices
PRICE_FILE = "./data/RV_IV_combo_1.csv"
 
LEFT_TAIL_QUANTILES = [0.01, 0.025, 0.05]
RIGHT_TAIL_QUANTILES = [0.95, 0.975, 0.99]
 
ES_COLS = {
    0.01: "ES_0.010",
    0.025: "ES_0.025",
    0.05: "ES_0.050",
    0.95: "ES_0.950",
    0.975: "ES_0.975",
    0.99: "ES_0.990"
}
 
PCT_DECIMALS = 1
TRADING_DAYS = 252
ZOOM_SHARE = 0.20   # 0.20 or 0.25, choose one that represents a reasonable zoom level for data
 
EURUSD_SPREAD_PIPS = 0.5
EURUSD_PIP_SIZE = 0.0001  # 1 pip for EUR/USD
SPREAD_IN_PRICE_UNITS = EURUSD_SPREAD_PIPS * EURUSD_PIP_SIZE  # = 0.00005
 
# ------------------ HELPERS ------------------
def standardize_date_column(df):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        raise KeyError("No 'Date' or 'date' column found.")
    return df
 
def historical_es(returns, q):
    q_tau = np.quantile(returns, q)
    if q < 0.5:
        tail = returns[returns < q_tau]
    else:
        tail = returns[returns > q_tau]
    return np.mean(tail)
 
def realized_es(returns, q):
    q_tau = np.quantile(returns, q)
    if q < 0.5:
        tail = returns[returns < q_tau]
    else:
        tail = returns[returns > q_tau]
    return np.mean(tail)
 
def hedge_ratio_eq34(es_pred, L):
    raw = (es_pred - L) / es_pred
    return np.clip(np.where(raw > 0, raw, 0.0), 0, 1)
 
def hedged_returns_with_cost(returns, hedge_ratio, cost_per_unit_hedge):
    # lyocsa et al.-style fixed percentage cost:
    # HR_t(L) = R_t * (1 - H_t(L)) - c * H_t(L)
    return returns * (1.0 - hedge_ratio) - cost_per_unit_hedge * hedge_ratio
 
def annualized_return_pct(returns, trading_days=252):
    """
    Arithmetic annualized return in percent.
    If returns are decimal log returns, this is:
        mean(daily return) * 252 * 100
    """
    return np.mean(returns) * trading_days * 100
 
def pct_formatter(decimals=1):
    return mtick.FuncFormatter(lambda x, pos: f"{x * 100:.{decimals}f}%")
 
def ret_formatter(decimals=1):
    return mtick.FuncFormatter(lambda x, pos: f"{x:.{decimals}f}")
 
def compute_fx_cost_from_close(price_df, oos_dates):
    """
    as Lyocsa et al.:
        c is approx. spread / average exchange rate over OOS
 
    Since returns are log returns in decimal form, we keep c in decimal form too.
    For tiny FX spreads this is a very close approximation.
    """
    px = standardize_date_column(price_df.copy())
 
    if "close" not in px.columns:
        raise KeyError("PRICE_FILE must contain a 'close' column.")
 
    px = px[["Date", "close"]].copy()
    px_oos = px[px["Date"].isin(oos_dates)].copy()
 
    if px_oos.empty:
        raise ValueError("No overlapping dates between price file and OOS prediction sample.")
 
    avg_close_oos = px_oos["close"].mean()
 
    # decimal return units
    c_decimal = SPREAD_IN_PRICE_UNITS / avg_close_oos
 
    # same number in percentage points
    c_percent = c_decimal * 100
 
    return c_decimal, c_percent, avg_close_oos, px_oos
 
# ------------------ LOAD ------------------
df1 = pd.read_csv(MODEL_1_FILE)
df2 = pd.read_csv(MODEL_2_FILE)
px = pd.read_csv(PRICE_FILE)
 
df1 = standardize_date_column(df1)
df2 = standardize_date_column(df2)
 
# just to check alignment and consistency of TrueY if it exists in both files
common_dates = set(df1["Date"]).intersection(set(df2["Date"]))
print(f"df1 dates: {len(df1)}, df2 dates: {len(df2)}, common: {len(common_dates)}")
 
if "TrueY" in df2.columns:
    merged_check = df1[["Date","TrueY"]].merge(df2[["Date","TrueY"]], on="Date", suffixes=("_1","_2"))
    diff = (merged_check["TrueY_1"] - merged_check["TrueY_2"]).abs()
    print(f"TrueY max diff between files: {diff.max():.2e}")
    print(f"TrueY mean diff between files: {diff.mean():.2e}")
 
cols_df1 = ["Date", "TrueY"] + list(ES_COLS.values())
cols_df2 = ["Date"] + list(ES_COLS.values())
 
df1 = df1[cols_df1].copy()
df2 = df2[cols_df2].copy()
 
df = pd.merge(df1, df2, on="Date", suffixes=("_m1", "_m2"))
returns_true = df["TrueY"].values
 
baseline_return = returns_true.mean() * 252 * 100
print(baseline_return)
 
# ------------------ TRADING COST ------------------
COST_DECIMAL, COST_PERCENT, AVG_CLOSE_OOS, px_oos = compute_fx_cost_from_close(px, set(df["Date"]))
 
print(f"Average OOS EUR/USD close: {AVG_CLOSE_OOS:.6f}")
print(f"Spread in price units:      {SPREAD_IN_PRICE_UNITS:.6f}")
print(f"Trading cost c (decimal):   {COST_DECIMAL:.8f}")
print(f"Trading cost c (% points):  {COST_PERCENT:.4f}%")
 
baseline_return = annualized_return_pct(returns_true, TRADING_DAYS)
print(f"Baseline annualized return (unhedged): {baseline_return:.3f}%")
 


 
# ------------------ PLOTTING FUNCTION ------------------
def make_tail_figure(quantiles, tail_name, cost_decimal):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("white")
 
    for ax in axes.flatten():
        ax.set_facecolor("white")
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="out", pad=4)
 
    legend_handles = None
    legend_labels = None
 
    for j, q in enumerate(quantiles):
        es_col = ES_COLS[q]
        ES_1 = df[f"{es_col}_m1"].values
        ES_2 = df[f"{es_col}_m2"].values
        ES_hist = historical_es(returns_true, q)
 
        span = ZOOM_SHARE * abs(ES_hist)
        L_grid = np.linspace(ES_hist - span, ES_hist + span, 140)
 
        es_vals_1 = []
        es_vals_2 = []
        ret_vals_1 = []
        ret_vals_2 = []
 
        for L in L_grid:
            H_1 = hedge_ratio_eq34(ES_1, L)
            H_2 = hedge_ratio_eq34(ES_2, L)
 
            # phedged returns with cost
            hr_1 = hedged_returns_with_cost(returns_true, H_1, cost_decimal)
            hr_2 = hedged_returns_with_cost(returns_true, H_2, cost_decimal)
 
            # top row: realized ES of hedged returns
            es_vals_1.append(realized_es(hr_1, q))
            es_vals_2.append(realized_es(hr_2, q))
 
            # bottom row: annualized mean return of hedged returns
            ret_vals_1.append(annualized_return_pct(hr_1, TRADING_DAYS))
            ret_vals_2.append(annualized_return_pct(hr_2, TRADING_DAYS))
 
        es_vals_1 = np.array(es_vals_1)
        es_vals_2 = np.array(es_vals_2)
        ret_vals_1 = np.array(ret_vals_1)
        ret_vals_2 = np.array(ret_vals_2)
 
        # ---------- TOP ROW: ES ----------
        ax_top = axes[0, j]
        ax_top.plot(L_grid, es_vals_1, color="#5DADE2", lw=1.4, label=MODEL_1_LABEL)
        ax_top.plot(L_grid, es_vals_2, color="#EC6D33", lw=1.4, label=MODEL_2_LABEL)
        ax_top.plot(L_grid, L_grid, "--", color="black", lw=1.4, label="Accepted Loss")
        ax_top.axvline(ES_hist, linestyle=":", color="black", lw=1.2, label="Historical ES")
 
        ax_top.set_title(f"{q*100:.1f}% ES")
        ax_top.set_xlabel("Accepted Loss (L)")
        ax_top.set_ylabel("Realized ES")
 
        ax_top.xaxis.set_major_formatter(pct_formatter(PCT_DECIMALS))
        ax_top.yaxis.set_major_formatter(pct_formatter(PCT_DECIMALS))
        ax_top.xaxis.set_major_locator(mtick.MaxNLocator(4))
        ax_top.yaxis.set_major_locator(mtick.MaxNLocator(5))
        ax_top.set_xlim(L_grid.min(), L_grid.max())
        ax_top.margins(x=0, y=0)
 
        # ---------- BOTTOM ROW: RETURNS ----------
        ax_bot = axes[1, j]
        ax_bot.plot(L_grid, ret_vals_1, color="#5DADE2", lw=1.4, label=MODEL_1_LABEL)
        ax_bot.plot(L_grid, ret_vals_2, color="#EC6D33", lw=1.4, label=MODEL_2_LABEL)
        ax_bot.axvline(ES_hist, linestyle=":", color="black", lw=1.2, label="Historical ES")
        ax_bot.axhline(0, color="black", lw=0.8)
 
        ax_bot.set_title(f"Return [%] for {q*100:.1f}% ES strategy", fontsize=10)
        ax_bot.set_xlabel("Accepted Loss (L)")
        ax_bot.set_ylabel("Annualized return [%]")
 
        ax_bot.xaxis.set_major_formatter(pct_formatter(PCT_DECIMALS))
        ax_bot.yaxis.set_major_formatter(ret_formatter(1))
        ax_bot.xaxis.set_major_locator(mtick.MaxNLocator(4))
        ax_bot.yaxis.set_major_locator(mtick.MaxNLocator(5))
        ax_bot.set_xlim(L_grid.min(), L_grid.max())
        ax_bot.margins(x=0, y=0)
 
        if legend_handles is None:
            legend_handles, legend_labels = ax_top.get_legend_handles_labels()
 
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_transform=fig.transFigure
    )
 
    #fig.suptitle(
    #    f"{tail_name} tail evaluation (with trading cost c = {COST_PERCENT:.4f}%)",
    #    y=0.98
    #)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()
 
# ------------------ RUN ------------------
make_tail_figure(LEFT_TAIL_QUANTILES, "Left", COST_DECIMAL)
make_tail_figure(RIGHT_TAIL_QUANTILES, "Right", COST_DECIMAL)
 
print("Mean hedge ratio LSTM:", np.mean(H_1))
print("Mean hedge ratio EGARCH:", np.mean(H_2))
 
print("Mean cost LSTM:", np.mean(COST_DECIMAL * H_1))
print("Mean cost EGARCH:", np.mean(COST_DECIMAL * H_2))
 
print("Cost ratio EGARCH / LSTM:",
      np.mean(COST_DECIMAL * H_2) / np.mean(COST_DECIMAL * H_1))
 