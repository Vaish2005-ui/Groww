"""
Task
“My task was to study how bank share prices and trading volumes behave around quarterly result announcements. 
I looked at 5 banks (Axis, ICICI, HDFC, SBI and Kotak) and at their last 4 results each, to see whether the stock reacts 
before the news or after it.”

Requirements
“I had to find each bank’s last four result dates, then measure the average price movement and the number of shares 
traded for the 10 days before and the 10 days after each result. I also had to show it in charts, one for the days before and 
one for the days after, for every bank, with a delay between data requests.”

What the code does
“The code reads the result dates from a file called events.csv. It downloads each bank’s daily price 
and volume from Yahoo Finance, pausing between downloads. For each result, it takes the 10 trading days 
before and the 10 trading days after, with the first trading day the market knows the result as Day 0.
If a result came out on a weekend, a holiday or after the market closed, it uses the next trading day.
It then averages the price change and the shares traded across the 4 quarters for each bank. It saves 
the numbers as CSV files and makes two charts per bank, a before chart and an after chart. In each 
chart the bars show shares traded and the line shows the cumulative price change.”
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

#SETTINGS (change these if needed)
EVENTS_FILE = "events.csv"   # list of banks and result dates
WINDOW = 10                  # number of trading days before and after the result
DATA_START = "2025-08-01"    # early enough to cover 10 days before the oldest result
DELAY_SECONDS = 2            # pause between downloads (avoid being blocked)
MAX_RETRIES = 3              # retry a failed download this many times
OUT_DIR = "event_output"     # folder where results and charts are saved

def get_prices(symbol):
    """Download daily Close and Volume for one symbol.
    Saved in a small CSV cache so re-running does not download again."""
    cache = os.path.join(OUT_DIR, "prices", f"{symbol}.csv")
    if os.path.exists(cache):
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if {"Close", "Volume"}.issubset(cached.columns):
            return cached
        # older cache without Volume: download again below

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # auto_adjust=True corrects prices for splits, bonus issues and dividends
            df = yf.Ticker(symbol).history(start=DATA_START, interval="1d", auto_adjust=True)
            break
        except Exception as e:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5 * attempt)

    time.sleep(DELAY_SECONDS)  # delay between API calls
    if df.empty:
        raise ValueError(f"no data for {symbol}")
    df.index = df.index.tz_localize(None).normalize()
    df = df[["Close", "Volume"]]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache)
    return df


def day0_position(dates, announce, after_market):
    """Find Day 0: the first trading day on which the result is known to the market."""
    announce = pd.Timestamp(announce).normalize()
    # announced during market hours on a trading day -> that same day is Day 0
    if announce in dates and not after_market:
        return dates.get_loc(announce)
    # otherwise (weekend, holiday, or after market close) -> next trading day
    later = dates[dates > announce]
    return dates.get_loc(later[0]) if len(later) else None


def analyse_event(df, announce, after_market):
    """Measure price and volume around one result announcement.
    Returns (metrics, price_path, volume_path) or None if there is not enough data."""
    dates = df.index
    i0 = day0_position(dates, announce, after_market)
    if i0 is None or i0 - WINDOW < 0 or i0 + WINDOW >= len(df):
        return None

    px = df["Close"].values
    vol = df["Volume"].values
    ks = list(range(-WINDOW, WINDOW + 1))   # -10 ... +10

    pre_avg_vol = vol[i0 - WINDOW:i0].mean()          # Day -10 ... Day -1
    post_avg_vol = vol[i0 + 1:i0 + WINDOW + 1].mean()  # Day +1 ... Day +10

    metrics = {
        "day0_date": dates[i0].date(),
        "window_start": dates[i0 - WINDOW].date(),      # calendar date of Day -10
        "window_end": dates[i0 + WINDOW].date(),        # calendar date of Day +10
        "pre_10d_%": (px[i0 - 1] / px[i0 - WINDOW] - 1) * 100,
        "day0_%": (px[i0] / px[i0 - 1] - 1) * 100,
        "post_10d_%": (px[i0 + WINDOW] / px[i0] - 1) * 100,
        "pre_avg_vol": pre_avg_vol,
        "day0_vol": vol[i0],
        "post_avg_vol": post_avg_vol,
        "day0_vol_x": vol[i0] / pre_avg_vol if pre_avg_vol else np.nan,
    }
    # price path: cumulative % change, with Day -10 = 0%
    price_path = pd.Series([(px[i0 + k] / px[i0 - WINDOW] - 1) * 100 for k in ks], index=ks)
    # volume path: shares traded on each day from Day -10 to Day +10
    vol_path = pd.Series([vol[i0 + k] for k in ks], index=ks)
    return metrics, price_path, vol_path


def make_chart(bank, kind, days, price_pct, vol_millions, n_events):
    """One chart: bars = shares traded per day, line = cumulative % price move.
    kind is "before" or "after"; the file is saved as  <kind>_<bank>.png"""
    days = list(days)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # bars: shares traded (millions). The red bar is Day 0 (the reaction day).
    colors = ["red" if d == 0 else "steelblue" for d in days]
    ax1.bar(days, vol_millions.values, color=colors)
    ax1.set_xticks(days)
    ax1.set_xlabel("Trading days relative to the result (Day 0 = reaction day)")
    ax1.set_ylabel("Shares traded (millions)")

    # line: cumulative % price move, on the right-hand axis
    ax2 = ax1.twinx()
    ax2.plot(days, price_pct.values, color="black", marker="o", linewidth=2)
    ax2.axhline(0, color="grey", linewidth=0.6)
    base = "Day -10" if kind == "before" else "Day 0"
    ax2.set_ylabel(f"Cumulative % price change ({base} = 0%)")

    if kind == "before":
        ax1.set_title(f"{bank}: 10 days BEFORE quarterly results (average of {n_events} results)")
    else:
        ax1.set_title(f"{bank}: 10 days AFTER quarterly results (average of {n_events} results)")

    ax1.legend(handles=[Patch(color="steelblue", label="Shares traded (left axis)"),
                        Line2D([0], [0], color="black", marker="o", label="Price change % (right axis)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{kind}_{bank}.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # comment="#" lets events.csv contain explanation lines starting with #
    events = pd.read_csv(EVENTS_FILE, comment="#")
    events["after_market"] = events["after_market"].astype(str).str.lower().isin(["yes", "y", "true", "1"])

    rows = []
    price_paths, vol_paths = {}, {}

    # ---- Step 1: analyse every bank and every result announcement ----
    for symbol, grp in events.groupby("symbol", sort=False):
        grp = grp.dropna(subset=["announce_date"])
        if grp.empty:
            print(f"\n{symbol}: no announce_date filled in events.csv yet, skipping")
            continue
        print(f"\n== {symbol} ==")
        try:
            df = get_prices(symbol)
        except Exception as e:
            print(f"  could not get prices: {e}")
            continue

        for _, ev in grp.iterrows():
            res = analyse_event(df, ev["announce_date"], ev["after_market"])
            if res is None:
                print(f"  {ev['quarter']}: not enough price data around {ev['announce_date']}, skipped")
                continue
            metrics, p_path, v_path = res
            rows.append({"bank": ev["bank"], "symbol": symbol, "quarter": ev["quarter"],
                         "announce_date": ev["announce_date"], **metrics})
            price_paths[(ev["bank"], ev["quarter"])] = p_path
            vol_paths[(ev["bank"], ev["quarter"])] = v_path
            print(f"  {ev['quarter']}: price pre {metrics['pre_10d_%']:+.2f}%  "
                  f"day0 {metrics['day0_%']:+.2f}%  post {metrics['post_10d_%']:+.2f}%  |  "
                  f"day0 volume {metrics['day0_vol_x']:.1f}x normal")

    if not rows:
        print("\nNothing to analyse. Fill in announce_date in events.csv and run again.")
        return

    # ---- Step 2: save the tables ----
    res = pd.DataFrame(rows)
    num_cols = ["pre_10d_%", "day0_%", "post_10d_%", "day0_vol_x"]
    res[num_cols] = res[num_cols].round(2)
    res[["pre_avg_vol", "day0_vol", "post_avg_vol"]] = res[["pre_avg_vol", "day0_vol", "post_avg_vol"]].round(0)
    res.to_csv(os.path.join(OUT_DIR, "event_results.csv"), index=False)

    cols = ["pre_10d_%", "day0_%", "post_10d_%", "pre_avg_vol", "day0_vol", "post_avg_vol", "day0_vol_x"]
    summary = res.groupby("bank", sort=False)[cols].mean().round(2)
    summary["events_used"] = res.groupby("bank", sort=False).size()
    summary.to_csv(os.path.join(OUT_DIR, "bank_summary.csv"))
    print("\nAverage per bank:\n", summary.to_string())

    banks = list(summary.index)

    # ---- Step 3: two charts per bank: "before" and "after" ----
    # remove charts made by older versions, so only the new ones remain
    for f in os.listdir(OUT_DIR):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(OUT_DIR, f))

    for bank in banks:
        n_events = int((res["bank"] == bank).sum())
        # average across the quarters: price path (Day -10 = 0%) and shares traded per day
        avg_price = pd.concat([p for (b, q), p in price_paths.items() if b == bank], axis=1).mean(axis=1)
        avg_vol = pd.concat([v for (b, q), v in vol_paths.items() if b == bank], axis=1).mean(axis=1) / 1e6

        # BEFORE chart: Day -10 ... Day -1 (price measured from Day -10)
        before_days = list(range(-WINDOW, 0))
        make_chart(bank, "before", before_days,
                   avg_price.loc[before_days], avg_vol.loc[before_days], n_events)

        # AFTER chart: Day 0 ... Day +10 (price re-based so Day 0 = 0%)
        after_days = list(range(0, WINDOW + 1))
        after_price = ((1 + avg_price.loc[after_days] / 100) / (1 + avg_price.loc[0] / 100) - 1) * 100
        make_chart(bank, "after", after_days,
                   after_price, avg_vol.loc[after_days], n_events)

    print(f"\nDone. CSVs and charts are in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()