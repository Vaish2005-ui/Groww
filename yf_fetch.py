import os
import time
import pandas as pd
import yfinance as yf

SYMBOLS = [ 
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS",
    "RBLBANK.NS", "YESBANK.NS",
  
    "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
    "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS",
   
    "AUBANK.NS",
]

PERIOD = "1y"
INTERVAL = "1d"
DELAY_SECONDS = 2
MAX_RETRIES = 3
RETRY_BACKOFF = 5
SKIP_IF_EXISTS = False

OUTPUT_DIR = "yf_data"

def fetch(symbol):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
        except Exception as e:
            print(f" attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    for symbol in SYMBOLS:
        path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
        print(f"\n== {symbol} ==")
 
        if SKIP_IF_EXISTS and os.path.exists(path):
            print("  already downloaded, skipping")
            continue
 
        try:
            df = fetch(symbol)
        except Exception as e:
            print(f"  giving up: {e}")
            time.sleep(DELAY_SECONDS)
            continue
 
        if df.empty:
            print("  no data returned (check the symbol)")
        else:
            df.index = df.index.tz_localize(None)       # drop timezone for cleaner CSV
            df.index.name = "date"
            df.to_csv(path)
            print(f"  saved {len(df)} rows -> {path}")
 
        time.sleep(DELAY_SECONDS)                       # delay between calls
 
 
if __name__ == "__main__":
    main()