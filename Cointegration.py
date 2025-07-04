import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import itertools
import time
import requests
from statsmodels.tsa.stattools import adfuller
import re
import os # Import the os module to handle file paths

# --- Configuration ---
SHORT_TERM_HOURS = 250
LONG_TERM_HOURS = 10000
SIGNIFICANCE_LEVEL = 0.05
MIN_DATA_POINTS = 100

# Bitunix API Configuration
BITUNIX_FUTURES_API_URL = "https://fapi.bitunix.com"


def get_bitunix_futures_symbols():
    """
    Fetches all USDT-margined perpetual futures symbols from the Bitunix exchange.
    """
    print("Fetching available futures symbols from Bitunix...")
    try:
        endpoint = "/api/v1/futures/market/tickers"
        resp = requests.get(BITUNIX_FUTURES_API_URL + endpoint)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        symbols = [item['symbol'] for item in data if item['symbol'].endswith('USDT')]
        print(f"Found {len(symbols)} USDT-margined futures symbols.")
        return symbols
    except Exception as e:
        print(f"Error fetching symbols from Bitunix: {e}")
        return []


def fetch_bitunix_kline(symbol, hours):
    """
    Fetches historical K-line (candlestick) data for a given symbol from Bitunix.
    """
    endpoint = "/api/v1/futures/market/kline"
    end_ms = int(datetime.datetime.now(pytz.utc).timestamp() * 1000)
    start_ms = end_ms - hours * 3600 * 1000

    params = {
        "symbol": symbol,
        "interval": "1h",  # 1 hour intervals
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1500
    }

    try:
        resp = requests.get(BITUNIX_FUTURES_API_URL + endpoint, params=params)
        resp.raise_for_status()
        raw = resp.json().get('data', [])
        if not raw:
            return None

        df = pd.DataFrame(raw, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'quoteVolume', 'numTrades', 'takerBuyVolume', 'takerBuyQuoteVolume'
        ])
        df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        df['close'] = df['close'].astype(float)
        return df['close']

    except Exception as e:
        # Suppress repetitive errors for cleaner output
        # print(f"Error fetching kline for {symbol}: {e}")
        return None


def test_cointegration(series1, series2):
    """
    Performs the Engle-Granger cointegration test on a pair of price series.
    Returns (p-value, hedge_ratio).
    """
    X = np.vstack([np.ones(len(series1)), series2.values]).T
    y = series1.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        spread = y - np.dot(X, beta)
        p_value = adfuller(spread, regression='c', autolag='BIC')[1]
        return p_value, beta[1]
    except Exception:
        return 1.0, None


def find_cointegrated_pairs(tickers, hours):
    """
    Scans all pairs for cointegration over a given lookback period.
    Returns dict of pair -> stats.
    """
    print(f"\n--- Scanning for cointegrated pairs over {hours} hours ---")
    data_dict = {}
    num_tickers = len(tickers)
    for i, t in enumerate(tickers, 1):
        print(f"Fetching [{i}/{num_tickers}] {t}...", end='\r', flush=True)
        series = fetch_bitunix_kline(t, hours)
        if series is not None and len(series) >= MIN_DATA_POINTS:
            data_dict[t] = series
        time.sleep(0.1)
    keys = list(data_dict.keys())
    print(f"\nFetched data for {len(keys)}/{num_tickers} symbols.")
    if len(keys) < 2:
        print("Not enough symbols to test pairs.")
        return {}
    pairs = list(itertools.combinations(keys, 2))
    stats = {}
    for i, (a, b) in enumerate(pairs, 1):
        df_pair = pd.concat([data_dict[a], data_dict[b]], axis=1, keys=[a, b]).dropna()
        if len(df_pair) < MIN_DATA_POINTS:
            continue
        p, hr = test_cointegration(df_pair[a], df_pair[b])
        if p <= SIGNIFICANCE_LEVEL:
            stats[tuple(sorted((a, b)))] = {'p_value': p, 'hedge_ratio': hr}
        print(f"Tested {i}/{len(pairs)} pairs, found {len(stats)} cointegrated", end='\r', flush=True)

    print(f"\nFound {len(stats)} cointegrated pairs for {hours} hours.")
    return stats


def visualize_relationship(pair, hours=SHORT_TERM_HOURS):
    """
    Plots normalized prices and spread for a given pair.
    """
    a, b = pair
    print(f"\nVisualizing {a}/{b}...")
    df = pd.concat(
        [fetch_bitunix_kline(a, hours), fetch_bitunix_kline(b, hours)],
        axis=1, keys=[a, b]
    ).dropna()
    if df.empty or len(df) < MIN_DATA_POINTS:
        print("Insufficient data for visualization.")
        return

    X = np.vstack([np.ones(len(df)), df[b].values]).T
    y = df[a].values
    
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    spread = y - np.dot(X, beta_full)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Short-Term Cointegration Analysis: {a} vs {b} ({hours}h)', fontsize=16)

    p1 = (df[a] / df[a].iloc[0]) * 100
    p2 = (df[b] / df[b].iloc[0]) * 100
    ax1.plot(p1, label=a)
    ax1.plot(p2, label=b, alpha=0.8)
    ax1.set_title('Normalized Prices (Indexed to 100)')
    ax1.set_ylabel('Indexed Price')
    ax1.legend()

    m, s = spread.mean(), spread.std()
    ax2.plot(df.index, spread, label='Spread')
    ax2.axhline(m, linestyle='--', color='black', label=f'Mean {m:.4f}')
    ax2.axhline(m + 2*s, color='red', linestyle=':', label='+2σ')
    ax2.axhline(m - 2*s, color='red', linestyle=':', label='-2σ')
    ax2.set_title(f'Spread (Hedge Ratio: {beta_full[1]:.4f})')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def save_results_to_csv(results_df):
    """
    Saves the results DataFrame to a CSV file in the same directory as the script.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cointegrated_pairs_{timestamp}.csv"
        full_path = os.path.join(script_dir, filename)
        
        results_df.to_csv(full_path, index=False)
        print(f"\n✅ Successfully saved results to: {full_path}")
        
    except Exception as e:
        print(f"\n❌ Error saving CSV file: {e}")
        print("Printing results to console as a fallback.")
        print(results_df.to_string())


def main():
    print("--- Starting Bitunix Futures Cointegration Analysis ---")
    symbols = get_bitunix_futures_symbols()
    if not symbols:
        print("Could not retrieve symbols. Exiting.")
        return

    long_stats = find_cointegrated_pairs(symbols, LONG_TERM_HOURS)
    short_stats = find_cointegrated_pairs(symbols, SHORT_TERM_HOURS)
    
    common = set(long_stats.keys()).intersection(short_stats.keys())

    print("\n" + "="*50)
    if common:
        results = []
        for i, pair in enumerate(sorted(list(common)), 1):
            # Get stats from both long and short term results
            long_stat = long_stats[pair]
            short_stat = short_stats.get(pair, {}) # Use .get for safety
            
            results.append({
                'Index': i,
                'Pair': f"{pair[0]}/{pair[1]}",
                'P-Value (Long)': long_stat.get('p_value'),
                'Hedge Ratio (Long)': long_stat.get('hedge_ratio'),
                'P-Value (Short)': short_stat.get('p_value')
            })
        df_res = pd.DataFrame(results)
        
        save_results_to_csv(df_res)

        print(f"\n✅ Found {len(results)} pairs cointegrated on BOTH timeframes:")
        # Print a clean, formatted table to the console
        print(df_res.to_string(index=False))

        sel = input("\nEnter pair numbers to visualize (or press Enter for all): ")
        try:
            if sel.strip():
                idxs = [int(x)-1 for x in re.findall(r'\d+', sel)]
            else:
                idxs = list(range(len(results)))
        except (EOFError, KeyboardInterrupt):
            return

        for i in idxs:
            if 0 <= i < len(results):
                pair_to_visualize = tuple(results[i]['Pair'].split('/'))
                visualize_relationship(pair_to_visualize)
    else:
        print("❌ No overlapping cointegrated pairs found.")

if __name__ == '__main__':
    main()
