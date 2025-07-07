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
import os
import sys

# --- Configuration ---
SHORT_TERM_HOURS = 338  # 338 hours = 2 weeks
LONG_TERM_HOURS = 8765  # 8765 hours = 1 year
MIN_LONG_TERM_CANDLES = LONG_TERM_HOURS  # Minimum required candles for long-term analysis
SIGNIFICANCE_LEVEL = 0.05
MIN_DATA_POINTS = 100
MAX_REQUESTS_PER_SYMBOL = 100  # Safety limit to prevent infinite loops
MAX_PAIRS_TO_PROCESS = 100000  # Safety limit for pair processing

# Bitunix API Configuration
BITUNIX_FUTURES_API_URL = "https://fapi.bitunix.com"


def debug_print(message):
    """Print debug messages with timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_bitunix_futures_symbols():
    """
    Fetches all USDT-margined perpetual futures symbols from the Bitunix exchange.
    """
    debug_print("Fetching available futures symbols from Bitunix...")
    try:
        endpoint = "/api/v1/futures/market/tickers"
        resp = requests.get(BITUNIX_FUTURES_API_URL + endpoint)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        symbols = [item['symbol'] for item in data if item['symbol'].endswith('USDT')]
        debug_print(f"Found {len(symbols)} USDT-margined futures symbols.")
        return symbols
    except Exception as e:
        debug_print(f"Error fetching symbols: {e}")
        return []


def fetch_bitunix_kline(symbol, hours):
    """
    Fetches historical K-line data with pagination to handle API limits.
    Returns a pandas Series of closing prices.
    Only prints total candles fetched per symbol.
    """
    endpoint = "/api/v1/futures/market/kline"
    end_ms = int(datetime.datetime.now(pytz.utc).timestamp() * 1000)
    start_ms = end_ms - hours * 3600 * 1000
    all_data = []
    current_end = end_ms
    request_count = 0

    while current_end > start_ms and request_count < MAX_REQUESTS_PER_SYMBOL:
        request_count += 1
        params = {
            "symbol": symbol,
            "interval": "1h",
            "endTime": current_end,
            "limit": 199  # Max allowed by API
        }
        try:
            resp = requests.get(BITUNIX_FUTURES_API_URL + endpoint, params=params)
            resp.raise_for_status()
            raw = resp.json().get('data', [])
            if not raw:
                break

            df_chunk = pd.DataFrame(raw, columns=['time', 'close'])
            df_chunk['time'] = pd.to_numeric(df_chunk['time'], errors='coerce')
            df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='ms', utc=True)
            df_chunk.set_index('time', inplace=True)
            df_chunk['close'] = df_chunk['close'].astype(float)
            all_data.insert(0, df_chunk)
            current_end = int(df_chunk.index.min().timestamp() * 1000) - 1
            time.sleep(0.2)
        except Exception as e:
            debug_print(f"Error fetching data for {symbol}: {e}")
            break

    if not all_data:
        return None

    full_series = pd.concat(all_data)
    full_series = full_series[~full_series.index.duplicated(keep='first')]
    full_series.sort_index(inplace=True)
    
    # Trim to exact requested time window
    start_dt = pd.to_datetime(start_ms, unit='ms', utc=True)
    end_dt = pd.to_datetime(end_ms, unit='ms', utc=True)
    full_series = full_series.loc[(full_series.index >= start_dt) & (full_series.index <= end_dt)]
    
    debug_print(f"{symbol} → fetched {len(full_series)} candles")
    return full_series['close']


def test_cointegration(series1, series2):
    """
    Performs the Engle-Granger cointegration test on a pair of price series.
    Returns (p-value, hedge_ratio).
    """
    aligned = pd.concat([series1, series2], axis=1, join='inner').dropna()
    if len(aligned) < MIN_DATA_POINTS:
        return 1.0, None

    X = np.vstack([np.ones(len(aligned)), aligned.iloc[:, 1].values]).T
    y = aligned.iloc[:, 0].values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        spread = y - np.dot(X, beta)
        p_value = adfuller(spread, regression='c', autolag='BIC')[1]
        return p_value, beta[1]
    except Exception as e:
        debug_print(f"Cointegration test failed: {e}")
        return 1.0, None


def find_cointegrated_pairs(tickers, hours):
    """
    Scans all pairs for cointegration over a given lookback period.
    Returns dict of pair -> stats.
    """
    debug_print(f"Scanning {len(tickers)} symbols for {hours}h cointegration...")
    data_dict = {}
    for t in tickers:
        series = fetch_bitunix_kline(t, hours)
        if series is not None and len(series) >= MIN_DATA_POINTS:
            data_dict[t] = series
    if len(data_dict) < 2:
        debug_print("Insufficient data for any pairs (need at least 2 symbols with data).")
        return {}

    stats = {}
    total_pairs = len(data_dict) * (len(data_dict) - 1) // 2
    debug_print(f"Starting analysis of {total_pairs} possible pairs...")
    
    pair_count = 0
    start_time = time.time()
    
    for i, (a, b) in enumerate(itertools.combinations(data_dict.keys(), 2)):
        pair_count += 1
        
        # Show progress every 100 pairs
        if pair_count % 100 == 0:
            elapsed = time.time() - start_time
            debug_print(f"Processed {pair_count}/{total_pairs} pairs ({elapsed:.1f}s elapsed)")
        
        # Safety check to prevent infinite processing
        if pair_count > MAX_PAIRS_TO_PROCESS:
            debug_print(f"Safety limit reached: Processed {MAX_PAIRS_TO_PROCESS} pairs. Stopping analysis.")
            break
            
        try:
            # Skip pairs with similar names to avoid correlated assets
            if a.split('USDT')[0] in b or b.split('USDT')[0] in a:
                continue
                
            # Align and clean data
            df_pair = pd.concat([data_dict[a], data_dict[b]], axis=1, join='inner', keys=[a, b]).dropna()
            if len(df_pair) < MIN_DATA_POINTS:
                continue
                
            # Test for cointegration
            p, hr = test_cointegration(df_pair[a], df_pair[b])
            if p <= SIGNIFICANCE_LEVEL:
                stats[tuple(sorted((a, b)))] = {
                    'p_value': p,
                    'hedge_ratio': hr,
                    'n_candles': len(df_pair)
                }
        except Exception as e:
            debug_print(f"Error processing pair {a}/{b}: {e}")
            continue

    debug_print(f"Found {len(stats)} cointegrated pairs for {hours}h.")
    return stats


def visualize_relationship(pair, hours=SHORT_TERM_HOURS):
    """
    Plots normalized prices and spread for a given pair.
    """
    a, b = pair
    debug_print(f"Visualizing {a}/{b}...")
    
    # Fetch data
    series_a = fetch_bitunix_kline(a, hours)
    series_b = fetch_bitunix_kline(b, hours)
    
    if series_a is None or series_b is None:
        debug_print("Error: Could not fetch data for visualization")
        return
        
    df = pd.concat([series_a, series_b], axis=1, keys=[a, b]).dropna()
    if df.empty or len(df) < MIN_DATA_POINTS:
        debug_print("Insufficient data for visualization.")
        return

    # Calculate spread
    X = np.vstack([np.ones(len(df)), df[b].values]).T
    y = df[a].values
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    spread = y - np.dot(X, beta_full)

    # Create plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Cointegration Analysis: {a} vs {b} ({hours}h)', fontsize=16)

    # Normalized prices
    p1 = (df[a] / df[a].iloc[0]) * 100
    p2 = (df[b] / df[b].iloc[0]) * 100
    ax1.plot(p1, label=a)
    ax1.plot(p2, label=b, alpha=0.8)
    ax1.set_title('Normalized Prices (Indexed to 100)')
    ax1.set_ylabel('Indexed Price')
    ax1.legend()

    # Spread
    m, s = spread.mean(), spread.std()
    ax2.plot(df.index, spread, label='Spread')
    ax2.axhline(m, linestyle='--', color='black', label=f'Mean {m:.4f}')
    ax2.axhline(m + 2*s, linestyle=':', color='red', label='+2σ')
    ax2.axhline(m - 2*s, linestyle=':', color='red', label='-2σ')
    ax2.set_title(f'Spread (Hedge Ratio: {beta_full[1]:.4f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def save_results_to_csv(results_df):
    """
    Saves the results DataFrame to a CSV file.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir:
            script_dir = os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cointegrated_pairs_{timestamp}.csv"
        full_path = os.path.join(script_dir, filename)
        results_df.to_csv(full_path, index=False)
        debug_print(f"Saved results to {full_path}")
        return full_path
    except Exception as e:
        debug_print(f"Error saving CSV: {e}")
        print(results_df.to_string())
        return None


def main():
    debug_print("=== Starting Cointegration Analysis ===")
    symbols = get_bitunix_futures_symbols()
    if not symbols:
        debug_print("No symbols, exiting.")
        return

    filtered_symbols = [s for s in symbols if s.endswith('USDT')]  # Limit to first 10 symbols for testing
    debug_print(f"Initial symbol count: {len(filtered_symbols)}")
    
    # Pre-filter symbols with sufficient long-term data
    symbols_with_sufficient_data = []
    symbol_counts_long = {}
    for t in filtered_symbols:
        series = fetch_bitunix_kline(t, LONG_TERM_HOURS)
        # FLEXIBLE THRESHOLD: Allow slightly less than exact candle count
        if series is not None and len(series) >= MIN_LONG_TERM_CANDLES - 10:  # Changed threshold
            symbols_with_sufficient_data.append(t)
            symbol_counts_long[t] = len(series)
            debug_print(f"{t} has sufficient long-term data: {len(series)} candles")
        else:
            count = len(series) if series is not None else 0
            debug_print(f"Skipping {t} - insufficient long-term data: {count} candles (required: {MIN_LONG_TERM_CANDLES})")
    
    if not symbols_with_sufficient_data:
        debug_print("No symbols with sufficient long-term data, exiting.")
        return

    debug_print(f"Proceeding with {len(symbols_with_sufficient_data)} symbols with sufficient long-term data")
    
    # Long-term analysis (only on filtered symbols)
    debug_print("Starting long-term cointegration analysis...")
    long_stats = find_cointegrated_pairs(symbols_with_sufficient_data, LONG_TERM_HOURS)
    debug_print(f"Long-term analysis complete. Found {len(long_stats)} cointegrated pairs.")

    # Short-term candle counts (only on filtered symbols)
    symbol_counts_short = {}
    for t in symbols_with_sufficient_data:
        series = fetch_bitunix_kline(t, SHORT_TERM_HOURS)
        symbol_counts_short[t] = len(series) if series is not None else 0

    # Short-term analysis
    debug_print("Starting short-term cointegration analysis...")
    short_stats = find_cointegrated_pairs(symbols_with_sufficient_data, SHORT_TERM_HOURS)
    debug_print(f"Short-term analysis complete. Found {len(short_stats)} cointegrated pairs.")

    common_pairs = set(long_stats.keys()).intersection(short_stats.keys())
    debug_print(f"Found {len(common_pairs)} pairs that are cointegrated in both timeframes")
    
    if not common_pairs:
        debug_print("No overlapping cointegrated pairs found.")
        return

    results = []
    for i, pair in enumerate(sorted(common_pairs), 1):
        a, b = pair
        long_data = long_stats[pair]
        short_data = short_stats[pair]
        results.append({
            'Index': i,
            'Pair': f"{a}/{b}",
            'Candles_A_Long': symbol_counts_long[a],
            'Candles_B_Long': symbol_counts_long[b],
            'Candles_A_Short': symbol_counts_short[a],
            'Candles_B_Short': symbol_counts_short[b],
            'P-Value_Long': long_data['p_value'],
            'Hedge_Ratio_Long': long_data['hedge_ratio'],
            'N_Candles_Long_Overlap': long_data['n_candles'],
            'P-Value_Short': short_data['p_value'],
            'Hedge_Ratio_Short': short_data['hedge_ratio'],
            'N_Candles_Short_Overlap': short_data['n_candles']
        })

    df_res = pd.DataFrame(results)
    csv_path = save_results_to_csv(df_res)
    
    if csv_path:
        debug_print(f"Results saved to: {csv_path}")
    
    debug_print("\nCointegrated pairs found:")
    print(df_res.to_string(index=False))

    # Visualization option
    if common_pairs:
        sel = input("\nEnter pair numbers to visualize (comma separated) or press Enter to skip: ")
        try:
            if sel.strip():
                idxs = [int(x)-1 for x in re.findall(r'\d+', sel)]
            else:
                return
        except:
            return

        for i in idxs:
            if 0 <= i < len(results):
                pair_to_visualize = tuple(results[i]['Pair'].split('/'))
                visualize_relationship(pair_to_visualize)
    else:
        debug_print("No pairs to visualize.")

if __name__ == '__main__':
    main()
