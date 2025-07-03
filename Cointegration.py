import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import itertools
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO
from statsmodels.tsa.stattools import adfuller
import re

# --- Configuration ---
SHORT_TERM_HOURS = 250
LONG_TERM_HOURS = 10000 
SIGNIFICANCE_LEVEL = 0.05
MIN_DATA_POINTS = 100
MAX_RETRIES = 3

def get_crypto_tickers():
    """
    Scrapes top cryptocurrency tickers from Yahoo Finance with a fallback list.
    """
    try:
        url = "https://finance.yahoo.com/cryptocurrencies?count=50"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        df = pd.read_html(StringIO(str(table)))[0]
        tickers = df['Symbol'].dropna().astype(str).tolist()
        return tickers
    except Exception as e:
        print(f"Warning: Ticker scraping failed ({e}). Using a fallback list.")
        return [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
            'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD',
            'TRX-USD', 'MATIC-USD', 'LTC-USD',
            'BCH-USD', 'UNI-USD', 'ATOM-USD', 'XLM-USD', 'ETC-USD'
        ]

def test_cointegration(series1, series2):
    """
    Performs the Engle-Granger cointegration test on a pair of price series.
    """
    X = np.vstack([np.ones(len(series1)), series2.values]).T
    y = series1.values
    
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        spread = y - np.dot(X, beta)
        adf_result = adfuller(spread, regression='c', autolag='BIC')
        return adf_result[1]
    except (np.linalg.LinAlgError, ValueError):
        return 1.0

def find_cointegrated_pairs(tickers, hours):
    """
    Scans all pairs for cointegration over a given lookback period.
    """
    print(f"\n--- Scanning for cointegrated pairs over {hours} hours ---")
    
    end_time = datetime.datetime.now(pytz.utc)
    start_time = end_time - datetime.timedelta(hours=hours)
    
    all_data = yf.download(
        tickers, start=start_time, end=end_time, interval='60m', 
        auto_adjust=True, progress=False
    )
    
    if all_data.empty:
        print("Failed to download any data.")
        return set()

    close_prices = all_data['Close'].dropna(axis=1, how='all')
    available_tickers = close_prices.columns.tolist()
    print(f"Successfully fetched data for {len(available_tickers)} tickers for this period.")

    if len(available_tickers) < 2:
        print("Not enough data to form pairs.")
        return set()

    pairs = list(itertools.combinations(available_tickers, 2))
    cointegrated_pairs = set()

    for i, (ticker1, ticker2) in enumerate(pairs):
        df_pair = close_prices[[ticker1, ticker2]].dropna()
        
        if len(df_pair) < MIN_DATA_POINTS:
            continue
            
        p_value = test_cointegration(df_pair[ticker1], df_pair[ticker2])
        
        if p_value is not None and p_value <= SIGNIFICANCE_LEVEL:
            cointegrated_pairs.add(tuple(sorted((ticker1, ticker2))))
            
        print(f"Tested {i+1}/{len(pairs)} pairs - Found: {len(cointegrated_pairs)} cointegrated", end='\r', flush=True)

    print(f"\nFound {len(cointegrated_pairs)} cointegrated pairs for the {hours}-hour period.")
    return cointegrated_pairs

def visualize_short_term_relationship(ticker1, ticker2):
    """
    Fetches SHORT-TERM data and creates the visualization plots for a confirmed pair.
    """
    print(f"\nVisualizing SHORT-TERM ({SHORT_TERM_HOURS} hours) relationship for {ticker1}/{ticker2}...")
    end_time = datetime.datetime.now(pytz.utc)
    # FIX: Changed to fetch short-term data for visualization
    start_time = end_time - datetime.timedelta(hours=SHORT_TERM_HOURS)
    pair_data_df = yf.download([ticker1, ticker2], start=start_time, end=end_time, interval='60m', auto_adjust=True, progress=False)['Close']
    pair_data_df.dropna(inplace=True)

    if pair_data_df.empty or len(pair_data_df) < MIN_DATA_POINTS:
        print(f"Could not fetch sufficient short-term data for {ticker1}/{ticker2} to visualize.")
        return

    # Calculate spread on the short-term data
    X = np.vstack([np.ones(len(pair_data_df)), pair_data_df[ticker2].values]).T
    y = pair_data_df[ticker1].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    spread = y - np.dot(X, beta)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # FIX: Updated plot title
    fig.suptitle(f'Short-Term Cointegration Analysis: {ticker1} vs {ticker2} ({SHORT_TERM_HOURS} Hours)', fontsize=16)
    
    # Plot 1: Normalized Prices
    normalized_price1 = (pair_data_df[ticker1] / pair_data_df[ticker1].iloc[0]) * 100
    normalized_price2 = (pair_data_df[ticker2] / pair_data_df[ticker2].iloc[0]) * 100
    
    ax1.plot(normalized_price1, label=ticker1, color='cyan')
    ax1.plot(normalized_price2, label=ticker2, color='magenta', alpha=0.8)
    ax1.set_title('Normalized Prices (Indexed to 100)')
    ax1.set_ylabel('Normalized Price')
    ax1.legend()
    
    # Plot 2: Spread
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    ax2.plot(pair_data_df.index, spread, label='Spread (Residuals)', color='green')
    ax2.axhline(spread_mean, color='black', linestyle='--', label=f'Mean: {spread_mean:.4f}')
    ax2.axhline(spread_mean + 2 * spread_std, color='red', linestyle=':', label='+2 Std Dev')
    ax2.axhline(spread_mean - 2 * spread_std, color='red', linestyle=':', label='-2 Std Dev')
    
    # FIX: Updated plot title
    ax2.set_title(f'Short-Term Spread (Hedge Ratio: {beta[1]:.4f})')
    ax2.set_ylabel('Spread Value')
    ax2.set_xlabel('Date')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def main():
    """
    Main function to run the two-stage filtering and visualization.
    """
    print("--- Starting Two-Stage Cointegration Analysis ---")
    tickers = get_crypto_tickers()
    
    long_term_pairs = find_cointegrated_pairs(tickers, LONG_TERM_HOURS)
    short_term_pairs = find_cointegrated_pairs(tickers, SHORT_TERM_HOURS)
    
    overlapping_pairs = long_term_pairs.intersection(short_term_pairs)
    
    print("\n" + "="*50)
    print("--- FINAL ANALYSIS COMPLETE ---")
    
    if overlapping_pairs:
        sorted_pairs = sorted(list(overlapping_pairs))
        
        print(f"✅ Found {len(sorted_pairs)} pairs cointegrated on BOTH timeframes:")
        for i, (ticker1, ticker2) in enumerate(sorted_pairs):
            print(f"  {i+1}: {ticker1}/{ticker2}")
        
        print("\nEnter the numbers of the pairs you want to visualize (e.g., '1 3 5' or '2,4').")
        print("Press Enter to visualize all, or type 'exit' to quit.")
        
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if user_input.lower() == 'exit':
            print("Exiting.")
            return
            
        selected_indices = []
        if user_input.strip() == '':
            selected_indices = range(len(sorted_pairs))
        else:
            raw_indices = re.split(r'[,\s]+', user_input.strip())
            for idx_str in raw_indices:
                if idx_str.isdigit():
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(sorted_pairs):
                        selected_indices.append(idx)
                    else:
                        print(f"Warning: Invalid number '{idx_str}' ignored (out of range).")
                elif idx_str:
                    print(f"Warning: Invalid input '{idx_str}' ignored.")

        if not selected_indices:
            print("No valid pairs selected. Exiting.")
            return

        print(f"\nVisualizing {len(selected_indices)} selected pair(s)...")
        for idx in selected_indices:
            ticker1, ticker2 = sorted_pairs[idx]
            # FIX: Changed function call to visualize the short-term relationship
            visualize_short_term_relationship(ticker1, ticker2)
            
    else:
        print("❌ No pairs were found to be cointegrated on both timeframes.")

if __name__ == "__main__":
    main()