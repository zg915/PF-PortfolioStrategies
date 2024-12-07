# %%
import yfinance as yf
import pandas as pd
import numpy as np

# %% [markdown]
# # Metrics function

# %%
def Calculate_Evaluation_metric(returns):
    """
    Calculate evaluation metrics for a given array of returns.
    
    Args:
    returns (list or np.array): Portfolio returns over T periods (monthly).
    
    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    T = len(returns)
    
    # Transform it back to dollar return
    returns = np.array(returns)/100 
    
    # Annualized Compound Return (CR)
    CR = (np.prod(1 + returns) ** (12 / T)) - 1
    
    # Average return
    mean_return = np.mean(returns)
    
    # Annualized Standard Deviation (SD)
    SD = np.sqrt(12 * np.mean((returns - mean_return) ** 2))
    
    # Annualized Downside Deviation (DD)
    downside_returns = np.minimum(0, returns)
    DD = np.sqrt(12 * np.mean(downside_returns ** 2))
    
    # Sharpe Ratio (ShR)
    AR = 12 * mean_return
    ShR = AR / SD if SD != 0 else np.nan
    
    # Sortino Ratio (SoR)
    SoR = AR / DD if DD != 0 else np.nan
    
    # Maximum Drawdown (MDD) and Average Drawdown (ADD)
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peak - cumulative_returns) / peak
    MDD = np.max(drawdowns)
    ADD = np.mean(drawdowns)
    
    # Return results as a dictionary (%)
    return {
        "Compound Return (CR)": CR * 100,
        "Standard Deviation (SD)": SD * 100,
        "Downside Deviation (DD)": DD * 100,
        "Sharpe Ratio (ShR)": ShR,  
        "Sortino Ratio (SoR)": SoR,  
        "Maximum Drawdown (MDD)": MDD * 100,
        "Average Drawdown (ADD)": ADD * 100
    }



# %% [markdown]
# # Using tickers to Test Buy and Hold 

# %%
if __name__ == "__main__":
    # List of yfinance-compatible tickers
    tickers = [
        "SPY",      # S&P 500 ETF (large-cap U.S. equities)
        "IWM",      # iShares Russell 2000 ETF (small-cap U.S. equities)
        "QQQ",      # Nasdaq 100 ETF (tech-heavy U.S. equities)
        "IEF",      # iShares 7-10 Year Treasury Bond ETF (intermediate bonds)
        "TLT",      # iShares 20+ Year Treasury Bond ETF (long-term bonds)
        "BND",      # Vanguard Total Bond Market ETF (broad bond market)
        "VNQ",      # Vanguard Real Estate ETF (U.S. REITs)
        "GLD",      # SPDR Gold Shares (gold commodity)
        "DBC",      # Invesco DB Commodity Index Tracking Fund (broad commodities)
        "VTI"       # Vanguard Total Stock Market ETF (overall U.S. equities)
    ]

    # Download monthly returns data for the last 14 years
    start_date = "2011-11-01"
    end_date = "2024-11-01"

    # Fetch monthly data for each ticker
    monthly_returns = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1mo', progress=False)['Adj Close']
        returns = data.pct_change().dropna() * 100  # Calculate monthly returns
        monthly_returns[ticker] = returns

    # Combine all into a single DataFrame
    monthly_returns_df = pd.DataFrame(monthly_returns)
    monthly_returns_df.index.name = "Date"

    # Abbreviation mapping for tickers
    abbreviation_mapping = {
        "SPY": "USE",     # Large-cap U.S. equities
        "IWM": "USSC",    # Small-cap U.S. equities
        "QQQ": "UST",     # Technology-focused U.S. equities
        "IEF": "USB",     # Intermediate-term U.S. bonds
        "TLT": "LTB",     # Long-term U.S. bonds
        "BND": "BB",      # Broad U.S. bond market
        "VNQ": "USR",     # U.S. REITs
        "GLD": "GC",      # Gold commodity
        "DBC": "BC",      # Broad commodities
        "VTI": "TSE"      # Total U.S. equities
    }

    # Rename columns based on the abbreviation mapping
    monthly_returns_df.rename(columns=abbreviation_mapping, inplace=True)
    monthly_returns_df

    # %% [markdown]
    # # Test Result

    # %%
    metrics = {}
    for ticker in monthly_returns_df.columns:
        metrics[ticker] = Calculate_Evaluation_metric(monthly_returns_df[ticker])

    evaluation_metrics_df = pd.DataFrame(metrics).T
    print(evaluation_metrics_df)


