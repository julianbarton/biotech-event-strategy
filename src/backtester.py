import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class BiotechEventBacktester:
    """
    Custom Event-Study Framework for FDA Catalyst Trading.
    
    Calculates Cumulative Abnormal Returns (CAR) around binary events
    to isolate the 'alpha' of the trial quality signal from broader market moves.
    """
    
    def __init__(self, benchmark_ticker='XBI', estimation_window=60, event_window=(-5, 1)):
        """
        :param benchmark_ticker: ETF to calculate 'Abnormal Returns' against (default: XBI for Biotech).
        :param estimation_window: Days used to calculate Beta before the event (avoiding look-ahead bias).
        :param event_window: Tuple (start, end) relative to event date. E.g. (-5, 1) means enter 5 days before, exit 1 day after.
        """
        self.benchmark_ticker = benchmark_ticker
        self.estimation_window = estimation_window
        self.event_window = event_window
        self.market_data = None
        self.results = []

    def fetch_market_data(self, tickers, start_date, end_date):
        """
        Fetches price data. 
        NOTE: In a production environment, this step requires a survivorship-bias-free database 
        (e.g., Norgate Data) to account for delisted biotechs. 
        For this prototype, we use yfinance with known limitations.
        """
        print(f"Fetching data for {len(tickers)} tickers + {self.benchmark_ticker}...")
        all_tickers = tickers + [self.benchmark_ticker]
        
        # Download adjusted close prices
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Calculate daily log returns
        self.market_data = np.log(data / data.shift(1)).dropna()
        print("Data fetch complete.")

    def calculate_expected_return(self, stock_ticker, event_date_idx):
        """
        Uses the Market Model (CAPM-lite) to predict what the stock *should* have done 
        based on XBI movements.
        """
        # Define the 'Estimation Window' (e.g., T-65 to T-5)
        # We STOP before the event window starts to avoid contaminating our Beta with the event volatility itself.
        est_start = event_date_idx - self.estimation_window - abs(self.event_window[0])
        est_end = event_date_idx - abs(self.event_window[0])
        
        if est_start < 0:
            return None # Not enough history

        # Slice data for regression
        y = self.market_data[stock_ticker].iloc[est_start:est_end].values.reshape(-1, 1)
        X = self.market_data[self.benchmark_ticker].iloc[est_start:est_end].values.reshape(-1, 1)
        
        # Calculate Beta (sensitivity to XBI) and Alpha (baseline drift)
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0][0]
        alpha = model.intercept_[0]
        
        return alpha, beta

    def run_event_study(self, events_df):
        """
        Core Logic: Iterates through your 'Quality Analyzer' signals.
        events_df must have columns: ['ticker', 'event_date', 'quality_score', 'catalyst_type']
        """
        print("Running Event Study Analysis...")
        
        for idx, row in events_df.iterrows():
            ticker = row['ticker']
            date = pd.to_datetime(row['event_date'])
            quality = row['quality_score']
            
            if ticker not in self.market_data.columns:
                continue
                
            # Find the index location of the event date
            try:
                event_idx = self.market_data.index.get_loc(date)
            except KeyError:
                continue
            
            # 1. Calculate Market Beta
            params = self.calculate_expected_return(ticker, event_idx)
            if not params: continue
            alpha, beta = params
            
            # 2. Calculate Abnormal Returns during the 'Trade Window'
            win_start = event_idx + self.event_window[0]
            win_end = event_idx + self.event_window[1]
            
            # Real Return of the stock
            real_returns = self.market_data[ticker].iloc[win_start:win_end+1]
            
            # Return of the Benchmark (XBI)
            market_returns = self.market_data[self.benchmark_ticker].iloc[win_start:win_end+1]
            
            # Expected Return = Alpha + Beta * Market_Return
            expected_returns = alpha + (beta * market_returns)
            
            # Abnormal Return = Real - Expected
            abnormal_returns = real_returns - expected_returns
            
            # Cumulative Abnormal Return (CAR) - This is your 'Edge'
            car = abnormal_returns.sum()
            
            self.results.append({
                'Ticker': ticker,
                'Event_Date': date,
                'Quality_Score': quality,
                'CAR': car,
                'Real_Return': real_returns.sum()
            })
            
        return pd.DataFrame(self.results)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # 1. Initialize Strategy
    backtester = BiotechEventBacktester(event_window=(-5, 1))
    
    # 2. Simulate Data from your BioTrial Analyzer
    dummy_events = pd.DataFrame({
        'ticker': ['ITCI', 'VRTX', 'BIIB'],
        'event_date': ['2024-12-20', '2024-11-15', '2024-10-01'],
        'quality_score': ['High', 'High', 'Low']
    })
    
    # 3. Fetch Data
    backtester.fetch_market_data(
        tickers=['ITCI', 'VRTX', 'BIIB'], 
        start_date='2024-01-01', 
        end_date='2025-01-01'
    )
    
    # 4. Run Backtest
    results = backtester.run_event_study(dummy_events)
    
    # 5. Simple Analysis
    print("\n--- Backtest Results ---")
    print(results.groupby('Quality_Score')['CAR'].mean())
