import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

# ---------------------------------------------------------
# 1. PARAMETERS & UNIVERSE SETUP
# ---------------------------------------------------------
START_DATE = "2022-01-01"
BACKTEST_START = "2023-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')
print("Enter Investment Amount")
INITIAL_CAPITAL = int(input())#100000.0
MAX_POSITIONS = 5  # Exactly 5 positions target

# Selected Strategy Parameters
MOMENTUM_DAYS = 20        # 20 Lookback Days
STOP_LOSS_PCT = 0.08      # 8% Hard Stop
TRAILING_STOP_PCT = 0.08  # 8% Trailing Stop

NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "M&M.NS", "NTPC.NS", "POWERGRID.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "BAJFINANCE.NS"
]

print("Downloading dataset...")
df_download = yf.download(NIFTY_50_TICKERS, start=START_DATE, end=END_DATE, progress=False)
prices = df_download['Close'].dropna(how='all', axis=1).ffill()

# ---------------------------------------------------------
# 2. STRATEGY ENGINE & RECOMMENDATION GENERATOR
# ---------------------------------------------------------
def run_backtest_and_signal(prices, momentum_days, stop_loss_pct, trailing_stop_pct, max_positions=5):
    ema_50 = prices.ewm(span=50, adjust=False).mean()
    
    portfolio = {}
    cash = INITIAL_CAPITAL
    portfolio_history = []
    
    trading_days = prices.loc[BACKTEST_START:END_DATE].index

    for current_date in trading_days:
        current_prices = prices.loc[current_date]
        
        # A. Exits (Hard & Trailing Stop Loss)
        tickers_to_remove = []
        for ticker, pos in portfolio.items():
            price = current_prices[ticker]
            if pd.isna(price):
                continue
            
            if price > pos['highest_price']:
                pos['highest_price'] = price
                
            hard_stop_price = pos['buy_price'] * (1 - stop_loss_pct)
            trailing_stop_price = pos['highest_price'] * (1 - trailing_stop_pct)
            
            if price <= hard_stop_price or price <= trailing_stop_price:
                cash += pos['shares'] * price
                tickers_to_remove.append(ticker)
                
        for t in tickers_to_remove:
            del portfolio[t]

        # B. Entries (Ranked by 20-Day Momentum & 50 EMA Filter)
        open_slots = max_positions - len(portfolio)
        if open_slots > 0 and cash > 10000:
            past_idx = prices.index.get_loc(current_date) - momentum_days
            if past_idx >= 0:
                past_prices = prices.iloc[past_idx]
                momentum_scores = (current_prices - past_prices) / past_prices
                
                trend_mask = current_prices > ema_50.loc[current_date]
                valid_scores = momentum_scores[trend_mask].drop(labels=list(portfolio.keys()), errors='ignore')
                
                top_candidates = valid_scores.nlargest(open_slots)
                allocation_per_slot = cash / open_slots
                
                for ticker, score in top_candidates.items():
                    price = current_prices[ticker]
                    if not pd.isna(price) and price > 0 and score > 0:
                        shares = int(allocation_per_slot // price)
                        if shares > 0:
                            cash -= shares * price
                            portfolio[ticker] = {
                                'buy_price': price,
                                'highest_price': price,
                                'buy_date': current_date,
                                'shares': shares
                            }

        # C. Valuation Tracking
        holdings_val = sum(pos['shares'] * current_prices[t] for t, pos in portfolio.items() if not pd.isna(current_prices[t]))
        portfolio_history.append(cash + holdings_val)

    # ---------------------------------------------------------
    # LIVE REBALANCING SIGNAL SHEET (EXACTLY 5 STOCKS)
    # ---------------------------------------------------------
    latest_prices = prices.iloc[-1]
    past_prices_latest = prices.iloc[-1 - momentum_days]
    
    momentum_scores_latest = (latest_prices - past_prices_latest) / past_prices_latest
    trend_mask_latest = trend_and_positive_mask = (latest_prices > ema_50.iloc[-1]) & (momentum_scores_latest > 0)
    
    
    # Filter candidates above 50 EMA
    valid_candidates = momentum_scores_latest[trend_mask_latest]
    
    # Fallback to top relative momentum if fewer than 5 stocks pass 50 EMA
    #if len(valid_candidates) < max_positions:
    #    top_recommendations = momentum_scores_latest.nlargest(max_positions)
    #else:
    top_recommendations = valid_candidates.nlargest(max_positions)
        
    slot_budget = INITIAL_CAPITAL / len(valid_candidates)#max_positions
    
    order_sheet = []
    
    for ticker, score in top_recommendations.items():
        price = latest_prices[ticker]
        above_ema = bool(trend_mask_latest[ticker])
        print(trend_mask_latest[ticker])
        shares = int(slot_budget // price)
        stop_price = round(price * (1 - stop_loss_pct), 2)
        
        order_sheet.append({
            'Ticker': ticker,
            'Price (INR)': round(price, 2),
            f'{momentum_days}D Momentum (%)': round(score * 100, 2),
            'Above 50 EMA': "YES" if above_ema else "NO (Fallback)",
            'Recommended Shares': shares,
            'Allocated Capital (INR)': round(shares * price, 2),
            'Hard Stop-Loss (INR)': stop_price
        })

    # Performance
    final_val = portfolio_history[-1]
    years = (prices.index[-1] - datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25
    cagr = (((final_val / INITIAL_CAPITAL) ** (1 / years)) - 1) * 100

    return pd.DataFrame(order_sheet), round(cagr, 2), round(final_val, 2)

# ---------------------------------------------------------
# 3. RUN STRATEGY WITH YOUR PARAMETERS
# ---------------------------------------------------------
recommendations_df, cagr, final_val = run_backtest_and_signal(
    prices, 
    momentum_days=MOMENTUM_DAYS, 
    stop_loss_pct=STOP_LOSS_PCT, 
    trailing_stop_pct=TRAILING_STOP_PCT,
    max_positions=MAX_POSITIONS
)

print("\n" + "="*80)
print(f"       LIVE EXECUTION SHEET (20-Day Momentum | 8% Stop | 8% Trailing)      ")
print("="*80)
print(f"Historical Backtest CAGR: {cagr}%")
print(f"Starting Capital: ₹{INITIAL_CAPITAL:,.2f}  |  Historical Growth Value: ₹{final_val:,.2f}")
print("-" * 80)
print(recommendations_df.to_string(index=False))
print("="*80)
