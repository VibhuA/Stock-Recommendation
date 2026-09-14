import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nifty 50 Momentum Strategy",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Nifty 50 Momentum Strategy Dashboard")
st.markdown("Automated backtesting and live trading recommendations using 20-Day Momentum, 50 EMA Trend Filter, and Trailing Stops.")

# ---------------------------------------------------------
# 2. SIDEBAR INPUTS & PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Strategy Parameters")

INITIAL_CAPITAL = st.sidebar.number_input(
    "Investment Amount (INR)",
    min_value=10000.0,
    max_value=100000000.0,
    value=500000.0,
    step=10000.0
)

MAX_POSITIONS = st.sidebar.number_input("Max Target Positions", min_value=1, max_value=10, value=5)
MOMENTUM_DAYS = st.sidebar.slider("Momentum Lookback (Days)", min_value=5, max_value=252, value=20)
STOP_LOSS_PCT = st.sidebar.slider("Hard Stop Loss (%)", min_value=0.01, max_value=0.20, value=0.08, step=0.01)
TRAILING_STOP_PCT = st.sidebar.slider("Trailing Stop Loss (%)", min_value=0.01, max_value=0.20, value=0.08, step=0.01)

START_DATE = "2022-01-01"
BACKTEST_START = "2023-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "M&M.NS", "NTPC.NS", "POWERGRID.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "BAJFINANCE.NS"
]

# ---------------------------------------------------------
# 3. DATA LOADING (CACHED FOR PERFORMANCE)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df_download = yf.download(NIFTY_50_TICKERS, start=START_DATE, end=END_DATE, progress=False)
    prices = df_download['Close'].dropna(how='all', axis=1).ffill()
    return prices

# ---------------------------------------------------------
# 4. BACKTEST ENGINE & SIGNAL GENERATOR
# ---------------------------------------------------------
def run_backtest_and_signal(prices, initial_capital, momentum_days, stop_loss_pct, trailing_stop_pct, max_positions=5):
    ema_50 = prices.ewm(span=50, adjust=False).mean()
    
    portfolio = {}
    cash = initial_capital
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
            ema_val_remove = ema_50.loc[current_date, ticker] # Fetch current day's EMA
            
            if price <= hard_stop_price or price <= trailing_stop_price or price <ema_val_remove:
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
    # LIVE REBALANCING SIGNAL SHEET
    # ---------------------------------------------------------
    latest_prices = prices.iloc[-1]
    past_prices_latest = prices.iloc[-1 - momentum_days]
    
    momentum_scores_latest = (latest_prices - past_prices_latest) / past_prices_latest
    trend_and_positive_mask = (latest_prices > ema_50.iloc[-1]) & (momentum_scores_latest > 0)
    trend_mask_latest = trend_and_positive_mask
    
    # Filter candidates above 50 EMA and with positive momentum
    valid_candidates = momentum_scores_latest[trend_mask_latest]
    
    top_recommendations = valid_candidates.nlargest(max_positions)
        
    slot_budget = initial_capital / (len(valid_candidates) if len(valid_candidates) > 0 else 1)
    
    order_sheet = []
    
    for ticker, score in top_recommendations.items():
        price = latest_prices[ticker]
        above_ema = bool(trend_mask_latest[ticker])
        shares = int(slot_budget // price) if price > 0 else 0
        stop_price = round(price * (1 - stop_loss_pct), 2)
        
        order_sheet.append({
            'Ticker': ticker,
            'Price (INR)': round(price, 2),
            f'{momentum_days}D Momentum (%)': round(score * 100, 2),
            'Above 50 EMA & Positive': "YES" if above_ema else "NO",
            'Recommended Shares': shares,
            'Allocated Capital (INR)': round(shares * price, 2),
            'Hard Stop-Loss (INR)': stop_price
        })

    # Performance Calculation
    final_val = portfolio_history[-1]
    years = (prices.index[-1] - datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25
    cagr = (((final_val / initial_capital) ** (1 / years)) - 1) * 100

    history_df = pd.DataFrame({'Portfolio Value': portfolio_history}, index=trading_days)

    return pd.DataFrame(order_sheet), round(cagr, 2), round(final_val, 2), history_df

# ---------------------------------------------------------
# 5. EXECUTION & DISPLAY
# ---------------------------------------------------------
if st.button("Run Strategy & Generate Signals", type="primary"):
    with st.spinner("Fetching market data and running backtest..."):
        prices = load_data()
        
        recommendations_df, cagr, final_val, history_df = run_backtest_and_signal(
            prices,
            initial_capital=INITIAL_CAPITAL,
            momentum_days=MOMENTUM_DAYS,
            stop_loss_pct=STOP_LOSS_PCT,
            trailing_stop_pct=TRAILING_STOP_PCT,
            max_positions=MAX_POSITIONS
        )

    # Metric Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Starting Capital", f"₹{INITIAL_CAPITAL:,.2f}")
    col2.metric("Historical Growth Value", f"₹{final_val:,.2f}")
    col3.metric("Backtest CAGR", f"{cagr}%")

    st.subheader("📋 Recommended Positions")
    if not recommendations_df.empty:
        st.dataframe(recommendations_df, use_container_width=True)
    else:
        st.warning("No stocks currently satisfy both the 50 EMA trend filter and positive momentum condition.")

    st.subheader("📈 Historical Equity Curve")
    st.line_chart(history_df)
