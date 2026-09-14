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
st.markdown("Automated backtesting and live trading recommendations using 20-Day Momentum, Stock & Index 50 EMA Trend Filters, and Trailing Stops.")

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

INDEX_TICKER = "^NSEI"

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
    all_tickers = NIFTY_50_TICKERS + [INDEX_TICKER]
    df_download = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)
    prices = df_download['Close'].dropna(how='all', axis=1).ffill()
    
    # Separate index data from stock data
    index_prices = prices[INDEX_TICKER]
    stock_prices = prices.drop(columns=[INDEX_TICKER])
    
    return stock_prices, index_prices

# ---------------------------------------------------------
# 4. BACKTEST ENGINE & SIGNAL GENERATOR
# ---------------------------------------------------------
def run_backtest_and_signal(prices, index_prices, initial_capital, momentum_days, stop_loss_pct, trailing_stop_pct, max_positions=5):
    ema_50 = prices.ewm(span=50, adjust=False).mean()
    index_ema_50 = index_prices.ewm(span=50, adjust=False).mean()
    
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
            ema_val_remove = ema_50.loc[current_date, ticker]
            
            if price <= hard_stop_price or price <= trailing_stop_price or price < ema_val_remove:
                cash += pos['shares'] * price
                tickers_to_remove.append(ticker)
                
        for t in tickers_to_remove:
            del portfolio[t]

        # B. Entries (Index Filter + Stock Filter + Momentum)
        # Check if the Nifty 50 index is trading above its 50 EMA
        is_index_bullish = index_prices.loc[current_date] > index_ema_50.loc[current_date]
        
        open_slots = max_positions - len(portfolio)
        
        # Only buy if the overall market/index trend is bullish
        if is_index_bullish and open_slots > 0 and cash > 10000:
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
    latest_index_price = index_prices.iloc[-1]
    latest_index_ema = index_ema_50.iloc[-1]
    is_index_bullish_now = latest_index_price > latest_index_ema

    latest_prices = prices.iloc[-1]
    past_prices_latest = prices.iloc[-1 - momentum_days]
    
    momentum_scores_latest = (latest_prices - past_prices_latest) / past_prices_latest
    trend_and_positive_mask = (latest_prices > ema_50.iloc[-1]) & (momentum_scores_latest > 0)
    
    order_sheet = []
    
    # If the index is not above its 50 EMA, reject all buy signals
    if is_index_bullish_now:
        valid_candidates = momentum_scores_latest[trend_and_positive_mask]
        top_recommendations = valid_candidates.nlargest(max_positions)
        slot_budget = initial_capital / (len(top_recommendations) if len(top_recommendations) > 0 else 1)
        
        for ticker, score in top_recommendations.items():
            price = latest_prices[ticker]
            shares = int(slot_budget // price) if price > 0 else 0
            stop_price = round(price * (1 - stop_loss_pct), 2)
            
            order_sheet.append({
                'Ticker': ticker,
                'Price (INR)': round(price, 2),
                f'{momentum_days}D Momentum (%)': round(score * 100, 2),
                'Above 50 EMA': "YES",
                'Recommended Shares': shares,
                'Allocated Capital (INR)': round(shares * price, 2),
                'Hard Stop-Loss (INR)': stop_price
            })

    # Performance Calculation
    final_val = portfolio_history[-1]
    years = (prices.index[-1] - datetime.strptime(BACKTEST_START, "%Y-%m-%d")).days / 365.25
    cagr = (((final_val / initial_capital) ** (1 / years)) - 1) * 100

    history_df = pd.DataFrame({'Portfolio Value': portfolio_history}, index=trading_days)

    return pd.DataFrame(order_sheet), round(cagr, 2), round(final_val, 2), history_df, is_index_bullish_now, round(latest_index_price, 2), round(latest_index_ema, 2)

# ---------------------------------------------------------
# 5. EXECUTION & DISPLAY
# ---------------------------------------------------------
if st.button("Run Strategy & Generate Signals", type="primary"):
    with st.spinner("Fetching market data and running backtest..."):
        prices, index_prices = load_data()
        
        recommendations_df, cagr, final_val, history_df, is_index_bullish, index_px, index_ema = run_backtest_and_signal(
            prices,
            index_prices,
            initial_capital=INITIAL_CAPITAL,
            momentum_days=MOMENTUM_DAYS,
            stop_loss_pct=STOP_LOSS_PCT,
            trailing_stop_pct=TRAILING_STOP_PCT,
            max_positions=MAX_POSITIONS
        )

    # Market Status Indicator
    st.subheader("🌐 Market Regimen (Nifty 50 Filter)")
    if is_index_bullish:
        st.success(f"**BULLISH REGIMEN**: Nifty 50 (`{index_px:,.2f}`) is ABOVE 50 EMA (`{index_ema:,.2f}`). Trade entries allowed.")
    else:
        st.error(f"**BEARISH REGIMEN**: Nifty 50 (`{index_px:,.2f}`) is BELOW 50 EMA (`{index_ema:,.2f}`). New purchases paused.")

    # Metric Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Starting Capital", f"₹{INITIAL_CAPITAL:,.2f}")
    col2.metric("Historical Growth Value", f"₹{final_val:,.2f}")
    col3.metric("Backtest CAGR", f"{cagr}%")

    st.subheader("📋 Recommended Positions")
    if not is_index_bullish:
        st.warning("No stock recommendations generated because the Nifty 50 Index is currently below its 50-day EMA.")
    elif not recommendations_df.empty:
        st.dataframe(recommendations_df, use_container_width=True)
    else:
        st.warning("No individual stocks currently meet both the 50 EMA trend filter and positive momentum conditions.")

    st.subheader("📈 Historical Equity Curve")
    st.line_chart(history_df)
