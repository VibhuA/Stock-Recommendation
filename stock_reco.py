import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="Momentum Trading Dashboard", layout="wide")
st.title("🚀 Nifty 50 Momentum Strategy")
st.write("Backtest results and live stock suggestions based on 6-month relative momentum.")

# --- PARAMETERS & TICKERS ---
SYMBOLS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
    'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS',
    'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS',
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS',
    'SBIN.NS',  'TCS.NS',  'TATASTEEL.NS', 'TECHM.NS',  'ULTRACEMCO.NS', 
]

# --- DATA FETCHING (CACHED) ---
@st.cache_data(ttl=3600)
def load_data(tickers):
    data = yf.download(tickers, start='2020-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))['Close']
    return data.ffill().bfill()

data = load_data(SYMBOLS)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Settings")
sl_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 3.0) / 100
tp_pct = st.sidebar.slider("Take Profit (%)", 5.0, 20.0, 10.0) / 100
fees_pct = st.sidebar.number_input("Transaction Fees (%)", value=0.3) / 100

# --- LIVE SUGGESTIONS SECTION ---
st.subheader("🎯 Live Suggestions for Today")
current_momentum = data.pct_change(125).iloc[-1]
top_5_now = current_momentum.nlargest(5)

cols = st.columns(5)
for i, (stock, val) in enumerate(top_5_now.items()):
    with cols[i]:
        st.metric(label=f"Rank {i+1}", value=stock, delta=f"{val*100:.2f}% Mom")

st.info(f"**Action:** Allocate 20% to each. Set SL at -{sl_pct*100}% and TP at +{tp_pct*100}%.")

# --- BACKTEST LOGIC ---
st.divider()
st.subheader("📈 Backtest Performance")

momentum = data.pct_change(125).fillna(0)
current_cash = 100.0
monthly_stats = []

for period, group in data.groupby(pd.Grouper(freq='MS')):
    if len(group) < 2: continue
    start_date = group.index[0]
    top_5 = momentum.loc[start_date].nlargest(5).index.tolist()
    
    cash_per_stock = current_cash / 5
    monthly_pnl = 0
    
    for stock in top_5:
        prices = group[stock]
        entry_price = prices.iloc[0]
        exit_price = prices.iloc[-1]
        
        pct_change = (prices / entry_price) - 1
        sl_hit = pct_change[pct_change <= -sl_pct]
        tp_hit = pct_change[pct_change >= tp_pct]
        
        if not sl_hit.empty and (tp_hit.empty or sl_hit.index[0] < tp_hit.index[0]):
            exit_price = entry_price * (1 - sl_pct)
        elif not tp_hit.empty:
            exit_price = entry_price * (1 + tp_pct)
            
        stock_return = (exit_price / entry_price) - 1 - fees_pct
        monthly_pnl += cash_per_stock * stock_return
        
    monthly_stats.append({
        'Month': period, 
        'Return%': (monthly_pnl/current_cash)*100,
        'Stocks': ", ".join(top_5)
    })
    current_cash += monthly_pnl

results_df = pd.DataFrame(monthly_stats).set_index('Month')

# Metrics Calculation
total_years = (data.index[-1] - data.index[0]).days / 365.25
cagr = (((current_cash / 100) ** (1 / total_years)) - 1) * 100
sharpe = (results_df['Return%'].mean() / results_df['Return%'].std()) * np.sqrt(12)

# Display Summary Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Final Portfolio Value", f"₹{current_cash:.2f}")
m2.metric("CAGR", f"{cagr:.2f}%")
m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# Charting
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results_df.index, results_df['Return%'].cumsum(), color='#1E88E5', linewidth=2)
ax.set_title("Cumulative Returns (%)")
ax.grid(True, alpha=0.2)
st.pyplot(fig)

# Table
st.write("### Monthly Breakdown")
st.dataframe(results_df[['Return%', 'Stocks']].sort_index(ascending=False), use_container_width=True)