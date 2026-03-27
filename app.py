import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from joblib import Parallel, delayed
import plotly.graph_objects as go

st.set_page_config(page_title="TA Directional Screener", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Directional Propensity Screener")
st.markdown("**Score**: –100 = strongly bearish → +100 = strongly bullish. Powered by trend, momentum, RSI, ADX+DMI, MACD & Stochastic.")

@st.cache_data(ttl=3600)  # cache 1 hour
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url)[0]['Symbol'].tolist()

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        return ticker, df
    except:
        return ticker, None

def compute_score(ticker, df):
    if df is None or len(df) < 50:
        return None
    df = df.copy()
    
    # TA indicators
    df['SMA50'] = ta.sma(df['Close'], 50)
    df['SMA200'] = ta.sma(df['Close'], 200)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    df['RSI'] = ta.rsi(df['Close'])
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, adx], axis=1)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, stoch], axis=1)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0
    # Trend (40%)
    score += 25 if latest['Close'] > latest.get('SMA200', 0) else -25
    score += 15 if latest.get('SMA50', 0) > latest.get('SMA200', 0) else -15
    score += 10 if latest['Close'] > latest.get('SMA50', 0) else -10
    # MACD (25%)
    score += 15 if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', 0) else -15
    score += 10 if latest.get('MACDh_12_26_9', 0) > prev.get('MACDh_12_26_9', 0) else -10
    # RSI (15%)
    score += 15 if latest['RSI'] < 30 else (-15 if latest['RSI'] > 70 else 5)
    # ADX + DMI (15%)
    if latest.get('ADX_14', 0) > 25:
        score += 15 if latest.get('DMP_14', 0) > latest.get('DMN_14', 0) else -15
    # Stochastic (5%)
    score += 5 if latest.get('STOCHk_14_3_3', 0) < 20 else (-5 if latest.get('STOCHk_14_3_3', 0) > 80 else 0)

    score = max(min(score, 100), -100)
    return {
        'ticker': ticker,
        'score': round(score, 1),
        'close': round(latest['Close'], 2),
        'volume': int(latest['Volume']),
        'rsi': round(latest['RSI'], 1),
        'adx': round(latest.get('ADX_14', 0), 1)
    }

if st.button("🚀 Run Full S&P 500 Scan", type="primary"):
    with st.spinner("Scanning ~500 tickers in parallel... (this takes ~30-60 seconds)"):
        tickers = get_sp500_tickers()
        data_list = Parallel(n_jobs=8)(delayed(fetch_data)(t) for t in tickers)
        
        results = []
        for ticker, df in data_list:
            score_dict = compute_score(ticker, df)
            if score_dict:
                results.append(score_dict)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
        
        st.success(f"✅ Scanned {len(df_results)} tickers!")
        
        # Main table
        st.dataframe(
            df_results.style.background_gradient(cmap='RdYlGn', subset=['score'])
            .format({'score': '{:.1f}', 'close': '${:.2f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        # Top 10 chart
        st.subheader("🔥 Top 10 Bullish Tickers")
        top10 = df_results.head(10)
        fig = go.Figure(go.Bar(x=top10['ticker'], y=top10['score'], marker_color='#00ff88'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Detail view
        selected = st.selectbox("Tap any ticker for 6-month chart", df_results['ticker'])
        if selected:
            detail_df = yf.download(selected, period="6mo")
            st.line_chart(detail_df['Close'])
            score = df_results[df_results['ticker'] == selected]['score'].values[0]
            st.metric("Current Propensity Score", f"{score}", delta=None)

# Sidebar info
st.sidebar.info("📱 Works great on mobile — just bookmark the page!\n\nBuilt live for you on March 27, 2026")
