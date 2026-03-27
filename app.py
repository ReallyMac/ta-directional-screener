import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
import numpy as np

st.set_page_config(page_title="TA Directional Screener", layout="wide")
st.title("📈 Full US Market Directional Propensity Screener")
st.markdown("**Now scans stocks + ETFs across NYSE/NASDAQ/AMEX** • Score -100 (bearish) → +100 (bullish) • Chart patterns + Elliott Wave bias included")

# ====================== TICKER UNIVERSES ======================
@st.cache_data(ttl=86400)
def get_all_us_tickers():
    urls = {
        "nasdaq": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq_tickers.txt",
        "nyse": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse_tickers.txt",
        "amex": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex_tickers.txt"
    }
    all_tickers = []
    for url in urls.values():
        try:
            df = pd.read_csv(url, header=None, names=["ticker"])
            all_tickers.extend(df["ticker"].str.strip().tolist())
        except:
            pass
    return sorted(list(set(all_tickers)))  # unique

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()

# ====================== FETCH & SCORE ======================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        return ticker, df
    except:
        return ticker, None

def detect_chart_patterns(df):
    close = df['Close'].values
    peaks, _ = signal.find_peaks(close, distance=5)
    troughs, _ = signal.find_peaks(-close, distance=5)
    patterns = []
    # Simple Double Top / Bottom
    if len(peaks) >= 2 and abs(close[peaks[-1]] - close[peaks[-2]]) / close[peaks[-2]] < 0.03:
        patterns.append("Double Top (Bearish)")
    if len(troughs) >= 2 and abs(close[troughs[-1]] - close[troughs[-2]]) / close[troughs[-2]] < 0.03:
        patterns.append("Double Bottom (Bullish)")
    # Head & Shoulders (very simplified)
    if len(peaks) >= 3 and close[peaks[-2]] > close[peaks[-1]] and close[peaks[-2]] > close[peaks[-3]]:
        patterns.append("Head & Shoulders (Bearish)")
    return patterns

def elliott_wave_bias(df):
    # Simple Fibonacci retracement bias + recent swing direction
    close = df['Close']
    recent_high = close[-60:].max()
    recent_low = close[-60:].min()
    retrace = (recent_high - close.iloc[-1]) / (recent_high - recent_low) if recent_high != recent_low else 0
    # Typical EW retracements
    bias = 0
    if 0.382 < retrace < 0.618:  # Wave 2 or 4 retracement → bullish continuation likely
        bias += 15
    elif retrace > 0.786:  # Deep correction → potential Wave C or new downtrend
        bias -= 10
    # Recent momentum
    if close.iloc[-1] > close.iloc[-20]:
        bias += 10
    return bias

def compute_score(ticker, df):
    if df is None or len(df) < 50:
        return None
    df = df.copy()
    
    # Core indicators
    df['SMA50'] = ta.sma(df['Close'], 50)
    df['SMA200'] = ta.sma(df['Close'], 200)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    df['RSI'] = ta.rsi(df['Close'])
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, adx], axis=1)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, stoch], axis=1)
    bb = ta.bbands(df['Close'])
    df = pd.concat([df, bb], axis=1)
    # Candlestick patterns (sum bullish/bearish)
    cdl = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="all")
    df = pd.concat([df, cdl], axis=1)
    bullish_cdl = df.filter(like='CDL_').iloc[-1].clip(lower=0).sum()
    bearish_cdl = df.filter(like='CDL_').iloc[-1].clip(upper=0).sum()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0
    # Trend (35%)
    score += 25 if latest['Close'] > latest.get('SMA200', 0) else -25
    score += 15 if latest.get('SMA50', 0) > latest.get('SMA200', 0) else -15
    score += 10 if latest['Close'] > latest.get('SMA50', 0) else -10
    # Momentum (25%)
    score += 15 if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', 0) else -15
    score += 10 if latest.get('MACDh_12_26_9', 0) > prev.get('MACDh_12_26_9', 0) else -10
    # RSI (10%)
    score += 15 if latest['RSI'] < 30 else (-15 if latest['RSI'] > 70 else 5)
    # ADX + DMI (10%)
    if latest.get('ADX_14', 0) > 25:
        score += 15 if latest.get('DMP_14', 0) > latest.get('DMN_14', 0) else -15
    # Stochastic (5%)
    score += 5 if latest.get('STOCHk_14_3_3', 0) < 20 else (-5 if latest.get('STOCHk_14_3_3', 0) > 80 else 0)
    # Bollinger (5%)
    score += 8 if latest['Close'] < latest.get('BBL_5_2.0', 0) else (-8 if latest['Close'] > latest.get('BBU_5_2.0', 0) else 0)
    # Candlestick patterns (5%)
    score += min(bullish_cdl * 2, 10) + max(bearish_cdl * 2, -10)
    # Chart formations
    patterns = detect_chart_patterns(df)
    score += 12 if any("Bullish" in p for p in patterns) else (-12 if any("Bearish" in p for p in patterns) else 0)
    # Elliott Wave bias
    score += elliott_wave_bias(df)

    score = max(min(score, 100), -100)
    return {
        'ticker': ticker,
        'score': round(score, 1),
        'close': round(latest['Close'], 2),
        'volume': int(latest['Volume']),
        'rsi': round(latest['RSI'], 1),
        'adx': round(latest.get('ADX_14', 0), 1),
        'patterns': patterns
    }

# ====================== UI ======================
universe = st.selectbox("Universe", ["S&P 500", "NASDAQ-100", "Full US Market (stocks + ETFs)"], index=2)

sidebar = st.sidebar
min_volume = sidebar.slider("Min Avg Daily Volume", 0, 50000000, 100000)
min_price = sidebar.slider("Min Price $", 0.0, 500.0, 2.0)
max_tickers = sidebar.slider("Max tickers to scan (for speed)", 500, 15000, 3000)

if st.button("🚀 Run Scan", type="primary"):
    with st.spinner("Fetching data & computing full TA (including patterns & Elliott bias)..."):
        if universe == "S&P 500":
            tickers = get_sp500_tickers()[:max_tickers]
        elif universe == "NASDAQ-100":
            tickers = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]['Ticker'].tolist()[:max_tickers]
        else:
            tickers = get_all_us_tickers()[:max_tickers]
        
        data_list = Parallel(n_jobs=8)(delayed(fetch_data)(t) for t in tickers)
        
        results = []
        for ticker, df in data_list:
            if df is None or len(df) < 50: continue
            score_dict = compute_score(ticker, df)
            if score_dict and score_dict['volume'] >= min_volume and score_dict['close'] >= min_price:
                results.append(score_dict)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
        
        st.success(f"✅ Scanned {len(df_results)} liquid tickers!")
        
        st.dataframe(
            df_results.style.background_gradient(cmap='RdYlGn', subset=['score'])
            .format({'score': '{:.1f}', 'close': '${:.2f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("🔥 Top 10 Bullish")
        top10 = df_results.head(10)
        fig = go.Figure(go.Bar(x=top10['ticker'], y=top10['score'], marker_color='#00ff88'))
        st.plotly_chart(fig, use_container_width=True)
        
        selected = st.selectbox("Drill into ticker (full overlays)", df_results['ticker'])
        if selected:
            detail_df = yf.download(selected, period="6mo")
            # Rich Plotly chart with overlays
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(f"{selected} Price + Overlays", "MACD + RSI", "Volume"))
            fig.add_trace(go.Candlestick(x=detail_df.index, open=detail_df['Open'], high=detail_df['High'],
                                         low=detail_df['Low'], close=detail_df['Close'], name="Price"), row=1, col=1)
            # Bollinger
            bb = ta.bbands(detail_df['Close'])
            fig.add_trace(go.Scatter(x=bb.index, y=bb['BBU_5_2.0'], line=dict(color='gray'), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=bb.index, y=bb['BBL_5_2.0'], line=dict(color='gray'), name="BB Lower"), row=1, col=1)
            # MACD & RSI
            macd = ta.macd(detail_df['Close'])
            fig.add_trace(go.Scatter(x=macd.index, y=macd['MACD_12_26_9'], name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=macd.index, y=macd['MACDs_12_26_9'], name="Signal"), row=2, col=1)
            fig.add_trace(go.Scatter(x=detail_df.index, y=ta.rsi(detail_df['Close']), name="RSI"), row=3, col=1)
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
            
            score_row = df_results[df_results['ticker'] == selected].iloc[0]
            st.metric("Propensity Score", f"{score_row['score']}")
            if score_row['patterns']:
                st.write("**Detected Chart Patterns**:", ", ".join(score_row['patterns']))

st.sidebar.info("📱 Mobile-friendly PWA • Full US scan with filters = practical & fast\n\nUpdated live March 27, 2026")
