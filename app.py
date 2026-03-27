import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
import numpy as np

st.set_page_config(page_title="TA Directional Screener", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Full US Market Directional Propensity Screener")
st.markdown("**Improved v2** • Hybrid TA + Fundamentals • Score breakdown • Patterns + Elliott Wave • Scans stocks + ETFs")

# ====================== TICKERS ======================
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
    return sorted(list(set(all_tickers)))

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'roe': info.get('returnOnEquity', 0) or 0,
            'debt_ratio': info.get('debtToEquity', 999) or 999,
            'profit_margin': info.get('profitMargins', 0) or 0
        }
    except:
        return {'roe': 0, 'debt_ratio': 999, 'profit_margin': 0}

# ====================== CORE FUNCTIONS ======================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        return ticker, df
    except:
        return ticker, None

def detect_chart_patterns(df):
    close = df['Close'].values
    peaks, _ = signal.find_peaks(close, distance=5, prominence=0.01)
    troughs, _ = signal.find_peaks(-close, distance=5, prominence=0.01)
    patterns = []
    # Double / Triple
    if len(peaks) >= 2 and abs(close[peaks[-1]] - close[peaks[-2]]) / close[peaks[-2]] < 0.03:
        patterns.append("Double Top (Bearish)")
    if len(peaks) >= 3 and abs(close[peaks[-1]] - close[peaks[-3]]) / close[peaks[-3]] < 0.03:
        patterns.append("Triple Top (Bearish)")
    if len(troughs) >= 2 and abs(close[troughs[-1]] - close[troughs[-2]]) / close[troughs[-2]] < 0.03:
        patterns.append("Double Bottom (Bullish)")
    # Head & Shoulders + Triangles (simplified)
    if len(peaks) >= 3 and close[peaks[-2]] > close[peaks[-1]] and close[peaks[-2]] > close[peaks[-3]]:
        patterns.append("Head & Shoulders (Bearish)")
    return patterns

def elliott_wave_bias(df):
    close = df['Close']
    high60 = close[-60:].max()
    low60 = close[-60:].min()
    retrace = (high60 - close.iloc[-1]) / (high60 - low60) if high60 != low60 else 0
    bias = 0
    if 0.382 < retrace < 0.618:
        bias += 18  # Wave 2/4 retracement → bullish continuation
    elif retrace > 0.786:
        bias -= 12
    if close.iloc[-1] > close.iloc[-20]:
        bias += 12
    return bias

def compute_score(ticker, df, use_fundamentals=True):
    if df is None or len(df) < 50:
        return None
    df = df.copy()

    # Technical indicators
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

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Component scores (for explainer)
    breakdown = {}
    score = 0.0

    # Trend (35%)
    trend_score = 25 if latest['Close'] > latest.get('SMA200', 0) else -25
    trend_score += 15 if latest.get('SMA50', 0) > latest.get('SMA200', 0) else -15
    trend_score += 10 if latest['Close'] > latest.get('SMA50', 0) else -10
    score += trend_score
    breakdown['Trend'] = trend_score

    # Momentum (25%)
    mom_score = 15 if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', 0) else -15
    mom_score += 10 if latest.get('MACDh_12_26_9', 0) > prev.get('MACDh_12_26_9', 0) else -10
    score += mom_score
    breakdown['Momentum'] = mom_score

    # RSI (10%)
    rsi_score = 15 if latest['RSI'] < 30 else (-15 if latest['RSI'] > 70 else 5)
    score += rsi_score
    breakdown['RSI'] = rsi_score

    # ADX + DMI (10%)
    adx_score = 15 if latest.get('ADX_14', 0) > 25 and latest.get('DMP_14', 0) > latest.get('DMN_14', 0) else (-15 if latest.get('ADX_14', 0) > 25 else 0)
    score += adx_score
    breakdown['ADX+DMI'] = adx_score

    # Bollinger + Stoch + Candles (10%)
    bb_score = 8 if latest['Close'] < latest.get('BBL_5_2.0', 0) else (-8 if latest['Close'] > latest.get('BBU_5_2.0', 0) else 0)
    stoch_score = 5 if latest.get('STOCHk_14_3_3', 0) < 20 else (-5 if latest.get('STOCHk_14_3_3', 0) > 80 else 0)
    score += bb_score + stoch_score
    breakdown['BB+Stoch'] = bb_score + stoch_score

    # Patterns + Elliott (10%)
    patterns = detect_chart_patterns(df)
    pattern_score = 12 if any("Bullish" in p for p in patterns) else (-12 if any("Bearish" in p for p in patterns) else 0)
    elliott_score = elliott_wave_bias(df)
    score += pattern_score + elliott_score
    breakdown['Patterns+Elliott'] = pattern_score + elliott_score

    # Fundamentals (25% only if enabled)
    if use_fundamentals:
        fund = get_fundamentals(ticker)
        fund_score = 12 if fund['roe'] > 0.15 else -8
        fund_score += 10 if fund['debt_ratio'] < 100 else -10
        fund_score += 8 if fund['profit_margin'] > 0.15 else -8
        score += fund_score
        breakdown['Fundamentals'] = fund_score
    else:
        breakdown['Fundamentals'] = 0

    score = max(min(score, 100), -100)
    return {
        'ticker': ticker,
        'score': round(score, 1),
        'close': round(latest['Close'], 2),
        'volume': int(latest['Volume']),
        'rsi': round(latest['RSI'], 1),
        'adx': round(latest.get('ADX_14', 0), 1),
        'patterns': patterns,
        'breakdown': breakdown
    }

# ====================== UI ======================
tab1, tab2, tab3 = st.tabs(["🚀 Scan", "📊 Results", "🔍 Detail"])

with tab1:
    universe = st.selectbox("Universe", ["S&P 500", "NASDAQ-100", "Full US Market (stocks + ETFs)"], index=2)
    use_fundamentals = st.sidebar.checkbox("Enable Hybrid Fundamentals (ROE + Debt + Margin)", value=True)
    min_volume = st.sidebar.slider("Min Avg Daily Volume", 0, 50000000, 100000)
    min_price = st.sidebar.slider("Min Price $", 0.0, 500.0, 2.0)
    max_tickers = st.sidebar.slider("Max tickers to scan", 500, 15000, 3000)

    if st.button("🚀 Run Full Scan", type="primary"):
        with st.spinner("Scanning..."):
            progress_bar = st.progress(0)
            if universe == "S&P 500":
                tickers = get_sp500_tickers()[:max_tickers]
            elif universe == "NASDAQ-100":
                tickers = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]['Ticker'].tolist()[:max_tickers]
            else:
                tickers = get_all_us_tickers()[:max_tickers]

            data_list = Parallel(n_jobs=8)(delayed(fetch_data)(t) for t in tickers)
            
            results = []
            for i, (ticker, df) in enumerate(data_list):
                if df is None or len(df) < 50: 
                    progress_bar.progress((i+1)/len(tickers))
                    continue
                score_dict = compute_score(ticker, df, use_fundamentals)
                if score_dict and score_dict['volume'] >= min_volume and score_dict['close'] >= min_price:
                    results.append(score_dict)
                progress_bar.progress((i+1)/len(tickers))

            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
            
            st.success(f"✅ Scanned {len(df_results)} liquid tickers!")
            st.session_state.df_results = df_results  # save for other tabs

with tab2:
    if 'df_results' in st.session_state:
        df_results = st.session_state.df_results
        st.dataframe(
            df_results.style.background_gradient(cmap='RdYlGn', subset=['score'])
            .format({'score': '{:.1f}', 'close': '${:.2f}'}),
            use_container_width=True,
            hide_index=True
        )
        if st.button("📤 Export Results to CSV"):
            st.download_button("Download CSV", df_results.to_csv(index=False), "screener_results.csv", "text/csv")
        
        st.subheader("🔥 Top 10 Bullish")
        top10 = df_results.head(10)
        fig = go.Figure(go.Bar(x=top10['ticker'], y=top10['score'], marker_color='#00ff88'))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if 'df_results' in st.session_state:
        df_results = st.session_state.df_results
        selected = st.selectbox("Select ticker for deep dive", df_results['ticker'])
        if selected:
            row = df_results[df_results['ticker'] == selected].iloc[0]
            detail_df = yf.download(selected, period="6mo")
            
            # Rich chart
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(f"{selected} Price + Overlays", "MACD + RSI", "Volume"))
            fig.add_trace(go.Candlestick(x=detail_df.index, open=detail_df['Open'], high=detail_df['High'],
                                         low=detail_df['Low'], close=detail_df['Close'], name="Price"), row=1, col=1)
            bb = ta.bbands(detail_df['Close'])
            fig.add_trace(go.Scatter(x=bb.index, y=bb['BBU_5_2.0'], line=dict(color='gray'), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=bb.index, y=bb['BBL_5_2.0'], line=dict(color='gray'), name="BB Lower"), row=1, col=1)
            macd = ta.macd(detail_df['Close'])
            fig.add_trace(go.Scatter(x=macd.index, y=macd['MACD_12_26_9'], name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=macd.index, y=macd['MACDs_12_26_9'], name="Signal"), row=2, col=1)
            fig.add_trace(go.Scatter(x=detail_df.index, y=ta.rsi(detail_df['Close']), name="RSI"), row=3, col=1)
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Overall Propensity Score", f"{row['score']}")
            st.write("**Score Breakdown**")
            for k, v in row['breakdown'].items():
                st.write(f"• {k}: **{v:+.0f}**")
            if row['patterns']:
                st.write("**Detected Patterns**:", ", ".join(row['patterns']))

st.sidebar.info("📱 Fully mobile-optimized PWA • Hybrid mode gives higher-conviction signals\n\nv2 pushed live March 27, 2026")
