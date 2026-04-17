import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========================== RETRY & SESSION CONFIG ==========================
def create_session_with_retries(retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    """Create a requests session with automatic retry and connection pooling"""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        status_forcelist=status_forcelist,
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=backoff_factor
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def retry_with_backoff(max_retries=3, base_wait=1):
    """Decorator for functions that may hit rate limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = base_wait
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                        if attempt < max_retries - 1:
                            st.warning(f"⏳ Rate limited. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            wait_time *= 2  # Exponential backoff
                        else:
                            st.error(f"❌ Rate limit exceeded after {max_retries} attempts. Please try again in a few minutes.")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator


# Session for connection pooling
@st.cache_resource
def get_session():
    """Cached session for connection reuse"""
    return create_session_with_retries(retries=3, backoff_factor=1.0)


# ========================== GREEKS CALCULATION ==========================
def black_scholes_greeks(S, K, T, r, sigma, q=0.0, option_type='put'):
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan, np.nan, np.nan, np.nan, np.nan  # delta, gamma, theta, vega
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    theta = - (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
            + q * K * np.exp(-r * T) * norm.cdf(-d2) \
            - r * S * np.exp(-q * T) * norm.cdf(-d1)
    
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    return delta, gamma, theta, vega


@retry_with_backoff(max_retries=3, base_wait=2)
@st.cache_data(ttl=7200, show_spinner=False)
def download_option_chain(ticker: str, r=0.042):
    session = get_session()
    etf = yf.Ticker(ticker, session=session)
    try:
        current_price = etf.history(period="1d", session=session)['Close'].iloc[-1]
    except:
        current_price = etf.info.get('regularMarketPrice') or etf.info.get('previousClose', None)
    
    q = etf.info.get('dividendYield', 0.0) or 0.0
    expirations = etf.options
    
    all_puts = []
    today = datetime.now().date()
    for exp_str in expirations:
        time.sleep(0.2)  # Small delay between expiration requests
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        T = dte / 365.25
        if T <= 0:
            continue
            
        chain = etf.option_chain(exp_str)
        puts = chain.puts.copy().reset_index()
        puts['expiry_dte'] = dte
        puts['T'] = T
        puts['type'] = 'put'
        puts['underlying_price'] = current_price
        puts['premium'] = (puts['bid'] + puts['ask']) / 2
        puts['premium'] = puts['premium'].fillna(puts['lastPrice'])
        
        def compute(row):
            d, g, th, v = black_scholes_greeks(
                current_price, row['strike'], row['T'], r, row['impliedVolatility'], q, 'put')
            return pd.Series([d, g, th, v])
        
        greeks = puts.apply(compute, axis=1)
        puts[['delta', 'gamma', 'theta', 'vega']] = greeks
        all_puts.append(puts)
    
    df = pd.concat(all_puts, ignore_index=True)
    df = df.dropna(subset=['gamma', 'delta', 'theta', 'vega', 'premium'])
    return df, current_price


# ========================== PRICE & VIX HISTORY ==========================
@retry_with_backoff(max_retries=3, base_wait=1)
@st.cache_data(ttl=7200, show_spinner=False)
def download_price_history(ticker: str, period: str = "1mo"):
    """Download 1-month price history for underlying asset"""
    session = get_session()
    etf = yf.Ticker(ticker, session=session)
    hist = etf.history(period=period, session=session)
    hist = hist[['Close']].reset_index()
    hist.columns = ['Date', 'Close']
    return hist


@retry_with_backoff(max_retries=3, base_wait=1)
@st.cache_data(ttl=7200, show_spinner=False)
def download_vix_history(period: str = "1mo"):
    """Download VIX history"""
    session = get_session()
    vix = yf.Ticker('^VIX', session=session)
    hist = vix.history(period=period, session=session)
    hist = hist[['Close']].reset_index()
    hist.columns = ['Date', 'VIX']
    return hist


def create_price_vix_chart(ticker: str, current_price: float):
    """Create dual-axis chart: Underlying Price (left) and VIX (right)"""
    price_hist = download_price_history(ticker, "1mo")
    vix_hist = download_vix_history("1mo")
    
    # Merge on date
    merged = pd.merge(price_hist, vix_hist, on='Date', how='outer').sort_values('Date')
    merged = merged.ffill().bfill()  # Forward/back fill to handle gaps
    
    # Calculate 1-month change
    price_1m_ago = price_hist.iloc[0]['Close']
    price_change_pct = ((current_price - price_1m_ago) / price_1m_ago) * 100
    
    fig = go.Figure()
    
    # Price trace (left axis)
    fig.add_trace(go.Scatter(
        x=merged['Date'],
        y=merged['Close'],
        name=f'{ticker} Price',
        line=dict(color='#1f77b4', width=2.5),
        yaxis='y1',
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # VIX trace (right axis)
    fig.add_trace(go.Scatter(
        x=merged['Date'],
        y=merged['VIX'],
        name='VIX',
        line=dict(color='#d62728', width=2, dash='dot'),
        yaxis='y2',
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>VIX:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Layout with dual y-axes
    fig.update_layout(
        title=f"{ticker} — 1-Month Price & VIX Overlay (Change: {price_change_pct:+.2f}%)",
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=f'{ticker} Price ($)',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
        ),
        yaxis2=dict(
            title='VIX',
            titlefont=dict(color='#d62728'),
            tickfont=dict(color='#d62728'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        height=450
    )
    
    return fig


# ========================== GENERIC CHART FUNCTION ==========================
def create_ratio_chart(df, etf_name, atm_premium, title, ratio_name, y_title, ratio_func):
    """Reusable chart creator"""
    etf_df = df[df['etf'] == etf_name].copy()
    unique_dtes = sorted(etf_df['expiry_dte'].unique())
    color_values = np.linspace(0, 1, len(unique_dtes))
    colors = px.colors.sample_colorscale("Viridis", color_values)
    color_map = dict(zip(unique_dtes, colors))
    
    fig = go.Figure()
    for dte in unique_dtes:
        sub = etf_df[etf_df['expiry_dte'] == dte].sort_values('premium').copy()
        if sub.empty:
            continue
            
        # Calculate the ratio safely
        sub['ratio_value'] = ratio_func(sub)
        
        fig.add_trace(go.Scatter(
            x=sub['premium'],
            y=sub['ratio_value'],
            mode='lines+markers',
            line=dict(color=color_map[dte]),
            marker=dict(size=4),
            name=f'DTE {dte}',
            hovertemplate=
                "<b>Strike:</b> %{customdata[0]:.2f}<br>" +
                "<b>Premium:</b> $%{x:.3f}<br>" +
                "<b>DTE:</b> %{customdata[1]}<br>" +
                f"<b>{ratio_name}:</b> %{{y:.6f}}<extra></extra>",
            customdata=sub[['strike', 'expiry_dte']].values
        ))
    
    fig.add_vline(x=atm_premium, line_dash="dash", line_color="red",
                  annotation_text=f"ATM ≈ ${atm_premium:.3f}", annotation_position="top right")
    
    fig.update_layout(
        title=f"{etf_name} — {title}",
        xaxis_title="Option Premium ($)",
        yaxis_title=y_title,
        legend_title="Expiration (DTE)",
        height=650
    )
    return fig

# ========================== LIQUIDITY ANALYSIS ==========================

def create_liquidity_chart(df, etf_name, atm_premium, metric='openInterest'):
    """Visualizes Volume or Open Interest with a Logarithmic Y-axis"""
    etf_df = df[df['etf'] == etf_name].copy()
    unique_dtes = sorted(etf_df['expiry_dte'].unique())
    color_map = dict(zip(unique_dtes, px.colors.sample_colorscale("Viridis", np.linspace(0, 1, len(unique_dtes)))))

    fig = go.Figure()
    for dte in unique_dtes:
        sub = etf_df[etf_df['expiry_dte'] == dte].sort_values('premium')
        
        # Note: We use the raw value for the bar height, 
        # Plotly's 'log' type handles the transformation visually.
        fig.add_trace(go.Bar(
            x=sub['premium'],
            y=sub[metric], 
            name=f'DTE {dte}',
            marker_color=color_map[dte],
            customdata=sub[['strike', 'volume', 'openInterest']].values,
            hovertemplate=
                "<b>Strike:</b> %{customdata[0]:.2f}<br>" +
                "<b>Premium:</b> $%{x:.3f}<br>" +
                "<b>Volume:</b> %{customdata[1]}<br>" +
                "<b>Open Interest:</b> %{customdata[2]}<extra></extra>"
        ))

    fig.update_layout(
        title=f"{etf_name} — {metric.capitalize()} Distribution (Log Scale)",
        xaxis_title="Option Premium ($)",
        yaxis_title=f"{metric.capitalize()} (Log)",
        yaxis_type="log",  # <--- This triggers the logarithmic scaling
        barmode='stack',
        height=500
    )
    return fig

# ========================== STREAMLIT APP ==========================
st.set_page_config(page_title="Options Convexity Tool", layout="wide")
st.title("📈 Options Convexity & Volatility Analysis Tool")
st.markdown("**Live analysis for GLD, SPY, and FEZ**")

tickers = ['GLD', 'SPY', 'FEZ']
selected_etf = st.sidebar.radio("**Select ETF**", tickers, horizontal=True)

with st.spinner(f"Downloading options chain for **{selected_etf}**..."):
    raw_df, current_price = download_option_chain(selected_etf)
    df = raw_df.copy()
    df['etf'] = selected_etf
    df['underlying_price'] = current_price
    df['moneyness'] = df['strike'] / current_price

    filtered = df[
        (df['type'] == 'put') &
        (df['expiry_dte'].between(19, 190)) &
        (df['premium'].between(0.001, 4.999)) &
        (df['moneyness'].between(0.80, 1.20))
    ].copy()

st.success(f"✅ Loaded {len(filtered):,} puts for **{selected_etf}** (Spot: ${current_price:,.2f})")

# ====================== PRICE & VIX SECTION ======================
st.subheader("📊 1-Month Price & Volatility Context")
with st.spinner(f"Loading price history and VIX data..."):
    price_vix_fig = create_price_vix_chart(selected_etf, current_price)
    st.plotly_chart(price_vix_fig, use_container_width=True)

# ATM Reference
atm_row = filtered.loc[(filtered['moneyness'] - 1).abs().idxmin()]
atm_premium = atm_row['premium']
st.info(f"**ATM Reference** — Premium ≈ **${atm_premium:.3f}** (Strike ${atm_row['strike']:.2f})")

# ====================== CHARTS ======================
st.subheader("💧 Liquidity Check — Volume & Open Interest")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Daily Volume** (Immediate Liquidity)")
    st.plotly_chart(create_liquidity_chart(filtered, selected_etf, atm_premium, 'volume'), use_container_width=True)

with col2:
    st.markdown("**Open Interest** (Market Depth)")
    st.plotly_chart(create_liquidity_chart(filtered, selected_etf, atm_premium, 'openInterest'), use_container_width=True)

# Advanced: Liquidity Filter Warning
median_oi = filtered['openInterest'].median()
low_liquidity_threshold = 50 # Example threshold

st.warning(f"Note: The median Open Interest for these puts is **{median_oi:.0f}**. Strikes with < {low_liquidity_threshold} OI may suffer from wide Bid-Ask spreads.")

st.subheader("Convexity per Dollar — Gamma / Premium CAP EFFICIENT: cheap lottery tickets with high explosive potential.")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma / Premium (Convexity per Dollar)",
        "Gamma / Premium",
        "Gamma / Premium",
        lambda x: x['gamma'] / x['premium']
    ), use_container_width=True
)

st.subheader("Convexity over decay — Gamma × |Delta| / |Theta|  STAY&TRADE: options that move fast but dont bleed out too quickly.")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma × |Delta| / |Theta|",
        "Gamma × |Δ| / |Θ|",
        "Gamma × |Delta| / |Theta|",
        lambda x: (x['gamma'] * np.abs(x['delta'])) / np.abs(x['theta'])
    ), use_container_width=True
)

st.subheader("Convexity over decay, premium, IV — Gamma × |Delta| / (Premium × |Theta| × Vega) ANTIFRAGILE: Undervalued convexity that isnt overly dependent on Volatility (Vega) staying high.")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma × |Delta| / (Premium × |Theta|)",
        "Gamma × |Δ| / (Prem × |Θ|)",
        "Gamma × |Delta| / (Premium × |Theta|)",
        lambda x: (x['gamma'] * np.abs(x['delta'])) / (x['premium'] * np.abs(x['theta'] * x['vega']))
    ), use_container_width=True
)

st.subheader("Volatility Play — Gamma × |Delta| × Vega / Premium LONG VOL BIAS: Expecting a massive crash where IV will explode.")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma × |Delta| × Vega / Premium",
        "Gamma × |Δ| × Vega / Prem",
        "Gamma × |Delta| × Vega / Premium",
        lambda x: (x['gamma'] * np.abs(x['delta']) * x['vega']) / x['premium']
    ), use_container_width=True
)

st.caption("Data source: Yahoo Finance • Hover shows Strike, Premium, DTE and metric")
