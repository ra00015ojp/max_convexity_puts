import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ========================== GREEKS CALCULATION ==========================
def black_scholes_greeks(S, K, T, r, sigma, q=0.0, option_type='put'):
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan, np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    color = gamma * ((d1 * d2 - 1) * (r - q + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T)) / (sigma**2 * T**2)
    return delta, gamma, color


@st.cache_data(ttl=3600, show_spinner=False)
def download_option_chain(ticker: str, r=0.042):
    etf = yf.Ticker(ticker)
    try:
        current_price = etf.history(period="1d")['Close'].iloc[-1]
    except:
        current_price = etf.info.get('regularMarketPrice') or etf.info.get('previousClose', None)
    
    q = etf.info.get('dividendYield', 0.0) or 0.0
    expirations = etf.options
    
    all_puts = []
    today = datetime.now().date()
    for exp_str in expirations:
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
            return pd.Series(black_scholes_greeks(
                current_price, row['strike'], row['T'], r, row['impliedVolatility'], q, 'put'))
        
        greeks = puts.apply(compute, axis=1)
        puts['delta'] = greeks[0]
        puts['gamma'] = greeks[1]
        puts['color'] = greeks[2]
        all_puts.append(puts)
    
    df = pd.concat(all_puts, ignore_index=True)
    df = df.dropna(subset=['gamma', 'delta', 'premium', 'impliedVolatility', 'color'])
    return df, current_price


# ========================== PLOT FUNCTIONS (one per ETF) ==========================
def plot_gamma_vs_premium(df, etf_name, atm_premium):
    etf_df = df[df['etf'] == etf_name]
    unique_dtes = sorted(etf_df['expiry_dte'].unique())
    color_values = np.linspace(0, 1, len(unique_dtes))
    colors = px.colors.sample_colorscale("Viridis", color_values)
    color_map = dict(zip(unique_dtes, colors))
    
    fig = go.Figure()
    for dte in unique_dtes:
        sub = etf_df[etf_df['expiry_dte'] == dte].sort_values('premium')
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['premium'], y=sub['gamma'],
            mode='lines+markers', line=dict(color=color_map[dte]),
            name=f'DTE {dte}',
            hovertemplate="<b>Strike:</b> %{customdata[0]:.2f}<br><b>Premium:</b> $%{x:.3f}<br><b>Gamma:</b> %{y:.6f}<extra></extra>",
            customdata=sub[['strike', 'expiry_dte']].values
        ))
    
    fig.add_vline(x=atm_premium, line_dash="dash", line_color="red",
                  annotation_text=f"ATM Premium ≈ ${atm_premium:.3f}", annotation_position="top right")
    
    fig.update_layout(
        title=f"{etf_name} — Set 1: Gamma vs Option Premium",
        xaxis_title="Option Premium ($)",
        yaxis_title="Raw Gamma (higher = stronger convexity)",
        legend_title="Expiration (DTE)",
        height=650
    )
    return fig


def plot_vol_surface(df, etf_name):
    etf_df = df[df['etf'] == etf_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=etf_df['moneyness'], y=etf_df['expiry_dte'], z=etf_df['impliedVolatility'],
        mode='markers',
        marker=dict(size=3, color=etf_df['impliedVolatility'], colorscale='RdYlBu_r', colorbar=dict(title="IV")),
        hovertemplate="<b>Strike:</b> %{customdata[0]:.2f}<br><b>Moneyness:</b> %{x:.3f}<br><b>DTE:</b> %{y}<br><b>IV:</b> %{z:.1%}<extra></extra>",
        customdata=etf_df[['strike']].values
    ))
    fig.add_trace(go.Scatter3d(
        x=[1.0, 1.0], y=[etf_df['expiry_dte'].min(), etf_df['expiry_dte'].max()],
        z=[etf_df['impliedVolatility'].min(), etf_df['impliedVolatility'].max()],
        mode='lines', line=dict(color='red', width=6, dash='dash'), name='ATM'
    ))
    fig.update_layout(
        title=f"{etf_name} — Set 2: 3D Volatility Surface Heatmap",
        scene=dict(xaxis_title='Moneyness (Strike/Spot)', yaxis_title='DTE', zaxis_title='Implied Volatility'),
        height=700
    )
    return fig


def plot_vol_smirk(df, etf_name):
    etf_df = df[df['etf'] == etf_name]
    unique_dtes = sorted(etf_df['expiry_dte'].unique())
    color_values = np.linspace(0, 1, len(unique_dtes))
    colors = px.colors.sample_colorscale("Viridis", color_values)
    color_map = dict(zip(unique_dtes, colors))
    
    fig = go.Figure()
    for dte in unique_dtes:
        sub = etf_df[etf_df['expiry_dte'] == dte].sort_values('moneyness')
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['moneyness'], y=sub['impliedVolatility'],
            mode='lines+markers', line=dict(color=color_map[dte]),
            name=f'DTE {dte}',
            hovertemplate="<b>Strike:</b> %{customdata[0]:.2f}<br><b>Moneyness:</b> %{x:.3f}<br><b>IV:</b> %{y:.1%}<extra></extra>",
            customdata=sub[['strike', 'expiry_dte']].values
        ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="ATM", annotation_position="top right")
    
    fig.update_layout(
        title=f"{etf_name} — Set 3: Volatility Smirk for Puts",
        xaxis_title="Moneyness (Strike / Spot) — lower = deeper OTM puts",
        yaxis_title="Implied Volatility",
        legend_title="Expiration (DTE)",
        height=650
    )
    return fig


# ========================== STREAMLIT APP ==========================
st.set_page_config(page_title="Options Convexity Tool", layout="wide")
st.title("📈 Options Convexity & Volatility Analysis")
st.markdown("**Live analysis of Gamma, Volatility Smirk, and 3D Volatility Surface** for GLD, SPY, and FEZ.")

tickers = ['GLD', 'SPY', 'FEZ']
selected_etf = st.sidebar.radio("Select ETF to analyse", tickers, horizontal=True)

st.sidebar.caption("Data refreshed automatically every hour.")

# Download data for the selected ETF only
with st.spinner(f"Downloading latest options chain for **{selected_etf}**..."):
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

st.success(f"✅ Loaded {len(filtered):,} put options for **{selected_etf}** (current price ${current_price:,.2f})")

# Find ATM premium for reference
atm_row = filtered.loc[(filtered['moneyness'] - 1).abs().idxmin()]
atm_premium = atm_row['premium']
st.info(f"**ATM reference** — Premium ≈ **${atm_premium:.3f}** (strike ${atm_row['strike']:.2f})")

# ====================== SET 1 ======================
st.subheader("Set 1 — Gamma vs Option Premium")
st.markdown("""
Raw **Gamma** measures convexity.  
Higher Gamma = faster acceleration of delta when the underlying moves → stronger convexity.  
The red dashed line marks the **ATM** premium.
""")
st.plotly_chart(plot_gamma_vs_premium(filtered, selected_etf, atm_premium), use_container_width=True)

# ====================== SET 2 ======================
st.subheader("Set 2 — 3D Volatility Surface Heatmap")
st.markdown("""
**Color = Implied Volatility** (red/hot = high IV).  
For puts you will often see **red at low strikes + short maturity** ("crashophobia").  
The red dashed line marks the **ATM** plane.
""")
st.plotly_chart(plot_vol_surface(filtered, selected_etf), use_container_width=True)

# ====================== SET 3 ======================
st.subheader("Set 3 — Volatility Smirk")
st.markdown("""
Puts show a **volatility smirk** (not a symmetric smile): lower strikes (OTM puts) have higher IV.  
This reflects strong market demand for **downside protection**.
The red dashed line is **ATM**.
""")
st.plotly_chart(plot_vol_smirk(filtered, selected_etf), use_container_width=True)

st.caption("Data source: Yahoo Finance • Built with Streamlit • Refresh page to update data")
