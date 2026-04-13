import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ========================== FULL GREEKS (including Vega) ==========================
def black_scholes_greeks(S, K, T, r, sigma, q=0.0, option_type='put'):
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan, np.nan, np.nan, np.nan, np.nan  # delta, gamma, theta, vega, color
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    # Theta for put
    theta = - (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
            + q * K * np.exp(-r * T) * norm.cdf(-d2) \
            - r * S * np.exp(-q * T) * norm.cdf(-d1)
    
    # Vega (same for call and put)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    color = gamma * ((d1 * d2 - 1) * (r - q + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T)) / (sigma**2 * T**2)
    
    return delta, gamma, theta, vega, color


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
        puts['theta'] = greeks[2]
        puts['vega'] = greeks[3]
        puts['color'] = greeks[4]
        all_puts.append(puts)
    
    df = pd.concat(all_puts, ignore_index=True)
    df = df.dropna(subset=['gamma', 'delta', 'theta', 'vega', 'premium'])
    return df, current_price


# ========================== CHART FUNCTIONS ==========================
def create_ratio_chart(df, etf_name, atm_premium, ratio_name, y_column_func, y_title):
    """Generic function to create ratio charts with consistent hover"""
    etf_df = df[df['etf'] == etf_name].copy()
    unique_dtes = sorted(etf_df['expiry_dte'].unique())
    color_values = np.linspace(0, 1, len(unique_dtes))
    colors = px.colors.sample_colorscale("Viridis", color_values)
    color_map = dict(zip(unique_dtes, colors))
    
    fig = go.Figure()
    for dte in unique_dtes:
        sub = etf_df[etf_df['expiry_dte'] == dte].sort_values('premium')
        if sub.empty:
            continue
        sub[y_column_func] = y_column_func(sub)   # Apply the lambda
        
        fig.add_trace(go.Scatter(
            x=sub['premium'],
            y=sub[y_column_func],
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
        title=f"{etf_name} — {ratio_name} vs Option Premium",
        xaxis_title="Option Premium ($)",
        yaxis_title=y_title,
        legend_title="Expiration (DTE)",
        height=650
    )
    return fig


# ========================== STREAMLIT APP ==========================
st.set_page_config(page_title="Options Convexity Tool", layout="wide")
st.title("📈 Options Convexity & Volatility Analysis Tool")
st.markdown("**Live multi-ratio analysis for GLD, SPY, and FEZ**")

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

# ATM Reference
atm_row = filtered.loc[(filtered['moneyness'] - 1).abs().idxmin()]
atm_premium = atm_row['premium']
st.info(f"**ATM Reference** — Premium ≈ **${atm_premium:.3f}** (Strike ${atm_row['strike']:.2f})")

# ====================== CHARTS ======================
st.subheader("Chart 1 — Gamma / Premium (Convexity per Dollar)")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma / Premium",
        lambda x: x['gamma'] / x['premium'],
        "Gamma / Premium (Higher = Better Convexity per $)"
    ), 
    use_container_width=True
)

st.subheader("Chart 2 — Gamma × |Delta| / |Theta| vs Premium")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma × |Δ| / |Θ|",
        lambda x: (x['gamma'] * np.abs(x['delta'])) / np.abs(x['theta']),
        "Gamma × |Delta| / |Theta|"
    ), 
    use_container_width=True
)

st.subheader("Chart 3 — Gamma × |Delta| / (Premium × |Theta|) vs Premium")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma × |Δ| / (Prem × |Θ|)",
        lambda x: (x['gamma'] * np.abs(x['delta'])) / (x['premium'] * np.abs(x['theta'])),
        "Gamma × |Delta| / (Premium × |Theta|)"
    ), 
    use_container_width=True
)

st.subheader("Chart 4 — Gamma × |Delta| × Vega / Premium vs Premium")
st.plotly_chart(
    create_ratio_chart(
        filtered, selected_etf, atm_premium,
        "Gamma × |Δ| × Vega / Prem",
        lambda x: (x['gamma'] * np.abs(x['delta']) * x['vega']) / x['premium'],
        "Gamma × |Delta| × Vega / Premium"
    ), 
    use_container_width=True
)

st.caption("Data source: Yahoo Finance • Hover shows Strike, Premium, DTE, and metric value")
