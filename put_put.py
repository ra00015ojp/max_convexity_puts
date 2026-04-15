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
            d, g, th, v = black_scholes_greeks(
                current_price, row['strike'], row['T'], r, row['impliedVolatility'], q, 'put')
            return pd.Series([d, g, th, v])

        greeks = puts.apply(compute, axis=1)
        puts[['delta', 'gamma', 'theta', 'vega']] = greeks
        all_puts.append(puts)

    df = pd.concat(all_puts, ignore_index=True)
    df = df.dropna(subset=['gamma', 'delta', 'theta', 'vega', 'premium'])
    return df, current_price


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
    """Visualizes Volume or Open Interest across premiums and DTE"""
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
        yaxis_title=metric.capitalize(),
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
