"""
Stock Prediction Dashboard
===========================
Walks through the same journey described in the slide:

  1. Blind / Penny-stock (luck & news)
  2. Fundamental analysis
  3. Technical analysis
  4. Quantitative analysis
  5. Sentiment analysis
  6. Seasonality and 7. Sector analysis
  8. Prediction - IA-LSTM + Bi-GRU model (ILBSMP, Gokaraju & Vara Prasad 2026)
  9. Combined decision (blend everything, weighted by YOUR own psychology)

Run with:
    pip install -r requirements.txt
    streamlit run stock_dashboard.py
"""

import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide", page_icon="📈")

st.title("📈 Stock Prediction Dashboard")
st.caption(
    "Different methods of understanding why stock prices move — "
    "and a way to combine them into one decision."
)

# ----------------------------------------------------------------------------
# Sidebar — inputs shared across every tab
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    exchange = st.selectbox(
        "Exchange",
        ["US (default)", "NSE (India)", "BSE (India)", "Other / already suffixed"],
        index=0,
    )
    raw_ticker = st.text_input(
        "Ticker symbol",
        value="AAPL" if exchange == "US (default)" else "RELIANCE",
    ).strip().upper()

    suffix_map = {"US (default)": "", "NSE (India)": ".NS", "BSE (India)": ".BO", "Other / already suffixed": ""}
    suffix = suffix_map[exchange]
    ticker = raw_ticker if raw_ticker.endswith(suffix) or not suffix else f"{raw_ticker}{suffix}"
    period = st.selectbox(
        "History window", ["6mo", "1y", "2y", "5y"], index=1
    )
    default_benchmark = "^NSEI" if suffix == ".NS" else "^BSESN" if suffix == ".BO" else "^GSPC"
    benchmark = st.text_input("Benchmark / index (for quant tab)", value=default_benchmark).strip().upper()
    st.markdown("---")
    st.caption(
        "This tool is educational, not financial advice. "
        "Markets are risky — combine methods with your own judgement."
    )


def _fetch_history(sym: str, per: str) -> pd.DataFrame:
    """
    Fetch price history from Yahoo with retries and a second method as fallback.
    Raises RuntimeError (with the underlying reason) if nothing comes back, so a failed
    request is NOT stored in Streamlit's cache.
    """
    last_err = None
    for attempt in range(3):
        try:
            df = yf.Ticker(sym).history(period=per)
            if df is not None and not df.empty:
                break
        except Exception as exc:  # rate limit, network, JSON errors ...
            last_err = exc
        try:
            df = yf.download(sym, period=per, progress=False)
            if df is not None and not df.empty:
                break
        except Exception as exc:
            last_err = exc
        df = None
        time.sleep(1.5 * (attempt + 1))

    if df is None or df.empty:
        reason = f" Yahoo said: {last_err}" if last_err else ""
        raise RuntimeError(
            f"Yahoo Finance returned no data for '{sym}'.{reason} If the symbol is right, Yahoo may be "
            f"rate-limiting this server — wait a few minutes and try again."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history(sym: str, per: str) -> pd.DataFrame:
    return _fetch_history(sym, per)


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(sym: str) -> dict:
    try:
        return yf.Ticker(sym).get_info()
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_news(sym: str) -> list:
    try:
        raw = yf.Ticker(sym).news or []
        out = []
        for item in raw:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            if title:
                out.append(title)
        return out[:12]
    except Exception:
        return []


def parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.20) -> pd.Series:
    """Classic Wilder parabolic SAR. Returns a Series aligned to df.index."""
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    sar = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=df.index)

    uptrend = close[1] >= close[0] if n > 1 else True
    af = af_step
    ep = high[0] if uptrend else low[0]
    sar[0] = low[0] if uptrend else high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar[i] = prev_sar + af * (ep - prev_sar)

        if uptrend:
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low[i]
                af = af_step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
        else:
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high[i]
                af = af_step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)

    return pd.Series(sar, index=df.index)


if not ticker:
    st.stop()

try:
    hist = load_price_history(ticker, period)
except Exception as e:
    st.error(
        f"Couldn't load price data for {ticker}. {e}\n\n"
        f"Also check the exchange dropdown matches the symbol (NSE needs .NS, BSE needs .BO)."
    )
    st.stop()

if hist.empty:
    st.error(
        f"No price data found for '{ticker}'. Double-check the symbol, and if it's an "
        f"NSE/BSE stock make sure the exchange dropdown matches (Yahoo Finance needs "
        f"a .NS or .BO suffix for Indian exchanges)."
    )
    st.stop()

info = load_info(ticker)
news_titles = load_news(ticker)
last_price = float(hist["Close"].iloc[-1])
prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_price
pct_change = (last_price - prev_price) / prev_price * 100 if prev_price else 0.0

currency_symbol = "₹" if suffix in (".NS", ".BO") else "$"

st.metric(
    f"{ticker} last close",
    f"{currency_symbol}{last_price:,.2f}",
    f"{pct_change:+.2f}%",
)

tabs = st.tabs(
    [
        "1️⃣ Blind / News",
        "2️⃣ Fundamental",
        "3️⃣ Technical",
        "4️⃣ Quantitative",
        "5️⃣ Sentiment",
        "6️⃣ Seasonality",
        "7️⃣ Sector",
        "8️⃣ Prediction",
        "🎯 Combined Decision",
    ]
)

scores = {}  # method -> score in [-1, 1], filled in as each tab renders

# ----------------------------------------------------------------------------
# 1) Blind / Penny-stock — luck & headlines, no real edge
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Blind / Penny-stock trading")
    st.write(
        "The starting point for most traders: no framework yet, decisions driven mostly "
        "by luck, tips, or whatever headline is trending. There's no real signal here — "
        "this tab exists to make that explicit, not to give you one."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Roll the dice** (illustrates a pure-luck 'signal'):")
        if st.button("🎲 Roll"):
            st.session_state["blind_roll"] = random.uniform(-1, 1)
        roll = st.session_state.get("blind_roll", 0.0)
        st.metric("Random signal", f"{roll:+.2f}")
        scores["Blind/Luck"] = roll

    with col2:
        st.write("**Latest headlines** (read them, don't trade on them blindly):")
        if news_titles:
            for t in news_titles[:6]:
                st.write(f"- {t}")
        else:
            st.info("No recent headlines available from the data source.")

# ----------------------------------------------------------------------------
# 2) Fundamental analysis — company & business quality
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Fundamental analysis")
    st.write("Studying the company and business behind the ticker: valuation, profitability, leverage.")

    def g(key, default=None):
        return info.get(key, default)

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_roce(sym: str):
        """ROCE = EBIT / (Total Assets - Current Liabilities), from latest annual filing."""
        try:
            tk = yf.Ticker(sym)
            income_stmt = tk.financials
            balance_sheet = tk.balance_sheet
            if income_stmt.empty or balance_sheet.empty:
                return None

            ebit = None
            for row_name in ["EBIT", "Ebit", "Operating Income"]:
                if row_name in income_stmt.index:
                    ebit = income_stmt.loc[row_name].iloc[0]
                    break
            if ebit is None or pd.isna(ebit):
                return None

            total_assets = (
                balance_sheet.loc["Total Assets"].iloc[0]
                if "Total Assets" in balance_sheet.index else None
            )
            current_liab = None
            for row_name in ["Current Liabilities", "Total Current Liabilities"]:
                if row_name in balance_sheet.index:
                    current_liab = balance_sheet.loc[row_name].iloc[0]
                    break
            if total_assets is None or current_liab is None or pd.isna(total_assets) or pd.isna(current_liab):
                return None

            capital_employed = total_assets - current_liab
            if capital_employed <= 0:
                return None
            return float(ebit / capital_employed * 100)
        except Exception:
            return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_cfo_vs_net_profit(sym: str):
        """Returns (cfo, net_profit) from the latest annual filing, or (None, None)."""
        try:
            tk = yf.Ticker(sym)
            cashflow = tk.cashflow
            income_stmt = tk.financials
            if cashflow.empty or income_stmt.empty:
                return None, None

            cfo = None
            for row_name in ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]:
                if row_name in cashflow.index:
                    cfo = cashflow.loc[row_name].iloc[0]
                    break

            net_profit = None
            for row_name in ["Net Income", "Net Income Common Stockholders"]:
                if row_name in income_stmt.index:
                    net_profit = income_stmt.loc[row_name].iloc[0]
                    break

            if cfo is None or net_profit is None or pd.isna(cfo) or pd.isna(net_profit):
                return None, None
            return float(cfo), float(net_profit)
        except Exception:
            return None, None

    pe = g("trailingPE")
    peg = g("pegRatio")
    roe = g("returnOnEquity")
    debt_to_equity = g("debtToEquity")
    profit_margin = g("profitMargins")
    rev_growth = g("revenueGrowth")
    roce = compute_roce(ticker)
    cfo, net_profit = compute_cfo_vs_net_profit(ticker)

    sector_lower = str(g("sector", "")).lower()
    industry_lower = str(g("industry", "")).lower()
    is_bank_or_nbfc = any(
        kw in sector_lower or kw in industry_lower
        for kw in ["bank", "nbfc", "financial services", "credit services", "insurance", "capital markets"]
    )
    roe_threshold = 0.10 if is_bank_or_nbfc else 0.12

    c1, c2, c3 = st.columns(3)
    c1.metric("Trailing P/E", f"{pe:.2f}" if pe else "N/A")
    c1.metric("PEG ratio", f"{peg:.2f}" if peg else "N/A")
    c2.metric("Return on equity", f"{roe*100:.1f}%" if roe else "N/A")
    c2.metric("Debt / Equity", f"{debt_to_equity:.1f}" if debt_to_equity else "N/A")
    c3.metric("Profit margin", f"{profit_margin*100:.1f}%" if profit_margin else "N/A")
    c3.metric("Revenue growth (YoY)", f"{rev_growth*100:.1f}%" if rev_growth else "N/A")
    if is_bank_or_nbfc:
        st.caption("Detected as a bank/NBFC/financial company — using a 10% minimum ROE threshold instead of 12%.")

    st.caption(
        f"Sector: {g('sector', 'N/A')} · Industry: {g('industry', 'N/A')} · "
        f"Market cap: {currency_symbol}{g('marketCap', 'N/A'):,}" if g("marketCap") else ""
    )

    st.markdown("---")
    st.markdown("**Buy criteria**")
    st.caption(
        "ROCE is calculated from the latest annual filing where available. "
        "Promoter/pledging % isn't provided by this data source, so enter it "
        "manually (check the shareholding pattern on the exchange/NSE-BSE filing or a site like Screener.in)."
    )

    crit1, crit2 = st.columns(2)
    with crit1:
        if roce is not None:
            st.metric("ROCE (min. 20% to buy)", f"{roce:.1f}%")
            if roce > 20:
                st.success("✅ Passes ROCE criterion")
            else:
                st.error("❌ Fails ROCE criterion")
        else:
            st.metric("ROCE (min. 20% to buy)", "N/A")
            st.warning("⚠️ ROCE couldn't be calculated from available data")

    with crit2:
        pledging_pct = st.number_input(
            "Promoter pledging % (max. 10% allowed)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.5,
        )
        if pledging_pct <= 10:
            st.success("✅ Passes pledging criterion")
        else:
            st.error("❌ Fails pledging criterion")

    crit3, crit4 = st.columns(2)
    with crit3:
        if cfo is not None and net_profit is not None:
            st.metric("CFO vs Net Profit", f"{currency_symbol}{cfo:,.0f} vs {currency_symbol}{net_profit:,.0f}")
            if cfo > net_profit:
                st.success("✅ CFO exceeds net profit — earnings quality looks healthy")
            else:
                st.error("❌ CFO below net profit — check earnings quality")
        else:
            st.metric("CFO vs Net Profit", "N/A")
            st.warning("⚠️ Couldn't be calculated from available data")

    with crit4:
        roe_label = f"ROE (min. {roe_threshold*100:.0f}% for {'banks/NBFC' if is_bank_or_nbfc else 'buying'})"
        if roe is not None:
            st.metric(roe_label, f"{roe*100:.1f}%")
            if roe > roe_threshold:
                st.success("✅ Passes ROE criterion")
            else:
                st.error("❌ Fails ROE criterion")
        else:
            st.metric(roe_label, "N/A")
            st.warning("⚠️ ROE not available")

    st.markdown("---")

    # simple, transparent scoring rubric — each factor contributes -1..+1
    def score_range(value, good_below=None, good_above=None, cap=1.0):
        if value is None:
            return 0.0
        if good_below is not None:
            return cap if value < good_below else -cap * 0.5
        if good_above is not None:
            return cap if value > good_above else -cap * 0.5
        return 0.0

    fund_components = {
        "P/E < 25 is favorable": score_range(pe, good_below=25),
        "PEG < 1.5 is favorable": score_range(peg, good_below=1.5),
        "ROE above threshold is favorable": score_range(roe, good_above=roe_threshold) if roe else 0.0,
        "Debt/Equity < 100 is favorable": score_range(debt_to_equity, good_below=100),
        "Profit margin > 10% is favorable": score_range(profit_margin, good_above=0.10) if profit_margin else 0.0,
        "Revenue growth > 5% is favorable": score_range(rev_growth, good_above=0.05) if rev_growth else 0.0,
        "ROCE > 20% is favorable (buy criterion)": (
            1.0 if roce > 20 else -1.0
        ) if roce is not None else 0.0,
        "Pledging <= 10% is favorable (buy criterion)": 1.0 if pledging_pct <= 10 else -1.0,
        "CFO > Net Profit is favorable (earnings quality)": (
            1.0 if cfo > net_profit else -1.0
        ) if (cfo is not None and net_profit is not None) else 0.0,
    }
    fund_score = float(np.mean(list(fund_components.values())))
    scores["Fundamental"] = fund_score

    with st.expander("How this score is built"):
        for label, val in fund_components.items():
            st.write(f"- {label}: {val:+.2f}")
    st.metric("Fundamental score", f"{fund_score:+.2f}", help="-1 (weak) to +1 (strong), simple rubric")

# ----------------------------------------------------------------------------
# 3) Technical analysis — price patterns & indicators
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Technical analysis")
    st.write("Historical price patterns and indicators: moving averages, RSI, MACD, candlesticks.")

    df = hist.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["SAR"] = parabolic_sar(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(width=1)))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SAR"], name="Parabolic SAR", mode="markers",
        marker=dict(size=4, color="purple", symbol="circle"),
    ))
    fig.update_layout(
        height=450, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10),
        yaxis_title=f"Price ({currency_symbol})",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Parabolic SAR dots below the candles suggest an uptrend (potential stop-loss level); "
        "dots above suggest a downtrend."
    )

    c1, c2 = st.columns(2)
    with c1:
        rsi_fig = go.Figure(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
        rsi_fig.update_layout(height=250, title="RSI (14)", margin=dict(t=30, b=10))
        st.plotly_chart(rsi_fig, use_container_width=True)
    with c2:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
        macd_fig.update_layout(height=250, title="MACD (12,26,9)", margin=dict(t=30, b=10))
        st.plotly_chart(macd_fig, use_container_width=True)

    latest = df.iloc[-1]
    tech_components = {
        "RSI oversold(<30)/overbought(>70)": (
            1.0 if latest["RSI"] < 30 else -1.0 if latest["RSI"] > 70 else 0.0
        ) if not np.isnan(latest["RSI"]) else 0.0,
        "MACD above signal (bullish)": 1.0 if latest["MACD"] > latest["MACD_signal"] else -1.0,
        "Price above SMA50 (uptrend)": 1.0 if latest["Close"] > latest["SMA50"] else -1.0
        if not np.isnan(latest["SMA50"]) else 0.0,
        "Golden/Death cross (SMA50 vs SMA200)": (
            1.0 if latest["SMA50"] > latest["SMA200"] else -1.0
        ) if not np.isnan(latest["SMA200"]) else 0.0,
        "Parabolic SAR (price above dot = bullish)": (
            1.0 if latest["Close"] > latest["SAR"] else -1.0
        ) if not np.isnan(latest["SAR"]) else 0.0,
    }
    tech_score = float(np.mean(list(tech_components.values())))
    scores["Technical"] = tech_score

    with st.expander("How this score is built"):
        for label, val in tech_components.items():
            st.write(f"- {label}: {val:+.2f}")
    st.metric("Technical score", f"{tech_score:+.2f}")

# ----------------------------------------------------------------------------
# 4) Quantitative — correlations, mean reversion, stat arb style signals
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Quantitative analysis")
    st.write("Correlations, mean reversion, and volatility-based signals — the statistical-arbitrage toolkit.")

    returns = hist["Close"].pct_change().dropna()
    zscore = (hist["Close"] - hist["Close"].rolling(20).mean()) / hist["Close"].rolling(20).std()
    latest_z = float(zscore.iloc[-1]) if not zscore.empty else 0.0
    volatility_annualized = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("20-day mean-reversion z-score", f"{latest_z:+.2f}")
    c2.metric("Annualized volatility", f"{volatility_annualized*100:.1f}%")

    corr_value = None
    try:
        bench_hist = load_price_history(benchmark, period)
        if isinstance(bench_hist.columns, pd.MultiIndex):
            bench_hist.columns = bench_hist.columns.get_level_values(0)
        joined = pd.DataFrame({
            ticker: hist["Close"],
            benchmark: bench_hist["Close"],
        }).dropna()
        corr_value = float(joined[ticker].pct_change().corr(joined[benchmark].pct_change()))
        c3.metric(f"Correlation vs {benchmark}", f"{corr_value:+.2f}")
    except Exception:
        c3.metric(f"Correlation vs {benchmark}", "N/A")

    z_fig = go.Figure(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
    z_fig.add_hline(y=2, line_dash="dot", line_color="red")
    z_fig.add_hline(y=-2, line_dash="dot", line_color="green")
    z_fig.update_layout(height=280, title="Price z-score vs 20-day mean", margin=dict(t=30, b=10))
    st.plotly_chart(z_fig, use_container_width=True)

    quant_components = {
        "Mean reversion (z-score extreme suggests pullback)": (
            -1.0 if latest_z > 2 else 1.0 if latest_z < -2 else 0.0
        ),
        "Momentum (20d return sign)": 1.0 if returns.tail(20).sum() > 0 else -1.0,
        "Low volatility favored slightly": 1.0 if volatility_annualized < 0.30 else -0.3,
    }
    quant_score = float(np.mean(list(quant_components.values())))
    scores["Quantitative"] = quant_score

    with st.expander("How this score is built"):
        for label, val in quant_components.items():
            st.write(f"- {label}: {val:+.2f}")
    st.metric("Quantitative score", f"{quant_score:+.2f}")

# ----------------------------------------------------------------------------
# 5) Sentiment — promoters, geopolitics, rates, flows
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Sentiment analysis")
    st.write(
        "Promoter/insider behavior, geopolitics, interest rates, and institutional flows are hard "
        "to fully quantify — headline tone gives a rough signal, and the sliders below let you "
        "encode your own macro read."
    )

    POSITIVE_WORDS = {"beat", "surge", "record", "growth", "upgrade", "rally", "strong", "profit", "gain", "buy"}
    NEGATIVE_WORDS = {"miss", "plunge", "cut", "downgrade", "loss", "fall", "weak", "lawsuit", "probe", "sell", "recall"}

    def headline_sentiment(titles):
        if not titles:
            return 0.0
        total = 0
        for t in titles:
            words = set(t.lower().split())
            total += len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
        return float(np.clip(total / max(len(titles), 1), -1, 1))

    news_score = headline_sentiment(news_titles)
    st.metric("Headline tone score (auto)", f"{news_score:+.2f}")
    if news_titles:
        st.caption("Based on simple keyword scan of the headlines shown in the first tab.")

    st.markdown("**Your own macro read** (institutional flows, rates, geopolitics, promoter activity):")
    macro = st.slider(
        "Bearish ← → Bullish", min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
        help="Use this to encode context the model can't see — rate decisions, insider buying/selling, geopolitical risk, etc."
    )

    sentiment_score = float(np.mean([news_score, macro]))
    scores["Sentiment"] = sentiment_score
    st.metric("Combined sentiment score", f"{sentiment_score:+.2f}")

# ----------------------------------------------------------------------------
# 6) Seasonality — historical monthly / day-of-week patterns
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Seasonality")
    st.write(
        "Some stocks show recurring calendar patterns — certain months or days that have "
        "historically tended to be stronger or weaker. This uses up to 10 years of history "
        "regardless of the sidebar's history window, since seasonality needs several years to mean anything."
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_long_history(sym: str) -> pd.DataFrame:
        return _fetch_history(sym, "10y")

    try:
        long_hist = load_long_history(ticker)
    except Exception:
        long_hist = pd.DataFrame()

    if long_hist.empty or len(long_hist) < 60:
        st.info("Not enough history available to compute seasonality for this ticker.")
        scores["Seasonality"] = 0.0
    else:
        daily_ret = long_hist["Close"].pct_change().dropna()
        years_covered = long_hist.index.year.nunique()
        st.caption(f"Based on {years_covered} year(s) of daily data.")

        # --- Monthly seasonality ---
        month_ret = daily_ret.groupby(daily_ret.index.month).mean() * 100
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_ret = month_ret.reindex(range(1, 13)).fillna(0.0)

        month_fig = go.Figure(go.Bar(
            x=month_names, y=month_ret.values,
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in month_ret.values],
        ))
        month_fig.update_layout(
            height=320, title="Average daily return by month (%)", margin=dict(t=30, b=10)
        )
        st.plotly_chart(month_fig, use_container_width=True)

        # --- Day-of-week seasonality ---
        dow_ret = daily_ret.groupby(daily_ret.index.dayofweek).mean() * 100
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        dow_ret = dow_ret.reindex(range(0, 5)).fillna(0.0)

        dow_fig = go.Figure(go.Bar(
            x=dow_names, y=dow_ret.values,
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in dow_ret.values],
        ))
        dow_fig.update_layout(
            height=280, title="Average daily return by weekday (%)", margin=dict(t=30, b=10)
        )
        st.plotly_chart(dow_fig, use_container_width=True)

        # --- Year x Month heatmap ---
        monthly_table = (
            long_hist["Close"].resample("M").last().pct_change().dropna() * 100
        )
        heat_df = pd.DataFrame({
            "Year": monthly_table.index.year,
            "Month": monthly_table.index.month,
            "Return": monthly_table.values,
        })
        pivot = heat_df.pivot(index="Year", columns="Month", values="Return")
        pivot = pivot.reindex(columns=range(1, 13))
        heat_fig = go.Figure(go.Heatmap(
            z=pivot.values, x=month_names, y=pivot.index.astype(str),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="%"),
        ))
        heat_fig.update_layout(height=320, title="Monthly return heatmap by year", margin=dict(t=30, b=10))
        st.plotly_chart(heat_fig, use_container_width=True)

        current_month = datetime.now().month
        current_month_avg = month_ret.get(current_month, 0.0)
        seasonality_score = float(np.clip(current_month_avg / 0.15, -1, 1))  # ~0.15%/day is a strong signal
        scores["Seasonality"] = seasonality_score

        with st.expander("How this score is built"):
            st.write(
                f"- Average daily return historically in {month_names[current_month-1]} "
                f"({years_covered} years): {current_month_avg:+.3f}%/day"
            )
        st.metric(f"Seasonality score ({month_names[current_month-1]})", f"{seasonality_score:+.2f}")

# ----------------------------------------------------------------------------
# 7) Sector — relative strength vs sector/peers
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Sector analysis")
    st.write(
        "How this stock is doing relative to its own sector — a stock can look strong or weak "
        "purely because its whole sector is in or out of favor."
    )

    sector_name = info.get("sector", "N/A")
    industry_name = info.get("industry", "N/A")
    st.caption(f"Sector: **{sector_name}** · Industry: **{industry_name}**")

    us_sector_etfs = {
        "Technology": "XLK", "Financial Services": "XLF", "Financials": "XLF",
        "Healthcare": "XLV", "Energy": "XLE", "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP", "Industrials": "XLI", "Utilities": "XLU",
        "Real Estate": "XLRE", "Basic Materials": "XLB", "Communication Services": "XLC",
    }
    nse_sector_indices = {
        "Technology": "^CNXIT", "Financial Services": "^NSEBANK", "Financials": "^NSEBANK",
        "Healthcare": "^CNXPHARMA", "Energy": "^CNXENERGY", "Consumer Cyclical": "^CNXFMCG",
        "Consumer Defensive": "^CNXFMCG", "Industrials": "^CNXINFRA", "Basic Materials": "^CNXMETAL",
        "Real Estate": "^CNXREALTY", "Communication Services": "^CNXMEDIA",
    }
    guess_map = nse_sector_indices if suffix in (".NS", ".BO") else us_sector_etfs
    default_sector_bm = guess_map.get(sector_name, benchmark)

    sector_benchmark = st.text_input(
        "Sector benchmark (ETF or index)", value=default_sector_bm,
        help="Auto-guessed from the stock's sector; edit if it doesn't fit.",
    ).strip().upper()

    peers_raw = st.text_input(
        "Peer tickers to compare (comma-separated, optional)",
        value="",
        help="e.g. MSFT,GOOGL,META  or  TCS.NS,INFY.NS",
    )
    peer_tickers = [p.strip().upper() for p in peers_raw.split(",") if p.strip()]

    compare_syms = [ticker, sector_benchmark] + peer_tickers
    norm_fig = go.Figure()
    sector_rel_return = None
    for sym in compare_syms:
        try:
            h = load_price_history(sym, period)
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            if h.empty:
                continue
            normalized = h["Close"] / h["Close"].iloc[0] * 100
            norm_fig.add_trace(go.Scatter(x=normalized.index, y=normalized, name=sym))
            if sym == ticker:
                stock_ret = normalized.iloc[-1] - 100
            if sym == sector_benchmark:
                sector_rel_return = normalized.iloc[-1] - 100
        except Exception:
            continue

    norm_fig.update_layout(
        height=380, title=f"Normalized performance (start = 100), {period}",
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(norm_fig, use_container_width=True)

    if sector_rel_return is not None:
        relative_strength = stock_ret - sector_rel_return
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{ticker} return", f"{stock_ret:+.1f}%")
        c2.metric(f"{sector_benchmark} return", f"{sector_rel_return:+.1f}%")
        c3.metric("Relative strength", f"{relative_strength:+.1f} pts")
        sector_score = float(np.clip(relative_strength / 15, -1, 1))  # ±15pts treated as strong outperformance
    else:
        st.info("Couldn't load the sector benchmark — try editing the ticker above.")
        sector_score = 0.0

    scores["Sector"] = sector_score
    with st.expander("How this score is built"):
        st.write("Relative strength = stock's period return minus sector benchmark's period return, scaled to ±1.")
    st.metric("Sector score", f"{sector_score:+.2f}")

# ----------------------------------------------------------------------------
# 8) Prediction — IA-LSTM + Bi-GRU with improved score-level fusion (ILBSMP)
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Prediction — IA-LSTM + Bi-GRU (ILBSMP)")
    st.write(
        "Implements the pipeline from Gokaraju & Vara Prasad (2026): min-max normalization → technical "
        "indicators (EMA, SMA, DEMA, WWS, MACD, RSI, DMI) → improved chi-square feature selection with "
        "Shannon entropy → IA-LSTM and Bi-GRU → tanh / Lehmer-mean score-level fusion."
    )

    try:
        import ilbsmp
        ilbsmp_error = None
    except Exception as exc:  # keep the rest of the dashboard alive
        ilbsmp = None
        ilbsmp_error = exc

    if ilbsmp is None:
        st.error(
            "Couldn't load `ilbsmp.py`. It must sit next to `stock_dashboard.py` "
            f"(same folder / same GitHub repo). Details: {ilbsmp_error}"
        )
    elif not ilbsmp.tensorflow_available():
        st.warning(
            "TensorFlow isn't installed in this environment, so the models can't be trained here. "
            "Locally: `pip install tensorflow-cpu==2.16.2` (Python 3.9–3.12). On Streamlit Cloud: choose "
            "Python 3.11 under *Advanced settings* when deploying, keep `tensorflow-cpu` in requirements.txt, "
            "then reboot the app."
        )
    else:
        r1 = st.columns(3)
        pred_period = r1[0].selectbox("Training history", ["2y", "5y", "10y"], index=1, key="pred_period")
        train_pct = r1[1].select_slider("Training split (%)", options=[60, 70, 80, 90], value=80, key="pred_split")
        lookback = r1[2].slider("Look-back window (days)", 1, 30, 10, key="pred_lookback")

        r2 = st.columns(3)
        n_keep = r2[0].slider("Features to keep (of 12)", 3, 12, 8, key="pred_nfeat")
        epochs = r2[1].slider("Epochs", 5, 100, 50, key="pred_epochs")
        target_label = r2[2].selectbox(
            "Prediction target",
            ["Next-day close (as in paper)", "Next-day return (more robust to trends)"],
            key="pred_target",
        )

        r3 = st.columns(2)
        conf_label = r3[0].selectbox(
            "Fusion confidence interval [α, β]", ["[0.6, 0.8]", "[0.6, 0.7]", "[0.5, 0.6]"], key="pred_conf"
        )
        lehmer_p = r3[1].slider("Lehmer mean power p", 1.0, 4.0, 2.0, 0.5, key="pred_p")
        conf_lo, conf_hi = [float(v) for v in conf_label.strip("[]").split(",")]

        run_pred = st.button("🚀 Train & predict", type="primary", key="pred_run")
        st.caption("Training runs on CPU and typically takes a minute or two; results are kept until you train again.")

        if run_pred:
            try:
                pred_df = load_price_history(ticker, pred_period)
                bar = st.progress(0.0, text="Preparing data…")

                def _progress(frac, msg):
                    bar.progress(float(min(max(frac, 0.0), 1.0)), text=msg)

                result = ilbsmp.run_ilbsmp(
                    pred_df,
                    train_pct=int(train_pct),
                    lookback=int(lookback),
                    n_features=int(n_keep),
                    epochs=int(epochs),
                    target_mode="return" if "return" in target_label else "price",
                    conf_lo=conf_lo,
                    conf_hi=conf_hi,
                    lehmer_p=float(lehmer_p),
                    progress_cb=_progress,
                )
                bar.empty()
                result["ticker"] = ticker
                st.session_state["ilbsmp_result"] = result
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        res = st.session_state.get("ilbsmp_result")
        if res and res.get("ticker") == ticker:
            st.success(
                f"Trained on {res['n_train']} days, tested on {res['n_test']} days "
                f"({res['n_rows']} usable rows). Target: "
                f"{'next-day return' if res['target_mode'] == 'return' else 'next-day close'}."
            )

            # --- next-day forecast -------------------------------------------------
            st.markdown(f"**Next-session forecast** (after the {res['last_date']:%d %b %Y} close)")
            fc_cols = st.columns(3)
            for col, name in zip(fc_cols, ["IA-LSTM", "Bi-GRU", "Fused (ILBSMP)"]):
                price = res["next_pred"][name]
                chg = (price / res["last_close"] - 1) * 100
                col.metric(name, f"{currency_symbol}{price:,.2f}", f"{chg:+.2f}%")

            # --- test-set metrics --------------------------------------------------
            st.markdown("**Test-set error metrics** (price units, held-out period)")
            st.dataframe(res["metrics"].round(4), use_container_width=True)
            fused_rmse = res["metrics"].loc["Fused (ILBSMP)", "RMSE"]
            naive_rmse = res["metrics"].loc["Naive (yesterday's close)", "RMSE"]
            if fused_rmse < naive_rmse:
                st.success(
                    f"The fused model's RMSE ({fused_rmse:.3f}) is below the naive 'tomorrow = today' "
                    f"baseline ({naive_rmse:.3f}) on this test period."
                )
            else:
                st.warning(
                    f"The fused model's RMSE ({fused_rmse:.3f}) does not beat the naive 'tomorrow = today' "
                    f"baseline ({naive_rmse:.3f}) on this test period — treat its forecast as weak evidence. "
                    "Next-day prices are very hard to beat with price history alone."
                )

            # --- actual vs predicted -----------------------------------------------
            pf = go.Figure()
            pf.add_trace(go.Scatter(x=res["test_dates"], y=res["y_true"], name="Actual", line=dict(color="black", width=2)))
            pf.add_trace(go.Scatter(x=res["test_dates"], y=res["pred_ia"], name="IA-LSTM", line=dict(width=1)))
            pf.add_trace(go.Scatter(x=res["test_dates"], y=res["pred_bigru"], name="Bi-GRU", line=dict(width=1)))
            pf.add_trace(go.Scatter(x=res["test_dates"], y=res["pred_fused"], name="Fused (ILBSMP)", line=dict(width=2, color="crimson")))
            pf.update_layout(
                height=400, margin=dict(t=30, b=10), title="Test period: actual vs predicted close",
                yaxis_title=f"Price ({currency_symbol})",
            )
            st.plotly_chart(pf, use_container_width=True)

            # --- details -----------------------------------------------------------
            with st.expander("Feature selection (improved chi-square × Shannon entropy)"):
                ft = res["feature_table"].copy()
                st.caption(
                    "Computed on training rows only. Score = chi-square(feature, binned target) × normalised "
                    "entropy of the feature; the top-ranked features are fed to the models."
                )
                bar_fig = go.Figure(go.Bar(
                    x=ft["score"], y=ft["feature"], orientation="h",
                    marker_color=["#2ca02c" if s else "#b0b0b0" for s in ft["selected"]],
                ))
                bar_fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(bar_fig, use_container_width=True)
                st.dataframe(ft.round(3), use_container_width=True, hide_index=True)

            with st.expander("Score-level fusion details"):
                fus = pd.DataFrame({
                    "Model": list(res["confidence"].keys()),
                    "Validation MSE (scaled)": [res["val_mse_scaled"][k] for k in res["confidence"]],
                    "Confidence": [res["confidence"][k] for k in res["confidence"]],
                    "Fusion weight W": [res["weights"][k] for k in res["confidence"]],
                })
                st.dataframe(fus.round(5), use_container_width=True, hide_index=True)
                st.caption(
                    "Confidence is mapped from each model's error on a validation slice it never trained on "
                    "(lowest error → upper end of the interval). Predictions are tanh-normalised, then combined "
                    "with a Lehmer mean, which weights higher-scoring predictions more."
                )

            with st.expander("Training curves"):
                for name, h in res["history"].items():
                    lc = go.Figure()
                    lc.add_trace(go.Scatter(y=h.get("loss", []), name="train loss"))
                    lc.add_trace(go.Scatter(y=h.get("val_loss", []), name="validation loss"))
                    lc.update_layout(height=250, title=name, margin=dict(t=30, b=10))
                    st.plotly_chart(lc, use_container_width=True)

            # --- score for the combined decision ----------------------------------
            expected_move = res["next_pred"]["Fused (ILBSMP)"] / res["last_close"] - 1
            prediction_score = float(np.clip(expected_move / 0.02, -1, 1))  # ±2% expected move = full signal
            scores["Prediction"] = prediction_score
            st.metric(
                "Prediction score (feeds Combined Decision)", f"{prediction_score:+.2f}",
                help="Fused next-day expected move scaled so ±2% maps to ±1.",
            )
        elif res:
            st.info(f"The stored result is for {res['ticker']}. Click **Train & predict** to train for {ticker}.")
        else:
            st.info("Choose your settings and click **Train & predict**.")

        st.caption(
            "Educational use only — not financial advice. The paper evaluates a single stock (Edwards Lifesciences, "
            "2000–2017); its reported errors should not be expected to carry over to other stocks or to live trading. "
            "Here, feature selection and scaling use training data only, which is stricter than normalising the whole dataset up front."
        )

# ----------------------------------------------------------------------------
# 9) Combined decision — weighted blend, tuned to the user's own psychology
# ----------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Combined decision")
    st.write(
        "No single method is complete on its own. Set weights that match your own trading style "
        "and time horizon — the blend below is only as good as the weights you choose."
    )

    default_weights = {
        "Blind/Luck": 0.0,
        "Fundamental": 0.2,
        "Technical": 0.2,
        "Quantitative": 0.2,
        "Sentiment": 0.15,
        "Seasonality": 0.1,
        "Sector": 0.15,
        "Prediction": 0.1,
    }

    weights = {}
    weight_items = list(default_weights.items())
    for start in range(0, len(weight_items), 4):
        weight_cols = st.columns(4)
        for col, (name, default) in zip(weight_cols, weight_items[start:start + 4]):
            with col:
                weights[name] = st.slider(name, 0.0, 1.0, default, 0.05)

    if "Prediction" not in scores:
        st.caption("The Prediction method is left out of the blend until you train the model in the Prediction tab.")

    # only methods that actually produced a score take part in the blend
    active_weights = {k: w for k, w in weights.items() if k in scores}
    total_weight = sum(active_weights.values()) or 1.0
    weighted_sum = sum(active_weights[k] * scores[k] for k in active_weights) / total_weight

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=weighted_sum,
        number={"valueformat": "+.2f"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [-1, -0.33], "color": "#f4a3a3"},
                {"range": [-0.33, 0.33], "color": "#f0f0a0"},
                {"range": [0.33, 1], "color": "#a3d9a5"},
            ],
        },
        title={"text": f"{ticker} weighted signal"},
    ))
    gauge.update_layout(height=320, margin=dict(t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    if weighted_sum > 0.33:
        verdict = "🟢 Leans BULLISH"
    elif weighted_sum < -0.33:
        verdict = "🔴 Leans BEARISH"
    else:
        verdict = "🟡 NEUTRAL / mixed signals"
    st.markdown(f"### {verdict}")

    breakdown = pd.DataFrame({
        "Method": list(scores.keys()),
        "Score": [scores[k] for k in scores],
        "Weight": [weights.get(k, 0.0) for k in scores],
    })
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.info(
        "This is a transparent, adjustable rubric — not a prediction engine. "
        "Treat it as a way to organize your own analysis across all these lenses, "
        "then decide what fits your own risk tolerance and investment cycle."
    )
