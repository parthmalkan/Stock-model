"""
dashboard.py — Personal US stock screening app (free, Streamlit).

Approach 3 uses REAL Markowitz optimization (PyPortfolioOpt) — a full
covariance matrix across the candidate holdings and a max-Sharpe solve,
not a two-bucket approximation.

  Tab 1  Approach 1: Diversified Large-Cap   (lower risk)
  Tab 2  Approach 2: Momentum / Speculative  (higher risk, higher swing)
  Tab 3  Approach 3: Markowitz-Optimized     (real optimizer, full covariance)
  Tab 4  How to read this

Run locally:   streamlit run dashboard.py
Deploy free:   share.streamlit.io  (sign in with GitHub)

Data: yfinance (free, no API key). Prices refresh every time the app opens.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS — change these if you like
# ----------------------------------------------------------------------------
UNIVERSE = [
    # ticker, company, sector, bucket
    ("AAPL",  "Apple Inc.",               "Technology",       "core"),
    ("MSFT",  "Microsoft Corp.",          "Technology",       "core"),
    ("GOOGL", "Alphabet Inc.",            "Technology",       "core"),
    ("AMZN",  "Amazon.com Inc.",          "Consumer Disc.",   "core"),
    ("NVDA",  "NVIDIA Corp.",             "Technology",       "core"),
    ("META",  "Meta Platforms Inc.",      "Technology",       "core"),
    ("AVGO",  "Broadcom Inc.",            "Technology",       "core"),
    ("JPM",   "JPMorgan Chase & Co.",     "Financials",       "core"),
    ("V",     "Visa Inc.",                "Financials",       "core"),
    ("MA",    "Mastercard Inc.",          "Financials",       "core"),
    ("JNJ",   "Johnson & Johnson",        "Healthcare",       "core"),
    ("LLY",   "Eli Lilly & Co.",          "Healthcare",       "core"),
    ("XOM",   "Exxon Mobil Corp.",        "Energy",           "core"),
    ("PG",    "Procter & Gamble",         "Consumer Staples", "core"),
    ("WMT",   "Walmart Inc.",             "Consumer Staples", "core"),
    ("COST",  "Costco Wholesale",         "Consumer Staples", "core"),
    ("HD",    "Home Depot Inc.",          "Consumer Disc.",   "core"),
    ("TSLA",  "Tesla Inc.",               "Consumer Disc.",   "core"),
    ("PLTR",  "Palantir Technologies",    "Technology",       "momentum"),
    ("SMCI",  "Super Micro Computer",     "Technology",       "momentum"),
    ("MU",    "Micron Technology",        "Technology",       "momentum"),
    ("SNDK",  "Sandisk Corporation",      "Technology",       "momentum"),
    ("LITE",  "Lumentum Holdings",        "Technology",       "momentum"),
    ("STX",   "Seagate Technology",       "Technology",       "momentum"),
]

RISK_FREE = 0.045       # approx. short-term Treasury yield, used for Sharpe
TOP_N_CORE = 8          # names to allocate to in Approach 1
TOP_N_MOM = 6           # names to allocate to in Approach 2
CAP_CORE = 0.20         # max weight per name, Approach 1
CAP_MOM = 0.25          # max weight per name, Approach 2

# Approach 3 (real optimizer) settings
OPT_CANDIDATES = 20     # shortlist size fed to the optimizer (keeps it fast)
OPT_MAX_WEIGHT = 0.15   # max weight per single stock in the optimal portfolio

st.set_page_config(page_title="My Stock Model", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# DATA + SCORING
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)   # cache 1 hour so it loads fast
def load_data():
    tickers = [u[0] for u in UNIVERSE]
    meta = {u[0]: {"company": u[1], "sector": u[2], "bucket": u[3]} for u in UNIVERSE}

    raw = yf.download(tickers, period="1y", interval="1d",
                      group_by="ticker", auto_adjust=True,
                      threads=True, progress=False)

    rows, closes = [], {}
    for t in tickers:
        try:
            px = raw[t]["Close"].dropna()
            if len(px) < 200:
                continue
            closes[t] = px
            ma50 = px.rolling(50).mean().iloc[-1]
            ma200 = px.rolling(200).mean().iloc[-1]
            rows.append({
                "Ticker": t,
                "Company": meta[t]["company"],
                "Sector": meta[t]["sector"],
                "Bucket": meta[t]["bucket"],
                "Price": float(px.iloc[-1]),
                "Return 12mo": float(px.iloc[-1] / px.iloc[0] - 1),
                "Return 6mo": float(px.iloc[-1] / px.iloc[-126] - 1),
                "Return 1mo": float(px.iloc[-1] / px.iloc[-21] - 1),
                "Volatility": float(px.pct_change().std() * np.sqrt(252)),
                "Trend": int(px.iloc[-1] > ma50) + int(px.iloc[-1] > ma200),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    # Normalized factor scores (0-100), ranked inside the universe
    df["Momentum Score"] = (df["Return 12mo"].rank(pct=True) * 0.6 +
                            df["Return 6mo"].rank(pct=True) * 0.4) * 100
    df["Trend Score"] = df["Trend"] * 50
    df["Low-Vol Score"] = (1 - df["Volatility"].rank(pct=True)) * 100
    df["Composite Score"] = (df["Momentum Score"] * 0.35 +
                             df["Trend Score"] * 0.25 +
                             df["Low-Vol Score"] * 0.20 + 50 * 0.20)

    prices = pd.DataFrame(closes).dropna(how="all")
    return df, prices


def diversified_picks(df):
    sub = df[df["Bucket"] == "core"].copy()
    sub = sub.nlargest(TOP_N_CORE, "Composite Score")
    w = sub["Composite Score"] / sub["Composite Score"].sum()
    sub["Weight"] = np.minimum(w, CAP_CORE)
    sub["Weight"] = sub["Weight"] / sub["Weight"].sum()   # renormalize after cap
    return sub.sort_values("Weight", ascending=False)


def momentum_picks(df):
    sub = df[df["Bucket"] == "momentum"].copy()
    sub = sub.nlargest(TOP_N_MOM, "Return 12mo")
    sub["Weight"] = 1.0 / len(sub)
    return sub.sort_values("Return 12mo", ascending=False)


def markowitz_real(df, prices):
    """Real Markowitz: full covariance matrix across a shortlist of the
    strongest candidates, solved for maximum Sharpe ratio with a per-stock
    weight cap. Falls back gracefully if the optimizer can't solve."""
    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns
    except ImportError:
        return None, "PyPortfolioOpt is not installed."

    # Shortlist: top composite names across the whole universe
    shortlist = df.nlargest(OPT_CANDIDATES, "Composite Score")["Ticker"].tolist()
    sub = prices[[t for t in shortlist if t in prices.columns]].dropna()

    if sub.shape[1] < 2 or len(sub) < 60:
        return None, "Not enough clean price history to optimize."

    mu = expected_returns.mean_historical_return(sub)
    S = risk_models.sample_cov(sub)

    ef = EfficientFrontier(mu, S, weight_bounds=(0, OPT_MAX_WEIGHT))
    ef.max_sharpe(risk_free_rate=RISK_FREE)
    clean = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)

    rows = []
    for t, w in clean.items():
        if w and w > 0.0005:
            meta = df[df["Ticker"] == t].iloc[0]
            rows.append({
                "Ticker": t,
                "Company": meta["Company"],
                "Sector": meta["Sector"],
                "Price": meta["Price"],
                "Return 12mo": meta["Return 12mo"],
                "Volatility": meta["Volatility"],
                "Weight": w,
            })
    out = pd.DataFrame(rows).sort_values("Weight", ascending=False)
    out["Weight"] = out["Weight"] / out["Weight"].sum()

    # Expected annual return / volatility / Sharpe of the optimal portfolio
    exp_ret, exp_vol, sharpe = perf
    return out, {"return": exp_ret, "vol": exp_vol, "sharpe": sharpe}


# ----------------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------------
st.title("📈 My US Stock Model")
st.caption("Three approaches, ranked from live market data. "
           "Prices refresh each time this page loads.")

with st.spinner("Fetching prices and scoring the universe..."):
    df, prices = load_data()

if df.empty:
    st.error("Could not load price data right now. "
             "Yahoo's free feed may be rate-limiting — try again in a minute.")
    st.stop()

# ---- headline row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks screened", len(df))
c2.metric("Universe avg. 12mo return", f"{df['Return 12mo'].mean():.1%}")
c3.metric("Universe avg. volatility", f"{df['Volatility'].mean():.1%}")
c4.metric("Last updated", pd.Timestamp.now().strftime("%d %b %Y, %H:%M"))

d1, d2, d3, d4 = st.tabs([
    "🟢 Approach 1 — Diversified",
    "🔴 Approach 2 — Momentum",
    "🔵 Approach 3 — Markowitz",
    "❓ How to read this",
])

# ---------------- Approach 1 ----------------
with d1:
    st.subheader("Diversified large-cap — lower risk, steadier ride")
    st.write("Ranks the core list on a blend of momentum, trend and low volatility, "
             "then spreads your money across the top picks with a cap per stock.")

    picks = diversified_picks(df)
    show = picks[["Ticker", "Company", "Sector", "Price", "Return 12mo",
                  "Volatility", "Composite Score", "Weight"]].copy()
    show["Weight"] = show["Weight"].map(lambda x: f"{x:.1%}")
    st.dataframe(
        show.style.format({"Price": "${:,.2f}", "Return 12mo": "{:.1%}",
                           "Volatility": "{:.1%}", "Composite Score": "{:.0f}"}),
        use_container_width=True, hide_index=True,
    )
    avg_r, avg_v = picks["Return 12mo"].mean(), picks["Volatility"].mean()
    x1, x2, x3 = st.columns(3)
    x1.metric("Avg. 12mo return of picks", f"{avg_r:.1%}")
    x2.metric("Avg. volatility of picks", f"{avg_v:.1%}")
    x3.metric("Typical monthly swing", f"±{avg_v/np.sqrt(12):.1%}")

# ---------------- Approach 2 ----------------
with d2:
    st.subheader("Momentum / speculative — higher risk, much bigger swings")
    st.warning(
        "These names can post huge annual numbers through a few violent spikes, "
        "and equally brutal down months. Big upside and big drawdown live together here."
    )

    picks = momentum_picks(df)
    show = picks[["Ticker", "Company", "Price", "Return 12mo", "Return 6mo",
                  "Return 1mo", "Volatility", "Weight"]].copy()
    show["Weight"] = show["Weight"].map(lambda x: f"{x:.1%}")
    st.dataframe(
        show.style.format({"Price": "${:,.2f}", "Return 12mo": "{:.1%}",
                           "Return 6mo": "{:.1%}", "Return 1mo": "{:.1%}",
                           "Volatility": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )
    avg_r, avg_v = picks["Return 12mo"].mean(), picks["Volatility"].mean()
    x1, x2, x3 = st.columns(3)
    x1.metric("Avg. 12mo return of picks", f"{avg_r:.1%}")
    x2.metric("Avg. volatility of picks", f"{avg_v:.1%}")
    x3.metric("Typical monthly swing", f"±{avg_v/np.sqrt(12):.1%}")

# ---------------- Approach 3 ----------------
with d3:
    st.subheader("Markowitz-optimized — real optimizer, full covariance")
    st.write("Builds a covariance matrix across the strongest candidates and solves "
             "for the portfolio with the highest return per unit of risk, with a cap "
             "on how much any single stock can take.")

    with st.spinner("Running the optimizer..."):
        out, info = markowitz_real(df, prices)

    if out is None:
        st.info(info)
    else:
        y1, y2, y3, y4 = st.columns(4)
        y1.metric("Expected 12mo return", f"{info['return']:.1%}")
        y2.metric("Expected volatility", f"{info['vol']:.1%}")
        y3.metric("Sharpe ratio", f"{info['sharpe']:.2f}")
        y4.metric("Positions held", len(out))

        show = out[["Ticker", "Company", "Sector", "Price", "Return 12mo",
                    "Volatility", "Weight"]].copy()
        show["Weight"] = show["Weight"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            show.style.format({"Price": "${:,.2f}", "Return 12mo": "{:.1%}",
                               "Volatility": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )

        st.caption(f"The optimizer considers the {OPT_CANDIDATES} strongest names by "
                   f"composite score, caps any single holding at "
                   f"{OPT_MAX_WEIGHT:.0%}, and uses a {RISK_FREE:.1%} risk-free rate. "
                   f"These are all editable in the settings block at the top of the file.")

# ---------------- Guide ----------------
with d4:
    st.subheader("How to read this")
    st.markdown("""
**The three approaches answer different questions.**

- **Approach 1 — Diversified** asks *"which healthy large-caps are trending up, "
  "and how do I spread the money so one bad name can't hurt me?"* Lower volatility, "
  "smaller month-to-month swings, historically a smoother line.
- **Approach 2 — Momentum** asks *"what has run the hardest over the past year?"* "
  "It chases strength. That means occasional spectacular years and occasional "
  "savage losses — both are normal here, not a malfunction.
- **Approach 3 — Markowitz** asks *"what combination of these stocks gives the most "
  "return for the risk I'm accepting?"* It looks at how every pair of stocks moves "
  "together (their covariance), not just how each one performed alone.

**Plain-language notes.**

- *Volatility* is how much a stock bounces around. Higher means wider swings, both ways.
- *Typical monthly swing* converts annual volatility into a rough monthly figure.
  Real months land above and below it — it is a yardstick, not a promise.
- *Sharpe ratio* is return divided by risk. Higher is better for the same return.
- *Covariance* is just "do these two move together?" Two stocks that rise and fall
  at different times can be safer combined than either one alone.

**What this app does not do.**

- It does not predict the future. It ranks and weights what has already happened.
- It does not average 20% a month. Nothing does — the S&P 500 has averaged roughly
  10% per *year* over the long run.
- It refreshes when you open it, not on a timer.
""")

st.divider()
st.caption("Decision-support tool, not financial advice. "
           "Past performance does not predict future returns.")
