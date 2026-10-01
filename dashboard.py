"""
dashboard.py — Personal US stock screening app (free, Streamlit).

Approach 3 uses REAL Markowitz optimization (PyPortfolioOpt) — a full
covariance matrix across the candidate holdings and a max-Sharpe solve,
not a two-bucket approximation.

  Tab 1  Approach 1: Diversified Large-Cap   (lower risk)
  Tab 2  Approach 2: Momentum / Speculative  (higher risk, higher swing)
  Tab 3  Approach 3: Markowitz-Optimized     (real optimizer, full covariance)
  Tab 4  Market Timing & News Sentiment      (rules-based signal, not a prediction)
  Tab 5  How to read this

Run locally:   streamlit run dashboard.py
Deploy free:   share.streamlit.io  (sign in with GitHub)

Data: yfinance (free, no API key) for prices/VIX; Google News RSS (free,
no API key) for headlines. Prices refresh every time the app opens.

IMPORTANT — what Tab 4 actually is:
It is a RULES-BASED SIGNAL, not a prediction. Market timing = VIX level +
S&P trend, scored against plain thresholds anyone can read. News sentiment =
counting positive/negative keywords in recent free headlines per stock. Both
are real and computed live, but neither forecasts price. No free data source
can reliably do that, and this app does not pretend otherwise.
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings
import urllib.request
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS — change these if you like
# ----------------------------------------------------------------------------
# SCREEN_MODE controls the universe:
#   "watchlist" -> the curated list below (~24 names). Loads in seconds.
#   "broad"     -> the full S&P 1500 (S&P 500 + 400 + 600) pulled free from
#                  Wikipedia, then pre-filtered so only tradable names get priced.
#                  Takes 1-3 minutes on first load; caches for an hour after.
#                  If it times out or errors, switch back to "watchlist".
SCREEN_MODE = "broad"

# VOLUME-SURGE FILTER (Option A):
#   Narrows the broad universe to the TOP_VOLUME_N names by how heavily
#   they are trading versus their own average, measured over the last
#   VOLUME_WINDOW_DAYS. A proxy for "what the market is focused on now".
#   IMPORTANT: this is TRADING VOLUME, not net buying. Volume counts buys
#   and sells alike, so it includes heavy selling and shorting. It is NOT
#   the same as "most purchased" -- use the custom list for that.
USE_VOLUME_SURGE = True
TOP_VOLUME_N = 250          # keep the top N by volume ratio
VOLUME_WINDOW_DAYS = 3      # recent window (trading days)
VOLUME_BASELINE_DAYS = 60   # baseline to compare against

# Times are shown in US Eastern (ET, America/New_York) so the clock matches
# the US market session. Python's zoneinfo ships with 3.9+ and needs no
# extra package; the fixed-offset fallback covers the rare missing tzdata.
try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    ET_TZ = None


def et_now():
    """Current time in US Eastern, falling back gracefully if tzdata is absent."""
    if ET_TZ is not None:
        return pd.Timestamp.now(tz=ET_TZ)
    return pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")


def et_stamp(fmt="%d %b %Y, %H:%M"):
    """Formatted ET timestamp, with the zone label so it is never ambiguous."""
    try:
        return et_now().strftime(fmt) + " ET"
    except Exception:
        return pd.Timestamp.now().strftime(fmt)


def market_session_note():
    """Plain-language note on whether the US market is open (regular hours)."""
    try:
        now = et_now()
    except Exception:
        return ""
    if now.weekday() >= 5:
        return "Weekend — US market closed; prices are the last close."
    mins = now.hour * 60 + now.minute
    if mins < 9 * 60 + 30:
        return "Pre-market — regular session opens 9:30am ET."
    if mins <= 16 * 60:
        return "US market open (regular session, 9:30am-4:00pm ET)."
    return "After hours — regular session closed at 4:00pm ET."

MIN_PRICE = 5.0            # skip penny stocks / near-zero names
MAX_BROAD_NAMES = 1500     # hard cap so the free host isn't overwhelmed

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
# UNIVERSE BUILDERS
# ----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)   # cache the name list a full day
def broad_universe():
    """Full S&P 1500 (500 + 400 + 600) pulled free from Wikipedia.
    No API key. Returns list of (ticker, company, sector, bucket).
    Bucket is 'core' for large-caps (S&P 500), 'momentum' for the rest --
    the smaller/mid names are where the bigger swings live."""
    sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "core"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "momentum"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "momentum"),
    ]
    out, seen, report = [], set(), []
    for url, bucket in sources:
        label = url.rsplit("/", 1)[-1]
        try:
            tables = pd.read_html(url)
            df = tables[0]
            # Column names vary slightly between the three pages, so find them
            tcol = next((c for c in df.columns if str(c).strip() in ("Symbol", "Ticker")), None)
            ccol = next((c for c in df.columns if str(c).strip() in
                         ("Security", "Company", "Company Name")), None)
            scol = next((c for c in df.columns if "Sector" in str(c)), None)
            if tcol is None or ccol is None:
                report.append(f"{label}: columns not found")
                continue
            got = 0
            for _, row in df.iterrows():
                tkr = str(row[tcol]).strip().replace(".", "-")
                if not tkr or tkr == "nan" or tkr in seen:
                    continue
                seen.add(tkr)
                out.append((tkr, str(row[ccol])[:60],
                            str(row[scol])[:30] if scol else "\u2014", bucket))
                got += 1
            report.append(f"{label}: {got} names")
        except Exception as e:
            report.append(f"{label}: FAILED - {e}")
        if len(out) >= MAX_BROAD_NAMES:
            break
    # Visible in the Streamlit logs so a failure is never silent again
    print("[universe] " + " | ".join(report) + f" | TOTAL {len(out)}")
    return out[:MAX_BROAD_NAMES]

# ----------------------------------------------------------------------------
# VOLUME-SURGE FILTER  (Option A -- a proxy, see the note at the top)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def volume_surge_tickers(tickers, top_n, window, baseline):
    """Rank tickers by recent traded volume vs their own average.

    Reads VOLUME ONLY -- it cannot tell a heavy buyer from a heavy seller,
    so treat it as "drawing attention", never as "being bought"."""
    ratios = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            raw = yf.download(batch, period="6mo", interval="1d",
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            for t in batch:
                try:
                    vol = raw[t]["Volume"].dropna()
                    if len(vol) < baseline:
                        continue
                    recent = vol.iloc[-window:].mean()
                    base = vol.iloc[-baseline:].mean()
                    if base and base > 0:
                        ratios[t] = float(recent / base)
                except Exception:
                    continue
        except Exception:
            continue
    if not ratios:
        return [], pd.DataFrame()
    s = pd.Series(ratios).sort_values(ascending=False)
    kept = s.head(top_n).index.tolist()
    print(f"[volume] scored {len(s)} names; kept top {len(kept)}")
    return kept, s.head(top_n).rename("Volume Ratio").reset_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_custom_tickers(raw_text):
    """Screen an explicit list of tickers (Option B -- e.g. a "most purchased"
    list typed in by you). Bad tickers are dropped, never faked."""
    parts = re.split(r"[,\s;]+", raw_text or "")
    wanted = [p.strip().upper().replace(".", "-") for p in parts if p.strip()]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return []
    out = []
    for i in range(0, len(wanted), 100):
        batch = wanted[i:i + 100]
        try:
            raw = yf.download(batch, period="1y", interval="1d",
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            for t in batch:
                try:
                    px = raw[t]["Close"].dropna()
                    if len(px) < 200 or float(px.iloc[-1]) < MIN_PRICE:
                        continue
                    out.append((t, t, "\u2014", "momentum"))
                except Exception:
                    continue
        except Exception:
            continue
    print(f"[custom] requested {len(wanted)}, usable {len(out)}")
    return out


# ----------------------------------------------------------------------------
# DATA + SCORING
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)   # cache 1 hour so it loads fast
def load_data(custom_names=None):
    if custom_names:
        names = custom_names
    elif SCREEN_MODE == "broad":
        names = broad_universe()
        if len(names) < 50:
            st.warning(f"Broad screen returned only {len(names)} names "
                       f"(check the app logs for which source failed). "
                       f"Falling back to the curated list.")
            names = UNIVERSE
        elif USE_VOLUME_SURGE:
            kept, _ = volume_surge_tickers(
                [u[0] for u in names],
                TOP_VOLUME_N, VOLUME_WINDOW_DAYS, VOLUME_BASELINE_DAYS)
            if kept:
                keep_set = set(kept)
                names = [u for u in names if u[0] in keep_set]
    else:
        names = UNIVERSE

    tickers = [u[0] for u in names]
    meta = {u[0]: {"company": u[1], "sector": u[2], "bucket": u[3]} for u in names}

    # Fetch in batches so one slow request can't sink the whole screen
    frames = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            raw = yf.download(batch, period="1y", interval="1d",
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            for t in batch:
                try:
                    px = raw[t]["Close"].dropna()
                    if len(px) >= 200 and float(px.iloc[-1]) >= MIN_PRICE:
                        frames[t] = px
                except Exception:
                    continue
        except Exception:
            continue

    rows, closes = [], {}
    for t, px in frames.items():
        closes[t] = px
        ma50 = px.rolling(50).mean().iloc[-1]
        ma200 = px.rolling(200).mean().iloc[-1]
        rows.append({
            "Ticker": t,
            "Company": meta.get(t, {}).get("company", t),
            "Sector": meta.get(t, {}).get("sector", "—"),
            "Bucket": meta.get(t, {}).get("bucket", "momentum"),
            "Price": float(px.iloc[-1]),
            "Return 12mo": float(px.iloc[-1] / px.iloc[0] - 1),
            "Return 6mo": float(px.iloc[-1] / px.iloc[-126] - 1),
            "Return 1mo": float(px.iloc[-1] / px.iloc[-21] - 1),
            "Volatility": float(px.pct_change().std() * np.sqrt(252)),
            "Trend": int(px.iloc[-1] > ma50) + int(px.iloc[-1] > ma200),
        })

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
    if len(sub) < TOP_N_CORE:          # broad mode may have fewer core names
        sub = df.copy()
    sub = sub.nlargest(TOP_N_CORE, "Composite Score")
    w = sub["Composite Score"] / sub["Composite Score"].sum()
    sub["Weight"] = np.minimum(w, CAP_CORE)
    sub["Weight"] = sub["Weight"] / sub["Weight"].sum()   # renormalize after cap
    return sub.sort_values("Weight", ascending=False)


def momentum_picks(df):
    sub = df[df["Bucket"] == "momentum"].copy()
    if len(sub) < TOP_N_MOM:
        sub = df.copy()
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
# MARKET TIMING — rules-based signal, not a prediction
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def market_timing_signal():
    """VIX level + S&P 500 trend, scored against plain, visible thresholds.
    This is a RULE, not a forecast: it tells you current conditions relative
    to historical norms, nothing about what happens next."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        spx = yf.Ticker("^GSPC").history(period="260d")["Close"]
        spx_now = spx.iloc[-1]
        spx_ma50 = spx.rolling(50).mean().iloc[-1]
        spx_ma200 = spx.rolling(200).mean().iloc[-1]
    except Exception:
        return None

    # VIX thresholds are the standard, widely-published bands
    if vix < 15:
        vix_label, vix_note = "Low / Complacent", "Calm markets — historically can precede surprises either way."
    elif vix < 20:
        vix_label, vix_note = "Normal", "Typical volatility range."
    elif vix < 30:
        vix_label, vix_note = "Elevated", "Markets pricing in real uncertainty."
    else:
        vix_label, vix_note = "High / Fear", "Historically often (not always) followed by a recovery — but can persist or worsen."

    trend_up = spx_now > spx_ma50 and spx_now > spx_ma200
    trend_label = "Uptrend (above both 50d & 200d avg)" if trend_up else (
        "Mixed / below long-term trend" if spx_now < spx_ma200 else "Mixed")

    # Simple combined score: trend matters more than VIX alone
    score = 50
    score += 20 if trend_up else -20
    score += 15 if vix < 20 else (0 if vix < 30 else -15)
    score = max(0, min(100, score))

    if score >= 65:
        regime, guidance = "Favorable", "Conditions historically associated with continuing to invest on schedule."
    elif score >= 40:
        regime, guidance = "Neutral", "No strong signal either way — sticking to your regular schedule is reasonable."
    else:
        regime, guidance = "Cautious", "Elevated fear and/or a weak trend. Some investors stay the course anyway (timing the market is notoriously hard); others reduce size this month. Your call, not the model's."

    return {
        "vix": vix, "vix_label": vix_label, "vix_note": vix_note,
        "spx_now": spx_now, "spx_ma50": spx_ma50, "spx_ma200": spx_ma200,
        "trend_label": trend_label, "score": score,
        "regime": regime, "guidance": guidance,
    }


# ----------------------------------------------------------------------------
# NEWS SENTIMENT — free headline keyword scoring per stock, not NLP magic
# ----------------------------------------------------------------------------
POS_WORDS = {"beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
             "upgrade", "upgraded", "record", "strong", "growth", "profit", "gain",
             "gains", "jump", "jumps", "outperform", "bullish", "buy", "raises",
             "raised", "exceeds", "optimis", "expand", "expands", "win", "wins"}
NEG_WORDS = {"miss", "misses", "plunge", "plunges", "slump", "slumps", "downgrade",
             "downgraded", "weak", "loss", "losses", "fall", "falls", "falling",
             "drop", "drops", "cut", "cuts", "lawsuit", "probe", "investigation",
             "bearish", "sell", "lowers", "lowered", "warns", "warning", "recall",
             "layoff", "layoffs", "decline", "declines"}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_headlines(ticker, company, max_items=10):
    """Free Google News RSS, no API key. Returns recent headline titles."""
    query = f"{ticker} {company} stock".replace(" ", "%20")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            tree = ET.fromstring(resp.read())
        items = tree.findall(".//item/title")
        return [i.text for i in items[:max_items] if i.text]
    except Exception:
        return []


def score_headline(text):
    words = set(re.findall(r"[a-z']+", text.lower()))
    pos = sum(1 for w in words if any(w.startswith(p) for p in POS_WORDS))
    neg = sum(1 for w in words if any(w.startswith(n) for n in NEG_WORDS))
    return pos - neg


@st.cache_data(ttl=1800, show_spinner=False)
def news_sentiment_table(tickers_companies):
    """Per-stock sentiment: average keyword score across recent free headlines.
    This is keyword counting, not language understanding — crude by design,
    and openly labeled as such. Headline tone is noisy and often LAGS price
    rather than leading it."""
    rows = []
    for ticker, company in tickers_companies:
        heads = fetch_headlines(ticker, company)
        if not heads:
            rows.append({"Ticker": ticker, "Headlines Found": 0,
                        "Sentiment Score": np.nan, "Sample Headline": "No headlines found"})
            continue
        scores = [score_headline(h) for h in heads]
        rows.append({
            "Ticker": ticker,
            "Headlines Found": len(heads),
            "Sentiment Score": float(np.mean(scores)),
            "Sample Headline": heads[0],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------------
st.title("📈 My US Stock Model")
_cap = "S&P 1500 (broad screen)" if SCREEN_MODE == "broad" else "curated watchlist"
st.caption(f"Screening: **{_cap}**. Three approaches, ranked from live market data. "
           f"Prices refresh each time this page loads. "
           f"Switch modes with SCREEN_MODE at the top of the file. "
           f"{market_session_note()}")

with st.expander("⚙️ Which stocks should it screen?", expanded=False):
    st.caption("Leave this alone to screen the full S&P 1500 (broad mode).")
    use_custom = st.checkbox(
        "Use my own ticker list instead (e.g. a 'most purchased' list)",
        value=False)
    custom_raw = ""
    if use_custom:
        custom_raw = st.text_area(
            "Paste tickers, separated by commas, spaces or new lines",
            placeholder="NVDA, TSLA, PLTR, AAPL, SOFI, F, AMD",
            height=80)
        st.caption("Screens exactly those names. Each needs 200+ days of "
                   f"history and a price above ${MIN_PRICE:.0f}; any that "
                   "fail are dropped.")
    _surge = "ON" if USE_VOLUME_SURGE else "OFF"
    st.caption(f"Volume-surge filter is {_surge} — keeps the top "
               f"{TOP_VOLUME_N} names by {VOLUME_WINDOW_DAYS}-day volume "
               f"versus their {VOLUME_BASELINE_DAYS}-day average. That "
               "ranks trading volume, not net buying.")

_custom_names = None
if use_custom and custom_raw.strip():
    with st.spinner("Looking up your ticker list..."):
        _custom_names = load_custom_tickers(custom_raw)
    if len(_custom_names) < 2:
        st.error("None of those tickers came back with usable price "
                 "history. Check the spelling and try again.")
        st.stop()
    st.success(f"Screening your list: {len(_custom_names)} usable names.")

with st.spinner("Screening the universe — this can take a minute or two in broad mode..."):
    df, prices = load_data(_custom_names)

if df.empty:
    st.error("Could not load price data right now. "
             "Yahoo's free feed may be rate-limiting — try again in a minute.")
    st.stop()

# ---- headline row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks screened", len(df))
c2.metric("Universe avg. 12mo return", f"{df['Return 12mo'].mean():.1%}")
c3.metric("Universe avg. volatility", f"{df['Volatility'].mean():.1%}")
c4.metric("Last updated (ET)", et_stamp())

d1, d2, d3, d4, d5, d6 = st.tabs([
    "🟢 Approach 1 — Diversified",
    "🔴 Approach 2 — Momentum",
    "🔵 Approach 3 — Markowitz",
    "📰 Market Timing & News",
    "✅ Summary — What to Buy",
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

# ---------------- Market Timing & News Sentiment ----------------
with d4:
    st.subheader("Market timing & news sentiment — signals, not predictions")
    st.info(
        "Read this first: nothing on this tab forecasts price. The timing gauge is a "
        "**rule** applied to two live indicators, and the stock sentiment column is "
        "**keyword counting** on free headlines. Both are useful context for *when* to "
        "deploy money and *what tone* surrounds a name. Neither knows what happens next."
    )

    # ---- Part A: market timing ----
    st.markdown("### Should I invest this month?")
    with st.spinner("Reading VIX and the S&P 500 trend..."):
        timing = market_timing_signal()

    if timing is None:
        st.warning("Couldn't reach the market indicators right now. Try reloading.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("VIX (fear gauge)", f"{timing['vix']:.1f}", timing["vix_label"])
        t2.metric("S&P 500 trend", timing["trend_label"].split(" (")[0])
        t3.metric("Timing score", f"{timing['score']:.0f}/100", timing["regime"])

        if timing["regime"] == "Favorable":
            st.success(f"**{timing['regime']}** — {timing['guidance']}")
        elif timing["regime"] == "Neutral":
            st.info(f"**{timing['regime']}** — {timing['guidance']}")
        else:
            st.warning(f"**{timing['regime']}** — {timing['guidance']}")

        st.write("**What produced that score:**")
        st.dataframe(pd.DataFrame({
            "Indicator": ["Read at (ET)", "VIX level", "VIX read", "S&P 500 vs 50-day avg",
                          "S&P 500 vs 200-day avg"],
            "Value": [et_stamp(), f"{timing['vix']:.1f}", timing["vix_label"],
                      f"{timing['spx_now']:,.0f} vs {timing['spx_ma50']:,.0f}",
                      f"{timing['spx_now']:,.0f} vs {timing['spx_ma200']:,.0f}"],
        }), use_container_width=True, hide_index=True)

        st.caption(f"{timing['vix_note']} Score = 50 to start, +20 for an uptrend "
                   f"(−20 otherwise), +15 for VIX under 20 (−15 above 30). "
                   f"Thresholds are visible in the code and you can change them.")

    st.divider()

    # ---- Part B: per-stock news sentiment ----
    st.markdown("### News tone on your watchlist")
    st.write("Counts positive vs negative words across the most recent free headlines "
             "for each stock. A negative score means the headlines skew bearish right "
             "now — not that the stock will fall.")

    with st.spinner("Scanning recent headlines..."):
        sent = news_sentiment_table([(u[0], u[1]) for u in UNIVERSE])

    if sent.empty:
        st.info("No headlines could be retrieved right now.")
    else:
        sent_sorted = sent.sort_values("Sentiment Score", ascending=False,
                                       na_position="last")
        st.dataframe(
            sent_sorted.style.format({"Sentiment Score": "{:+.1f}"}, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )

        rated = sent_sorted.dropna(subset=["Sentiment Score"])
        if not rated.empty:
            p1, p2 = st.columns(2)
            p1.metric("Most positive tone right now", rated.iloc[0]["Ticker"],
                      f"{rated.iloc[0]['Sentiment Score']:+.1f}")
            p2.metric("Most negative tone right now", rated.iloc[-1]["Ticker"],
                      f"{rated.iloc[-1]['Sentiment Score']:+.1f}")

        st.caption("Scores run from roughly −3 (heavy negative tone) to +3 (heavy "
                   "positive). This is keyword matching, not language understanding — "
                   "treat it as a quick read of headline mood, and check the sample "
                   "headline yourself before acting on any single row.")

# ---------------- Summary & Buy Plan ----------------
with d5:
    st.subheader("What to buy, and is now a good time")

    # ---- Part A: is today a good time ----
    timing = market_timing_signal()
    if timing is None:
        st.warning("Couldn't read the market indicators right now — timing score unavailable.")
        timing_score, timing_regime, timing_label = 50, "Unknown", "n/a"
    else:
        timing_score = timing["score"]
        timing_regime = timing["regime"]
        timing_label = f"{timing_score:.0f}/100"

    st.markdown("#### 1. Is today a good time to invest?")
    a1, a2, a3 = st.columns(3)
    a1.metric("Timing score", timing_label, timing_regime if timing else "")
    a2.metric("Today (ET)", et_stamp())
    a2.caption(market_session_note())
    a3.metric("Recommendation from the score",
              "Full amount" if timing_score >= 65
              else ("Normal amount" if timing_score >= 40 else "Reduced amount"))

    gauge = int(round(timing_score))
    pct_good = gauge / 100
    st.progress(pct_good)
    st.write(f"**{gauge}% favourable** on this rule "
             f"({timing_regime} regime). Read it as opinion from a fixed rule, "
             f"not a forecast.")

    if timing:
        if timing_regime == "Favorable":
            st.success("Conditions look favourable on this rule — investing the "
                       "full monthly amount is reasonable.")
        elif timing_regime == "Neutral":
            st.info("No strong signal either way — sticking to your usual schedule "
                    "is reasonable.")
        else:
            st.warning("Conditions look cautious on this rule. Some investors stay the "
                       "course anyway (timing is notoriously hard); others reduce size "
                       "this month. Your call.")

    st.divider()

    # ---- Part B: the buy plan ----
    st.markdown("#### 2. What to buy this month")

    contribution = st.number_input(
        "How much are you investing this month? ($)",
        min_value=50, max_value=100000, value=1000, step=50,
    )

    plan_source = st.radio(
        "Build the buy list from which approach?",
        ["Approach 1 — Diversified", "Approach 2 — Momentum",
         "Approach 3 — Markowitz", "Blended (80% Diversified / 20% Momentum)"],
        horizontal=False,
    )

    def to_buy(weights_df, total):
        out = weights_df.copy()
        out["Amount ($)"] = (out["Weight"] * total).round(0)
        out["Shares (approx.)"] = (out["Amount ($)"] / out["Price"]).round(3)
        return out[["Ticker", "Company", "Price", "Weight", "Amount ($)", "Shares (approx.)"]]

    if plan_source.startswith("Approach 1"):
        base = diversified_picks(df)
    elif plan_source.startswith("Approach 2"):
        base = momentum_picks(df)
    elif plan_source.startswith("Approach 3"):
        opt, _ = markowitz_real(df, prices)
        base = opt if opt is not None else diversified_picks(df)
    else:
        core = diversified_picks(df).copy()
        core["Weight"] = core["Weight"] * 0.8
        mom = momentum_picks(df).copy()
        mom["Weight"] = mom["Weight"] * 0.2
        base = pd.concat([core, mom])

    plan = to_buy(base, contribution)
    shown = plan.copy()
    shown["Weight"] = shown["Weight"].map(lambda x: f"{x:.1%}")
    shown["Amount ($)"] = shown["Amount ($)"].map(lambda x: f"${x:,.0f}")
    st.dataframe(shown.style.format({"Price": "${:,.2f}", "Shares (approx.)": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    st.metric("Total being invested", f"${plan['Amount ($)'].sum():,.0f}")
    st.caption(f"Prices and levels read at {et_stamp()}. "
           "Amounts are rounded to the nearest dollar, so they may total a few "
               "dollars either side of your contribution. Reopen the app right before "
               "you trade so the prices you act on are current.")

    st.download_button(
        "Download this buy list as CSV",
        plan.to_csv(index=False).encode("utf-8"),
        file_name="buy_list.csv",
        mime="text/csv",
    )

    if timing and timing_regime == "Cautious":
        st.info("The timing rule currently reads caution. A common middle path is to "
                "buy the list above at a reduced size this month and keep the rest "
                "aside for next month — that keeps you investing without ignoring the "
                "signal.")

# ---------------- Guide ----------------
with d6:
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


