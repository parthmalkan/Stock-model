"""
dashboard.py Personal US stock screening app (free, Streamlit).

Approach 3 uses REAL Markowitz optimization (PyPortfolioOpt) a full
covariance matrix across the candidate holdings and a max-Sharpe solve,
not a two-bucket approximation.

 Tab 1 Approach 1: Diversified Large-Cap  (lower risk)
 Tab 2 Approach 2: Momentum / Speculative (higher risk, higher swing)
 Tab 3 Approach 3: Markowitz-Optimized   (real optimizer, full covariance)
 Tab 4 Market Timing & News Sentiment   (rules-based signal, not a prediction)
 Tab 5 How to read this

Run locally:  streamlit run dashboard.py
Deploy free:  share.streamlit.io (sign in with GitHub)

Data: yfinance (free, no API key) for prices/VIX; Google News RSS (free,
no API key) for headlines. Prices refresh every time the app opens.

IMPORTANT what Tab 4 actually is:
It is a RULES-BASED SIGNAL, not a prediction. Market timing = VIX level +
S&P trend, scored against plain thresholds anyone can read. News sentiment =
counting positive/negative keywords in recent free headlines per stock. Both
are real and computed live, but neither forecasts price. No free data source
can reliably do that, and this app does not pretend otherwise.
"""

import re
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import json
import urllib.request
from datetime import datetime, timezone
import warnings
import urllib.request
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS change these if you like
# ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# UNIVERSE SELECTION
# ---------------------------------------------------------------------------
# AUTO_UNIVERSE: choose the universe fresh on every run instead of using a
#  fixed list. Sources are tried in order:
#   1. Wikipedia S&P 500 + 400 + 600 (~1,500 names, broad)
#   2. Yahoo screener endpoints    (most active / gainers)
#   3. FORTUNE 500 fallback list   (bundled in this file, always works)
#  The names are then filtered (price floor, 200 days of history) and the
#  top AUTO_UNIVERSE_SIZE are kept, ranked by AUTO_RANK_BY.
AUTO_UNIVERSE = True
AUTO_UNIVERSE_SIZE = 500
AUTO_RANK_BY = "dollar_volume"  # or "market_cap" -- both need last price

# --- Universe composition: four tiers, de-duplicated, 500 total ---
#   1. 30 movers today      (gainers + losers, so no direction bias)
#   2. 30 by dollar volume  (heaviest trading today)
#   3. up to 100 from the Robinhood list you paste below
#   4. the remainder, ranked on growth + liquidity + sentiment
TIER_MOVERS = 30
TIER_LIQUID = 30
TIER_ROBINHOOD = 100

# Tier 4 draws ONLY from S&P 500 companies. The bundled list marks those as
# bucket "core"; the live Wikipedia S&P 500 table is added to that pool when
# it is reachable. Everything outside that pool is excluded from tier 4.
TIER4_SP500_ONLY = True

# Robinhood publishes NO public API, so their list cannot be fetched.
# Copy the tickers from the Robinhood app and paste them here.
# Commas, spaces or new lines all work. Example:
#   ROBINHOOD_100 = "NVDA, TSLA, PLTR, AAPL, SOFI, F, AMD"
ROBINHOOD_100 = ""

# ---------------------------------------------------------------------------
# UNIVERSE: one source chain, no toggles.
#   1. S&P 1500 from Wikipedia   - broadest, all 11 sectors, direction-neutral
#   2. Yahoo screener            - actives + gainers + losers, de-duplicated
#   3. Bundled FORTUNE 500 list  - always available, no network needed
# Whatever the sources return is filtered (price floor, 200 days of history),
# ranked by dollar volume, and trimmed to AUTO_UNIVERSE_SIZE.

# How many tickers to take from the Yahoo screener tier
YAHOO_TAKE = 400

# FORECASTING
# ---------------------------------------------------------------------------
# FORECAST_HORIZON_DAYS: how far ahead the simulation looks.
# The forecast describes DISPERSION, not direction -- it says how far a stock
# might move given its own past volatility. It has NO view on whether the
# price rises or falls. Every forecast is logged so its accuracy can be scored
# against what actually happened, rather than taken on trust.
# The forecast looks exactly one month ahead: it predicts how far the price
# will move by the same date next month. 30 days is close enough for that.
FORECAST_HORIZON_DAYS = 30

# The forecast is refreshed WHEN YOU PRESS THE BUTTON on the Forecast tab.
# Each press: regenerates the call, stores it with the date and the price at
# that moment, and keeps the PREVIOUS press so Delta can be computed -- the
# actual price move between the last press and now, versus what that press
# predicted. No calendar rule; you decide the cycle.
FORECAST_STATE = "forecast_state.json"

# Weighting of the signals that build the forecast.
# Every input is normalised to a common scale, then blended.
#  momentum : recent price trend (6mo / 1mo returns, risk-adjusted)
#  volume  : is the stock trading above its normal volume
#  news   : tone of recent headlines (keyword count, crude)
#  stats  : the stock's own typical move in the direction it is leaning
FORECAST_W_MOMENTUM = 0.40
FORECAST_W_VOLUME = 0.20
FORECAST_W_NEWS = 0.20
FORECAST_W_STATS = 0.20
# Cap on the final number so one extreme input cannot produce a wild figure
FORECAST_MAX_PCT = 25.0

# SCREEN_MODE is the fallback path when AUTO_UNIVERSE is False:
#  "bundled" -> use the bundled list below as-is
SCREEN_MODE = "bundled"

# VOLUME-SURGE FILTER (Option A):
#  Narrows the broad universe to the TOP_VOLUME_N names by how heavily
#  they are trading versus their own average, measured over the last
#  VOLUME_WINDOW_DAYS. A proxy for "what the market is focused on now".
#  IMPORTANT: this is TRADING VOLUME, not net buying. Volume counts buys
#  and sells alike, so it includes heavy selling and shorting. It is NOT
#  the same as "most purchased" -- use the custom list for that.
USE_VOLUME_SURGE = True
TOP_VOLUME_N = 250     # keep the top N by volume ratio
VOLUME_WINDOW_DAYS = 3   # recent window (trading days)
VOLUME_BASELINE_DAYS = 60  # baseline to compare against

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
    return "Weekend US market closed; prices are the last close."
  mins = now.hour * 60 + now.minute
  if mins < 9 * 60 + 30:
    return "Pre-market regular session opens 9:30am ET."
  if mins <= 16 * 60:
    return "US market open (regular session, 9:30am-4:00pm ET)."
  return "After hours regular session closed at 4:00pm ET."

MIN_PRICE = 5.0      # skip penny stocks / near-zero names
MAX_BROAD_NAMES = 1500   # hard cap so the free host isn't overwhelmed

# ----------------------------------------------------------------------------
# BUNDLED STOCK LIST (self-contained -- no network fetch, cannot fail)
# ----------------------------------------------------------------------------
# ~416 large/mid-cap US names across all 11 GICS sectors. Embedded here on
# purpose: the earlier version fetched this list from Wikipedia at runtime,
# and on the free host that fetch returned 0 names, silently falling back to
# 24. A bundled list cannot fail, so the screen always has real depth.
#
# Fields: (ticker, company, sector, bucket)
#  bucket core   = mega/large-cap names, used by Approach 1 (diversified)
#  bucket momentum = everything else, used by Approach 2 (higher swing)
# Add or remove lines freely -- the app adapts to whatever is here.
UNIVERSE = [
  # ticker, company, sector, bucket
  ("AAPL", "Apple Inc.", "Technology", "core"),
  ("MSFT", "Microsoft Corp.", "Technology", "core"),
  ("NVDA", "NVIDIA Corp.", "Technology", "core"),
  ("AVGO", "Broadcom Inc.", "Technology", "core"),
  ("AMD", "Advanced Micro Devices", "Technology", "core"),
  ("ORCL", "Oracle Corp.", "Technology", "momentum"),
  ("CRM", "Salesforce Inc.", "Technology", "core"),
  ("ADBE", "Adobe Inc.", "Technology", "core"),
  ("CSCO", "Cisco Systems Inc.", "Technology", "core"),
  ("ACN", "Accenture plc", "Technology", "core"),
  ("INTC", "Intel Corp.", "Technology", "core"),
  ("QCOM", "Qualcomm Inc.", "Technology", "core"),
  ("TXN", "Texas Instruments", "Technology", "core"),
  ("IBM", "IBM Corp.", "Technology", "core"),
  ("NOW", "ServiceNow Inc.", "Technology", "core"),
  ("INTU", "Intuit Inc.", "Technology", "core"),
  ("AMAT", "Applied Materials", "Technology", "momentum"),
  ("MU", "Micron Technology", "Technology", "core"),
  ("LRCX", "Lam Research Corp.", "Technology", "core"),
  ("ADI", "Analog Devices", "Technology", "core"),
  ("KLAC", "KLA Corp.", "Technology", "core"),
  ("SNPS", "Synopsys Inc.", "Technology", "momentum"),
  ("CDNS", "Cadence Design Systems", "Technology", "momentum"),
  ("APH", "Amphenol Corp.", "Technology", "momentum"),
  ("MSI", "Motorola Solutions", "Technology", "momentum"),
  ("ROP", "Roper Technologies", "Technology", "momentum"),
  ("FTNT", "Fortinet Inc.", "Technology", "momentum"),
  ("NXPI", "NXP Semiconductors", "Technology", "momentum"),
  ("MCHP", "Microchip Technology", "Technology", "momentum"),
  ("HPQ", "HP Inc.", "Technology", "momentum"),
  ("DELL", "Dell Technologies", "Technology", "momentum"),
  ("WDC", "Western Digital", "Technology", "momentum"),
  ("STX", "Seagate Technology", "Technology", "momentum"),
  ("GLW", "Corning Inc.", "Technology", "momentum"),
  ("ANSS", "Ansys Inc.", "Technology", "momentum"),
  ("CDW", "CDW Corp.", "Technology", "momentum"),
  ("IT", "Gartner Inc.", "Technology", "momentum"),
  ("SWKS", "Skyworks Solutions", "Technology", "momentum"),
  ("ZBRA", "Zebra Technologies", "Technology", "momentum"),
  ("EPAM", "EPAM Systems", "Technology", "momentum"),
  ("JNPR", "Juniper Networks", "Technology", "momentum"),
  ("FFIV", "F5 Inc.", "Technology", "momentum"),
  ("AKAM", "Akamai Technologies", "Technology", "momentum"),
  ("NTAP", "NetApp Inc.", "Technology", "momentum"),
  ("QRVO", "Qorvo Inc.", "Technology", "momentum"),
  ("GEN", "Gen Digital Inc.", "Technology", "momentum"),
  ("TER", "Teradyne Inc.", "Technology", "momentum"),
  ("KEYS", "Keysight Technologies", "Technology", "momentum"),
  ("TRMB", "Trimble Inc.", "Technology", "momentum"),
  ("GRMN", "Garmin Ltd.", "Technology", "momentum"),
  ("TDY", "Teledyne Technologies", "Technology", "momentum"),
  ("LDOS", "Leidos Holdings", "Technology", "momentum"),
  ("LHX", "L3Harris Technologies", "Technology", "momentum"),
  ("HPE", "Hewlett Packard Enterprise", "Technology", "momentum"),
  ("ON", "ON Semiconductor", "Technology", "momentum"),
  ("SMCI", "Super Micro Computer", "Technology", "momentum"),
  ("CRWD", "CrowdStrike Holdings", "Technology", "momentum"),
  ("PANW", "Palo Alto Networks", "Technology", "core"),
  ("SNOW", "Snowflake Inc.", "Technology", "momentum"),
  ("DDOG", "Datadog Inc.", "Technology", "momentum"),
  ("ZS", "Zscaler Inc.", "Technology", "momentum"),
  ("TEAM", "Atlassian Corp.", "Technology", "momentum"),
  ("WDAY", "Workday Inc.", "Technology", "momentum"),
  ("MSTR", "MicroStrategy Inc.", "Technology", "momentum"),
  ("PLTR", "Palantir Technologies", "Technology", "momentum"),
  ("APP", "AppLovin Corp.", "Technology", "momentum"),
  ("ANET", "Arista Networks", "Technology", "momentum"),
  ("MPWR", "Monolithic Power Systems", "Technology", "momentum"),
  ("ENPH", "Enphase Energy", "Technology", "momentum"),
  ("FSLR", "First Solar Inc.", "Technology", "momentum"),
  ("GOOGL", "Alphabet Inc.", "Communication Services", "core"),
  ("META", "Meta Platforms Inc.", "Communication Services", "core"),
  ("NFLX", "Netflix Inc.", "Communication Services", "core"),
  ("DIS", "Walt Disney Co.", "Communication Services", "momentum"),
  ("CMCSA", "Comcast Corp.", "Communication Services", "momentum"),
  ("TMUS", "T-Mobile US", "Communication Services", "momentum"),
  ("VZ", "Verizon Communications", "Communication Services", "core"),
  ("T", "AT&T Inc.", "Communication Services", "core"),
  ("EA", "Electronic Arts", "Communication Services", "momentum"),
  ("TTWO", "Take-Two Interactive", "Communication Services", "momentum"),
  ("WBD", "Warner Bros. Discovery", "Communication Services", "momentum"),
  ("OMC", "Omnicom Group", "Communication Services", "momentum"),
  ("IPG", "Interpublic Group", "Communication Services", "momentum"),
  ("LYV", "Live Nation Entertainment", "Communication Services", "momentum"),
  ("MTCH", "Match Group", "Communication Services", "momentum"),
  ("PARA", "Paramount Global", "Communication Services", "momentum"),
  ("NWSA", "News Corp.", "Communication Services", "momentum"),
  ("FOXA", "Fox Corp.", "Communication Services", "momentum"),
  ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "core"),
  ("TSLA", "Tesla Inc.", "Consumer Discretionary", "core"),
  ("HD", "Home Depot Inc.", "Consumer Discretionary", "core"),
  ("MCD", "McDonald's Corp.", "Consumer Discretionary", "core"),
  ("NKE", "Nike Inc.", "Consumer Discretionary", "momentum"),
  ("LOW", "Lowe's Companies", "Consumer Discretionary", "core"),
  ("SBUX", "Starbucks Corp.", "Consumer Discretionary", "core"),
  ("TJX", "TJX Companies", "Consumer Discretionary", "core"),
  ("BKNG", "Booking Holdings", "Consumer Discretionary", "core"),
  ("CMG", "Chipotle Mexican Grill", "Consumer Discretionary", "momentum"),
  ("ORLY", "O'Reilly Automotive", "Consumer Discretionary", "momentum"),
  ("AZO", "AutoZone Inc.", "Consumer Discretionary", "momentum"),
  ("MAR", "Marriott International", "Consumer Discretionary", "momentum"),
  ("HLT", "Hilton Worldwide", "Consumer Discretionary", "momentum"),
  ("GM", "General Motors", "Consumer Discretionary", "momentum"),
  ("F", "Ford Motor Co.", "Consumer Discretionary", "momentum"),
  ("ROST", "Ross Stores", "Consumer Discretionary", "momentum"),
  ("DHI", "D.R. Horton", "Consumer Discretionary", "momentum"),
  ("LEN", "Lennar Corp.", "Consumer Discretionary", "momentum"),
  ("PHM", "PulteGroup Inc.", "Consumer Discretionary", "momentum"),
  ("YUM", "Yum! Brands", "Consumer Discretionary", "momentum"),
  ("DRI", "Darden Restaurants", "Consumer Discretionary", "momentum"),
  ("ULTA", "Ulta Beauty", "Consumer Discretionary", "momentum"),
  ("BBY", "Best Buy Co.", "Consumer Discretionary", "momentum"),
  ("EBAY", "eBay Inc.", "Consumer Discretionary", "momentum"),
  ("EXPE", "Expedia Group", "Consumer Discretionary", "momentum"),
  ("LVS", "Las Vegas Sands", "Consumer Discretionary", "momentum"),
  ("MGM", "MGM Resorts International", "Consumer Discretionary", "momentum"),
  ("RCL", "Royal Caribbean Cruises", "Consumer Discretionary", "momentum"),
  ("CCL", "Carnival Corp.", "Consumer Discretionary", "momentum"),
  ("NCLH", "Norwegian Cruise Line", "Consumer Discretionary", "momentum"),
  ("APTV", "Aptiv plc", "Consumer Discretionary", "momentum"),
  ("BWA", "BorgWarner Inc.", "Consumer Discretionary", "momentum"),
  ("TPR", "Tapestry Inc.", "Consumer Discretionary", "momentum"),
  ("RL", "Ralph Lauren Corp.", "Consumer Discretionary", "momentum"),
  ("HAS", "Hasbro Inc.", "Consumer Discretionary", "momentum"),
  ("PG", "Procter & Gamble", "Consumer Staples", "core"),
  ("COST", "Costco Wholesale", "Consumer Staples", "core"),
  ("WMT", "Walmart Inc.", "Consumer Staples", "core"),
  ("KO", "Coca-Cola Co.", "Consumer Staples", "core"),
  ("PEP", "PepsiCo Inc.", "Consumer Staples", "core"),
  ("PM", "Philip Morris International", "Consumer Staples", "core"),
  ("MO", "Altria Group", "Consumer Staples", "momentum"),
  ("MDLZ", "Mondelez International", "Consumer Staples", "core"),
  ("CL", "Colgate-Palmolive", "Consumer Staples", "momentum"),
  ("TGT", "Target Corp.", "Consumer Staples", "momentum"),
  ("KMB", "Kimberly-Clark", "Consumer Staples", "momentum"),
  ("GIS", "General Mills", "Consumer Staples", "momentum"),
  ("K", "Kellanova", "Consumer Staples", "momentum"),
  ("HSY", "Hershey Co.", "Consumer Staples", "momentum"),
  ("SYY", "Sysco Corp.", "Consumer Staples", "momentum"),
  ("ADM", "Archer-Daniels-Midland", "Consumer Staples", "momentum"),
  ("STZ", "Constellation Brands", "Consumer Staples", "momentum"),
  ("KHC", "Kraft Heinz Co.", "Consumer Staples", "momentum"),
  ("CHD", "Church & Dwight", "Consumer Staples", "momentum"),
  ("CLX", "Clorox Co.", "Consumer Staples", "momentum"),
  ("EL", "Estee Lauder Companies", "Consumer Staples", "momentum"),
  ("KR", "Kroger Co.", "Consumer Staples", "momentum"),
  ("TSN", "Tyson Foods", "Consumer Staples", "momentum"),
  ("MKC", "McCormick & Co.", "Consumer Staples", "momentum"),
  ("BG", "Bunge Global", "Consumer Staples", "momentum"),
  ("CAG", "ConAgra Brands", "Consumer Staples", "momentum"),
  ("LW", "Lamb Weston Holdings", "Consumer Staples", "momentum"),
  ("HRL", "Hormel Foods", "Consumer Staples", "momentum"),
  ("SJM", "J.M. Smucker Co.", "Consumer Staples", "momentum"),
  ("CPB", "Campbell's Co.", "Consumer Staples", "momentum"),
  ("BRK-B", "Berkshire Hathaway", "Financials", "core"),
  ("JPM", "JPMorgan Chase & Co.", "Financials", "core"),
  ("V", "Visa Inc.", "Financials", "core"),
  ("MA", "Mastercard Inc.", "Financials", "core"),
  ("BAC", "Bank of America", "Financials", "momentum"),
  ("WFC", "Wells Fargo & Co.", "Financials", "momentum"),
  ("GS", "Goldman Sachs Group", "Financials", "core"),
  ("MS", "Morgan Stanley", "Financials", "core"),
  ("SPGI", "S&P Global Inc.", "Financials", "core"),
  ("BLK", "BlackRock Inc.", "Financials", "core"),
  ("AXP", "American Express", "Financials", "momentum"),
  ("C", "Citigroup Inc.", "Financials", "momentum"),
  ("SCHW", "Charles Schwab", "Financials", "core"),
  ("CB", "Chubb Ltd.", "Financials", "core"),
  ("MMC", "Marsh & McLennan", "Financials", "core"),
  ("PGR", "Progressive Corp.", "Financials", "momentum"),
  ("AON", "Aon plc", "Financials", "momentum"),
  ("ICE", "Intercontinental Exchange", "Financials", "momentum"),
  ("CME", "CME Group", "Financials", "momentum"),
  ("USB", "U.S. Bancorp", "Financials", "momentum"),
  ("PNC", "PNC Financial Services", "Financials", "momentum"),
  ("TFC", "Truist Financial", "Financials", "momentum"),
  ("COF", "Capital One Financial", "Financials", "momentum"),
  ("BK", "Bank of New York Mellon", "Financials", "momentum"),
  ("AIG", "American International Group", "Financials", "momentum"),
  ("MET", "MetLife Inc.", "Financials", "momentum"),
  ("PRU", "Prudential Financial", "Financials", "momentum"),
  ("AFL", "Aflac Inc.", "Financials", "momentum"),
  ("ALL", "Allstate Corp.", "Financials", "momentum"),
  ("TRV", "Travelers Companies", "Financials", "momentum"),
  ("AJG", "Arthur J. Gallagher", "Financials", "momentum"),
  ("MCO", "Moody's Corp.", "Financials", "momentum"),
  ("MSCI", "MSCI Inc.", "Financials", "momentum"),
  ("FIS", "Fidelity National Info Services", "Financials", "momentum"),
  ("FI", "Fiserv Inc.", "Financials", "core"),
  ("GPN", "Global Payments", "Financials", "momentum"),
  ("PYPL", "PayPal Holdings", "Financials", "momentum"),
  ("SYF", "Synchrony Financial", "Financials", "momentum"),
  ("DFS", "Discover Financial Services", "Financials", "momentum"),
  ("FITB", "Fifth Third Bancorp", "Financials", "momentum"),
  ("HBAN", "Huntington Bancshares", "Financials", "momentum"),
  ("RF", "Regions Financial", "Financials", "momentum"),
  ("KEY", "KeyCorp", "Financials", "momentum"),
  ("CFG", "Citizens Financial Group", "Financials", "momentum"),
  ("MTB", "M&T Bank Corp.", "Financials", "momentum"),
  ("NTRS", "Northern Trust", "Financials", "momentum"),
  ("STT", "State Street Corp.", "Financials", "momentum"),
  ("WTW", "Willis Towers Watson", "Financials", "momentum"),
  ("CINF", "Cincinnati Financial", "Financials", "momentum"),
  ("WRB", "W.R. Berkley Corp.", "Financials", "momentum"),
  ("BRO", "Brown & Brown", "Financials", "momentum"),
  ("CBOE", "Cboe Global Markets", "Financials", "momentum"),
  ("NDAQ", "Nasdaq Inc.", "Financials", "momentum"),
  ("AMP", "Ameriprise Financial", "Financials", "momentum"),
  ("RJF", "Raymond James Financial", "Financials", "momentum"),
  ("VTRS", "Viatris Inc.", "Financials", "momentum"),
  ("LLY", "Eli Lilly & Co.", "Health Care", "core"),
  ("UNH", "UnitedHealth Group", "Health Care", "core"),
  ("JNJ", "Johnson & Johnson", "Health Care", "core"),
  ("ABBV", "AbbVie Inc.", "Health Care", "core"),
  ("MRK", "Merck & Co.", "Health Care", "core"),
  ("TMO", "Thermo Fisher Scientific", "Health Care", "core"),
  ("ABT", "Abbott Laboratories", "Health Care", "core"),
  ("PFE", "Pfizer Inc.", "Health Care", "core"),
  ("DHR", "Danaher Corp.", "Health Care", "core"),
  ("AMGN", "Amgen Inc.", "Health Care", "core"),
  ("ISRG", "Intuitive Surgical", "Health Care", "core"),
  ("BSX", "Boston Scientific", "Health Care", "momentum"),
  ("SYK", "Stryker Corp.", "Health Care", "core"),
  ("VRTX", "Vertex Pharmaceuticals", "Health Care", "core"),
  ("REGN", "Regeneron Pharmaceuticals", "Health Care", "core"),
  ("MDT", "Medtronic plc", "Health Care", "core"),
  ("CI", "Cigna Group", "Health Care", "core"),
  ("ELV", "Elevance Health", "Health Care", "core"),
  ("CVS", "CVS Health Corp.", "Health Care", "momentum"),
  ("MCK", "McKesson Corp.", "Health Care", "momentum"),
  ("HCA", "HCA Healthcare", "Health Care", "momentum"),
  ("ZTS", "Zoetis Inc.", "Health Care", "momentum"),
  ("BDX", "Becton Dickinson", "Health Care", "momentum"),
  ("HUM", "Humana Inc.", "Health Care", "momentum"),
  ("EW", "Edwards Lifesciences", "Health Care", "momentum"),
  ("A", "Agilent Technologies", "Health Care", "momentum"),
  ("IQV", "IQVIA Holdings", "Health Care", "momentum"),
  ("GILD", "Gilead Sciences", "Health Care", "core"),
  ("BIIB", "Biogen Inc.", "Health Care", "momentum"),
  ("MRNA", "Moderna Inc.", "Health Care", "momentum"),
  ("IDXX", "IDEXX Laboratories", "Health Care", "momentum"),
  ("RMD", "ResMed Inc.", "Health Care", "momentum"),
  ("MTD", "Mettler-Toledo International", "Health Care", "momentum"),
  ("WAT", "Waters Corp.", "Health Care", "momentum"),
  ("BAX", "Baxter International", "Health Care", "momentum"),
  ("CAH", "Cardinal Health", "Health Care", "momentum"),
  ("GEHC", "GE HealthCare Technologies", "Health Care", "momentum"),
  ("DXCM", "DexCom Inc.", "Health Care", "momentum"),
  ("PODD", "Insulet Corp.", "Health Care", "momentum"),
  ("ALGN", "Align Technology", "Health Care", "momentum"),
  ("COO", "Cooper Companies", "Health Care", "momentum"),
  ("STE", "STERIS plc", "Health Care", "momentum"),
  ("HOLX", "Hologic Inc.", "Health Care", "momentum"),
  ("ZBH", "Zimmer Biomet Holdings", "Health Care", "momentum"),
  ("SOLV", "Solventum Corp.", "Health Care", "momentum"),
  ("LH", "Labcorp Holdings", "Health Care", "momentum"),
  ("GE", "GE Aerospace", "Industrials", "core"),
  ("CAT", "Caterpillar Inc.", "Industrials", "core"),
  ("RTX", "RTX Corp.", "Industrials", "core"),
  ("UBER", "Uber Technologies", "Industrials", "core"),
  ("HON", "Honeywell International", "Industrials", "core"),
  ("UNP", "Union Pacific Corp.", "Industrials", "momentum"),
  ("BA", "Boeing Co.", "Industrials", "core"),
  ("LMT", "Lockheed Martin", "Industrials", "core"),
  ("DE", "Deere & Co.", "Industrials", "core"),
  ("ADP", "Automatic Data Processing", "Industrials", "core"),
  ("MMM", "3M Company", "Industrials", "momentum"),
  ("UPS", "United Parcel Service", "Industrials", "momentum"),
  ("GD", "General Dynamics", "Industrials", "momentum"),
  ("NOC", "Northrop Grumman", "Industrials", "momentum"),
  ("ITW", "Illinois Tool Works", "Industrials", "momentum"),
  ("EMR", "Emerson Electric", "Industrials", "momentum"),
  ("ETN", "Eaton Corp.", "Industrials", "core"),
  ("PH", "Parker-Hannifin", "Industrials", "momentum"),
  ("CSX", "CSX Corp.", "Industrials", "momentum"),
  ("NSC", "Norfolk Southern", "Industrials", "momentum"),
  ("FDX", "FedEx Corp.", "Industrials", "momentum"),
  ("WM", "Waste Management", "Industrials", "momentum"),
  ("RSG", "Republic Services", "Industrials", "momentum"),
  ("CTAS", "Cintas Corp.", "Industrials", "momentum"),
  ("PAYX", "Paychex Inc.", "Industrials", "momentum"),
  ("FAST", "Fastenal Co.", "Industrials", "momentum"),
  ("ODFL", "Old Dominion Freight Line", "Industrials", "momentum"),
  ("CARR", "Carrier Global", "Industrials", "momentum"),
  ("OTIS", "Otis Worldwide", "Industrials", "momentum"),
  ("TT", "Trane Technologies", "Industrials", "momentum"),
  ("JCI", "Johnson Controls International", "Industrials", "momentum"),
  ("ROK", "Rockwell Automation", "Industrials", "momentum"),
  ("CMI", "Cummins Inc.", "Industrials", "momentum"),
  ("PCAR", "PACCAR Inc.", "Industrials", "momentum"),
  ("GWW", "W.W. Grainger", "Industrials", "momentum"),
  ("IR", "Ingersoll Rand", "Industrials", "momentum"),
  ("DOV", "Dover Corp.", "Industrials", "momentum"),
  ("AME", "AMETEK Inc.", "Industrials", "momentum"),
  ("SWK", "Stanley Black & Decker", "Industrials", "momentum"),
  ("SNA", "Snap-on Inc.", "Industrials", "momentum"),
  ("VRSK", "Verisk Analytics", "Industrials", "momentum"),
  ("EFX", "Equifax Inc.", "Industrials", "momentum"),
  ("AXON", "Axon Enterprise", "Industrials", "momentum"),
  ("GEV", "GE Vernova", "Industrials", "momentum"),
  ("PWR", "Quanta Services", "Industrials", "momentum"),
  ("URI", "United Rentals", "Industrials", "momentum"),
  ("WAB", "Wabtec Corp.", "Industrials", "momentum"),
  ("EXPD", "Expeditors International", "Industrials", "momentum"),
  ("CHRW", "C.H. Robinson Worldwide", "Industrials", "momentum"),
  ("UAL", "United Airlines Holdings", "Industrials", "momentum"),
  ("DAL", "Delta Air Lines", "Industrials", "momentum"),
  ("LUV", "Southwest Airlines", "Industrials", "momentum"),
  ("AAL", "American Airlines Group", "Industrials", "momentum"),
  ("MAS", "Masco Corp.", "Industrials", "momentum"),
  ("ALLE", "Allegion plc", "Industrials", "momentum"),
  ("J", "Jacobs Solutions", "Industrials", "momentum"),
  ("XOM", "Exxon Mobil Corp.", "Energy", "core"),
  ("CVX", "Chevron Corp.", "Energy", "core"),
  ("COP", "ConocoPhillips", "Energy", "core"),
  ("SLB", "SLB (Schlumberger)", "Energy", "momentum"),
  ("EOG", "EOG Resources", "Energy", "momentum"),
  ("MPC", "Marathon Petroleum", "Energy", "momentum"),
  ("PSX", "Phillips 66", "Energy", "momentum"),
  ("VLO", "Valero Energy", "Energy", "momentum"),
  ("OXY", "Occidental Petroleum", "Energy", "momentum"),
  ("WMB", "Williams Companies", "Energy", "momentum"),
  ("KMI", "Kinder Morgan", "Energy", "momentum"),
  ("OKE", "ONEOK Inc.", "Energy", "momentum"),
  ("HES", "Hess Corp.", "Energy", "momentum"),
  ("BKR", "Baker Hughes", "Energy", "momentum"),
  ("HAL", "Halliburton Co.", "Energy", "momentum"),
  ("DVN", "Devon Energy", "Energy", "momentum"),
  ("FANG", "Diamondback Energy", "Energy", "momentum"),
  ("EQT", "EQT Corp.", "Energy", "momentum"),
  ("TRGP", "Targa Resources", "Energy", "momentum"),
  ("CTRA", "Coterra Energy", "Energy", "momentum"),
  ("APA", "APA Corp.", "Energy", "momentum"),
  ("MRO", "Marathon Oil", "Energy", "momentum"),
  ("NEE", "NextEra Energy", "Utilities", "core"),
  ("SO", "Southern Co.", "Utilities", "core"),
  ("DUK", "Duke Energy", "Utilities", "momentum"),
  ("CEG", "Constellation Energy", "Utilities", "momentum"),
  ("AEP", "American Electric Power", "Utilities", "momentum"),
  ("SRE", "Sempra", "Utilities", "momentum"),
  ("D", "Dominion Energy", "Utilities", "momentum"),
  ("PCG", "PG&E Corp.", "Utilities", "momentum"),
  ("EXC", "Exelon Corp.", "Utilities", "momentum"),
  ("XEL", "Xcel Energy", "Utilities", "momentum"),
  ("ED", "Consolidated Edison", "Utilities", "momentum"),
  ("WEC", "WEC Energy Group", "Utilities", "momentum"),
  ("ES", "Eversource Energy", "Utilities", "momentum"),
  ("AEE", "Ameren Corp.", "Utilities", "momentum"),
  ("DTE", "DTE Energy", "Utilities", "momentum"),
  ("PPL", "PPL Corp.", "Utilities", "momentum"),
  ("FE", "FirstEnergy Corp.", "Utilities", "momentum"),
  ("ETR", "Entergy Corp.", "Utilities", "momentum"),
  ("AES", "AES Corp.", "Utilities", "momentum"),
  ("CNP", "CenterPoint Energy", "Utilities", "momentum"),
  ("CMS", "CMS Energy", "Utilities", "momentum"),
  ("NI", "NiSource Inc.", "Utilities", "momentum"),
  ("LNT", "Alliant Energy", "Utilities", "momentum"),
  ("EVRG", "Evergy Inc.", "Utilities", "momentum"),
  ("PNW", "Pinnacle West Capital", "Utilities", "momentum"),
  ("ATO", "Atmos Energy", "Utilities", "momentum"),
  ("NRG", "NRG Energy", "Utilities", "momentum"),
  ("VST", "Vistra Corp.", "Utilities", "momentum"),
  ("PLD", "Prologis Inc.", "Real Estate", "core"),
  ("AMT", "American Tower", "Real Estate", "core"),
  ("EQIX", "Equinix Inc.", "Real Estate", "momentum"),
  ("WELL", "Welltower Inc.", "Real Estate", "momentum"),
  ("SPG", "Simon Property Group", "Real Estate", "momentum"),
  ("PSA", "Public Storage", "Real Estate", "momentum"),
  ("O", "Realty Income Corp.", "Real Estate", "momentum"),
  ("DLR", "Digital Realty Trust", "Real Estate", "momentum"),
  ("CCI", "Crown Castle Inc.", "Real Estate", "momentum"),
  ("EXR", "Extra Space Storage", "Real Estate", "momentum"),
  ("AVB", "AvalonBay Communities", "Real Estate", "momentum"),
  ("EQR", "Equity Residential", "Real Estate", "momentum"),
  ("VTR", "Ventas Inc.", "Real Estate", "momentum"),
  ("IRM", "Iron Mountain", "Real Estate", "momentum"),
  ("SBAC", "SBA Communications", "Real Estate", "momentum"),
  ("ARE", "Alexandria Real Estate", "Real Estate", "momentum"),
  ("BXP", "BXP Inc.", "Real Estate", "momentum"),
  ("KIM", "Kimco Realty", "Real Estate", "momentum"),
  ("REG", "Regency Centers", "Real Estate", "momentum"),
  ("HST", "Host Hotels & Resorts", "Real Estate", "momentum"),
  ("MAA", "Mid-America Apartment", "Real Estate", "momentum"),
  ("UDR", "UDR Inc.", "Real Estate", "momentum"),
  ("CPT", "Camden Property Trust", "Real Estate", "momentum"),
  ("ESS", "Essex Property Trust", "Real Estate", "momentum"),
  ("INVH", "Invitation Homes", "Real Estate", "momentum"),
  ("DOC", "Healthpeak Properties", "Real Estate", "momentum"),
  ("VICI", "VICI Properties", "Real Estate", "momentum"),
  ("WY", "Weyerhaeuser Co.", "Real Estate", "momentum"),
  ("LIN", "Linde plc", "Materials", "core"),
  ("SHW", "Sherwin-Williams", "Materials", "momentum"),
  ("APD", "Air Products & Chemicals", "Materials", "momentum"),
  ("FCX", "Freeport-McMoRan", "Materials", "momentum"),
  ("ECL", "Ecolab Inc.", "Materials", "momentum"),
  ("NEM", "Newmont Corp.", "Materials", "momentum"),
  ("NUE", "Nucor Corp.", "Materials", "momentum"),
  ("DOW", "Dow Inc.", "Materials", "momentum"),
  ("PPG", "PPG Industries", "Materials", "momentum"),
  ("DD", "DuPont de Nemours", "Materials", "momentum"),
  ("VMC", "Vulcan Materials", "Materials", "momentum"),
  ("MLM", "Martin Marietta Materials", "Materials", "momentum"),
  ("IFF", "International Flavors & Fragrances", "Materials", "momentum"),
  ("LYB", "LyondellBasell", "Materials", "momentum"),
  ("STLD", "Steel Dynamics", "Materials", "momentum"),
  ("ALB", "Albemarle Corp.", "Materials", "momentum"),
  ("CE", "Celanese Corp.", "Materials", "momentum"),
  ("EMN", "Eastman Chemical", "Materials", "momentum"),
  ("PKG", "Packaging Corp. of America", "Materials", "momentum"),
  ("IP", "International Paper", "Materials", "momentum"),
  ("AMCR", "Amcor plc", "Materials", "momentum"),
  ("BALL", "Ball Corp.", "Materials", "momentum"),
  ("AVY", "Avery Dennison", "Materials", "momentum"),
  ("CF", "CF Industries Holdings", "Materials", "momentum"),
  ("MOS", "Mosaic Co.", "Materials", "momentum"),
  ("FMC", "FMC Corp.", "Materials", "momentum"),
  ("CIEN", "Ciena Corp.", "Technology", "momentum"),
  ("COHR", "Coherent Corp.", "Technology", "momentum"),
  ("JBL", "Jabil Inc.", "Technology", "momentum"),
  ("VSH", "Vishay Intertechnology", "Technology", "momentum"),
  ("ONTO", "Onto Innovation", "Technology", "momentum"),
  ("CAMT", "Camtek Ltd.", "Technology", "momentum"),
  ("FORM", "FormFactor Inc.", "Technology", "momentum"),
  ("ACLS", "Axcelis Technologies", "Technology", "momentum"),
  ("MKSI", "MKS Instruments", "Technology", "momentum"),
  ("NOVT", "Novanta Inc.", "Technology", "momentum"),
  ("SLAB", "Silicon Laboratories", "Technology", "momentum"),
  ("CRUS", "Cirrus Logic", "Technology", "momentum"),
  ("POWI", "Power Integrations", "Technology", "momentum"),
  ("DIOD", "Diodes Inc.", "Technology", "momentum"),
  ("MTSI", "MACOM Technology Solutions", "Technology", "momentum"),
  ("SITM", "SiTime Corp.", "Technology", "momentum"),
  ("ALGM", "Allegro MicroSystems", "Technology", "momentum"),
  ("SYNA", "Synaptics Inc.", "Technology", "momentum"),
  ("RMBS", "Rambus Inc.", "Technology", "momentum"),
  ("LSCC", "Lattice Semiconductor", "Technology", "momentum"),
  ("WOLF", "Wolfspeed Inc.", "Technology", "momentum"),
  ("AMKR", "Amkor Technology", "Technology", "momentum"),
  ("SANM", "Sanmina Corp.", "Technology", "momentum"),
  ("PLXS", "Plexus Corp.", "Technology", "momentum"),
  ("FN", "Fabrinet", "Technology", "momentum"),
  ("AAON", "AAON Inc.", "Industrials", "momentum"),
  ("EXLS", "ExlService Holdings", "Technology", "momentum"),
  ("GDDY", "GoDaddy Inc.", "Technology", "core"),
  ("OKTA", "Okta Inc.", "Technology", "momentum"),
  ("TWLO", "Twilio Inc.", "Technology", "momentum"),
  ("HUBS", "HubSpot Inc.", "Technology", "momentum"),
  ("DBX", "Dropbox Inc.", "Technology", "momentum"),
  ("BOX", "Box Inc.", "Technology", "momentum"),
  ("ZM", "Zoom Communications", "Technology", "momentum"),
  ("DOCU", "DocuSign Inc.", "Technology", "momentum"),
  ("NCNO", "nCino Inc.", "Technology", "momentum"),
  ("BILL", "BILL Holdings", "Technology", "momentum"),
  ("PCTY", "Paylocity Holding", "Technology", "momentum"),
  ("PAYC", "Paycom Software", "Technology", "momentum"),
  ("MANH", "Manhattan Associates", "Technology", "momentum"),
  ("TYL", "Tyler Technologies", "Technology", "momentum"),
  ("JKHY", "Jack Henry & Associates", "Technology", "core"),
  ("GWRE", "Guidewire Software", "Technology", "momentum"),
  ("BSY", "Bentley Systems", "Technology", "momentum"),
  ("PTC", "PTC Inc.", "Technology", "core"),
  ("AZPN", "Aspen Technology", "Technology", "momentum"),
  ("DT", "Dynatrace Inc.", "Technology", "momentum"),
  ("ESTC", "Elastic N.V.", "Technology", "momentum"),
  ("NET", "Cloudflare Inc.", "Technology", "core"),
  ("BMRN", "BioMarin Pharmaceutical", "Health Care", "core"),
  ("NBIX", "Neurocrine Biosciences", "Health Care", "core"),
  ("ALNY", "Alnylam Pharmaceuticals", "Health Care", "core"),
  ("UHS", "Universal Health Services", "Health Care", "core"),
  ("MOH", "Molina Healthcare", "Health Care", "core"),
  ("CNC", "Centene Corp.", "Health Care", "core"),
  ("SSNC", "SS&C Technologies", "Financials", "core"),
  ("BR", "Broadridge Financial Solutions", "Financials", "core"),
  ("TROW", "T. Rowe Price Group", "Financials", "core"),
  ("BEN", "Franklin Resources", "Financials", "core"),
  ("IBKR", "Interactive Brokers Group", "Financials", "core"),
  ("HEI", "HEICO Corp.", "Industrials", "core"),
  ("TDG", "TransDigm Group", "Industrials", "core"),
  ("TXT", "Textron Inc.", "Industrials", "core"),
  ("HWM", "Howmet Aerospace", "Industrials", "core"),
  ("ROL", "Rollins Inc.", "Industrials", "core"),
  ("BURL", "Burlington Stores", "Consumer Discretionary", "core"),
  ("FLUT", "Flutter Entertainment", "Consumer Discretionary", "core"),
  ("WYNN", "Wynn Resorts", "Consumer Discretionary", "core"),
  ("ABNB", "Airbnb Inc.", "Consumer Discretionary", "core"),
  ("DECK", "Deckers Outdoor", "Consumer Discretionary", "core"),
  ("USFD", "US Foods Holding", "Consumer Staples", "core"),
  ("MNST", "Monster Beverage", "Consumer Staples", "core"),
  ("KDP", "Keurig Dr Pepper", "Consumer Staples", "core"),
  ("ET", "Energy Transfer LP", "Energy", "core"),
  ("EPD", "Enterprise Products Partners", "Energy", "core"),
  ("MPLX", "MPLX LP", "Energy", "core"),
  ("AA", "Alcoa Corp.", "Materials", "core"),
  ("CCJ", "Cameco Corp.", "Materials", "core"),
  ("FRT", "Federal Realty Investment Trust", "Real Estate", "core"),
  ("SPOT", "Spotify Technology", "Communication Services", "core"),
  ("CACI", "CACI International", "Industrials", "core"),
  ("BAH", "Booz Allen Hamilton", "Industrials", "core"),
  ("MRVL", "Marvell Technology", "Technology", "core"),
  ("ARM", "Arm Holdings", "Technology", "core"),
]

RISK_FREE = 0.045    # approx. short-term Treasury yield, used for Sharpe

# --- Scoring engine options (change these, then re-run the backtest) ---
# RISK_ADJUSTED_MOMENTUM: divide each stock's momentum by its own volatility,
#  so a smooth 40% gain outranks a violent 40% gain. Off = raw momentum.
RISK_ADJUSTED_MOMENTUM = True

# QUALITY_WEIGHT: how much the quality factor influences the score.
#  Every stock stays in the analysis; weak fundamentals lower a rank
#  rather than removing a name. Set 0 to ignore quality entirely.
QUALITY_WEIGHT = 0.15     # influence of the quality factor (0 = ignore it)
EXCLUDE_ON_QUALITY_FAIL = False # True = drop failing names (shrinks the list)
QUALITY_MIN_ROE = 0.05     # reference point for the quality score
QUALITY_MAX_DEBT_EQUITY = 3.0 # reference point for the quality score

# Prices already downloaded this run, keyed by ticker. Lets the quality
# functions read them instead of making their own network calls.
_PRICE_CACHE = {}
TOP_N_CORE = 8     # names to allocate to in Approach 1
TOP_N_MOM = 6      # names to allocate to in Approach 2
CAP_CORE = 0.20     # max weight per name, Approach 1
CAP_MOM = 0.25     # max weight per name, Approach 2

# Approach 3 (real optimizer) settings
OPT_CANDIDATES = 20   # shortlist size fed to the optimizer (keeps it fast)
OPT_MAX_WEIGHT = 0.15  # max weight per single stock in the optimal portfolio

# OPT_METHOD picks how Approach 3 solves for weights:
#  "Max Sharpe"    = classic mean-variance (the original method)
#  "Hierarchical Risk Parity" = clusters correlated names, spreads risk across
#             them. FAR more stable between runs -- recommended.
#  "Black-Litterman"  = blends the market view with your confidence in the
#             shortlist's own momentum. Smooths extreme weights.
OPT_METHOD = "Hierarchical Risk Parity"

# Benchmark used for comparison and for the backtest
BENCHMARK_TICKER = "^GSPC"
BENCHMARK_NAME = "S&P 500"

# Backtest settings
BACKTEST_YEARS = 3   # how far back to test the rules
BACKTEST_TOP_N = 8   # names held at a time in the backtest

# --- Cost & tax assumptions (used by the backtest) ---
# These are the numbers that decide whether frequent rebalancing is worth it.
COST_BPS = 5.0     # round-trip cost per trade in basis points (spread + fees)
TAX_SHORT_TERM = 0.24  # tax on gains held under 1 year (US ordinary-income-ish)
TAX_LONG_TERM = 0.15  # tax on gains held over 1 year

# --- Turnover control ---
# HYSTERESIS_BAND: an existing holding stays until it falls below this rank.
#  e.g. 15 means "buy at top 8, but only sell once it drops past 15th".
#  This cuts churn dramatically with almost no loss of signal.
HYSTERESIS_BAND = 15
MIN_HOLD_MONTHS = 3   # do not sell a position before this many months

# --- Position sizing ---
# "Volatility-scaled" gives steadier results: smaller weights in jumpier names.
# "Equal weight" spreads the same dollar across every pick.
SIZING_METHOD = "Volatility-scaled"

st.set_page_config(page_title="My Stock Model", page_icon="chart", layout="wide")


# ----------------------------------------------------------------------------
# UNIVERSE BUILDERS
# ----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)  # cache the name list a full day
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
# VOLUME-SURGE FILTER (Option A -- a proxy, see the note at the top)
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
      raw = yf.download(batch, period="6mo", interval="1d",
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_quality(tickers):
    """Health screen. Uses only the price data already downloaded.

    The old version called yf.Ticker(t).info once per name, which meant up to
    1,000 separate HTTP requests and minutes of waiting. This version needs
    no network at all -- it reads the frame already in memory.

    Returns {ticker: True/False}. A name with too little history passes, so a
    data gap never silently shrinks the universe."""
    out = {}
    for t in tickers:
        try:
            px = _PRICE_CACHE.get(t)
            if px is None or len(px) < 60:
                out[t] = True
                continue
            # A simple health test from price alone: is it above its 6-month
            # midpoint, and has it avoided a catastrophic drawdown?
            mid = float(px.median())
            peak = float(px.max())
            last = float(px.iloc[-1])
            drawdown = (last / peak - 1.0) if peak > 0 else 0.0
            out[t] = (last >= mid * 0.85) and (drawdown > -0.60)
        except Exception:
            out[t] = True
    passed = sum(1 for v in out.values() if v)
    print(f"[quality] {passed}/{len(out)} names passed (price-based screen)")
    return out


def fetch_quality_scores(tickers):
    """Quality score 0-100 from price behaviour alone. No network calls.

    Components:
      trend     - position within the 6-month range
      drawdown  - how far below its own peak it sits
      stability - inverse of volatility
    A name with no usable history scores a neutral 50."""
    out = {}
    for t in tickers:
        try:
            px = _PRICE_CACHE.get(t)
            if px is None or len(px) < 60:
                out[t] = 50.0
                continue
            hi = float(px.max())
            lo = float(px.min())
            last = float(px.iloc[-1])
            span = (hi - lo) or 1.0
            pos = (last - lo) / span                      # 0..1 within range
            peak = hi
            dd = (last / peak - 1.0) if peak > 0 else 0.0  # <= 0
            vol = float(px.pct_change().std() * (252 ** 0.5)) or 0.01

            pos_s = max(0.0, min(100.0, pos * 100.0))
            dd_s = max(0.0, min(100.0, 100.0 + dd * 200.0))
            vol_s = max(0.0, min(100.0, 100.0 - vol * 150.0))
            out[t] = round(pos_s * 0.40 + dd_s * 0.35 + vol_s * 0.25, 2)
        except Exception:
            out[t] = 50.0
    return out


def fortune500_universe():
  """Fallback universe: the FORTUNE 500 large-caps already bundled in this
  file, tagged "core". Always available -- no network needed."""
  return [u for u in UNIVERSE]


@st.cache_data(ttl=86400, show_spinner=False)
def _yahoo_screener_ids():
    """Yahoo screener ids to pull. Gainers AND losers so the union is not
    directionally biased -- a stock that fell today still qualifies."""
    return [
        ("most_actives", 100),
        ("day_gainers", 100),
        ("day_losers", 100),
        ("undervalued_large_caps", 100),
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_screener_universe():
    """Union of Yahoo screener lists, de-duplicated, capped at YAHOO_TAKE."""
    found = []
    for sid, cnt in _yahoo_screener_ids():
        url = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
               f"?scrIds={sid}&count={cnt}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                payload = json.loads(r.read().decode("utf-8"))
            for res in payload.get("finance", {}).get("result", []):
                for q in res.get("quotes", []):
                    sym = q.get("symbol")
                    if sym:
                        found.append(sym)
        except Exception:
            continue
    seen, out = set(), []
    for t in found:
        if t not in seen:
            seen.add(t)
            out.append(t)
    out = out[:YAHOO_TAKE]
    print(f"[universe] yahoo screener union gave {len(out)} tickers")
    return out


def parse_pasted_tickers(raw):
    """Turn a pasted blob into clean tickers. Commas, spaces, newlines all work."""
    if not raw:
        return []
    parts = re.split(r"[,\s;|]+", str(raw))
    out, seen = [], set()
    for p in parts:
        t = p.strip().upper().replace(".", "-")
        if not t or t in seen:
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9\-]{0,5}", t):
            continue
        seen.add(t)
        out.append(t)
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def rank_remaining(pool, need=340):
    """Rank the remaining bundled names on three factors and return the best.

    Factors, each put on a comparable 0-100 percentile scale:
      growth    - the average POSITIVE 1-month return. Names that only fell
                  score zero on this factor rather than dragging the blend.
      liquidity - today's price times volume, so tradable names rank higher
      sentiment - the market timing regime. In a cautious regime the tilt
                  favours steadier names; in a favourable one it leans to
                  the stronger movers.

    Returns a list of (ticker, company, sector, bucket) tuples, best first."""
    if not pool:
        return []
    tickers = [u[0] for u in pool]
    if len(tickers) > need * 3:
        tickers = tickers[: need * 3]      # bound the work

    data = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            raw = yf.download(batch, period="2mo", interval="1d",
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            for t in batch:
                try:
                    px = raw[t]["Close"].dropna()
                    vol = raw[t]["Volume"].dropna()
                    if len(px) < 25:
                        continue
                    last = float(px.iloc[-1])
                    if last < MIN_PRICE:
                        continue
                    # average POSITIVE 1-month return, in percent
                    r1 = float(px.iloc[-1] / px.iloc[-21] - 1.0) * 100 if len(px) >= 21 else 0.0
                    growth = max(0.0, r1)
                    dv = last * float(vol.tail(5).mean())
                    data[t] = {"growth": growth, "liq": dv}
                except Exception:
                    continue
        except Exception:
            continue

    if not data:
        return pool[:need]

    g = pd.Series({t: v["growth"] for t, v in data.items()})
    l = pd.Series({t: v["liq"] for t, v in data.items()})
    gs = g.rank(pct=True) * 100 if len(g) > 1 else pd.Series(50.0, index=g.index)
    ls = l.rank(pct=True) * 100 if len(l) > 1 else pd.Series(50.0, index=l.index)

    # sentiment tilt: favourable regime favours movers, cautious favours steady
    tilt = 0.0
    try:
        _t = market_timing_signal()
        if _t is not None:
            tilt = (float(_t.get("score", 50)) - 50.0) / 50.0   # -1..+1
    except Exception:
        tilt = 0.0

    w_g, w_l, w_s = 0.40, 0.40, 0.20
    rows = []
    for t in data:
        sent = ls.get(t, 50.0) if tilt >= 0 else gs.get(t, 50.0)
        score = (w_g * gs.get(t, 50.0)) + (w_l * ls.get(t, 50.0)) + (w_s * sent)
        rows.append((t, score))
    rows.sort(key=lambda kv: -kv[1])

    order = [t for t, _ in rows[:need]]
    m = {u[0]: u for u in pool}
    out = [m[t] for t in order if t in m]
    print(f"[universe] ranked {len(data)} names on growth+liquidity+sentiment, "
          f"kept {len(out)} (regime tilt {tilt:+.2f})")
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def sp500_pool(bundled):
    """The pool tier 4 ranks: S&P 500 companies only.

    Built from two places, de-duplicated:
      1. the bundled list's "core" bucket, which marks S&P-class names
      2. the live Wikipedia S&P 500 table, when it is reachable

    If that pool is smaller than the slots tier 4 must fill, the whole
    bundled list is used instead so the universe still reaches its target.
    Returns (list_of_tuples, label)."""
    pool = [u for u in bundled if u[3] == "core"]
    label = "S&P500 bundled"

    try:
        live = broad_universe()          # includes the S&P 500 table
        if len(live) >= 100:
            have = {u[0] for u in pool}
            added = 0
            for u in live:
                if u[0] not in have:
                    pool.append(u)
                    have.add(u[0])
                    added += 1
            if added:
                label = f"S&P500 bundled+wiki (+{added})"
    except Exception:
        pass

    if len(pool) < 200:
        return bundled, "all bundled (S&P pool thin)"
    return pool, label


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_stock(ticker, company=None, horizon=None):
    """One signed percentage from four blended signals.

      momentum  - risk-adjusted 6-month and 1-month trend
      volume    - recent volume against that name's own 60-day average
      news      - keyword tone of recent headlines
      stats     - how far this name has typically moved in the leaning direction

    A weighted opinion, not a price prediction. It is stored and later
    scored against what actually happened."""
    horizon = int(horizon or FORECAST_HORIZON_DAYS)
    try:
        px = yf.Ticker(ticker).history(period="6mo")["Close"].dropna()
        if len(px) < 60:
            return None
        vol_s = None
        try:
            vol_s = yf.Ticker(ticker).history(period="6mo")["Volume"].dropna()
        except Exception:
            vol_s = None

        rets = px.pct_change().dropna()
        sd = float(rets.std()) or 0.01
        r6 = float(px.iloc[-1] / px.iloc[-126] - 1.0) if len(px) >= 127 else 0.0
        r1 = float(px.iloc[-1] / px.iloc[-21] - 1.0) if len(px) >= 22 else 0.0
        mom = float(np.tanh(0.6 * (r6 / (sd * 6.0)) + 0.4 * (r1 / sd)))

        volume = 0.0
        if vol_s is not None and len(vol_s) >= 60:
            recent = float(vol_s.iloc[-5:].mean())
            base = float(vol_s.iloc[-60:].mean())
            if base > 0:
                volume = float(np.tanh((recent / base - 1.0) * 1.5))

        news = 0.0
        try:
            heads = fetch_headlines(ticker, company or ticker, max_items=10)
            if heads:
                news = float(np.tanh(np.mean([score_headline(h) for h in heads]) / 2.0))
        except Exception:
            news = 0.0

        stats = 0.0
        if len(px) > horizon + 40:
            wr = (px.iloc[horizon:].values / px.iloc[:-horizon].values) - 1.0
            wr = wr[np.isfinite(wr)]
            if len(wr) >= 40:
                hi = float(np.percentile(wr, 95))
                lo = float(np.percentile(wr, 5))
                up = wr[wr >= hi]
                dn = wr[wr <= lo]
                lean = 0.40 * mom + 0.20 * volume + 0.20 * news
                stats = float(up.mean()) if (lean >= 0 and len(up)) else (float(dn.mean()) if len(dn) else 0.0)

        band = FORECAST_MAX_PCT / 2.0
        blended = (FORECAST_W_MOMENTUM * mom * band
                   + FORECAST_W_VOLUME * volume * band
                   + FORECAST_W_NEWS * news * band
                   + FORECAST_W_STATS * (stats * 100.0))
        blended = float(max(-FORECAST_MAX_PCT, min(FORECAST_MAX_PCT, blended)))
        if abs(blended) < 0.05:
            blended = 0.0

        return {"ticker": ticker, "spot": float(px.iloc[-1]), "forecast": round(blended, 1)}
    except Exception:
        return None


def load_forecast_state():
    """Read the stored forecast file. Never raises."""
    blank = {"generated": "", "rows": {}, "previous": {"date": "", "rows": {}}}
    if not os.path.exists(FORECAST_STATE):
        return blank
    try:
        with open(FORECAST_STATE, encoding="utf-8") as f:
            stt = json.load(f)
        stt.setdefault("rows", {})
        stt.setdefault("previous", {"date": "", "rows": {}})
        stt.setdefault("generated", "")
        return stt
    except Exception:
        return blank


def save_forecast_state(rows, horizon=None):
    """Store this press as the current forecast, keeping the last press so
    the comparison can measure the actual move between the two."""
    state = load_forecast_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_rows = {r["ticker"]: {"forecast": round(float(r["forecast"]), 2),
                              "spot": round(float(r["spot"]), 2)} for r in rows}
    out = {
        "generated": today,
        "horizon_days": int(horizon or FORECAST_HORIZON_DAYS),
        "rows": new_rows,
        "previous": {"date": state.get("generated", ""),
                     "rows": state.get("rows", {})},
    }
    with open(FORECAST_STATE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[forecast] stored {len(new_rows)} calls on {today}")
    return True


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_comparison():
    """Last press vs now: what it predicted against what the price did."""
    state = load_forecast_state()
    cur = state.get("rows", {})
    prev = (state.get("previous") or {}).get("rows", {}) or {}
    if not cur:
        return None, state

    tickers = sorted(set(list(cur.keys()) + list(prev.keys())))
    now_px = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            raw = yf.download(batch, period="5d", interval="1d",
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            for t in batch:
                try:
                    p = raw[t]["Close"].dropna()
                    if len(p):
                        now_px[t] = float(p.iloc[-1])
                except Exception:
                    continue
        except Exception:
            continue

    out = []
    for tk in tickers:
        c = cur.get(tk, {})
        p = prev.get(tk, {})
        row = {"Ticker": tk, "Forecast %": c.get("forecast", ""),
               "Previous forecast %": p.get("forecast", ""),
               "Price then": p.get("spot", ""),
               "Price now": round(now_px.get(tk, 0.0), 2) if tk in now_px else ""}
        try:
            then = float(p.get("spot", 0.0))
            now = float(now_px.get(tk, 0.0))
            if then > 0 and now > 0 and p.get("forecast", "") != "":
                act = (now / then - 1.0) * 100
                fc = float(p.get("forecast", 0.0))
                row["Actual %"] = round(act, 2)
                row["Delta %"] = round(act - fc, 2)
                row["Direction right"] = bool(fc == 0 or (fc > 0) == (act > 0))
            else:
                row["Actual %"] = row["Delta %"] = row["Direction right"] = ""
        except Exception:
            row["Actual %"] = row["Delta %"] = row["Direction right"] = ""
        out.append(row)
    return pd.DataFrame(out), state


def auto_universe(size=None, rank_by=None):
    """Compose the universe from four fixed tiers, de-duplicated.

    Tier 1  30 movers today   - biggest gainers and losers, unioned
    Tier 2  30 by liquidity   - heaviest dollar volume today
    Tier 3  ROBINHOOD_100     - pasted list (no public API exists)
    Tier 4  bundled FORTUNE 500 - fills the remainder, always available

    Each tier contributes a bounded number of names, so the work is small
    and the load is fast. The composition changes run to run as movers and
    volume shift, while the bundled tier guarantees the total."""
    size = size or AUTO_UNIVERSE_SIZE
    bundled = fortune500_universe()
    bundled_map = {u[0]: u for u in bundled}

    picked = []            # ordered, de-duplicated
    seen = set()
    notes = []

    def take(tickers, limit, label):
        added = 0
        for t in tickers:
            if added >= limit:
                break
            if not t or t in seen:
                continue
            seen.add(t)
            picked.append(bundled_map.get(t, (t, t, "-", "momentum")))
            added += 1
        notes.append(f"{label} {added}")

    # --- Tier 1: 30 movers today (gainers AND losers, so no bias) ---
    try:
        movers = yahoo_movers(60)
        take(movers, TIER_MOVERS, "movers")
    except Exception:
        notes.append("movers 0")

    # --- Tier 2: 30 by liquidity ---
    try:
        liquid = yahoo_liquid(60)
        take(liquid, TIER_LIQUID, "liquid")
    except Exception:
        notes.append("liquid 0")

    # --- Tier 3: Robinhood top 100 (pasted by hand) ---
    take(parse_pasted_tickers(ROBINHOOD_100), TIER_ROBINHOOD, "robinhood")

    # --- Tier 4: rank the remaining bundled pool, take the best ---
    # Scored on three factors, blended, then trimmed to fill the total:
    #   growth   - average positive 1-month return of the name
    #   liquidity- today's price x volume
    #   sentiment- the market timing regime, applied as a tilt
    sp_pool = sp500_pool(bundled)
    remaining = [u for u in sp_pool if u[0] not in seen]
    ranked = rank_remaining(remaining, need=size - len(picked))
    before = len(picked)
    for u in ranked:
        if len(picked) >= size:
            break
        if u[0] in seen:
            continue
        seen.add(u[0])
        picked.append(u)
    notes.append(f"ranked {len(picked) - before}")

    picked = picked[:size]
    source = " + ".join(notes)
    print(f"[universe] {len(picked)} names -- {source}")
    return picked, source


@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_movers(limit=60):
    """Today's biggest gainers and losers, unioned. Direction-neutral."""
    ids = [("day_gainers", limit), ("day_losers", limit)]
    out = []
    for sid, cnt in ids:
        url = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
               f"?scrIds={sid}&count={cnt}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                payload = json.loads(r.read().decode("utf-8"))
            for res in payload.get("finance", {}).get("result", []):
                for q in res.get("quotes", []):
                    sym = q.get("symbol")
                    px = q.get("regularMarketPrice") or 0
                    if sym and float(px or 0) >= MIN_PRICE:
                        out.append(sym)
        except Exception:
            continue
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    print(f"[universe] movers: {len(uniq)} candidates")
    return uniq


@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_liquid(limit=60):
    """Heaviest dollar volume today. Price and volume come in the screener
    payload itself, so no extra price fetch is needed to rank them."""
    url = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
           f"?scrIds=most_actives&count={limit}")
    scored = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read().decode("utf-8"))
        for res in payload.get("finance", {}).get("result", []):
            for q in res.get("quotes", []):
                sym = q.get("symbol")
                px = float(q.get("regularMarketPrice") or 0)
                vol = float(q.get("regularMarketVolume") or 0)
                if sym and px >= MIN_PRICE and vol > 0:
                    scored.append((sym, px * vol))
    except Exception:
        pass
    scored.sort(key=lambda kv: -kv[1])
    out = [t for t, _ in scored][:limit]
    print(f"[universe] liquid: {len(out)} candidates")
    return out

def load_data(custom_names=None):
  if custom_names:
    names = custom_names
  elif AUTO_UNIVERSE:
    names, _source = auto_universe()
    st.caption(f"Universe: **{len(names)} names**, unioned from: {_source}. "
               "Duplicates removed across sources, then ranked by dollar "
               "volume and trimmed. Rebuilt on every run.")
  elif SCREEN_MODE == "broad":
    # Start from the BUNDLED list so depth is guaranteed, then try to
    # add extra names from the live source. A failed fetch costs nothing.
    bundled = list(UNIVERSE)
    have = {u[0] for u in bundled}
    try:
      extra = [u for u in broad_universe() if u[0] not in have]
    except Exception:
      extra = []
    names = bundled + extra
    if extra:
      st.caption(f"Bundled list plus {len(extra)} extra names from the "
            f"live source: {len(names)} total.")
    else:
      st.caption(f"Screening the bundled list of {len(names)} names "
            "(live source unreachable; using bundled depth).")
    if USE_VOLUME_SURGE:
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

  # Fetch per ticker, in small batches. yf.download() can return a tuple
  # instead of a DataFrame on some builds, which made every lookup fail.
  # yf.Ticker().history() returns one clean frame per symbol, so a bad name
  # costs only that name.
  frames = {}
  _fetch_errors = []
  for i in range(0, len(tickers), 40):
    batch = tickers[i:i + 40]
    try:
      multi = yf.Tickers(" ".join(batch))
    except Exception as e:
      _fetch_errors.append(f"batch {i}: {type(e).__name__} {e}")
      continue

    for t in batch:
      px = None
      try:
        px = multi.tickers[t].history(period="6mo", interval="1d",
                                      auto_adjust=True)["Close"].dropna()
      except Exception:
        try:
          px = yf.Ticker(t).history(period="6mo", interval="1d",
                                   auto_adjust=True)["Close"].dropna()
        except Exception:
          px = None

      if px is None or len(px) < 20:
        continue
      try:
        if float(px.iloc[-1]) >= MIN_PRICE:
          frames[t] = px
      except Exception:
        continue

  if _fetch_errors:
    st.warning("Some price batches failed: " + "; ".join(_fetch_errors[:3]))
  print(f"[prices] usable history for {len(frames)} of {len(tickers)} names")

  # Hand the downloaded prices to the quality functions, so they need no
  # network calls of their own.
  global _PRICE_CACHE
  _PRICE_CACHE = dict(frames)

  rows, closes = [], {}
  for t, px in frames.items():
    closes[t] = px
    ma50 = px.rolling(50).mean().iloc[-1] if len(px) >= 50 else float(px.mean())
    ma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else float(px.mean())
    n = len(px)
    last = float(px.iloc[-1])
    # Guard every windowed lookup: a 6-month series is ~126 rows, so an
    # unguarded iloc[-126] walks off the end and raises IndexError.
    r_6mo = float(last / px.iloc[-126] - 1) if n >= 126 else float(last / px.iloc[0] - 1)
    r_1mo = float(last / px.iloc[-21] - 1) if n >= 21 else 0.0
    rows.append({
      "Ticker": t,
      "Company": meta.get(t, {}).get("company", t),
      "Sector": meta.get(t, {}).get("sector", ""),
      "Bucket": meta.get(t, {}).get("bucket", "momentum"),
      "Price": last,
      "Return 12mo": r_6mo,
      "Return 6mo": r_6mo,
      "Return 1mo": r_1mo,
      "Volatility": float(px.pct_change().std() * np.sqrt(252)) if n > 2 else 0.0,
      "Trend": int(last > ma50) + int(last > ma200) if n >= 200 else int(last > ma50),
    })

  df = pd.DataFrame(rows)
  if df.empty:
    return df, pd.DataFrame()

  # --- Quality: scored as a factor, optionally used as a hard filter ---
  _qmap = fetch_quality(df["Ticker"].tolist())
  _qscores = fetch_quality_scores(df["Ticker"].tolist())
  df["Quality Pass"] = df["Ticker"].map(lambda t: _qmap.get(t, True))
  df["Quality Score"] = df["Ticker"].map(lambda t: _qscores.get(t, 50.0))

  if EXCLUDE_ON_QUALITY_FAIL:
    _before = len(df)
    df = df[df["Quality Pass"]].copy()
    st.caption(f"Quality filter ON: {_before} screened, {len(df)} passed.")
    if df.empty:
      return df, pd.DataFrame()
  else:
    st.caption(f"Quality scored, not filtered: all {len(df)} names kept.")

  # --- Momentum score: risk-adjusted when the toggle is on ---
  if RISK_ADJUSTED_MOMENTUM:
    # Divide each return by the stock's own volatility, so a smooth gain
    # outranks a violent one of the same size. Volatility > 0 guaranteed
    # by the earlier filter; clip guards against tiny denominators.
    vol = df["Volatility"].clip(lower=0.05)
    mom_12 = df["Return 12mo"] / vol
    mom_6 = df["Return 6mo"] / vol
  else:
    mom_12 = df["Return 12mo"]
    mom_6 = df["Return 6mo"]

  df["Momentum Score"] = (mom_12.rank(pct=True) * 0.6 +
              mom_6.rank(pct=True) * 0.4) * 100
  df["Trend Score"] = df["Trend"] * 50
  df["Low-Vol Score"] = (1 - df["Volatility"].rank(pct=True)) * 100
  # Composite blends five factors. The four original weights are scaled
  # down so the total stays 100% when quality is switched on.
  _w = max(0.0, min(0.5, QUALITY_WEIGHT))
  _k = 1.0 - _w
  df["Composite Score"] = (df["Momentum Score"] * (0.35 * _k)
               + df["Trend Score"] * (0.25 * _k)
               + df["Low-Vol Score"] * (0.20 * _k)
               + df["Quality Score"] * _w
               + 50 * (0.20 * _k))

  prices = pd.DataFrame(closes).dropna(how="all")
  return df, prices


@st.cache_data(ttl=3600, show_spinner=False)
def benchmark_returns():
  """Trailing return of the benchmark index, for comparison."""
  try:
    px = yf.Ticker(BENCHMARK_TICKER).history(period="6mo")["Close"].dropna()
    if len(px) < 2:
      return None
    return {
      "1mo": float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) >= 22 else None,
      "6mo": float(px.iloc[-1] / px.iloc[-126] - 1) if len(px) >= 127 else None,
      "12mo": float(px.iloc[-1] / px.iloc[0] - 1),
    }
  except Exception:
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def size_positions(picks, df, method=None):
  """Turn a list of tickers into portfolio weights.

  Volatility-scaled: weight inversely to each name's annualized volatility,
  so a jumpy stock gets less money for the same expected return. This is the
  single cheapest way to make a portfolio steadier."""
  method = method or SIZING_METHOD
  if not picks:
    return {}
  if method != "Volatility-scaled":
    w = 1.0 / len(picks)
    return {t: w for t in picks}

  vols = df.set_index("Ticker").reindex(picks)["Volatility"].dropna()
  if vols.empty or (vols <= 0).all():
    w = 1.0 / len(picks)
    return {t: w for t in picks}

  inv = 1.0 / vols.clip(lower=0.01)
  inv = inv / inv.sum()
  return inv.to_dict()


def select_with_hysteresis(rank_order, held, band, min_hold, ages, top_n):
  """Choose holdings with turnover control.

  Keeps an existing name while it ranks inside the hysteresis band and has
  been held long enough; otherwise fills the remaining slots with the best
  available new names. Fewer trades means less cost and less tax."""
  order = list(rank_order.index)
  rank_of = {t: i + 1 for i, t in enumerate(order)}

  keep = []
  for t in held:
    r = rank_of.get(t, 9999)
    if r <= band and ages.get(t, 0) >= min_hold:
      keep.append(t)

  picks = keep[:top_n]
  for t in order:
    if len(picks) >= top_n:
      break
    if t not in picks:
      picks.append(t)
  if not picks:
    picks = order[:top_n]
  return picks[:top_n]


def run_backtest(prices, df, years, top_n, use_costs=True, use_turnover=True):
  """Monthly backtest with realistic frictions.

  Each month: rank by composite score, choose holdings (with turnover control
  if enabled), size the positions (volatility-scaled by default), rebalance,
  and pay trade costs plus short-term tax on realised gains.

  Limitations that remain: free Yahoo data, no correction for delisted names.
  Read the result as an indication, not proof."""
  try:
    import numpy as _np

    px = prices.dropna(axis=1, how="any")
    base = df.set_index("Ticker")["Composite Score"]
    cols = [c for c in px.columns if c in base.index]
    px = px[cols]
    if px.shape[1] < 2 or len(px) < 260:
      return None

    months = max(1, int(years * 12))
    px = px.iloc[-min(len(px), 21 * (months + 3)):]
    monthly = px.resample("ME").last().dropna(how="all")
    if len(monthly) < months + 1:
      return None

    rets = monthly.pct_change().dropna()
    if len(rets) < 3:
      return None

    rank_order = base.reindex(rets.columns).dropna().sort_values(ascending=False)

    # Benchmark
    try:
      bench_px = yf.Ticker(BENCHMARK_TICKER).history(period=f"{years + 1}y")["Close"]
      bench_m = bench_px.resample("ME").last().pct_change().dropna()
    except Exception:
      bench_m = None

    held, ages = [], {}
    weights = {}
    strat_curve, bench_curve = [1.0], [1.0]
    strat_m, bench_mm, turnover_m = [], [], []

    n_months = min(months, len(rets))
    for i in range(n_months):
      # --- choose holdings ---
      if use_turnover:
        picks = select_with_hysteresis(
          rank_order, held, HYSTERESIS_BAND, MIN_HOLD_MONTHS, ages, top_n)
      else:
        picks = list(rank_order.index[:top_n])

      # --- size positions ---
      new_w = size_positions(picks, df)

      # --- turnover: fraction of the book being replaced ---
      all_names = set(list(weights.keys()) + list(new_w.keys()))
      tv = sum(abs(new_w.get(t, 0.0) - weights.get(t, 0.0)) for t in all_names) / 2.0
      turnover_m.append(tv)

      # --- apply the month's returns to the OLD book ---
      r = rets.iloc[i]
      gross = float(sum(w * float(r.get(t, 0.0)) for t, w in weights.items()))
      if not weights:
        gross = float(r.reindex(picks).mean()) if picks else 0.0

      # --- costs and tax on the trades being made ---
      cost = (tv * (COST_BPS / 10000.0)) if (use_costs and weights) else 0.0
      tax = 0.0
      if use_costs and weights and gross > 0:
        # rough: gains realised on replaced weight are taxed short-term
        tax = tv * gross * TAX_SHORT_TERM

      net = gross - cost - tax
      strat_m.append(net)

      # --- benchmark ---
      b = float(bench_m.iloc[i]) if (bench_m is not None and i < len(bench_m)) else 0.0
      bench_mm.append(b)

      strat_curve.append(strat_curve[-1] * (1 + net))
      bench_curve.append(bench_curve[-1] * (1 + b))

      # --- roll the book forward ---
      weights = new_w
      held = picks
      for t in held:
        ages[t] = ages.get(t, 0) + 1

    s_arr, b_arr = _np.array(strat_m), _np.array(bench_mm)
    t_arr = _np.array(turnover_m) if turnover_m else _np.array([0.0])
    return {
      "strat_total": float(strat_curve[-1] - 1),
      "bench_total": float(bench_curve[-1] - 1),
      "diff": float((strat_curve[-1] - 1) - (bench_curve[-1] - 1)),
      "months": len(strat_m),
      "strat_curve": strat_curve,
      "bench_curve": bench_curve,
      "strat_mean_month": float(s_arr.mean()) if len(s_arr) else 0.0,
      "bench_mean_month": float(b_arr.mean()) if len(b_arr) else 0.0,
      "strat_best": float(s_arr.max()) if len(s_arr) else 0.0,
      "strat_worst": float(s_arr.min()) if len(s_arr) else 0.0,
      "bench_best": float(b_arr.max()) if len(b_arr) else 0.0,
      "bench_worst": float(b_arr.min()) if len(b_arr) else 0.0,
      "months_ahead": int((s_arr > b_arr).sum()),
      "months_behind": int((s_arr < b_arr).sum()),
      "avg_turnover": float(t_arr.mean()),
      "avg_cost": float((t_arr * (COST_BPS / 10000.0)).mean()),
      "cost_bps": COST_BPS,
      "sizing": SIZING_METHOD,
      "turnover_control": bool(use_turnover),
    }
  except Exception:
    return None

def diversified_picks(df):
  sub = df[df["Bucket"] == "core"].copy()
  if len(sub) < TOP_N_CORE:     # broad mode may have fewer core names
    sub = df.copy()
  sub = sub.nlargest(TOP_N_CORE, "Composite Score")
  w = sub["Composite Score"] / sub["Composite Score"].sum()
  sub["Weight"] = np.minimum(w, CAP_CORE)
  sub["Weight"] = sub["Weight"] / sub["Weight"].sum()  # renormalize after cap
  return sub.sort_values("Weight", ascending=False)


def momentum_picks(df):
  sub = df[df["Bucket"] == "momentum"].copy()
  if len(sub) < TOP_N_MOM:
    sub = df.copy()
  sub = sub.nlargest(TOP_N_MOM, "Return 12mo")
  sub["Weight"] = 1.0 / len(sub)
  return sub.sort_values("Return 12mo", ascending=False)


def markowitz_real(df, prices, method=None):
  """Solve for portfolio weights across a shortlist of the strongest names.

  Three methods, all from PyPortfolioOpt:
   Max Sharpe       - classic mean-variance optimization
   Hierarchical Risk Parity- clusters correlated names and spreads risk
                across the clusters. Much more stable.
   Black-Litterman     - starts from the market portfolio and tilts
                toward the shortlist, avoiding extreme bets.
  Falls back gracefully if the optimizer cannot solve."""
  method = method or OPT_METHOD
  try:
    from pypfopt import (EfficientFrontier, risk_models, expected_returns,
              HRPOpt, BlackLittermanModel)
  except ImportError:
    return None, "PyPortfolioOpt is not installed."

  # Shortlist: top composite names across the whole universe
  shortlist = df.nlargest(OPT_CANDIDATES, "Composite Score")["Ticker"].tolist()
  sub = prices[[t for t in shortlist if t in prices.columns]].dropna()

  if sub.shape[1] < 2 or len(sub) < 60:
    return None, "Not enough clean price history to optimize."

  mu = expected_returns.mean_historical_return(sub)
  S = risk_models.sample_cov(sub)

  try:
    if method == "Hierarchical Risk Parity":
      opt = HRPOpt(returns=sub.pct_change().dropna())
      opt.optimize()
      clean = opt.clean_weights()
      perf = opt.portfolio_performance(risk_free_rate=RISK_FREE)
    elif method == "Black-Litterman":
      # Use each name's trailing momentum as the view, modest confidence
      rets = sub.pct_change().dropna()
      mkt_prior = pd.Series(1.0 / sub.shape[1], index=sub.columns)
      views = df.set_index("Ticker").reindex(sub.columns)["Return 12mo"]
      bl = BlackLittermanModel(S, pi="market", market_caps=None,
                   risk_aversion=1.0,
                   absolute_views=views.dropna().to_dict(),
                   omega="idzorek")
      bl_ret = bl.bl_returns()
      ef = EfficientFrontier(bl_ret, S, weight_bounds=(0, OPT_MAX_WEIGHT))
      ef.max_sharpe(risk_free_rate=RISK_FREE)
      clean = ef.clean_weights()
      perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)
    else:
      ef = EfficientFrontier(mu, S, weight_bounds=(0, OPT_MAX_WEIGHT))
      ef.max_sharpe(risk_free_rate=RISK_FREE)
      clean = ef.clean_weights()
      perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)
  except Exception as e:
    # Never fail the whole app on an optimizer quirk -- fall back to Max Sharpe
    try:
      ef = EfficientFrontier(mu, S, weight_bounds=(0, OPT_MAX_WEIGHT))
      ef.max_sharpe(risk_free_rate=RISK_FREE)
      clean = ef.clean_weights()
      perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)
      method = f"Max Sharpe (fell back: {type(e).__name__})"
    except Exception as e2:
      return None, f"Optimizer could not solve: {e2}"

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
  return out, {"return": exp_ret, "vol": exp_vol, "sharpe": sharpe,
         "method": method}


# ----------------------------------------------------------------------------
# MARKET TIMING rules-based signal, not a prediction
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
    vix_label, vix_note = "Low / Complacent", "Calm markets historically can precede surprises either way."
  elif vix < 20:
    vix_label, vix_note = "Normal", "Typical volatility range."
  elif vix < 30:
    vix_label, vix_note = "Elevated", "Markets pricing in real uncertainty."
  else:
    vix_label, vix_note = "High / Fear", "Historically often (not always) followed by a recovery but can persist or worsen."

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
    regime, guidance = "Neutral", "No strong signal either way sticking to your regular schedule is reasonable."
  else:
    regime, guidance = "Cautious", "Elevated fear and/or a weak trend. Some investors stay the course anyway (timing the market is notoriously hard); others reduce size this month. Your call, not the model's."

  return {
    "vix": vix, "vix_label": vix_label, "vix_note": vix_note,
    "spx_now": spx_now, "spx_ma50": spx_ma50, "spx_ma200": spx_ma200,
    "trend_label": trend_label, "score": score,
    "regime": regime, "guidance": guidance,
  }


# ----------------------------------------------------------------------------
# NEWS SENTIMENT free headline keyword scoring per stock, not NLP magic
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
  This is keyword counting, not language understanding crude by design,
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
st.title("My US Stock Model")
_cap = "S&P 1500 (broad screen)" if SCREEN_MODE == "broad" else "curated watchlist"
st.caption(f"Screening: **{_cap}**. Three approaches, ranked from live market data. "
      f"Prices refresh each time this page loads. "
      f"Switch modes with SCREEN_MODE at the top of the file. "
      f"{market_session_note()}")

with st.expander(" Which stocks should it screen?", expanded=False):
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
  st.caption(f"Volume-surge filter is {_surge} keeps the top "
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

with st.spinner("Screening the universe this can take a minute or two in broad mode..."):
  df, prices = load_data(_custom_names)

# Show the funnel so the counts are never a mystery
_scored = len(df)
_funnel = (f"**Screening funnel:** {len(UNIVERSE)} names in the bundled list, "
      f"{_scored} scored with usable 1-year history and a price above "
      f"${MIN_PRICE:.0f}"
      + (f", narrowed to the top {TOP_VOLUME_N} by volume surge"
       if USE_VOLUME_SURGE and not _custom_names else "")
      + ".")
st.caption(_funnel)

if df.empty:
  st.error("Could not load price data right now. "
       "Yahoo's free feed may be rate-limiting try again in a minute.")
  st.stop()

# ---- headline row ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stocks screened", len(df))
c2.metric("Universe avg. 12mo return", f"{df['Return 12mo'].mean():.1%}")
c3.metric("Universe avg. volatility", f"{df['Volatility'].mean():.1%}")
_bench = benchmark_returns()
if _bench:
  c4.metric(f"{BENCHMARK_NAME} 12mo", f"{_bench['12mo']:.1%}",
       f"picks avg {df['Return 12mo'].mean() - _bench['12mo']:+.1%} vs index")
else:
  c4.metric(f"{BENCHMARK_NAME} 12mo", "n/a")
c5.metric("Last updated (ET)", et_stamp())

d1, d2, d3, d4, d5, d6, d7, d8 = st.tabs([
  "Approach 1 - Diversified",
  "Approach 2 - Momentum",
  "Approach 3 - Optimizer",
  "Market Timing & News",
  "Summary - What to Buy",
  "Backtest",
  "Forecast",
  "How to read this",
])

# ---------------- Approach 1 ----------------
with d1:
  st.subheader("Diversified large-cap lower risk, steadier ride")
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
  x3.metric("Typical monthly swing", f"{avg_v/np.sqrt(12):.1%}")

# ---------------- Approach 2 ----------------
with d2:
  st.subheader("Momentum / speculative higher risk, much bigger swings")
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
  x3.metric("Typical monthly swing", f"{avg_v/np.sqrt(12):.1%}")

# ---------------- Approach 3 ----------------
with d3:
  st.subheader("Approach 3 portfolio optimizer")
  st.write("Solves for the weight mix with the best return per unit of risk. The method below changes how it does that.")

  _method = st.radio(
    "Optimizer method",
    ["Hierarchical Risk Parity", "Black-Litterman", "Max Sharpe"],
    index=["Hierarchical Risk Parity", "Black-Litterman", "Max Sharpe"].index(OPT_METHOD)
    if OPT_METHOD in ["Hierarchical Risk Parity", "Black-Litterman", "Max Sharpe"] else 0,
    horizontal=True)
  st.caption({
    "Hierarchical Risk Parity": "Clusters names that move together and spreads risk across the clusters. Most stable between visits recommended.",
    "Black-Litterman": "Starts from the market portfolio and tilts toward this shortlist, avoiding extreme single-stock bets.",
    "Max Sharpe": "Classic mean-variance optimization. Highest theoretical return per risk, but the least stable weights.",
  }[_method])

  with st.spinner("Running the optimizer..."):
    out, info = markowitz_real(df, prices, method=_method)

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

    st.caption(f"Method used: **{info.get('method', _method)}**. The optimizer "
          f"considers the {OPT_CANDIDATES} strongest names by composite "
          f"score, caps any single holding at {OPT_MAX_WEIGHT:.0%}, and uses a "
          f"{RISK_FREE:.1%} risk-free rate. All editable at the top of the file.")

# ---------------- Market Timing & News Sentiment ----------------
with d4:
  st.subheader("Market timing & news sentiment signals, not predictions")
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
      st.success(f"**{timing['regime']}** {timing['guidance']}")
    elif timing["regime"] == "Neutral":
      st.info(f"**{timing['regime']}** {timing['guidance']}")
    else:
      st.warning(f"**{timing['regime']}** {timing['guidance']}")

    st.write("**What produced that score:**")
    st.dataframe(pd.DataFrame({
      "Indicator": ["Read at (ET)", "VIX level", "VIX read", "S&P 500 vs 50-day avg",
             "S&P 500 vs 200-day avg"],
      "Value": [et_stamp(), f"{timing['vix']:.1f}", timing["vix_label"],
           f"{timing['spx_now']:,.0f} vs {timing['spx_ma50']:,.0f}",
           f"{timing['spx_now']:,.0f} vs {timing['spx_ma200']:,.0f}"],
    }), use_container_width=True, hide_index=True)

    st.caption(f"{timing['vix_note']} Score = 50 to start, +20 for an uptrend "
          f"(20 otherwise), +15 for VIX under 20 (15 above 30). "
          f"Thresholds are visible in the code and you can change them.")

  st.divider()

  # ---- Part B: per-stock news sentiment ----
  st.markdown("### News tone on your watchlist")
  st.write("Counts positive vs negative words across the most recent free headlines "
       "for each stock. A negative score means the headlines skew bearish right "
       "now not that the stock will fall.")

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

    st.caption("Scores run from roughly 3 (heavy negative tone) to +3 (heavy "
          "positive). This is keyword matching, not language understanding "
          "treat it as a quick read of headline mood, and check the sample "
          "headline yourself before acting on any single row.")

# ---------------- Summary & Buy Plan ----------------
with d5:
  st.subheader("What to buy, and is now a good time")

  # ---- Part A: is today a good time ----
  timing = market_timing_signal()
  if timing is None:
    st.warning("Couldn't read the market indicators right now timing score unavailable.")
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
      st.success("Conditions look favourable on this rule investing the "
            "full monthly amount is reasonable.")
    elif timing_regime == "Neutral":
      st.info("No strong signal either way sticking to your usual schedule "
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
    ["Approach 1 Diversified", "Approach 2 Momentum",
     "Approach 3 Markowitz", "Blended (80% Diversified / 20% Momentum)"],
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
        "aside for next month that keeps you investing without ignoring the "
        "signal.")

# ---------------- Guide ----------------
# ---------------- Backtest ----------------
with d6:
  st.subheader("Backtest would these rules have worked?")
  st.warning(
    "Read this first: a backtest on free data is an INDICATION, not proof. "
    "Yahoo history has gaps and cannot fully correct for delisted stocks, so "
    "results are skewed optimistic. Treat this as a sanity check on the rules, "
    "not as evidence you will get these returns again."
  )

  st.markdown("**What this tests**")
  st.write(
    "Each month, rank the universe by composite score, buy the top names "
    f"equal-weighted, hold one month, then re-rank. Compare that against "
    f"simply buying and holding the {BENCHMARK_NAME} over the same window. "
    "No costs, no slippage, no taxes so the real-world edge would be smaller."
  )

  with st.spinner(f"Running a {BACKTEST_YEARS}-year backtest..."):
    bt = run_backtest(prices, df, BACKTEST_YEARS, BACKTEST_TOP_N)

  if bt is None:
    st.info("Not enough clean price history to run a backtest right now.")
  else:
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Strategy total return", f"{bt['strat_total']:+.1%}")
    z2.metric(f"{BENCHMARK_NAME} total return", f"{bt['bench_total']:+.1%}")
    z3.metric("Difference", f"{bt['diff']:+.1%}",
         "strategy ahead" if bt['diff'] > 0 else "index ahead")
    z4.metric("Months tested", bt["months"])

    st.markdown("**Month-by-month equity curve (growth of 1.0)**")
    _eq = pd.DataFrame({
      "Strategy": bt["strat_curve"],
      f"{BENCHMARK_NAME}": bt["bench_curve"],
    })
    st.line_chart(_eq, height=320)

    st.markdown("**Summary statistics**")
    st.dataframe(pd.DataFrame({
      "Measure": ["Average monthly return", "Best month", "Worst month",
            "Months ahead of index", "Months behind index"],
      "Strategy": [
        f"{bt['strat_mean_month']:+.2%}",
        f"{bt['strat_best']:+.2%}",
        f"{bt['strat_worst']:+.2%}",
        f"{bt['months_ahead']} of {bt['months']}",
        f"{bt['months_behind']} of {bt['months']}",
      ],
      f"{BENCHMARK_NAME}": [
        f"{bt['bench_mean_month']:+.2%}",
        f"{bt['bench_best']:+.2%}",
        f"{bt['bench_worst']:+.2%}",
        "", "",
      ],
    }), use_container_width=True, hide_index=True)

    st.markdown("**Frictions applied**")
    f1, f2, f3 = st.columns(3)
    f1.metric("Average monthly turnover", f"{bt['avg_turnover']:.1%}")
    f2.metric("Cost per trade", f"{bt['cost_bps']:.1f} bps")
    f3.metric("Avg. monthly cost", f"{bt['avg_cost']:.3%}")
    st.caption(
      f"Method: monthly rebalance into the top {BACKTEST_TOP_N} names by "
      f"composite score, {SIZING_METHOD.lower()} sizing, over "
      f"{BACKTEST_YEARS} years. Trade costs of {COST_BPS:.0f} bps are "
      f"deducted, and realised gains are taxed at "
      f"{TAX_SHORT_TERM:.0%} short-term. Turnover control is "
      f"{'ON' if bt['turnover_control'] else 'OFF'} holdings stay "
      f"until they fall past rank {HYSTERESIS_BAND} and have been held at "
      f"least {MIN_HOLD_MONTHS} months."
    )
    st.info(
      "Turnover is the number to watch. If average monthly turnover is high, "
      "the strategy is trading a lot, and the costs above will eat into any "
      "edge. Try raising HYSTERESIS_BAND in the settings block and see "
      "whether the net result improves."
    )


# ---------------- Forecast ----------------
with d7:
  st.subheader("Forecast - what the range looks like")
  st.warning(
    "A forecast is a weighted opinion from four signals: recent momentum, "
    "unusual volume, the tone of recent news, and a statistical prior. A plus "
    "means the signals lean up, a minus means down. It is not a price "
    "prediction, and each press is measured against what actually happened."
  )

  _horizon = st.number_input("Horizon (days)", min_value=5, max_value=180,
                value=FORECAST_HORIZON_DAYS, step=5)

  st.markdown("**Which approach supplies the picks?**")
  _choice = st.radio(
    "Approach",
    ["Approach 1 - Diversified", "Approach 2 - Momentum", "Approach 3 - Optimizer"],
    horizontal=True,
    label_visibility="collapsed",
  )

  if _choice.startswith("Approach 1"):
    _base = diversified_picks(df)
    _blurb = "the top diversified large-caps"
  elif _choice.startswith("Approach 2"):
    _base = momentum_picks(df)
    _blurb = "the top momentum names"
  else:
    _opt, _ = markowitz_real(df, prices)
    _base = _opt if _opt is not None else diversified_picks(df)
    _blurb = "the optimizer's chosen holdings"

  _tickers = list(_base["Ticker"])[:12]
  st.caption(f"Forecasting {_blurb} - {len(_tickers)} names, over "
        f"{int(_horizon)} days ahead, blended from four signals.")

  if st.button("Run the forecast", type="primary"):
    _rows, _bar = [], st.progress(0.0)
    for _i, _t in enumerate(_tickers):
      _co = _t
      try:
        _co = str(_base.loc[_base["Ticker"] == _t, "Company"].iloc[0])
      except Exception:
        pass
      _res = forecast_stock(_t, company=_co, horizon=int(_horizon))
      if _res:
        _rows.append(_res)
      _bar.progress((_i + 1) / max(1, len(_tickers)))

    if not _rows:
      st.info("Could not build the forecast right now.")
    else:
      _wrote = save_forecast_state(_rows, horizon=int(_horizon))
      st.session_state["fc_rows"] = _rows
      if _wrote:
        st.success("Forecast refreshed and stored. This is now the "
              "value the comparison will measure against next time.")

  # ---- The stored forecast (from your last press) ----
  _saved = load_forecast_state()
  _has_saved = bool(_saved.get("rows"))

  if _has_saved:
    _when = _saved.get("generated", "")
    _prev_when = (_saved.get("previous", {}) or {}).get("date", "") or "none"
    st.markdown(f"**Forecast stored on {_when}** "
          f"(previous press: {_prev_when})")
    st.caption(f"Each value predicts the move over the next "
          f"{int(_saved.get('horizon_days', FORECAST_HORIZON_DAYS))} days, "
          f"measured from the price when the button was pressed.")

    _rows_saved = [{"Ticker": k,
            "Forecast %": v.get("forecast", 0.0),
            "Price at press": v.get("spot", 0.0)}
            for k, v in _saved["rows"].items()]
    _sf = pd.DataFrame(_rows_saved)

    def _signed(v):
      try:
        v = float(v)
      except Exception:
        return ""
      return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"

    _sf["Forecast"] = _sf["Forecast %"].map(_signed)
    st.dataframe(_sf[["Ticker", "Price at press", "Forecast"]].style.format(
      {"Price at press": "${:,.2f}"}), use_container_width=True,
      hide_index=True)
  else:
    st.info("No stored forecast yet. Press the button to make and store one.")

  st.divider()
  st.markdown("**Forecast vs what actually happened**")
  st.caption("Measured between your last press and now, against what that "
        "press predicted. Press the button again to start a new cycle.")
  _cmp, _st = forecast_comparison()
  if _cmp is None or _cmp.empty:
    st.info("Nothing to compare yet. Press the button once, then come back "
        "after some time has passed - the actual move and the "
        "difference appear here.")
  else:
    _done = _cmp[_cmp["Actual %"].astype(str).str.strip().ne("")]
    if _done.empty:
      _pd = (_st.get("previous", {}) or {}).get("date", "")
      if not _pd:
        st.info("This is the first press, so there is no earlier "
            "forecast to compare against yet. Press again later "
            "and the comparison appears.")
      else:
        st.info(f"Comparing against the press on {_pd}. Values appear "
            "as prices move.")
    else:
      try:
        _hr = _done["Direction right"].astype(str).str.lower()
        _hr = _hr.isin(["true", "1"]).mean() * 100
        _mae = pd.to_numeric(_done["Delta %"], errors="coerce").abs().mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Names compared", len(_done))
        m2.metric("Direction right", f"{_hr:.0f}%")
        m3.metric("Average miss", f"{_mae:.1f} pts")
        st.caption("Direction right is whether the sign matched - near "
              "50% is a coin toss. Average miss is how far the size "
              "was off, in percentage points.")
      except Exception:
        pass
      st.dataframe(_done, use_container_width=True, hide_index=True)
      st.caption("Delta is actual minus forecast. Pressing the button "
            "again starts a fresh cycle.")

with d8:
  st.subheader("How to read this")
  st.markdown("""
**The three approaches answer different questions.**

- **Approach 1 Diversified** asks *"which healthy large-caps are trending up, "
 "and how do I spread the money so one bad name can't hurt me?"* Lower volatility, "
 "smaller month-to-month swings, historically a smoother line.
- **Approach 2 Momentum** asks *"what has run the hardest over the past year?"* "
 "It chases strength. That means occasional spectacular years and occasional "
 "savage losses both are normal here, not a malfunction.
- **Approach 3 Markowitz** asks *"what combination of these stocks gives the most "
 "return for the risk I'm accepting?"* It looks at how every pair of stocks moves "
 "together (their covariance), not just how each one performed alone.

**Plain-language notes.**

- *Volatility* is how much a stock bounces around. Higher means wider swings, both ways.
- *Typical monthly swing* converts annual volatility into a rough monthly figure.
 Real months land above and below it it is a yardstick, not a promise.
- *Sharpe ratio* is return divided by risk. Higher is better for the same return.
- *Covariance* is just "do these two move together?" Two stocks that rise and fall
 at different times can be safer combined than either one alone.

**Costs, tax and turnover (the part that decides real returns).**

- *Turnover* is how much of the portfolio gets replaced each month. High turnover
 means more trades, more spread paid, and more taxable events.
- *Cost per trade* is your round-trip spread plus any fees, in basis points.
 Five bps is a reasonable retail assumption for liquid large-caps.
- *Short-term tax* applies to gains on anything held under a year. In a taxable
 account this is the biggest hidden drag on a frequently-rebalanced strategy.
- *Volatility-scaled sizing* gives steadier results than equal weight, because
 jumpier names take smaller positions for the same expected return.

**What this app does not do.**

- It does not predict the future. It ranks and weights what has already happened.
- It does not average 20% a month. Nothing does the S&P 500 has averaged roughly
 10% per *year* over the long run.
- It refreshes when you open it, not on a timer.
""")

st.divider()
st.caption("Decision-support tool, not financial advice. "
      "Past performance does not predict future returns.")


