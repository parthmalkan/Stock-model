"""
dashboard.py ÃÂ¢ÃÂÃÂ Personal US stock screening app (free, Streamlit).

Approach 3 uses REAL Markowitz optimization (PyPortfolioOpt) ÃÂ¢ÃÂÃÂ a full
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

IMPORTANT ÃÂ¢ÃÂÃÂ what Tab 4 actually is:
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
# SETTINGS ÃÂ¢ÃÂÃÂ change these if you like
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
        return "Weekend ÃÂ¢ÃÂÃÂ US market closed; prices are the last close."
    mins = now.hour * 60 + now.minute
    if mins < 9 * 60 + 30:
        return "Pre-market ÃÂ¢ÃÂÃÂ regular session opens 9:30am ET."
    if mins <= 16 * 60:
        return "US market open (regular session, 9:30am-4:00pm ET)."
    return "After hours ÃÂ¢ÃÂÃÂ regular session closed at 4:00pm ET."

MIN_PRICE = 5.0            # skip penny stocks / near-zero names
MAX_BROAD_NAMES = 1500     # hard cap so the free host isn't overwhelmed

# ----------------------------------------------------------------------------
# BUNDLED STOCK LIST  (self-contained -- no network fetch, cannot fail)
# ----------------------------------------------------------------------------
# ~416 large/mid-cap US names across all 11 GICS sectors. Embedded here on
# purpose: the earlier version fetched this list from Wikipedia at runtime,
# and on the free host that fetch returned 0 names, silently falling back to
# 24. A bundled list cannot fail, so the screen always has real depth.
#
# Fields: (ticker, company, sector, bucket)
#   bucket core     = mega/large-cap names, used by Approach 1 (diversified)
#   bucket momentum = everything else, used by Approach 2 (higher swing)
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
    ("CFLT", "Confluent Inc.", "Technology", "momentum"),
    ("MDB", "MongoDB Inc.", "Technology", "momentum"),
    ("NET", "Cloudflare Inc.", "Technology", "core"),
    ("S", "SentinelOne Inc.", "Technology", "momentum"),
    ("TENB", "Tenable Holdings", "Technology", "momentum"),
    ("QLYS", "Qualys Inc.", "Technology", "momentum"),
    ("VRNS", "Varonis Systems", "Technology", "momentum"),
    ("CYBR", "CyberArk Software", "Technology", "momentum"),
    ("HIMS", "Hims & Hers Health", "Health Care", "momentum"),
    ("VKTX", "Viking Therapeutics", "Health Care", "momentum"),
    ("CRSP", "CRISPR Therapeutics", "Health Care", "momentum"),
    ("NTLA", "Intellia Therapeutics", "Health Care", "momentum"),
    ("BEAM", "Beam Therapeutics", "Health Care", "momentum"),
    ("SRPT", "Sarepta Therapeutics", "Health Care", "momentum"),
    ("BMRN", "BioMarin Pharmaceutical", "Health Care", "core"),
    ("NBIX", "Neurocrine Biosciences", "Health Care", "core"),
    ("EXEL", "Exelixis Inc.", "Health Care", "momentum"),
    ("JAZZ", "Jazz Pharmaceuticals", "Health Care", "momentum"),
    ("RARE", "Ultragenyx Pharmaceutical", "Health Care", "momentum"),
    ("IONS", "Ionis Pharmaceuticals", "Health Care", "momentum"),
    ("ARWR", "Arrowhead Pharmaceuticals", "Health Care", "momentum"),
    ("ALNY", "Alnylam Pharmaceuticals", "Health Care", "core"),
    ("UTHR", "United Therapeutics", "Health Care", "momentum"),
    ("AMED", "Amedisys Inc.", "Health Care", "momentum"),
    ("CHE", "Chemed Corp.", "Health Care", "momentum"),
    ("ENSG", "Ensign Group", "Health Care", "momentum"),
    ("ACHC", "Acadia Healthcare", "Health Care", "momentum"),
    ("DVA", "DaVita Inc.", "Health Care", "momentum"),
    ("THC", "Tenet Healthcare", "Health Care", "momentum"),
    ("UHS", "Universal Health Services", "Health Care", "core"),
    ("MOH", "Molina Healthcare", "Health Care", "core"),
    ("CNC", "Centene Corp.", "Health Care", "core"),
    ("GMED", "Globus Medical", "Health Care", "momentum"),
    ("PEN", "Penumbra Inc.", "Health Care", "momentum"),
    ("NVCR", "NovoCure Ltd.", "Health Care", "momentum"),
    ("HAE", "Haemonetics Corp.", "Health Care", "momentum"),
    ("TFX", "Teleflex Inc.", "Health Care", "momentum"),
    ("BRKR", "Bruker Corp.", "Health Care", "momentum"),
    ("CRL", "Charles River Laboratories", "Health Care", "momentum"),
    ("MEDP", "Medpace Holdings", "Health Care", "momentum"),
    ("DOCS", "Doximity Inc.", "Health Care", "momentum"),
    ("RCM", "R1 RCM Inc.", "Health Care", "momentum"),
    ("OMCL", "Omnicell Inc.", "Health Care", "momentum"),
    ("MASI", "Masimo Corp.", "Health Care", "momentum"),
    ("ICUI", "ICU Medical", "Health Care", "momentum"),
    ("PRGO", "Perrigo Co.", "Health Care", "momentum"),
    ("SOFI", "SoFi Technologies", "Financials", "momentum"),
    ("AFRM", "Affirm Holdings", "Financials", "momentum"),
    ("UPST", "Upstart Holdings", "Financials", "momentum"),
    ("TOST", "Toast Inc.", "Financials", "momentum"),
    ("HQY", "HealthEquity Inc.", "Financials", "momentum"),
    ("EEFT", "Euronet Worldwide", "Financials", "momentum"),
    ("WEX", "WEX Inc.", "Financials", "momentum"),
    ("SSNC", "SS&C Technologies", "Financials", "core"),
    ("BR", "Broadridge Financial Solutions", "Financials", "core"),
    ("SEIC", "SEI Investments", "Financials", "momentum"),
    ("TROW", "T. Rowe Price Group", "Financials", "core"),
    ("BEN", "Franklin Resources", "Financials", "core"),
    ("IVZ", "Invesco Ltd.", "Financials", "momentum"),
    ("AMG", "Affiliated Managers Group", "Financials", "momentum"),
    ("VOYA", "Voya Financial", "Financials", "momentum"),
    ("LNC", "Lincoln National", "Financials", "momentum"),
    ("GL", "Globe Life Inc.", "Financials", "momentum"),
    ("UNM", "Unum Group", "Financials", "momentum"),
    ("AIZ", "Assurant Inc.", "Financials", "momentum"),
    ("ERIE", "Erie Indemnity", "Financials", "momentum"),
    ("KNSL", "Kinsale Capital Group", "Financials", "momentum"),
    ("RLI", "RLI Corp.", "Financials", "momentum"),
    ("SIGI", "Selective Insurance", "Financials", "momentum"),
    ("THG", "Hanover Insurance Group", "Financials", "momentum"),
    ("AFG", "American Financial Group", "Financials", "momentum"),
    ("ORI", "Old Republic International", "Financials", "momentum"),
    ("FNF", "Fidelity National Financial", "Financials", "momentum"),
    ("FAF", "First American Financial", "Financials", "momentum"),
    ("MTG", "MGIC Investment", "Financials", "momentum"),
    ("RDN", "Radian Group", "Financials", "momentum"),
    ("ESNT", "Essent Group", "Financials", "momentum"),
    ("PFSI", "PennyMac Financial Services", "Financials", "momentum"),
    ("COOP", "Mr. Cooper Group", "Financials", "momentum"),
    ("EWBC", "East West Bancorp", "Financials", "momentum"),
    ("ZION", "Zions Bancorporation", "Financials", "momentum"),
    ("CMA", "Comerica Inc.", "Financials", "momentum"),
    ("WAL", "Western Alliance Bancorp", "Financials", "momentum"),
    ("FHN", "First Horizon Corp.", "Financials", "momentum"),
    ("SNV", "Synovus Financial", "Financials", "momentum"),
    ("VLY", "Valley National Bancorp", "Financials", "momentum"),
    ("ONB", "Old National Bancorp", "Financials", "momentum"),
    ("UMBF", "UMB Financial", "Financials", "momentum"),
    ("CBSH", "Commerce Bancshares", "Financials", "momentum"),
    ("WTFC", "Wintrust Financial", "Financials", "momentum"),
    ("IBKR", "Interactive Brokers Group", "Financials", "core"),
    ("HEI", "HEICO Corp.", "Industrials", "core"),
    ("TDG", "TransDigm Group", "Industrials", "core"),
    ("CW", "Curtiss-Wright", "Industrials", "momentum"),
    ("CR", "Crane Co.", "Industrials", "momentum"),
    ("ITT", "ITT Inc.", "Industrials", "momentum"),
    ("ESAB", "ESAB Corp.", "Industrials", "momentum"),
    ("MIDD", "Middleby Corp.", "Industrials", "momentum"),
    ("NDSN", "Nordson Corp.", "Industrials", "momentum"),
    ("GGG", "Graco Inc.", "Industrials", "momentum"),
    ("AOS", "A.O. Smith Corp.", "Industrials", "momentum"),
    ("LECO", "Lincoln Electric Holdings", "Industrials", "momentum"),
    ("TTC", "Toro Co.", "Industrials", "momentum"),
    ("DCI", "Donaldson Co.", "Industrials", "momentum"),
    ("MSA", "MSA Safety", "Industrials", "momentum"),
    ("RBC", "RBC Bearings", "Industrials", "momentum"),
    ("TXT", "Textron Inc.", "Industrials", "core"),
    ("HWM", "Howmet Aerospace", "Industrials", "core"),
    ("SPXC", "SPX Technologies", "Industrials", "momentum"),
    ("AIT", "Applied Industrial Technologies", "Industrials", "momentum"),
    ("WSO", "Watsco Inc.", "Industrials", "momentum"),
    ("POOL", "Pool Corp.", "Industrials", "momentum"),
    ("SITE", "SiteOne Landscape Supply", "Industrials", "momentum"),
    ("BECN", "Beacon Roofing Supply", "Industrials", "momentum"),
    ("BLD", "TopBuild Corp.", "Industrials", "momentum"),
    ("IBP", "Installed Building Products", "Industrials", "momentum"),
    ("CSL", "Carlisle Companies", "Industrials", "momentum"),
    ("GATX", "GATX Corp.", "Industrials", "momentum"),
    ("WERN", "Werner Enterprises", "Industrials", "momentum"),
    ("KNX", "Knight-Swift Transportation", "Industrials", "momentum"),
    ("SNDR", "Schneider National", "Industrials", "momentum"),
    ("XPO", "XPO Inc.", "Industrials", "momentum"),
    ("SAIA", "Saia Inc.", "Industrials", "momentum"),
    ("LSTR", "Landstar System", "Industrials", "momentum"),
    ("ARCB", "ArcBest Corp.", "Industrials", "momentum"),
    ("TFII", "TFI International", "Industrials", "momentum"),
    ("ALK", "Alaska Air Group", "Industrials", "momentum"),
    ("JBLU", "JetBlue Airways", "Industrials", "momentum"),
    ("SKYW", "SkyWest Inc.", "Industrials", "momentum"),
    ("ALGT", "Allegiant Travel", "Industrials", "momentum"),
    ("ROL", "Rollins Inc.", "Industrials", "core"),
    ("MMS", "Maximus Inc.", "Industrials", "momentum"),
    ("ABM", "ABM Industries", "Industrials", "momentum"),
    ("GEO", "GEO Group", "Industrials", "momentum"),
    ("CXW", "CoreCivic Inc.", "Industrials", "momentum"),
    ("UNF", "UniFirst Corp.", "Industrials", "momentum"),
    ("CTRE", "CareTrust REIT", "Real Estate", "momentum"),
    ("LRN", "Stride Inc.", "Consumer Discretionary", "momentum"),
    ("ATGE", "Adtalem Global Education", "Consumer Discretionary", "momentum"),
    ("LOPE", "Grand Canyon Education", "Consumer Discretionary", "momentum"),
    ("CHGG", "Chegg Inc.", "Consumer Discretionary", "momentum"),
    ("CHWY", "Chewy Inc.", "Consumer Discretionary", "momentum"),
    ("ETSY", "Etsy Inc.", "Consumer Discretionary", "momentum"),
    ("W", "Wayfair Inc.", "Consumer Discretionary", "momentum"),
    ("RH", "RH (Restoration Hardware)", "Consumer Discretionary", "momentum"),
    ("WSM", "Williams-Sonoma", "Consumer Discretionary", "momentum"),
    ("BBWI", "Bath & Body Works", "Consumer Discretionary", "momentum"),
    ("ANF", "Abercrombie & Fitch", "Consumer Discretionary", "momentum"),
    ("AEO", "American Eagle Outfitters", "Consumer Discretionary", "momentum"),
    ("URBN", "Urban Outfitters", "Consumer Discretionary", "momentum"),
    ("GPS", "Gap Inc.", "Consumer Discretionary", "momentum"),
    ("M", "Macy's Inc.", "Consumer Discretionary", "momentum"),
    ("JWN", "Nordstrom Inc.", "Consumer Discretionary", "momentum"),
    ("KSS", "Kohl's Corp.", "Consumer Discretionary", "momentum"),
    ("DKS", "Dick's Sporting Goods", "Consumer Discretionary", "momentum"),
    ("ASO", "Academy Sports & Outdoors", "Consumer Discretionary", "momentum"),
    ("FIVE", "Five Below", "Consumer Discretionary", "momentum"),
    ("BURL", "Burlington Stores", "Consumer Discretionary", "core"),
    ("PLNT", "Planet Fitness", "Consumer Discretionary", "momentum"),
    ("DKNG", "DraftKings Inc.", "Consumer Discretionary", "momentum"),
    ("FLUT", "Flutter Entertainment", "Consumer Discretionary", "core"),
    ("PENN", "PENN Entertainment", "Consumer Discretionary", "momentum"),
    ("BYD", "Boyd Gaming", "Consumer Discretionary", "momentum"),
    ("CHDN", "Churchill Downs", "Consumer Discretionary", "momentum"),
    ("WYNN", "Wynn Resorts", "Consumer Discretionary", "core"),
    ("CZR", "Caesars Entertainment", "Consumer Discretionary", "momentum"),
    ("H", "Hyatt Hotels", "Consumer Discretionary", "momentum"),
    ("WH", "Wyndham Hotels & Resorts", "Consumer Discretionary", "momentum"),
    ("CHH", "Choice Hotels International", "Consumer Discretionary", "momentum"),
    ("TNL", "Travel + Leisure Co.", "Consumer Discretionary", "momentum"),
    ("ABNB", "Airbnb Inc.", "Consumer Discretionary", "core"),
    ("CVNA", "Carvana Co.", "Consumer Discretionary", "momentum"),
    ("KMX", "CarMax Inc.", "Consumer Discretionary", "momentum"),
    ("PAG", "Penske Automotive Group", "Consumer Discretionary", "momentum"),
    ("AN", "AutoNation Inc.", "Consumer Discretionary", "momentum"),
    ("LAD", "Lithia Motors", "Consumer Discretionary", "momentum"),
    ("GPI", "Group 1 Automotive", "Consumer Discretionary", "momentum"),
    ("ABG", "Asbury Automotive Group", "Consumer Discretionary", "momentum"),
    ("SAH", "Sonic Automotive", "Consumer Discretionary", "momentum"),
    ("LEA", "Lear Corp.", "Consumer Discretionary", "momentum"),
    ("GNTX", "Gentex Corp.", "Consumer Discretionary", "momentum"),
    ("THO", "Thor Industries", "Consumer Discretionary", "momentum"),
    ("WGO", "Winnebago Industries", "Consumer Discretionary", "momentum"),
    ("LCII", "LCI Industries", "Consumer Discretionary", "momentum"),
    ("PATK", "Patrick Industries", "Consumer Discretionary", "momentum"),
    ("FOXF", "Fox Factory Holding", "Consumer Discretionary", "momentum"),
    ("MODG", "Topgolf Callaway Brands", "Consumer Discretionary", "momentum"),
    ("MAT", "Mattel Inc.", "Consumer Discretionary", "momentum"),
    ("YETI", "YETI Holdings", "Consumer Discretionary", "momentum"),
    ("SKX", "Skechers USA", "Consumer Discretionary", "momentum"),
    ("DECK", "Deckers Outdoor", "Consumer Discretionary", "core"),
    ("CROX", "Crocs Inc.", "Consumer Discretionary", "momentum"),
    ("BIRK", "Birkenstock Holding", "Consumer Discretionary", "momentum"),
    ("SIG", "Signet Jewelers", "Consumer Discretionary", "momentum"),
    ("LEVI", "Levi Strauss & Co.", "Consumer Discretionary", "momentum"),
    ("PVH", "PVH Corp.", "Consumer Discretionary", "momentum"),
    ("VFC", "V.F. Corp.", "Consumer Discretionary", "momentum"),
    ("HBI", "Hanesbrands Inc.", "Consumer Discretionary", "momentum"),
    ("COLM", "Columbia Sportswear", "Consumer Discretionary", "momentum"),
    ("KTB", "Kontoor Brands", "Consumer Discretionary", "momentum"),
    ("UA", "Under Armour Inc.", "Consumer Discretionary", "momentum"),
    ("USFD", "US Foods Holding", "Consumer Staples", "core"),
    ("PFGC", "Performance Food Group", "Consumer Staples", "momentum"),
    ("CASY", "Casey's General Stores", "Consumer Staples", "momentum"),
    ("SFM", "Sprouts Farmers Market", "Consumer Staples", "momentum"),
    ("GO", "Grocery Outlet Holding", "Consumer Staples", "momentum"),
    ("CHEF", "Chefs' Warehouse", "Consumer Staples", "momentum"),
    ("SPB", "Spectrum Brands", "Consumer Staples", "momentum"),
    ("ENR", "Energizer Holdings", "Consumer Staples", "momentum"),
    ("NWL", "Newell Brands", "Consumer Staples", "momentum"),
    ("HELE", "Helen of Troy", "Consumer Staples", "momentum"),
    ("EPC", "Edgewell Personal Care", "Consumer Staples", "momentum"),
    ("COTY", "Coty Inc.", "Consumer Staples", "momentum"),
    ("ELF", "e.l.f. Beauty", "Consumer Staples", "momentum"),
    ("IPAR", "Inter Parfums", "Consumer Staples", "momentum"),
    ("FRPT", "Freshpet Inc.", "Consumer Staples", "momentum"),
    ("SMPL", "Simply Good Foods", "Consumer Staples", "momentum"),
    ("CELH", "Celsius Holdings", "Consumer Staples", "momentum"),
    ("MNST", "Monster Beverage", "Consumer Staples", "core"),
    ("KDP", "Keurig Dr Pepper", "Consumer Staples", "core"),
    ("FIZZ", "National Beverage", "Consumer Staples", "momentum"),
    ("COKE", "Coca-Cola Consolidated", "Consumer Staples", "momentum"),
    ("PRMW", "Primo Brands", "Consumer Staples", "momentum"),
    ("CHK", "Expand Energy", "Energy", "momentum"),
    ("RRC", "Range Resources", "Energy", "momentum"),
    ("CNX", "CNX Resources", "Energy", "momentum"),
    ("AR", "Antero Resources", "Energy", "momentum"),
    ("SM", "SM Energy", "Energy", "momentum"),
    ("MTDR", "Matador Resources", "Energy", "momentum"),
    ("PR", "Permian Resources", "Energy", "momentum"),
    ("CIVI", "Civitas Resources", "Energy", "momentum"),
    ("OVV", "Ovintiv Inc.", "Energy", "momentum"),
    ("DTM", "DT Midstream", "Energy", "momentum"),
    ("AM", "Antero Midstream", "Energy", "momentum"),
    ("ENLC", "EnLink Midstream", "Energy", "momentum"),
    ("PAA", "Plains All American Pipeline", "Energy", "momentum"),
    ("ET", "Energy Transfer LP", "Energy", "core"),
    ("EPD", "Enterprise Products Partners", "Energy", "core"),
    ("MPLX", "MPLX LP", "Energy", "core"),
    ("SUN", "Sunoco LP", "Energy", "momentum"),
    ("DINO", "HF Sinclair", "Energy", "momentum"),
    ("PBF", "PBF Energy", "Energy", "momentum"),
    ("CVI", "CVR Energy", "Energy", "momentum"),
    ("DK", "Delek US Holdings", "Energy", "momentum"),
    ("WFRD", "Weatherford International", "Energy", "momentum"),
    ("NOV", "NOV Inc.", "Energy", "momentum"),
    ("FTI", "TechnipFMC", "Energy", "momentum"),
    ("CHX", "ChampionX Corp.", "Energy", "momentum"),
    ("HP", "Helmerich & Payne", "Energy", "momentum"),
    ("PTEN", "Patterson-UTI Energy", "Energy", "momentum"),
    ("NE", "Noble Corp.", "Energy", "momentum"),
    ("RIG", "Transocean Ltd.", "Energy", "momentum"),
    ("VAL", "Valaris Ltd.", "Energy", "momentum"),
    ("TDW", "Tidewater Inc.", "Energy", "momentum"),
    ("UGI", "UGI Corp.", "Utilities", "momentum"),
    ("NFG", "National Fuel Gas", "Utilities", "momentum"),
    ("SWX", "Southwest Gas Holdings", "Utilities", "momentum"),
    ("OGS", "ONE Gas Inc.", "Utilities", "momentum"),
    ("SR", "Spire Inc.", "Utilities", "momentum"),
    ("NJR", "New Jersey Resources", "Utilities", "momentum"),
    ("BKH", "Black Hills Corp.", "Utilities", "momentum"),
    ("OTTR", "Otter Tail Corp.", "Utilities", "momentum"),
    ("MGEE", "MGE Energy", "Utilities", "momentum"),
    ("IDA", "IDACORP Inc.", "Utilities", "momentum"),
    ("POR", "Portland General Electric", "Utilities", "momentum"),
    ("AVA", "Avista Corp.", "Utilities", "momentum"),
    ("NWE", "NorthWestern Energy", "Utilities", "momentum"),
    ("ALE", "ALLETE Inc.", "Utilities", "momentum"),
    ("PNM", "PNM Resources", "Utilities", "momentum"),
    ("UTL", "Unitil Corp.", "Utilities", "momentum"),
    ("CWEN", "Clearway Energy", "Utilities", "momentum"),
    ("ORA", "Ormat Technologies", "Utilities", "momentum"),
    ("RS", "Reliance Inc.", "Materials", "momentum"),
    ("CMC", "Commercial Metals", "Materials", "momentum"),
    ("ATI", "ATI Inc.", "Materials", "momentum"),
    ("CRS", "Carpenter Technology", "Materials", "momentum"),
    ("HCC", "Warrior Met Coal", "Materials", "momentum"),
    ("AMR", "Alpha Metallurgical Resources", "Materials", "momentum"),
    ("ARCH", "Arch Resources", "Materials", "momentum"),
    ("BTU", "Peabody Energy", "Materials", "momentum"),
    ("CEIX", "CONSOL Energy", "Materials", "momentum"),
    ("RYI", "Ryerson Holding", "Materials", "momentum"),
    ("WOR", "Worthington Enterprises", "Materials", "momentum"),
    ("MTUS", "Metallus Inc.", "Materials", "momentum"),
    ("KALU", "Kaiser Aluminum", "Materials", "momentum"),
    ("CENX", "Century Aluminum", "Materials", "momentum"),
    ("AA", "Alcoa Corp.", "Materials", "core"),
    ("MP", "MP Materials", "Materials", "momentum"),
    ("UEC", "Uranium Energy", "Materials", "momentum"),
    ("CCJ", "Cameco Corp.", "Materials", "core"),
    ("SQM", "Sociedad Quimica y Minera", "Materials", "momentum"),
    ("IPI", "Intrepid Potash", "Materials", "momentum"),
    ("LXU", "LSB Industries", "Materials", "momentum"),
    ("MEOH", "Methanex Corp.", "Materials", "momentum"),
    ("OLN", "Olin Corp.", "Materials", "momentum"),
    ("TROX", "Tronox Holdings", "Materials", "momentum"),
    ("HUN", "Huntsman Corp.", "Materials", "momentum"),
    ("ASH", "Ashland Inc.", "Materials", "momentum"),
    ("RPM", "RPM International", "Materials", "momentum"),
    ("SXT", "Sensient Technologies", "Materials", "momentum"),
    ("HWKN", "Hawkins Inc.", "Materials", "momentum"),
    ("KWR", "Quaker Chemical", "Materials", "momentum"),
    ("BCPC", "Balchem Corp.", "Materials", "momentum"),
    ("NEU", "NewMarket Corp.", "Materials", "momentum"),
    ("CBT", "Cabot Corp.", "Materials", "momentum"),
    ("SLVM", "Sylvamo Corp.", "Materials", "momentum"),
    ("SON", "Sonoco Products", "Materials", "momentum"),
    ("SEE", "Sealed Air", "Materials", "momentum"),
    ("GEF", "Greif Inc.", "Materials", "momentum"),
    ("BERY", "Berry Global Group", "Materials", "momentum"),
    ("REYN", "Reynolds Consumer Products", "Materials", "momentum"),
    ("CLW", "Clearwater Paper", "Materials", "momentum"),
    ("MATV", "Mativ Holdings", "Materials", "momentum"),
    ("GLT", "Glatfelter Corp.", "Materials", "momentum"),
    ("VNO", "Vornado Realty Trust", "Real Estate", "momentum"),
    ("SLG", "SL Green Realty", "Real Estate", "momentum"),
    ("KRC", "Kilroy Realty", "Real Estate", "momentum"),
    ("DEI", "Douglas Emmett", "Real Estate", "momentum"),
    ("HPP", "Hudson Pacific Properties", "Real Estate", "momentum"),
    ("CUZ", "Cousins Properties", "Real Estate", "momentum"),
    ("HIW", "Highwoods Properties", "Real Estate", "momentum"),
    ("PGRE", "Paramount Group", "Real Estate", "momentum"),
    ("ESRT", "Empire State Realty Trust", "Real Estate", "momentum"),
    ("PDM", "Piedmont Office Realty", "Real Estate", "momentum"),
    ("OFC", "Corporate Office Properties", "Real Estate", "momentum"),
    ("FRT", "Federal Realty Investment Trust", "Real Estate", "core"),
    ("AKR", "Acadia Realty Trust", "Real Estate", "momentum"),
    ("UE", "Urban Edge Properties", "Real Estate", "momentum"),
    ("ROIC", "Retail Opportunity Investments", "Real Estate", "momentum"),
    ("SITC", "SITE Centers", "Real Estate", "momentum"),
    ("BRX", "Brixmor Property Group", "Real Estate", "momentum"),
    ("KRG", "Kite Realty Group", "Real Estate", "momentum"),
    ("NNN", "NNN REIT", "Real Estate", "momentum"),
    ("ADC", "Agree Realty", "Real Estate", "momentum"),
    ("EPRT", "Essential Properties Realty", "Real Estate", "momentum"),
    ("GTY", "Getty Realty", "Real Estate", "momentum"),
    ("LXP", "LXP Industrial Trust", "Real Estate", "momentum"),
    ("STAG", "STAG Industrial", "Real Estate", "momentum"),
    ("TRNO", "Terreno Realty", "Real Estate", "momentum"),
    ("PLYM", "Plymouth Industrial REIT", "Real Estate", "momentum"),
    ("EGP", "EastGroup Properties", "Real Estate", "momentum"),
    ("FR", "First Industrial Realty", "Real Estate", "momentum"),
    ("REXR", "Rexford Industrial Realty", "Real Estate", "momentum"),
    ("COLD", "Americold Realty Trust", "Real Estate", "momentum"),
    ("LTC", "LTC Properties", "Real Estate", "momentum"),
    ("OHI", "Omega Healthcare Investors", "Real Estate", "momentum"),
    ("SBRA", "Sabra Health Care REIT", "Real Estate", "momentum"),
    ("NHI", "National Health Investors", "Real Estate", "momentum"),
    ("AHR", "American Healthcare REIT", "Real Estate", "momentum"),
    ("MPW", "Medical Properties Trust", "Real Estate", "momentum"),
    ("AMH", "American Homes 4 Rent", "Real Estate", "momentum"),
    ("ELS", "Equity LifeStyle Properties", "Real Estate", "momentum"),
    ("SUI", "Sun Communities", "Real Estate", "momentum"),
    ("CUBE", "CubeSmart", "Real Estate", "momentum"),
    ("NSA", "National Storage Affiliates", "Real Estate", "momentum"),
    ("VCSA", "Vacasa Inc.", "Real Estate", "momentum"),
    ("RHP", "Ryman Hospitality Properties", "Real Estate", "momentum"),
    ("PEB", "Pebblebrook Hotel Trust", "Real Estate", "momentum"),
    ("RLJ", "RLJ Lodging Trust", "Real Estate", "momentum"),
    ("APLE", "Apple Hospitality REIT", "Real Estate", "momentum"),
    ("DRH", "DiamondRock Hospitality", "Real Estate", "momentum"),
    ("SHO", "Sunstone Hotel Investors", "Real Estate", "momentum"),
    ("XHR", "Xenia Hotels & Resorts", "Real Estate", "momentum"),
    ("PK", "Park Hotels & Resorts", "Real Estate", "momentum"),
    ("CLDT", "Chatham Lodging Trust", "Real Estate", "momentum"),
    ("ROKU", "Roku Inc.", "Communication Services", "momentum"),
    ("SPOT", "Spotify Technology", "Communication Services", "core"),
    ("PINS", "Pinterest Inc.", "Communication Services", "momentum"),
    ("SNAP", "Snap Inc.", "Communication Services", "momentum"),
    ("RDDT", "Reddit Inc.", "Communication Services", "momentum"),
    ("BMBL", "Bumble Inc.", "Communication Services", "momentum"),
    ("IAC", "IAC Inc.", "Communication Services", "momentum"),
    ("ANGI", "Angi Inc.", "Communication Services", "momentum"),
    ("TRIP", "TripAdvisor Inc.", "Communication Services", "momentum"),
    ("YELP", "Yelp Inc.", "Communication Services", "momentum"),
    ("ZD", "Ziff Davis", "Communication Services", "momentum"),
    ("CARG", "CarGurus Inc.", "Communication Services", "momentum"),
    ("SCHL", "Scholastic Corp.", "Communication Services", "momentum"),
    ("NYT", "New York Times Co.", "Communication Services", "momentum"),
    ("GCI", "Gannett Co.", "Communication Services", "momentum"),
    ("LEE", "Lee Enterprises", "Communication Services", "momentum"),
    ("SIRI", "Sirius XM Holdings", "Communication Services", "momentum"),
    ("LSXMA", "Liberty Media", "Communication Services", "momentum"),
    ("FWONA", "Formula One Group", "Communication Services", "momentum"),
    ("WMG", "Warner Music Group", "Communication Services", "momentum"),
    ("MSGS", "Madison Square Garden Sports", "Communication Services", "momentum"),
    ("EDR", "Endeavor Group Holdings", "Communication Services", "momentum"),
    ("CABO", "Cable One", "Communication Services", "momentum"),
    ("LBRDA", "Liberty Broadband", "Communication Services", "momentum"),
    ("ATUS", "Altice USA", "Communication Services", "momentum"),
    ("LUMN", "Lumen Technologies", "Communication Services", "momentum"),
    ("FYBR", "Frontier Communications", "Communication Services", "momentum"),
    ("SHEN", "Shenandoah Telecommunications", "Communication Services", "momentum"),
    ("CCOI", "Cogent Communications", "Communication Services", "momentum"),
    ("GSAT", "Globalstar Inc.", "Communication Services", "momentum"),
    ("IRDM", "Iridium Communications", "Communication Services", "momentum"),
    ("VSAT", "Viasat Inc.", "Communication Services", "momentum"),
    ("ASTS", "AST SpaceMobile", "Communication Services", "momentum"),
    ("LUNR", "Intuitive Machines", "Industrials", "momentum"),
    ("RKLB", "Rocket Lab USA", "Industrials", "momentum"),
    ("PL", "Planet Labs", "Industrials", "momentum"),
    ("SPCE", "Virgin Galactic", "Industrials", "momentum"),
    ("KTOS", "Kratos Defense & Security", "Industrials", "momentum"),
    ("AVAV", "AeroVironment", "Industrials", "momentum"),
    ("CACI", "CACI International", "Industrials", "core"),
    ("SAIC", "Science Applications International", "Industrials", "momentum"),
    ("BAH", "Booz Allen Hamilton", "Industrials", "core"),
    ("ICFI", "ICF International", "Industrials", "momentum"),
    ("TTEK", "Tetra Tech", "Industrials", "momentum"),
    ("ACM", "AECOM", "Industrials", "momentum"),
    ("FLR", "Fluor Corp.", "Industrials", "momentum"),
    ("KBR", "KBR Inc.", "Industrials", "momentum"),
    ("MTZ", "MasTec Inc.", "Industrials", "momentum"),
    ("PRIM", "Primoris Services", "Industrials", "momentum"),
    ("MYRG", "MYR Group", "Industrials", "momentum"),
    ("DY", "Dycom Industries", "Industrials", "momentum"),
    ("STRL", "Sterling Infrastructure", "Industrials", "momentum"),
    ("ROAD", "Construction Partners", "Industrials", "momentum"),
    ("APG", "APi Group", "Industrials", "momentum"),
    ("ARLO", "Arlo Technologies", "Industrials", "momentum"),
    ("NVEI", "Nuvei Corp.", "Financials", "momentum"),
    ("COUR", "Coursera Inc.", "Consumer Discretionary", "momentum"),
    ("DUOL", "Duolingo Inc.", "Consumer Discretionary", "momentum"),
    ("ASAN", "Asana Inc.", "Technology", "momentum"),
    ("PATH", "UiPath Inc.", "Technology", "momentum"),
    ("AI", "C3.ai Inc.", "Technology", "momentum"),
    ("BBAI", "BigBear.ai Holdings", "Technology", "momentum"),
    ("SOUN", "SoundHound AI", "Technology", "momentum"),
    ("IONQ", "IonQ Inc.", "Technology", "momentum"),
    ("RGTI", "Rigetti Computing", "Technology", "momentum"),
    ("QUBT", "Quantum Computing Inc.", "Technology", "momentum"),
    ("ARQQ", "Arqit Quantum", "Technology", "momentum"),
    ("CRDO", "Credo Technology", "Technology", "momentum"),
    ("MRVL", "Marvell Technology", "Technology", "core"),
    ("ALAB", "Astera Labs", "Technology", "momentum"),
    ("ARM", "Arm Holdings", "Technology", "core"),
    ("GTLB", "GitLab Inc.", "Technology", "momentum"),
    ("BRZE", "Braze Inc.", "Technology", "momentum"),
    ("AMPL", "Amplitude Inc.", "Technology", "momentum"),
    ("APPF", "AppFolio Inc.", "Technology", "momentum"),
    ("BL", "BlackLine Inc.", "Technology", "momentum"),
    ("WK", "Workiva Inc.", "Technology", "momentum"),
    ("ALRM", "Alarm.com Holdings", "Technology", "momentum"),
    ("SMAR", "Smartsheet Inc.", "Technology", "momentum"),
    ("MNDY", "monday.com Ltd.", "Technology", "momentum"),
    ("KLAR", "Klarna Group", "Financials", "momentum"),
    ("CIRCLE", "Circle Internet Group", "Financials", "momentum"),
    ("CRCL", "Circle Internet Group", "Financials", "momentum"),
]

RISK_FREE = 0.045       # approx. short-term Treasury yield, used for Sharpe

# --- Scoring engine options (change these, then re-run the backtest) ---
# RISK_ADJUSTED_MOMENTUM: divide each stock's momentum by its own volatility,
#   so a smooth 40% gain outranks a violent 40% gain. Off = raw momentum.
RISK_ADJUSTED_MOMENTUM = True

# QUALITY_GATE: drop names that fail basic health checks before ranking.
#   Removes the classic momentum failure mode -- buying a stock that is up
#   because it is cheap and broken. Needs Yahoo fundamentals, which are
#   patchy for smaller names; failures are skipped rather than faked.
QUALITY_WEIGHT = 0.15          # influence of the quality factor (0 = ignore it)
EXCLUDE_ON_QUALITY_FAIL = False # True = drop failing names (shrinks the list)
QUALITY_MIN_ROE = 0.05         # reference point for the quality score
QUALITY_MAX_DEBT_EQUITY = 3.0  # reference point for the quality score
TOP_N_CORE = 8          # names to allocate to in Approach 1
TOP_N_MOM = 6           # names to allocate to in Approach 2
CAP_CORE = 0.20         # max weight per name, Approach 1
CAP_MOM = 0.25          # max weight per name, Approach 2

# Approach 3 (real optimizer) settings
OPT_CANDIDATES = 20     # shortlist size fed to the optimizer (keeps it fast)
OPT_MAX_WEIGHT = 0.15   # max weight per single stock in the optimal portfolio

# OPT_METHOD picks how Approach 3 solves for weights:
#   "Max Sharpe"        = classic mean-variance (the original method)
#   "Hierarchical Risk Parity" = clusters correlated names, spreads risk across
#                         them. FAR more stable between runs -- recommended.
#   "Black-Litterman"   = blends the market view with your confidence in the
#                         shortlist's own momentum. Smooths extreme weights.
OPT_METHOD = "Hierarchical Risk Parity"

# Benchmark used for comparison and for the backtest
BENCHMARK_TICKER = "^GSPC"
BENCHMARK_NAME = "S&P 500"

# Backtest settings
BACKTEST_YEARS = 3      # how far back to test the rules
BACKTEST_TOP_N = 8      # names held at a time in the backtest

# --- Cost & tax assumptions (used by the backtest) ---
# These are the numbers that decide whether frequent rebalancing is worth it.
COST_BPS = 5.0          # round-trip cost per trade in basis points (spread + fees)
TAX_SHORT_TERM = 0.24   # tax on gains held under 1 year (US ordinary-income-ish)
TAX_LONG_TERM = 0.15    # tax on gains held over 1 year

# --- Turnover control ---
# HYSTERESIS_BAND: an existing holding stays until it falls below this rank.
#   e.g. 15 means "buy at top 8, but only sell once it drops past 15th".
#   This cuts churn dramatically with almost no loss of signal.
HYSTERESIS_BAND = 15
MIN_HOLD_MONTHS = 3     # do not sell a position before this many months

# --- Position sizing ---
# "Volatility-scaled" gives steadier results: smaller weights in jumpier names.
# "Equal weight" spreads the same dollar across every pick.
SIZING_METHOD = "Volatility-scaled"

st.set_page_config(page_title="My Stock Model", page_icon="ÃÂ°ÃÂÃÂÃÂ", layout="wide")


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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_quality(tickers):
    """Basic health checks per ticker: profitable, not over-levered.

    Returns {ticker: True/False}. A ticker whose fundamentals cannot be read
    is treated as PASS so news/coverage gaps never silently shrink the list."""
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            roe = info.get("returnOnEquity")
            de = info.get("debtToEquity")
            margin = info.get("profitMargins")
            ok = True
            if roe is not None and roe < QUALITY_MIN_ROE:
                ok = False
            if de is not None and (de / 100.0) > QUALITY_MAX_DEBT_EQUITY:
                ok = False
            if margin is not None and margin < QUALITY_MIN_MARGIN:
                ok = False
            out[t] = ok
        except Exception:
            out[t] = True
    passed = sum(1 for v in out.values() if v)
    print(f"[quality] {passed}/{len(out)} names passed the health checks")
    return out


# ----------------------------------------------------------------------------
# DATA + SCORING
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)   # cache 1 hour so it loads fast
def load_data(custom_names=None):
    if custom_names:
        names = custom_names
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
            "Sector": meta.get(t, {}).get("sector", "ÃÂ¢ÃÂÃÂ"),
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

    # --- Quality gate: drop unhealthy names before ranking ---
    if QUALITY_GATE:
        qmap = fetch_quality([u[0] for u in names])
        before = len(df)
        df = df[df["Ticker"].map(lambda t: qmap.get(t, True))].copy()
        st.caption(f"Quality gate: {before} names screened, {len(df)} passed. "
                   "A name that cannot be read is kept rather than dropped.")
        if df.empty:
            return df, pd.DataFrame()

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
        px = yf.Ticker(BENCHMARK_TICKER).history(period="1y")["Close"].dropna()
        if len(px) < 2:
            return None
        return {
            "1mo": float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) > 21 else None,
            "6mo": float(px.iloc[-1] / px.iloc[-126] - 1) if len(px) > 126 else None,
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


def markowitz_real(df, prices, method=None):
    """Solve for portfolio weights across a shortlist of the strongest names.

    Three methods, all from PyPortfolioOpt:
      Max Sharpe              - classic mean-variance optimization
      Hierarchical Risk Parity- clusters correlated names and spreads risk
                                across the clusters. Much more stable.
      Black-Litterman         - starts from the market portfolio and tilts
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
# MARKET TIMING ÃÂ¢ÃÂÃÂ rules-based signal, not a prediction
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
        vix_label, vix_note = "Low / Complacent", "Calm markets ÃÂ¢ÃÂÃÂ historically can precede surprises either way."
    elif vix < 20:
        vix_label, vix_note = "Normal", "Typical volatility range."
    elif vix < 30:
        vix_label, vix_note = "Elevated", "Markets pricing in real uncertainty."
    else:
        vix_label, vix_note = "High / Fear", "Historically often (not always) followed by a recovery ÃÂ¢ÃÂÃÂ but can persist or worsen."

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
        regime, guidance = "Neutral", "No strong signal either way ÃÂ¢ÃÂÃÂ sticking to your regular schedule is reasonable."
    else:
        regime, guidance = "Cautious", "Elevated fear and/or a weak trend. Some investors stay the course anyway (timing the market is notoriously hard); others reduce size this month. Your call, not the model's."

    return {
        "vix": vix, "vix_label": vix_label, "vix_note": vix_note,
        "spx_now": spx_now, "spx_ma50": spx_ma50, "spx_ma200": spx_ma200,
        "trend_label": trend_label, "score": score,
        "regime": regime, "guidance": guidance,
    }


# ----------------------------------------------------------------------------
# NEWS SENTIMENT ÃÂ¢ÃÂÃÂ free headline keyword scoring per stock, not NLP magic
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
    This is keyword counting, not language understanding ÃÂ¢ÃÂÃÂ crude by design,
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
st.title("ÃÂ°ÃÂÃÂÃÂ My US Stock Model")
_cap = "S&P 1500 (broad screen)" if SCREEN_MODE == "broad" else "curated watchlist"
st.caption(f"Screening: **{_cap}**. Three approaches, ranked from live market data. "
           f"Prices refresh each time this page loads. "
           f"Switch modes with SCREEN_MODE at the top of the file. "
           f"{market_session_note()}")

with st.expander("ÃÂ¢ÃÂÃÂÃÂ¯ÃÂ¸ÃÂ Which stocks should it screen?", expanded=False):
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
    st.caption(f"Volume-surge filter is {_surge} ÃÂ¢ÃÂÃÂ keeps the top "
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

with st.spinner("Screening the universe ÃÂ¢ÃÂÃÂ this can take a minute or two in broad mode..."):
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
             "Yahoo's free feed may be rate-limiting ÃÂ¢ÃÂÃÂ try again in a minute.")
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

d1, d2, d3, d4, d5, d6, d7 = st.tabs([
    "ÃÂ°ÃÂÃÂÃÂ¢ Approach 1 ÃÂ¢ÃÂÃÂ Diversified",
    "ÃÂ°ÃÂÃÂÃÂ´ Approach 2 ÃÂ¢ÃÂÃÂ Momentum",
    "ÃÂ°ÃÂÃÂÃÂµ Approach 3 ÃÂ¢ÃÂÃÂ Optimizer",
    "ÃÂ°ÃÂÃÂÃÂ° Market Timing & News",
    "ÃÂ¢ÃÂÃÂ Summary ÃÂ¢ÃÂÃÂ What to Buy",
    "ÃÂ°ÃÂÃÂ§ÃÂª Backtest",
    "ÃÂ¢ÃÂÃÂ How to read this",
])

# ---------------- Approach 1 ----------------
with d1:
    st.subheader("Diversified large-cap ÃÂ¢ÃÂÃÂ lower risk, steadier ride")
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
    x3.metric("Typical monthly swing", f"ÃÂÃÂ±{avg_v/np.sqrt(12):.1%}")

# ---------------- Approach 2 ----------------
with d2:
    st.subheader("Momentum / speculative ÃÂ¢ÃÂÃÂ higher risk, much bigger swings")
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
    x3.metric("Typical monthly swing", f"ÃÂÃÂ±{avg_v/np.sqrt(12):.1%}")

# ---------------- Approach 3 ----------------
with d3:
    st.subheader("Approach 3 ÃÂ¢ÃÂÃÂ portfolio optimizer")
    st.write("Solves for the weight mix with the best return per unit of risk. The method below changes how it does that.")

    _method = st.radio(
        "Optimizer method",
        ["Hierarchical Risk Parity", "Black-Litterman", "Max Sharpe"],
        index=["Hierarchical Risk Parity", "Black-Litterman", "Max Sharpe"].index(OPT_METHOD)
        if OPT_METHOD in ["Hierarchical Risk Parity", "Black-Litterman", "Max Sharpe"] else 0,
        horizontal=True)
    st.caption({
        "Hierarchical Risk Parity": "Clusters names that move together and spreads risk across the clusters. Most stable between visits ÃÂ¢ÃÂÃÂ recommended.",
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
    st.subheader("Market timing & news sentiment ÃÂ¢ÃÂÃÂ signals, not predictions")
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
            st.success(f"**{timing['regime']}** ÃÂ¢ÃÂÃÂ {timing['guidance']}")
        elif timing["regime"] == "Neutral":
            st.info(f"**{timing['regime']}** ÃÂ¢ÃÂÃÂ {timing['guidance']}")
        else:
            st.warning(f"**{timing['regime']}** ÃÂ¢ÃÂÃÂ {timing['guidance']}")

        st.write("**What produced that score:**")
        st.dataframe(pd.DataFrame({
            "Indicator": ["Read at (ET)", "VIX level", "VIX read", "S&P 500 vs 50-day avg",
                          "S&P 500 vs 200-day avg"],
            "Value": [et_stamp(), f"{timing['vix']:.1f}", timing["vix_label"],
                      f"{timing['spx_now']:,.0f} vs {timing['spx_ma50']:,.0f}",
                      f"{timing['spx_now']:,.0f} vs {timing['spx_ma200']:,.0f}"],
        }), use_container_width=True, hide_index=True)

        st.caption(f"{timing['vix_note']} Score = 50 to start, +20 for an uptrend "
                   f"(ÃÂ¢ÃÂÃÂ20 otherwise), +15 for VIX under 20 (ÃÂ¢ÃÂÃÂ15 above 30). "
                   f"Thresholds are visible in the code and you can change them.")

    st.divider()

    # ---- Part B: per-stock news sentiment ----
    st.markdown("### News tone on your watchlist")
    st.write("Counts positive vs negative words across the most recent free headlines "
             "for each stock. A negative score means the headlines skew bearish right "
             "now ÃÂ¢ÃÂÃÂ not that the stock will fall.")

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

        st.caption("Scores run from roughly ÃÂ¢ÃÂÃÂ3 (heavy negative tone) to +3 (heavy "
                   "positive). This is keyword matching, not language understanding ÃÂ¢ÃÂÃÂ "
                   "treat it as a quick read of headline mood, and check the sample "
                   "headline yourself before acting on any single row.")

# ---------------- Summary & Buy Plan ----------------
with d5:
    st.subheader("What to buy, and is now a good time")

    # ---- Part A: is today a good time ----
    timing = market_timing_signal()
    if timing is None:
        st.warning("Couldn't read the market indicators right now ÃÂ¢ÃÂÃÂ timing score unavailable.")
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
            st.success("Conditions look favourable on this rule ÃÂ¢ÃÂÃÂ investing the "
                       "full monthly amount is reasonable.")
        elif timing_regime == "Neutral":
            st.info("No strong signal either way ÃÂ¢ÃÂÃÂ sticking to your usual schedule "
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
        ["Approach 1 ÃÂ¢ÃÂÃÂ Diversified", "Approach 2 ÃÂ¢ÃÂÃÂ Momentum",
         "Approach 3 ÃÂ¢ÃÂÃÂ Markowitz", "Blended (80% Diversified / 20% Momentum)"],
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
                "aside for next month ÃÂ¢ÃÂÃÂ that keeps you investing without ignoring the "
                "signal.")

# ---------------- Guide ----------------
# ---------------- Backtest ----------------
with d6:
    st.subheader("Backtest ÃÂ¢ÃÂÃÂ would these rules have worked?")
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
        "No costs, no slippage, no taxes ÃÂ¢ÃÂÃÂ so the real-world edge would be smaller."
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
                "ÃÂ¢ÃÂÃÂ", "ÃÂ¢ÃÂÃÂ",
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
            f"{'ON' if bt['turnover_control'] else 'OFF'} ÃÂ¢ÃÂÃÂ holdings stay "
            f"until they fall past rank {HYSTERESIS_BAND} and have been held at "
            f"least {MIN_HOLD_MONTHS} months."
        )
        st.info(
            "Turnover is the number to watch. If average monthly turnover is high, "
            "the strategy is trading a lot, and the costs above will eat into any "
            "edge. Try raising HYSTERESIS_BAND in the settings block and see "
            "whether the net result improves."
        )


with d7:
    st.subheader("How to read this")
    st.markdown("""
**The three approaches answer different questions.**

- **Approach 1 ÃÂ¢ÃÂÃÂ Diversified** asks *"which healthy large-caps are trending up, "
  "and how do I spread the money so one bad name can't hurt me?"* Lower volatility, "
  "smaller month-to-month swings, historically a smoother line.
- **Approach 2 ÃÂ¢ÃÂÃÂ Momentum** asks *"what has run the hardest over the past year?"* "
  "It chases strength. That means occasional spectacular years and occasional "
  "savage losses ÃÂ¢ÃÂÃÂ both are normal here, not a malfunction.
- **Approach 3 ÃÂ¢ÃÂÃÂ Markowitz** asks *"what combination of these stocks gives the most "
  "return for the risk I'm accepting?"* It looks at how every pair of stocks moves "
  "together (their covariance), not just how each one performed alone.

**Plain-language notes.**

- *Volatility* is how much a stock bounces around. Higher means wider swings, both ways.
- *Typical monthly swing* converts annual volatility into a rough monthly figure.
  Real months land above and below it ÃÂ¢ÃÂÃÂ it is a yardstick, not a promise.
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
- It does not average 20% a month. Nothing does ÃÂ¢ÃÂÃÂ the S&P 500 has averaged roughly
  10% per *year* over the long run.
- It refreshes when you open it, not on a timer.
""")

st.divider()
st.caption("Decision-support tool, not financial advice. "
           "Past performance does not predict future returns.")


