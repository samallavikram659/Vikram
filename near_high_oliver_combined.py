"""
=============================================================================
NEAR HIGH + OLIVER KELL COMBINED BACKTEST  --  PORTFOLIO MODE  (Yahoo Finance)
=============================================================================
DATA      : Yahoo Finance (yfinance) — no login required.
            NSE universe from NSE public CSV (~2500 EQ stocks).
            Batch downloads 50 stocks per request (~2-3 min first run).

UNIVERSE  : Nifty 500 stocks (default) or All NSE EQ-series (~2500)
            Excludes: ETFs, Liquid Funds, Bonds, Index Funds

PORTFOLIO : INITIAL_CAPITAL = Rs 1,00,000 (1 lakh)
            MAX_POSITIONS   = 3 (max 3 stocks held at any time)
            Compounding allocation: current total equity / 3 per position,
              recomputed before every entry (mark-to-market cash + open
              positions). Position size grows with profits, shrinks with
              drawdowns -- capital recycled on exit.

SIMULATION: Two-pass approach
            Pass 1 - Fetch data + compute indicators for ALL stocks
            Pass 2 - Iterate day-by-day across all trading dates
              (0) Fill pending entries from yesterday's scan at TODAY's open
                  (models running the scan after market close, placing the
                  order overnight, and it filling at next day's open)
              (a) Check all open positions for exits
              (b) Scan for new entry signals (if open < MAX_POSITIONS),
                  queue accepted candidates for tomorrow's open (not filled
                  today -- no same-day lookahead)
              (c) On ties, pick highest momentum (closest to reference high)

ENTRY     : EMA alignment (Close > EMA5 > EMA10 > EMA20 > EMA50 > EMA150 > EMA200)
          + Close within 5-10% BELOW the key high (1Y / 5Y / ATH)
          + Oliver Kell signal: Wedge_Pop OR Crossback
          + Filled at NEXT trading day's Open (signal detected on prior close)

EXIT      : (checked in priority order, both stop and target modeled as
             resting LIMIT sell orders -- fill only if price actually
             trades at that level, not on a gap through it)
            1. Stop loss 5% below entry price (LIMIT order: fills only if
               Low <= stop <= High; a gap-down below stop leaves it unfilled
               and the position stays open, exposed until price recovers to
               the stop level or another exit rule triggers)
            2. Target 10% above ref high (LIMIT order: fills if High >= target)
            3. Oliver Trailing EMA stop (EMA20 early, EMA10 after MIN_HOLD_DAYS)
            4. Extension exit (Close > EMA20 + 3*ATR)

COSTS     : Slippage (0.05% default) applied to market-style fills only --
              next-day-open entries and Oliver_EMA_Stop/Extension_Exit/
              End_Of_Data closes. Stop_Loss/Target_Hit limit fills get 0%
              slippage by design.
            Statutory + broker charges modeled per NSE delivery trade:
              brokerage (0 by default -- most discount brokers), STT (0.1%
              both legs), exchange transaction charges, SEBI fees, stamp
              duty (buy side), GST on brokerage+exchange+SEBI, and a flat
              DP charge on the sell side. See buy_cost()/sell_cost() and
              the config block for exact rates -- adjust to match your broker.

SIGNAL TYPES: 'NearHigh_Wedge_1Y', 'NearHigh_Wedge_5Y', 'NearHigh_Wedge_ATH',
              'NearHigh_Crossback_1Y', 'NearHigh_Crossback_5Y', 'NearHigh_Crossback_ATH'

RESULTS   : CSV saved with timestamp; full performance report printed
=============================================================================
"""

import os
import pickle
import time
import datetime
import warnings
import requests
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================
INITIAL_CAPITAL      = 100_000               # Rs 1 lakh
MAX_POSITIONS        = 3                     # max concurrent positions
STOP_LOSS_PCT        = 0.05                  # 5% stop loss
TARGET_ABOVE_HIGH_PCT = 0.10                 # 10% above ref high
NEAR_HIGH_LOWER      = 0.90                  # 10% below high (lower bound)
NEAR_HIGH_UPPER      = 0.95                  # 5% below high (upper bound)
LOOKBACK_DAYS        = 365                   # backtest window in days
FETCH_YEARS          = 6                     # years of history to fetch
RANK_METHOD          = "momentum"            # "roc" | "rs_nifty" | "near_ath" | "volume_surge" | "momentum"
ROC_PERIOD           = 90                    # days for ROC and momentum calculation
RS_RANK_PERIOD       = 90                    # days for RS vs Nifty ranking
FRESH_HIGH_DAYS      = 200                   # skip stock if it touched ref high within this many bars

# Oliver Kell specific config
MINI_BASE_BARS       = 7                     # window for mini-base detection
MINI_BASE_MIN        = 4                     # minimum qualifying bars in base
EXTENSION_MULT       = 3.0                   # ATR multiplier for extension exit
MIN_HOLD_DAYS        = 2                     # hold for at least this many bars before trailing stop
RS_LOOKBACK          = 20                    # days for relative-strength calculation

CACHE_DIR            = "cache_near_high_yf"     # primary cache dir for this script
CACHE_FALLBACK_DIRS  = ["cache_ath_yf"]         # reuse data fetched by ath_signal_backtest.py
CACHE_MAX_AGE_HOURS  = 168                      # use cache up to 7 days old
MIN_DATA_DAYS        = 252                      # minimum bars needed
BATCH_SIZE           = 50                       # stocks per yfinance batch request

# Derived
PER_POSITION_CAPITAL = INITIAL_CAPITAL / MAX_POSITIONS  # ~33,333 per slot

# Sector filters
USE_SECTOR_FILTER    = True   # enable sector trend + diversity filters
MAX_SAME_SECTOR      = 1      # max positions from the same sector at once
SECTOR_TREND_EMA     = 50     # EMA period for sector index trend detection

# =============================================================================
# SLIPPAGE & TRANSACTION COSTS (NSE equity DELIVERY trades, India)
# =============================================================================
# Slippage: applied to fills that behave like market orders (next-day-open
# entry, and exits that aren't resting limit orders). Stop-loss/target exits
# are modeled as limit orders (see backtest_portfolio) and fill at the exact
# limit price, so slippage does NOT apply to those -- that's the point of a
# limit order.
SLIPPAGE_PCT         = 0.05   # % adverse slippage on market-style fills (buy higher / sell lower)

# Statutory / broker charges. Defaults match Angel One's iTrade Prime plan
# (zero brokerage on equity delivery) as of mid-2026 -- adjust BROKERAGE_PCT /
# BROKERAGE_FLAT if you're on a different plan or broker.
BROKERAGE_PCT        = 0.0    # brokerage % of turnover per order (0 = free delivery)
BROKERAGE_FLAT       = 0.0    # flat Rs per order (use if your broker charges a flat fee instead)
STT_PCT              = 0.1    # Securities Transaction Tax %, charged on BOTH buy and sell (delivery)
EXCHANGE_TXN_PCT     = 0.00297  # NSE transaction charges % (Rs 2.97/lakh/side, effective Oct 2024)
SEBI_PCT             = 0.0001   # SEBI turnover fees % (Rs 10 per crore)
STAMP_DUTY_PCT       = 0.015    # stamp duty %, BUY side only
GST_PCT              = 18.0     # GST % on (brokerage + exchange charges + SEBI fees + DP charge)
DP_CHARGE_PER_SELL   = 20.0     # Rs per scrip on SELL side, DP charge BEFORE GST (Angel One: Rs 20 + GST)


def buy_cost(turnover: float) -> float:
    """Total charges for the buy leg of a delivery trade (Rs)."""
    brokerage = BROKERAGE_FLAT + turnover * BROKERAGE_PCT / 100
    stt       = turnover * STT_PCT / 100
    exch      = turnover * EXCHANGE_TXN_PCT / 100
    sebi      = turnover * SEBI_PCT / 100
    stamp     = turnover * STAMP_DUTY_PCT / 100
    gst       = (brokerage + exch + sebi) * GST_PCT / 100
    return brokerage + stt + exch + sebi + stamp + gst


def sell_cost(turnover: float) -> float:
    """Total charges for the sell leg of a delivery trade (Rs)."""
    brokerage = BROKERAGE_FLAT + turnover * BROKERAGE_PCT / 100
    stt       = turnover * STT_PCT / 100
    exch      = turnover * EXCHANGE_TXN_PCT / 100
    sebi      = turnover * SEBI_PCT / 100
    gst       = (brokerage + exch + sebi + DP_CHARGE_PER_SELL) * GST_PCT / 100
    return brokerage + stt + exch + sebi + gst + DP_CHARGE_PER_SELL

# =============================================================================
# ETF / FUND EXCLUSION PATTERNS
# =============================================================================
ETF_PATTERNS = [
    "ETF", "FUND", "NIFTY", "SENSEX", "LIQUID", "GILT", "BOND",
    "INDEX", "BEES", "JUNIOR", "CPSE",
]

def is_etf(name: str) -> bool:
    n = name.upper()
    return any(p in n for p in ETF_PATTERNS)

# =============================================================================
# SECTOR CONFIGURATION
# =============================================================================
SECTOR_INDEX_MAP = {
    "Banks":   "^NSEBANK",
    "IT":       "^CNXIT",
    "Pharma":   "^CNXPHARMA",
    "Auto":     "^CNXAUTO",
    "FMCG":     "^CNXFMCG",
    "Metal":    "^CNXMETAL",
    "Realty":   "^CNXREALTY",
    "Energy":   "^CNXENERGY",
    "Media":    "^CNXMEDIA",
}

SECTOR_KEYWORDS = [
    ("Banks",   ["bank"]),
    ("Finance",  ["financ", "nbfc", "housing fin", "insurance", "microfinance"]),
    ("IT",       ["software", "it-software", "informat", "computer"]),
    ("Pharma",   ["pharma", "drug", "medicine", "health", "hospital", "biotech", "diagnostic"]),
    ("Auto",     ["auto", "vehicle", "tyre"]),
    ("FMCG",     ["fmcg", "consumer good", "food & beverage", "beverage", "tobacco", "edible oil"]),
    ("Metal",    ["metal", "steel", "alumin", "copper", "zinc", "iron & steel", "mining", "mineral"]),
    ("Realty",   ["realty", "real estate", "construction", "cement"]),
    ("Energy",   ["oil & gas", "power", "energy", "petro", "refin", "electricity"]),
    ("Media",    ["media", "entertainment", "telecom", "broadcasting", "publishing"]),
]

def classify_industry(industry: str) -> str:
    """Map NSE industry string to a broad sector name via keyword matching."""
    ind_lower = industry.lower()
    for sector, keywords in SECTOR_KEYWORDS:
        if any(kw in ind_lower for kw in keywords):
            return sector
    return industry  # fall back to raw industry name

# =============================================================================
# NIFTY 500 UNIVERSE (Nifty50 + NiftyNext50 + Midcap150 + Smallcap250)
# Updated list of ~500 NSE trading symbols for focused backtesting
# =============================================================================
NIFTY_500 = [
    # --- Nifty 50 ---
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH",
    "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "WIPRO", "ONGC", "NTPC", "TECHM", "POWERGRID", "NESTLEIND", "TATAMOTORS",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "TATASTEEL", "BAJAJFINSV",
    "COALINDIA", "HINDALCO", "GRASIM", "CIPLA", "DRREDDY", "APOLLOHOSP",
    "EICHERMOT", "HEROMOTOCO", "DIVISLAB", "TATACONSUM", "BPCL", "BRITANNIA",
    "SHRIRAMFIN", "M&M", "BAJAJ-AUTO", "INDUSINDBK", "SBILIFE", "HDFCLIFE",
    "LTIM", "TRENT", "BEL", "ETERNAL",
    # --- Nifty Next 50 ---
    "ADANIPOWER", "DMART", "VEDL", "HAL", "HINDZINC", "IOC", "ADANIGREEN",
    "TVSMOTOR", "VBL", "ABB", "ADANIENSO", "CHOLAFIN", "DLF", "TORNTPHARM",
    "LODHA", "MAXHEALTH", "INDIGO", "AMBUJACEM", "GODREJCP", "PIDILITIND",
    "BOSCHLTD", "SIEMENS", "HAVELLS", "JINDALSTEL", "ICICIPRULI",
    "ICICIGI", "DABUR", "MOTHERSON", "BANKBARODA", "PFC", "RECLTD",
    "TATAPOWER", "CANBK", "NAUKRI", "MARICO", "ACC", "SRF", "COLPAL",
    "PAGEIND", "OFSS", "MCDOWELL-N", "BERGEPAINT", "ATGL", "CONCOR",
    "INDUSTOWER", "CGPOWER", "PGHH", "NHPC", "IRFC", "INDIANB",
    # --- Nifty Midcap 150 ---
    "FEDERALBNK", "SUZLON", "GICRE", "MCX", "BHEL", "POLYCAB", "PBFINTECH",
    "IDFCFIRSTB", "HDFCAMC", "BHARATFORG", "AUROPHARMA", "LUPIN", "MFSL",
    "PETRONET", "VOLTAS", "TATAELXSI", "OBEROIRLTY", "BALKRISIND",
    "MPHASIS", "ASTRAL", "JUBLFOOD", "SOLARINDS", "MRF", "ABCAPITAL",
    "PIIND", "PERSISTENT", "LTTS", "SAIL", "LICHSGFIN", "IRCTC",
    "CUMMINSIND", "COROMANDEL", "IPCALAB", "ASHOKLEY", "BATAINDIA",
    "KAJARIACER", "GLAND", "HONAUT", "AUBANK", "DIXON", "LALPATHLAB",
    "KEI", "ESCORTS", "ALKEM", "MUTHOOTFIN", "NAVINFLUOR", "METROPOLIS",
    "ZYDUSLIFE", "CROMPTON", "SUNDRMFAST", "BIOCON", "PRESTIGE", "TATACOMM",
    "EXIDEIND", "SYNGENE", "APLAPOLLO", "CRISIL", "SUNTV", "SCHAEFFLER",
    "THERMAX", "DELHIVERY", "ENDURANCE", "SONACOMS", "ATUL", "MSUMI",
    "DEEPAKNTR", "GRINDWELL", "TORNTPOWER", "EMAMILTD", "JKCEMENT",
    "CARBORUNIV", "RELAXO", "AJANTPHARM", "SUMICHEM", "PHOENIXLTD",
    "KPITTECH", "BRIGADE", "FORTIS", "GLENMARK", "PFIZER",
    "GMRAIRPORT", "SRIRAMFIN", "IIFL", "SUPREMEIND", "AIAENG", "TIMKEN",
    "AAVAS", "WHIRLPOOL", "NATCOPHARM", "CDSL", "JBCHEPHARM", "ZEEL",
    "NIACL", "PNBHOUSING", "LINDEINDIA", "CESC", "RADICO", "RBLBANK",
    "MGL", "BANKINDIA", "TATACHEM", "CANFINHOME", "FINEORG", "SUNDARMFIN",
    "AFFLE", "MANAPPURAM", "AMARAJABAT", "SHREECEM", "RAMCOCEM", "LAURUSLABS",
    "NMDC", "APOLLOTYRE", "INDHOTEL", "GODREJPROP", "LODHA", "PVRINOX",
    "ABBOTINDIA", "SANOFI", "GILLETTE", "ZOMATO", "JIOFIN", "PAYTM",
    "POLICYBZR", "KAYNES", "LLOYDSME", "SONATSOFTW", "STARHEALTH",
    "MAZDOCK", "COCHINSHIP", "GRSE", "DALBHARAT",
    # --- Nifty Smallcap 250 ---
    "IDBI", "IOB", "CENTRALBK", "UCOBANK", "MAHABANK", "J&KBANK",
    "KTKBANK", "KARURVYSYA", "CUB", "DCBBANK", "EQUITAS", "UJJIVAN",
    "CREDITACC", "SPANDANA", "MANAPPURAM", "PNBHOUSING", "AAVAS",
    "HOMEFIRST", "APTUS", "REPCO", "CANARABNK",
    "CGCL", "CHOLAHLDNG", "BSE", "CAMS", "KFINTECH", "ANGELONE",
    "NYKAA", "CARTRADE", "EASEMYTRIP",
    "HUDCO", "IRCON", "RVNL", "IREDA", "SJVN", "NLCINDIA", "NHPC",
    "JSWENERGY", "TATAPOWER", "ADANIGREEN", "TORNTPOWER",
    "POWERMECH", "KALPATPOWR", "KEC", "JYOTISTRUC",
    "BAJAJELEC", "BLUESTARCO", "VOLTAS", "AMBER", "ELGIEQUIP",
    "THERMAX", "CUMMINSIND", "KIRLOSENG", "TRITURBINE", "GREAVESCOT",
    "MAHSEAMLES", "RATNAMANI", "GPPL", "GUJGASLTD",
    "IGL", "GSPL", "GAIL", "OIL", "MRPL", "CHENNPETRO", "HINDPETRO",
    "CASTROLIND", "GULFOILLUB", "GNFC", "GSFC", "CHAMBLFERT",
    "DEEPAKFERT", "COROMANDEL", "RALLIS", "UPL", "PIIND",
    "BAYERCROP", "DHANUKA", "SHARDACROP",
    "ERIS", "GRANULES", "CAPLIPOINT", "STRIDES", "IPCA",
    "LAURUS", "GLENMARK", "AUROPHARMA", "ALKEM",
    "NATCOPHARM", "BIOCON", "SYNGENE", "THYROCARE", "POLYMED",
    "MAXHEALTH", "MEDANTA", "RAINBOW", "NH", "KIMS", "YATHARTH",
    "ASIANPAINT", "BERGEPAINT", "KANSAINER", "AKZOINDIA",
    "CENTURYPLY", "GREENPANEL", "CERA", "KAJARIACER", "SHREECEM",
    "ULTRACEMCO", "AMBUJACEM", "ACC", "JKCEMENT", "RAMCOCEM",
    "JKLAKSHMI", "HEIDELBERG", "DALBHARAT", "STARCEMENT", "BIRLACORPN",
    "JSWSTEEL", "TATASTEEL", "HINDALCO", "VEDL",
    "NMDC", "NATIONALUM", "HINDCOPPER", "MOIL", "COALINDIA",
    "WELCORP", "JINDALSAW", "APL",
    "TATAMOTORS", "M&M", "MARUTI", "EICHERMOT", "HEROMOTOCO",
    "BAJAJ-AUTO", "TVSMOTOR", "ASHOKLEY", "FORCEMOT", "ESCORTS",
    "SONACOMS", "ENDURANCE", "SUNDRMFAST", "MOTHERSON",
    "BOSCHLTD", "EXIDEIND", "AMARAJABAT", "CEATLTD", "APOLLOTYRE",
    "BALKRISIND", "MRF", "JKTYRE",
    "GRINDWELL", "CARBORUNIV", "SUPRAJIT", "SCHAEFFLER",
    "SKFINDIA", "TIMKEN", "FIVESTAR",
    "TITAN", "KALYANAJEW", "SENCO", "PCJEWELLER", "RAJESHEXPO",
    "TASTYBITE", "EIDPARRY", "DALMIASUGAR", "BALRAMCHIN", "RENUKA",
    "BRITANNIA", "NESTLEIND", "TATACONSUM", "HINDUNILVR", "ITC",
    "COLPAL", "DABUR", "MARICO", "GODREJCP", "EMAMILTD",
    "VBL", "CCL", "RADICO", "UBL", "MCDOWELL-N",
    "JYOTHYLAB", "HATSUN", "HERITGFOOD", "BIKAJI", "GODFRYPHLP",
    "VSTIND",
    "DMART", "TRENT", "SHOPERSTOP", "VMART", "TITAN",
    "PAGEIND", "LUXIND", "RUPA", "BATA",
    "ZOMATO", "SWIGGY", "NYKAA", "DELHIVERY", "CARTRADE",
    "INDIGO", "SPICEJET", "IRCTC",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
    "PHOENIXLTD", "LODHA", "SOBHA", "SUNTECK", "MAHLIFE",
    "KOLTEPATIL", "ANANTRAJ", "RAYMOND",
    "LT", "LTIM", "LTTS", "HCLTECH", "WIPRO", "TECHM",
    "INFY", "TCS", "MPHASIS", "PERSISTENT", "COFORGE",
    "TATAELXSI", "KPITTECH", "CYIENT", "ECLERX",
    "NAUKRI", "OFSS", "INTELLECT", "MASTEK", "SONATSOFTW",
    "HAPPSTMNDS", "NEWGEN", "ROUTE", "LATENTVIEW", "TANLA",
    "NETWORK18", "TV18BRDCST", "SUNTV", "PVRINOX", "ZEEL",
    "STAR", "SAREGAMA", "TIPS",
    "BHARTIARTL", "INDUSTOWER", "HFCL", "STLTECH", "RAILTEL",
    "TTML", "ROUTE",
    "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI", "GICRE",
    "NIACL", "STARHEALTH", "MAXFIN", "GODIGIT",
    "ISEC", "HDFCAMC", "MFSL", "CAMS", "KFINTECH",
    "BSE", "MCX", "CDSL", "ANGELONE",
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "CHOLAFIN",
    "M&MFIN", "MANAPPURAM", "MUTHOOTFIN", "IIFL", "POONAWALLA",
    "ABCAPITAL", "LICHSGFIN", "PFC", "RECLTD", "IRFC",
    "HUDCO", "CANFINHOME", "AAVAS", "HOMEFIRST", "APTUS",
    "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK",
    "AUBANK", "EQUITAS", "UJJIVAN", "CREDITACC", "DCBBANK",
    "IDBI", "IOB", "CENTRALBK", "INDIANB", "MAHABANK",
    "BANKINDIA", "UCOBANK", "BANKBARODA", "CANBK", "PNB", "SBIN",
    "KOTAKBANK", "HDFCBANK", "ICICIBANK", "AXISBANK", "INDUSINDBK",
    "J&KBANK", "KTKBANK", "KARURVYSYA", "CUB",
    "POWERGRID", "NTPC", "NHPC", "SJVN", "NLCINDIA",
    "TATAPOWER", "JSWENERGY", "ADANIGREEN", "ADANIPOWER", "TORNTPOWER",
    "BHEL", "SIEMENS", "ABB", "CGPOWER", "HAVELLS", "POLYCAB",
    "SUZLON", "KAYNES", "DIXON", "AFFLE",
    "MAZDOCK", "COCHINSHIP", "GRSE", "BDL", "BEL", "HAL",
    "PARAS", "SOLARINDS", "DATAPATTNS",
    "BPCL", "IOC", "HINDPETRO", "ONGC", "OIL", "GAIL",
    "PETRONET", "IGL", "MGL", "ATGL",
    "CONCOR", "IRCON", "RVNL", "IRCTC",
    "GODREJIND", "PIDILITIND", "SRF", "ATUL", "DEEPAKNTR",
    "NAVINFLUOR", "FINEORG", "CLEAN", "GALAXYSURF",
    "LALPATHLAB", "METROPOLIS", "FORTIS",
    "ICRA", "CRISIL",
    "MMTC", "MOIL", "NMDC", "NFL", "RCF",
    "FACT", "RITES", "NBCC", "NCC", "BDL",
    "ITI", "RAILTEL", "IRCON", "RVNL",
    "PTC", "SJVN", "NHPC", "POWERGRID", "NTPC",
    "TATACHEM", "GRASIM", "HINDALCO", "JSWSTEEL", "TATASTEEL",
    "APLAPOLLO", "ASTRAL", "SUPREMEIND",
    "KEI", "POLYCAB", "HAVELLS", "CROMPTON", "WHIRLPOOL",
    "BLUESTARCO", "VOLTAS", "BAJAJELEC", "AMBER",
    "DIXON", "KAYNES", "HONAUT",
    "RELIANCE", "ADANIENT", "ADANIPORTS", "ADANIPOWER", "ADANIGREEN",
    "ADANIENSO", "ATGL",
    # --- Additional Smallcap / Midcap stocks to complete ~500 ---
    "AARTI", "AARTIIND", "ABSLAMC", "ALOKINDS", "ANANDRATHI", "APARINDS",
    "ARE&M", "ARIES", "ASAHIINDIA", "ASTRAZEN", "BEML", "BLUEDART",
    "BORORENEW", "BSOFT", "CAMPUS", "CAPLIN", "CENTURYTEX", "CHALET",
    "CHAMBAL", "CHOICEIN", "CMSINFO", "CONCORDBIO", "CRAFTSMAN",
    "DBCORP", "DCMSHRIRAM", "DELTACORP", "DEVYANI", "EDELWEISS", "ELECON",
    "ENGINERSIN", "FDC", "FINCABLES", "FINPIPE", "FLUOROCHEM",
    "GARFIBRES", "GESHIP", "GHCL", "GMDCLTD", "GOCOLORS", "GODREJAGRO",
    "GOODYEAR", "GRPLTD", "GUJALKALI", "HBLPOWER", "HGS", "HIKAL",
    "HINDWAREAP", "IBULHSGFIN", "IFBIND", "INDIAMART", "INDIGOPNTS",
    "INOXWIND", "JAMNAAUTO", "JBM", "JMFINANCIL", "JSL", "JTEKTINDIA",
    "JUBLINGREA", "JUBLPHARMA", "JUSTDIAL", "KENNAMET", "KNRCON",
    "KPRMILL", "KRBL", "KRSNAA", "LAXMIMACH", "LEMONTREE", "MAPMYINDIA",
    "MAXIND", "MIDHANI", "MINDACORP", "MOTILALOFS", "NESCO", "NILKAMAL",
    "OLECTRA", "ORIENTELEC", "PIRAMAL", "POLYMED", "PPLPHARMA",
    "PRINCEPIPE", "PRSMJOHNSN", "QUESS", "REDINGTON", "ROSSARI",
    "SAPPHIRE", "SHILPAMED", "SHYAMMETL", "SIS", "SOUTHBANK",
    "SUDARSCHEM", "SWANENERGY", "SYMPHONY", "TARSONS", "TEAMLEASE",
    "TECHNOE", "TEGA", "TIINDIA", "TRIDENT", "TTKPRESTIG", "UFLEX",
    "UTIAMC", "VAIBHAVGBL", "VARUN", "VIPIND", "VRLLOG", "WABAG",
    "WESTLIFE", "WOCKPHARMA", "WONDERLA", "ZENSARTECH",
]

# De-duplicate the list (some stocks appear in multiple sub-indices)
NIFTY_500 = sorted(set(NIFTY_500))

# =============================================================================
# NSE UNIVERSE - Yahoo Finance
# =============================================================================
def get_nse_universe(cache_file: str = "nse_universe_yf.pkl") -> list:
    """Return list of NSE EQ symbols. Downloads from NSE public CSV with 24h cache."""
    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 24:
            with open(cache_file, "rb") as fh:
                symbols = pickle.load(fh)
            print(f"  NSE universe: {len(symbols):,} symbols (cache, {age_h:.1f}h old)")
            return symbols

    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        resp = requests.get(
            url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        resp.raise_for_status()
        eq_df = pd.read_csv(StringIO(resp.text))
        if "SYMBOL" not in eq_df.columns:
            raise ValueError("SYMBOL column missing")

        symbols = [str(s).strip() for s in eq_df["SYMBOL"] if not is_etf(str(s))]
        symbols = sorted(set(symbols))

        with open(cache_file, "wb") as fh:
            pickle.dump(symbols, fh)
        print(f"  NSE universe: {len(symbols):,} symbols (from NSE public CSV)")
        return symbols

    except Exception as e:
        print(f"  WARNING: NSE CSV unavailable ({e}). Using NIFTY_500 fallback.")
        return list(NIFTY_500)


# =============================================================================
# BATCH OHLCV FETCH - Yahoo Finance
# =============================================================================
def fetch_batch_yf(symbols: list, years: int = FETCH_YEARS) -> dict:
    """Download adjusted OHLCV for a batch of NSE symbols via yfinance."""
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    tickers_ns = [f"{s}.NS" for s in symbols]

    try:
        raw = yf.download(
            tickers_ns,
            start=start, end=end,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    if raw is None or raw.empty:
        return {}

    results = {}
    multi = len(symbols) > 1

    for sym, ticker_ns in zip(symbols, tickers_ns):
        try:
            df = raw[ticker_ns].copy() if multi else raw.copy()

            if df.empty:
                continue

            df.columns = [str(c).lower() for c in df.columns]
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                continue

            df = df[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(subset=["close"], inplace=True)
            df = df[(df["close"] > 0) & (df["volume"] >= 0)]

            if len(df) >= MIN_DATA_DAYS:
                results[sym] = df

        except Exception:
            continue

    return results


# =============================================================================
# NIFTY REFERENCE DATA (for RS calculation)
# =============================================================================
def fetch_nifty_close(years: int = FETCH_YEARS) -> pd.Series:
    """Fetch Nifty 50 index close prices from Yahoo Finance."""
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    try:
        raw = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"].squeeze()
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_convert(None)
        close.name = "close"
        return close
    except Exception:
        return pd.Series(dtype=float)

# =============================================================================
# SECTOR MAP  (symbol -> broad sector)
# =============================================================================
def get_sector_map(cache_file: str = "sector_map_yf.pkl") -> dict:
    """Download Nifty 500 constituent list from NSE to get stock->sector mapping."""
    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 168:
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)
    sector_map = {}
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        resp = requests.get(
            url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        ind_col = next((c for c in df.columns if "industr" in c.lower()), None)
        if sym_col and ind_col:
            for _, row in df.iterrows():
                sym = str(row[sym_col]).strip()
                ind = str(row[ind_col]).strip()
                if sym:
                    sector_map[sym] = classify_industry(ind)
            with open(cache_file, "wb") as fh:
                pickle.dump(sector_map, fh)
            print(f"  Sector map: {len(sector_map):,} stocks mapped to {len(set(sector_map.values()))} sectors")
        else:
            print(f"  WARNING: Sector CSV columns not as expected. Sector filter partial.")
    except Exception as e:
        print(f"  WARNING: Could not load sector map ({e}). Sector filters may be limited.")
    return sector_map

# =============================================================================
# SECTOR TRENDS  (sector -> daily trending bool)
# =============================================================================
def fetch_sector_trends(years: int = FETCH_YEARS) -> dict:
    """
    Fetch NSE sector index data and compute trending status (close > EMA50) per day.
    Returns: {sector_name: pd.Series(bool, index=DatetimeIndex)}
    """
    sector_trends = {}
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")

    tickers = list(SECTOR_INDEX_MAP.values())
    sectors = list(SECTOR_INDEX_MAP.keys())

    try:
        raw = yf.download(
            tickers, start=start, auto_adjust=True,
            group_by="ticker", threads=True, progress=False,
        )
        if raw is None or raw.empty:
            return sector_trends

        for sector, ticker in zip(sectors, tickers):
            try:
                close = raw[ticker]["Close"].squeeze() if len(tickers) > 1 else raw["Close"].squeeze()
                close = close.dropna()
                if close.empty:
                    continue
                close.index = pd.to_datetime(close.index)
                if close.index.tz is not None:
                    close.index = close.index.tz_convert(None)
                ema = close.ewm(span=SECTOR_TREND_EMA, adjust=False).mean()
                sector_trends[sector] = (close > ema)
            except Exception:
                continue
    except Exception as e:
        print(f"  WARNING: sector trend fetch failed ({e}). Trend filter disabled.")

    if sector_trends:
        trending_now = sum(1 for s in sector_trends.values() if not s.empty and bool(s.iloc[-1]))
        print(f"  Sector trends: {len(sector_trends)} sectors tracked, {trending_now} currently trending")
    return sector_trends

# =============================================================================
# INDICATOR COMPUTATION
# =============================================================================
def compute_indicators(df: pd.DataFrame, nifty_close: pd.Series | None = None) -> pd.DataFrame | None:
    if len(df) < MIN_DATA_DAYS:
        return None

    d = df.copy()

    # Normalize to timezone-naive (cached files may have UTC+05:30 from yfinance)
    if d.index.tz is not None:
        d.index = d.index.tz_convert(None)
    if nifty_close is not None and nifty_close.index.tz is not None:
        nifty_close = nifty_close.copy()
        nifty_close.index = nifty_close.index.tz_convert(None)

    # ---- EMAs on Close ----
    d["EMA5"]   = d["close"].ewm(span=5,   adjust=False).mean()
    d["EMA10"]  = d["close"].ewm(span=10,  adjust=False).mean()
    d["EMA20"]  = d["close"].ewm(span=20,  adjust=False).mean()
    d["EMA50"]  = d["close"].ewm(span=50,  adjust=False).mean()
    d["EMA150"] = d["close"].ewm(span=150, adjust=False).mean()
    d["EMA200"] = d["close"].ewm(span=200, adjust=False).mean()

    # ---- ATR14 ----
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(span=14, adjust=False).mean()

    # ---- Key highs (on High column) ----
    d["High_1Y"] = d["high"].rolling(252,  min_periods=200).max()
    d["High_5Y"] = d["high"].rolling(1260, min_periods=252).max()
    d["ATH"]     = d["high"].cummax()

    # ---- EMA alignment ----
    d["EMA_Aligned"] = (
        (d["close"] > d["EMA5"])    &
        (d["EMA5"]  > d["EMA10"])   &
        (d["EMA10"] > d["EMA20"])   &
        (d["EMA20"] > d["EMA50"])   &
        (d["EMA50"] > d["EMA150"])  &
        (d["EMA150"]> d["EMA200"])
    )

    # ---- Near-high conditions (raw proximity) ----
    d["Near_1Y"]  = (d["close"] >= d["High_1Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_1Y"] * NEAR_HIGH_UPPER)
    d["Near_5Y"]  = (d["close"] >= d["High_5Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_5Y"] * NEAR_HIGH_UPPER)
    d["Near_ATH"] = (d["close"] >= d["ATH"]     * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["ATH"]      * NEAR_HIGH_UPPER)

    # Fresh-high filter: exclude if stock already touched ref high within last FRESH_HIGH_DAYS bars
    touched_1Y  = (d["high"] >= d["High_1Y"].shift(1)).fillna(False)
    touched_5Y  = (d["high"] >= d["High_5Y"].shift(1)).fillna(False)
    touched_ATH = (d["high"] >= d["ATH"].shift(1)).fillna(False)

    d["Touched_1Y_Recent"]  = touched_1Y.rolling(FRESH_HIGH_DAYS, min_periods=1).sum() > 0
    d["Touched_5Y_Recent"]  = touched_5Y.rolling(FRESH_HIGH_DAYS, min_periods=1).sum() > 0
    d["Touched_ATH_Recent"] = touched_ATH.rolling(FRESH_HIGH_DAYS, min_periods=1).sum() > 0

    # Override: near-high only valid if the high hasn't been touched recently
    d["Near_1Y"]  = d["Near_1Y"]  & ~d["Touched_1Y_Recent"]
    d["Near_5Y"]  = d["Near_5Y"]  & ~d["Touched_5Y_Recent"]
    d["Near_ATH"] = d["Near_ATH"] & ~d["Touched_ATH_Recent"]

    d["Near_High"] = d["Near_1Y"] | d["Near_5Y"] | d["Near_ATH"]

    # ---- Weekly EMAs (resampled to week-end, forward-filled to daily) ----
    weekly = d["close"].resample("W").last().ffill()
    w_ema10 = weekly.ewm(span=10, adjust=False).mean()
    w_ema20 = weekly.ewm(span=20, adjust=False).mean()
    d["W_EMA10"] = w_ema10.reindex(d.index, method="ffill")
    d["W_EMA20"] = w_ema20.reindex(d.index, method="ffill")
    d["Weekly_Uptrend"] = d["close"] > d["W_EMA20"]

    # ---- Volume indicators ----
    d["Vol20_Avg"]    = d["volume"].rolling(20, min_periods=10).mean()
    d["Vol_Above_Avg"] = d["volume"] > d["Vol20_Avg"]

    # ---- Relative Strength vs Nifty ----
    if nifty_close is not None and not nifty_close.empty:
        nifty_aligned = nifty_close.reindex(d.index, method="ffill")
        stock_ret  = d["close"].pct_change(RS_LOOKBACK)
        nifty_ret  = nifty_aligned.pct_change(RS_LOOKBACK)
        d["RS_Leader"] = stock_ret > nifty_ret
    else:
        d["RS_Leader"] = True   # fallback when Nifty data unavailable

    # ---- Mini-Base (volatility contraction) ----
    # Check if the last MINI_BASE_BARS bars have ATR contracting (each < previous)
    atr_declining = d["ATR14"].diff() < 0
    d["Mini_Base"] = (
        atr_declining.rolling(MINI_BASE_BARS - 1, min_periods=MINI_BASE_BARS - 1).sum()
        >= MINI_BASE_MIN
    ).fillna(False)

    # ---- Swing High (10-bar rolling max, shifted by 1 to avoid lookahead) ----
    d["Swing_Hi10"] = d["high"].rolling(10).max().shift(1)

    # ---- Wedge Pop ----
    prev_mini = d["Mini_Base"].shift(1).fillna(False)
    d["Wedge_Pop"] = (
        (d["close"]  > d["Swing_Hi10"]) &
        prev_mini &
        d["EMA_Aligned"] &
        d["Weekly_Uptrend"] &
        d["Vol_Above_Avg"] &
        d["RS_Leader"]
    )

    # ---- Crossback (bounce off EMA20 from above) ----
    # Price dipped to EMA20 and recovered: previous low touched EMA20, current close above EMA20
    prev_low    = d["low"].shift(1)
    prev_ema20  = d["EMA20"].shift(1)
    # Touched EMA20 = prev_low <= prev_EMA20 <= prev_close (dipped to EMA20 but didn't close below)
    touched_ema20 = (prev_low <= prev_ema20) & (d["close"].shift(1) >= prev_ema20)
    d["Crossback"] = (
        touched_ema20 &
        (d["close"] > d["EMA20"]) &
        d["Weekly_Uptrend"] &
        d["Vol_Above_Avg"] &
        d["RS_Leader"]
    )

    # ---- Extension exit indicator ----
    d["Extension"] = d["close"] > (d["EMA20"] + EXTENSION_MULT * d["ATR14"])

    # ---- Combined Signal ----
    d["Oliver_Signal"] = d["Wedge_Pop"] | d["Crossback"]
    d["Signal"]        = d["EMA_Aligned"] & d["Near_High"] & d["Oliver_Signal"]

    # Ranking indicators
    d["ROC_90"]        = d["close"].pct_change(ROC_PERIOD) * 100
    d["Momentum_90"]   = d["close"] / d["close"].shift(ROC_PERIOD)
    d["Vol_Surge"]     = d["volume"] / d["volume"].rolling(20, min_periods=10).mean()
    d["Pct_From_ATH"]  = (d["ATH"] - d["close"]) / d["ATH"] * 100

    d.dropna(subset=["EMA200", "High_1Y"], inplace=True)
    return d if len(d) > 0 else None


# =============================================================================
# RANK SCORE FUNCTION
# =============================================================================
def rank_score(row, close_price, ref_high, nifty_close=None, day=None):
    if RANK_METHOD == "roc":
        val = row.get("ROC_90", 0)
        return float(val) if pd.notna(val) else 0.0

    elif RANK_METHOD == "rs_nifty":
        stock_ret = float(row.get("ROC_90", 0)) if pd.notna(row.get("ROC_90", 0)) else 0.0
        if nifty_close is not None and not nifty_close.empty and day is not None:
            try:
                nifty_at = nifty_close.asof(day)
                nifty_past = nifty_close.asof(day - pd.Timedelta(days=RS_RANK_PERIOD))
                if pd.notna(nifty_at) and pd.notna(nifty_past) and nifty_past > 0:
                    nifty_ret = (nifty_at - nifty_past) / nifty_past * 100
                    return stock_ret - nifty_ret
            except Exception:
                pass
        return stock_ret

    elif RANK_METHOD == "near_ath":
        val = row.get("Pct_From_ATH", 100)
        return -float(val) if pd.notna(val) else -100.0

    elif RANK_METHOD == "volume_surge":
        val = row.get("Vol_Surge", 0)
        return float(val) if pd.notna(val) else 0.0

    else:  # "momentum" (default - 90 day momentum)
        val = row.get("Momentum_90", 0)
        return float(val) if pd.notna(val) else 0.0


# =============================================================================
# REFERENCE HIGH SELECTION
# =============================================================================
def get_ref_high(row) -> tuple[float, str]:
    """
    Return (ref_high_price, ref_high_type) using the smallest triggered high.
    """
    candidates = []
    if row["Near_1Y"]  and pd.notna(row["High_1Y"]):
        candidates.append((row["High_1Y"], "1Y"))
    if row["Near_5Y"]  and pd.notna(row["High_5Y"]):
        candidates.append((row["High_5Y"], "5Y"))
    if row["Near_ATH"] and pd.notna(row["ATH"]):
        candidates.append((row["ATH"],     "ATH"))
    if not candidates:
        return float("nan"), "None"
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def signal_type_label(ref_high_type: str, oliver_type: str) -> str:
    return f"NearHigh_{oliver_type}_{ref_high_type}"

# =============================================================================
# MOMENTUM SCORE (for ranking same-day entry candidates)
# =============================================================================
# =============================================================================
# PORTFOLIO-BASED BACKTEST
# =============================================================================
def backtest_portfolio(
    stock_data: dict[str, pd.DataFrame],
    nifty_close=None,
    sector_map: dict | None = None,
    sector_trends: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run day-by-day portfolio simulation across all stocks.

    Args:
        stock_data: {symbol: DataFrame} where each DataFrame has all indicators computed.

    Returns:
        (trades, equity_curve) where:
          trades      = list of trade dicts
          equity_curve = list of {Date, Equity, Drawdown_Pct, Open_Positions, ...}
    """
    today    = datetime.date.today()
    bt_start = today - datetime.timedelta(days=LOOKBACK_DAYS)

    # ---- Collect all unique trading dates across all stocks ----
    all_dates = set()
    stock_bt_data = {}  # {symbol: DataFrame} restricted to backtest window

    for symbol, df in stock_data.items():
        bt_df = df[df.index.date >= bt_start].copy()
        if bt_df.empty:
            continue
        stock_bt_data[symbol] = bt_df
        all_dates.update(bt_df.index)

    if not all_dates:
        return [], [], {}

    all_dates = sorted(all_dates)

    # ---- Portfolio state ----
    cash = float(INITIAL_CAPITAL)
    open_positions = {}  # symbol -> position dict
    trades = []
    equity_curve = []
    peak_equity = float(INITIAL_CAPITAL)
    max_concurrent = 0

    # Signals detected after today's close are queued here and filled at
    # NEXT trading day's open (no same-day lookahead entry).
    pending_entries = []

    # Diagnostics: trace where candidates drop out of the entry funnel
    funnel = {
        "signals_raw": 0,          # rows with Signal True on a scanned day (pre-filter)
        "lost_sector_trend": 0,    # skipped: sector index not trending
        "lost_sector_diversity_scan": 0,  # skipped at scan time: sector slot taken
        "queued": 0,               # accepted into pending_entries
        "lost_no_data_next_day": 0,       # stock didn't trade on the fill day
        "lost_sector_diversity_fill": 0,  # skipped at fill time: sector slot taken
        "lost_capital": 0,         # not enough cash / <1 share at fill time
        "filled": 0,               # successfully entered
    }

    # Pre-build per-stock row lookups for fast access
    # For each stock, build a dict: {timestamp -> row_as_dict}
    stock_row_lookup = {}
    for symbol, bt_df in stock_bt_data.items():
        row_dict = {}
        for ts, row in bt_df.iterrows():
            row_dict[ts] = row
        stock_row_lookup[symbol] = row_dict

    # ---- Day-by-day simulation ----
    for dt in all_dates:
        # ================================================================
        # STEP 0: FILL PENDING ENTRIES (signals detected at yesterday's
        # close, executed at today's open — models placing the order after
        # market close and having it filled the next trading day)
        # ================================================================
        if pending_entries:
            open_positions_value = 0.0
            for symbol, pos in open_positions.items():
                row = stock_row_lookup.get(symbol, {}).get(dt)
                if row is not None:
                    open_positions_value += pos["shares"] * float(row["close"])
                else:
                    open_positions_value += pos["allocated_capital"]

            current_equity  = cash + open_positions_value
            position_size   = current_equity / MAX_POSITIONS
            available_fills = MAX_POSITIONS - len(open_positions)

            from collections import Counter
            sectors_held_fill = Counter(pos["sector"] for pos in open_positions.values())

            filled = 0
            for cand in pending_entries:
                if filled >= available_fills:
                    break

                symbol = cand["symbol"]
                if symbol in open_positions:
                    continue

                row = stock_row_lookup.get(symbol, {}).get(dt)
                if row is None:
                    funnel["lost_no_data_next_day"] += 1
                    continue  # stock didn't trade today; order can't fill, signal lost

                sec = cand["sector"]
                if USE_SECTOR_FILTER and sectors_held_fill.get(sec, 0) >= MAX_SAME_SECTOR:
                    funnel["lost_sector_diversity_fill"] += 1
                    continue  # sector slot already taken by another fill today

                raw_open = float(row["open"])
                if raw_open <= 0:
                    funnel["lost_capital"] += 1
                    continue
                # Market-style fill at next day's open: adverse slippage applies
                ep = raw_open * (1 + SLIPPAGE_PCT / 100)

                sp = round(ep * (1 - STOP_LOSS_PCT), 2)
                tp = round(cand["ref_high"] * (1 + TARGET_ABOVE_HIGH_PCT), 2)

                alloc = min(position_size, cash)
                if alloc < 1.0:
                    funnel["lost_capital"] += 1
                    continue

                shares = int(alloc / ep)
                if shares < 1:
                    funnel["lost_capital"] += 1
                    continue

                actual_cost = shares * ep
                entry_charges = buy_cost(actual_cost)
                total_outlay  = actual_cost + entry_charges
                if total_outlay > cash:
                    funnel["lost_capital"] += 1
                    continue

                cash -= total_outlay
                sectors_held_fill[sec] += 1
                filled += 1
                funnel["filled"] += 1

                open_positions[symbol] = {
                    "entry_price":      ep,
                    "stop_price":       sp,
                    "target_price":     tp,
                    "ref_high":         cand["ref_high"],
                    "ref_high_type":    cand["ref_high_type"],
                    "oliver_type":      cand["oliver_type"],
                    "entry_date":       dt,
                    "bars_held":        0,
                    "shares":           shares,
                    "allocated_capital": actual_cost,
                    "entry_cost":       entry_charges,
                    "sector":           sec,
                }

            pending_entries = []

        # ================================================================
        # STEP 1: CHECK ALL OPEN POSITIONS FOR EXIT CONDITIONS
        # ================================================================
        symbols_to_close = []

        for symbol, pos in open_positions.items():
            if symbol not in stock_row_lookup:
                continue
            row = stock_row_lookup[symbol].get(dt)
            if row is None:
                continue  # stock did not trade this day

            pos["bars_held"] += 1
            bars_held = pos["bars_held"]
            exit_reason = None
            exit_price  = 0.0

            # 1. Stop loss as a LIMIT sell order: fills only if price actually
            #    traded at/through the stop level today (Low <= stop <= High).
            #    A gap-down that skips over the level entirely (High < stop)
            #    leaves the limit order unfilled -- position stays open and
            #    is re-checked the next day (real gap risk of GTT/limit stops).
            if row["low"] <= pos["stop_price"] <= row["high"]:
                exit_price  = pos["stop_price"]
                exit_reason = "Stop_Loss"

            # 2. Target as a LIMIT sell order: fills if High >= target
            elif row["high"] >= pos["target_price"]:
                exit_price  = pos["target_price"]
                exit_reason = "Target_Hit"

            # 3. Oliver trailing EMA stop (only after MIN_HOLD_DAYS)
            elif bars_held > MIN_HOLD_DAYS:
                trail_ema = row["EMA10"] if bars_held > MIN_HOLD_DAYS else row["EMA20"]
                if row["close"] < trail_ema:
                    exit_price  = float(row["close"])
                    exit_reason = "Oliver_EMA_Stop"

            # 4. Extension exit (only after MIN_HOLD_DAYS)
            if exit_reason is None and bars_held > MIN_HOLD_DAYS:
                if bool(row["Extension"]):
                    exit_price  = float(row["close"])
                    exit_reason = "Extension_Exit"

            if exit_reason:
                # Stop_Loss / Target_Hit are resting LIMIT orders -> fill at
                # the exact limit price, no slippage. Oliver_EMA_Stop /
                # Extension_Exit close out at market (close price) -> slippage applies.
                if exit_reason not in ("Stop_Loss", "Target_Hit"):
                    exit_price = exit_price * (1 - SLIPPAGE_PCT / 100)

                invested       = pos["allocated_capital"] + pos["entry_cost"]
                sell_turnover  = exit_price * pos["shares"]
                exit_charges   = sell_cost(sell_turnover)
                net_proceeds   = sell_turnover - exit_charges
                pnl_rs         = net_proceeds - invested
                pnl_pct        = pnl_rs / invested * 100
                total_costs    = pos["entry_cost"] + exit_charges

                trades.append({
                    "Symbol":         symbol,
                    "Sector":         pos.get("sector", "Unknown"),
                    "Entry_Date":     pos["entry_date"],
                    "Exit_Date":      dt,
                    "Signal_Type":    signal_type_label(pos["ref_high_type"], pos["oliver_type"]),
                    "Entry_Price":    round(pos["entry_price"], 2),
                    "Exit_Price":     round(exit_price, 2),
                    "Ref_High":       round(pos["ref_high"], 2),
                    "Ref_High_Type":  pos["ref_high_type"],
                    "Oliver_Type":    pos["oliver_type"],
                    "Target_Price":   round(pos["target_price"], 2),
                    "Stop_Price":     round(pos["stop_price"], 2),
                    "Shares":         pos["shares"],
                    "PnL_Rs":         round(pnl_rs, 2),
                    "PnL_Pct":        round(pnl_pct, 2),
                    "Hold_Days":      bars_held,
                    "Exit_Reason":    exit_reason,
                    "Entry_Month":    pos["entry_date"].strftime("%Y-%m"),
                    "Allocated_Capital": round(pos["allocated_capital"], 2),
                    "Costs_Rs":       round(total_costs, 2),
                })

                # Credit net sale proceeds (invested amount was already
                # deducted from cash at entry time, including entry costs)
                cash += net_proceeds
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del open_positions[s]

        # ================================================================
        # STEP 2: SCAN FOR NEW ENTRY SIGNALS (after today's close)
        # Accepted candidates are queued into pending_entries and filled at
        # NEXT trading day's open (STEP 0) -- not entered same-day.
        # ================================================================
        available_slots = MAX_POSITIONS - len(open_positions)

        if available_slots > 0:
            # Track how many positions we hold per sector (for diversity cap),
            # plus how many pending slots this scan has already claimed today.
            from collections import Counter
            sectors_held = Counter(pos["sector"] for pos in open_positions.values())

            # Compute which sectors are trending TODAY — once per day, not per candidate
            # (avoids O(candidates) asof() calls; replaces with O(sectors) = ~9 calls/day)
            ts_key = pd.Timestamp(dt)
            trending_sectors = set()
            if USE_SECTOR_FILTER and sector_trends:
                for _sec, _ts in sector_trends.items():
                    if not _ts.empty:
                        val = _ts.asof(ts_key)
                        if pd.notna(val) and bool(val):
                            trending_sectors.add(_sec)

            entry_candidates = []

            for symbol, row_dict in stock_row_lookup.items():
                if symbol in open_positions:
                    continue  # already holding this stock
                row = row_dict.get(dt)
                if row is None:
                    continue
                if not row["Signal"]:
                    continue
                funnel["signals_raw"] += 1

                rh, rh_type = get_ref_high(row)
                if pd.isna(rh) or rh <= 0:
                    continue

                # Determine Oliver sub-type (Wedge_Pop takes priority)
                if bool(row["Wedge_Pop"]):
                    ok_type = "Wedge"
                elif bool(row["Crossback"]):
                    ok_type = "Crossback"
                else:
                    continue

                close_price = float(row["close"])
                score = rank_score(row, close_price, rh, nifty_close, pd.Timestamp(dt))

                entry_candidates.append({
                    "symbol":      symbol,
                    "close":       close_price,
                    "ref_high":    rh,
                    "ref_high_type": rh_type,
                    "oliver_type": ok_type,
                    "rank_score":  score,
                    "row":         row,
                })

            # Sort by ranking method (highest score first)
            entry_candidates.sort(key=lambda x: x["rank_score"], reverse=True)

            # Iterate ALL candidates; sector filters may skip top-ranked ones.
            # Queue accepted candidates for tomorrow's open -- no same-day fill.
            slots_queued = 0
            for cand in entry_candidates:
                if slots_queued >= available_slots:
                    break

                symbol = cand["symbol"]

                # Determine sector for this candidate
                sec = (sector_map or {}).get(symbol, symbol)

                # --- Sector filters (when enabled) ---
                if USE_SECTOR_FILTER:
                    # 1. Trend check: sector index must be above its EMA
                    #    trending_sectors was precomputed once for today (not per candidate)
                    if sec in sector_trends and sec not in trending_sectors:
                        funnel["lost_sector_trend"] += 1
                        continue

                    # 2. Diversity check: cap same-sector positions (including
                    #    other candidates already queued from this same scan)
                    if sectors_held.get(sec, 0) >= MAX_SAME_SECTOR:
                        funnel["lost_sector_diversity_scan"] += 1
                        continue

                sectors_held[sec] += 1
                slots_queued += 1
                funnel["queued"] += 1

                pending_entries.append({
                    "symbol":        symbol,
                    "ref_high":      cand["ref_high"],
                    "ref_high_type": cand["ref_high_type"],
                    "oliver_type":   cand["oliver_type"],
                    "sector":        sec,
                })

        # ================================================================
        # STEP 3: TRACK EQUITY CURVE
        # ================================================================
        positions_value = 0.0
        for symbol, pos in open_positions.items():
            row = stock_row_lookup.get(symbol, {}).get(dt)
            if row is not None:
                positions_value += pos["shares"] * float(row["close"])
            else:
                # Use last known price (entry price as fallback)
                positions_value += pos["allocated_capital"]

        total_equity = cash + positions_value
        peak_equity = max(peak_equity, total_equity)
        drawdown_pct = (total_equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0

        n_open = len(open_positions)
        max_concurrent = max(max_concurrent, n_open)

        equity_curve.append({
            "Date":            dt,
            "Equity":          round(total_equity, 2),
            "Cash":            round(cash, 2),
            "Positions_Value": round(positions_value, 2),
            "Open_Positions":  n_open,
            "Peak_Equity":     round(peak_equity, 2),
            "Drawdown_Pct":    round(drawdown_pct, 2),
        })

    # ================================================================
    # CLOSE ANY REMAINING OPEN POSITIONS AT END OF DATA
    # ================================================================
    if open_positions and all_dates:
        last_date = all_dates[-1]
        for symbol, pos in list(open_positions.items()):
            row = stock_row_lookup.get(symbol, {}).get(last_date)
            if row is not None:
                ep_exit = float(row["close"]) * (1 - SLIPPAGE_PCT / 100)  # market-style forced close
            else:
                ep_exit = pos["entry_price"]  # fallback

            invested      = pos["allocated_capital"] + pos["entry_cost"]
            sell_turnover = ep_exit * pos["shares"]
            exit_charges  = sell_cost(sell_turnover)
            net_proceeds  = sell_turnover - exit_charges
            pnl_rs        = net_proceeds - invested
            pnl_pct       = pnl_rs / invested * 100
            total_costs   = pos["entry_cost"] + exit_charges

            trades.append({
                "Symbol":         symbol,
                "Sector":         pos.get("sector", "Unknown"),
                "Entry_Date":     pos["entry_date"],
                "Exit_Date":      last_date,
                "Signal_Type":    signal_type_label(pos["ref_high_type"], pos["oliver_type"]),
                "Entry_Price":    round(pos["entry_price"], 2),
                "Exit_Price":     round(ep_exit, 2),
                "Ref_High":       round(pos["ref_high"], 2),
                "Ref_High_Type":  pos["ref_high_type"],
                "Oliver_Type":    pos["oliver_type"],
                "Target_Price":   round(pos["target_price"], 2),
                "Stop_Price":     round(pos["stop_price"], 2),
                "Shares":         pos["shares"],
                "PnL_Rs":         round(pnl_rs, 2),
                "PnL_Pct":        round(pnl_pct, 2),
                "Hold_Days":      pos["bars_held"],
                "Exit_Reason":    "End_Of_Data",
                "Entry_Month":    pos["entry_date"].strftime("%Y-%m"),
                "Allocated_Capital": round(pos["allocated_capital"], 2),
                "Costs_Rs":       round(total_costs, 2),
            })

    return trades, equity_curve, funnel

# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def performance_report(
    trades_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    label: str = "NEAR HIGH + OLIVER KELL PORTFOLIO BACKTEST",
):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)

    if trades_df.empty:
        print("  No trades executed.")
        print(sep)
        return

    total  = len(trades_df)
    wins   = trades_df[trades_df["PnL_Rs"] > 0]
    losses = trades_df[trades_df["PnL_Rs"] <= 0]
    win_rate = len(wins) / total * 100 if total > 0 else 0

    gross_profit  = wins["PnL_Rs"].sum()
    gross_loss    = losses["PnL_Rs"].sum()
    net_pnl       = trades_df["PnL_Rs"].sum()
    total_return  = net_pnl / INITIAL_CAPITAL * 100
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = wins["PnL_Pct"].mean()   if not wins.empty   else 0
    avg_loss = losses["PnL_Pct"].mean() if not losses.empty else 0
    best  = trades_df.loc[trades_df["PnL_Rs"].idxmax()]  if total > 0 else None
    worst = trades_df.loc[trades_df["PnL_Rs"].idxmin()]  if total > 0 else None

    # Drawdown from equity curve (proper day-by-day)
    if not equity_curve_df.empty:
        max_dd = equity_curve_df["Drawdown_Pct"].min()
        final_equity = equity_curve_df["Equity"].iloc[-1]
    else:
        max_dd = 0.0
        final_equity = INITIAL_CAPITAL + net_pnl

    avg_hold = trades_df["Hold_Days"].mean()

    # Portfolio-specific stats
    if not equity_curve_df.empty:
        max_concurrent = equity_curve_df["Open_Positions"].max()
        days_with_positions = (equity_curve_df["Open_Positions"] > 0).sum()
        total_days = len(equity_curve_df)
        capital_utilization = days_with_positions / total_days * 100 if total_days > 0 else 0
        avg_positions = equity_curve_df["Open_Positions"].mean()
    else:
        max_concurrent = 0
        capital_utilization = 0
        avg_positions = 0

    print(f"  PORTFOLIO CONFIGURATION:")
    print(f"  Initial Capital    : Rs {INITIAL_CAPITAL:>14,.0f}")
    print(f"  Max Positions      : {MAX_POSITIONS:>12d}")
    print(f"  Starting Per-Position Size: Rs {PER_POSITION_CAPITAL:>8,.0f}  (compounds with equity thereafter)")

    print(f"\n  PERFORMANCE SUMMARY:")
    print(f"  Final Equity       : Rs {final_equity:>14,.0f}")
    print(f"  Net P&L            : Rs {net_pnl:>14,.0f}")
    print(f"  Total Return       : {total_return:>11.2f}%")
    print(f"  Total Trades       : {total:>12d}")
    print(f"  Win Rate           : {win_rate:>11.2f}%")
    print(f"  Max Drawdown       : {max_dd:>11.2f}%")
    print(f"  Profit Factor      : {profit_factor:>12.2f}")
    print(f"  Avg Win            : {avg_win:>11.2f}%")
    print(f"  Avg Loss           : {avg_loss:>11.2f}%")
    print(f"  Avg Hold Days      : {avg_hold:>12.1f}")
    if best is not None:
        print(f"  Best Trade         : {best['Symbol']}  Rs {best['PnL_Rs']:,.0f}  ({best['PnL_Pct']:.1f}%)")
    if worst is not None:
        print(f"  Worst Trade        : {worst['Symbol']}  Rs {worst['PnL_Rs']:,.0f}  ({worst['PnL_Pct']:.1f}%)")

    if "Costs_Rs" in trades_df.columns:
        total_costs   = trades_df["Costs_Rs"].sum()
        total_turnover = (trades_df["Entry_Price"] * trades_df["Shares"]).sum() + \
                         (trades_df["Exit_Price"]  * trades_df["Shares"]).sum()
        cost_pct_turnover = total_costs / total_turnover * 100 if total_turnover > 0 else 0
        pnl_before_costs  = net_pnl + total_costs
        print(f"\n  SLIPPAGE & COST IMPACT:")
        print(f"  Slippage Assumption: {SLIPPAGE_PCT:.2f}%  (market-style fills only; limit fills get 0%)")
        print(f"  Total Costs Paid   : Rs {total_costs:>14,.0f}  (brokerage+STT+exchange+SEBI+stamp+GST+DP)")
        print(f"  Costs % of Turnover: {cost_pct_turnover:>11.3f}%")
        print(f"  Avg Cost / Trade   : Rs {total_costs / total:>14,.0f}")
        print(f"  P&L Before Costs   : Rs {pnl_before_costs:>14,.0f}")
        print(f"  P&L After Costs    : Rs {net_pnl:>14,.0f}  (costs ate {total_costs / abs(pnl_before_costs) * 100 if pnl_before_costs != 0 else 0:.1f}% of gross P&L)")

    print(f"\n  PORTFOLIO-SPECIFIC STATS:")
    print(f"  Max Concurrent Pos : {max_concurrent:>12d}")
    print(f"  Avg Concurrent Pos : {avg_positions:>12.2f}")
    print(f"  Capital Utilization: {capital_utilization:>11.2f}%  (days with >= 1 position)")
    print(f"  Trading Days       : {len(equity_curve_df) if not equity_curve_df.empty else 0:>12d}")

    print(f"\n  EXIT REASON BREAKDOWN:")
    for reason, cnt in trades_df["Exit_Reason"].value_counts().items():
        avg_p = trades_df[trades_df["Exit_Reason"] == reason]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Exit_Reason"] == reason) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {reason:<22}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    print(f"\n  NEAR-HIGH TYPE BREAKDOWN:")
    for ht, cnt in trades_df["Ref_High_Type"].value_counts().items():
        avg_p = trades_df[trades_df["Ref_High_Type"] == ht]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Ref_High_Type"] == ht) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {ht:<6}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    print(f"\n  OLIVER SIGNAL TYPE BREAKDOWN:")
    for ot, cnt in trades_df["Oliver_Type"].value_counts().items():
        avg_p = trades_df[trades_df["Oliver_Type"] == ot]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Oliver_Type"] == ot) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {ot:<12}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    print(f"\n  COMBINED SIGNAL TYPE BREAKDOWN:")
    for st, cnt in trades_df["Signal_Type"].value_counts().items():
        avg_p = trades_df[trades_df["Signal_Type"] == st]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Signal_Type"] == st) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {st:<30}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    if "Sector" in trades_df.columns:
        print(f"\n  SECTOR BREAKDOWN (sectors with >= 3 trades):")
        vc = trades_df["Sector"].value_counts()
        other = trades_df[trades_df["Sector"].isin(vc[vc < 3].index)]
        for sec, cnt in vc[vc >= 3].items():
            sub    = trades_df[trades_df["Sector"] == sec]
            avg_p  = sub["PnL_Pct"].mean()
            pnl_rs = sub["PnL_Rs"].sum()
            wr     = (sub[sub["PnL_Rs"] > 0].shape[0] / cnt * 100)
            print(f"    {sec:<22}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%  PnL Rs {pnl_rs:>10,.0f}")
        if not other.empty:
            o_cnt  = len(other)
            o_avg  = other["PnL_Pct"].mean()
            o_pnl  = other["PnL_Rs"].sum()
            o_wr   = (other[other["PnL_Rs"] > 0].shape[0] / o_cnt * 100)
            print(f"    {'Other (niche)':<22}: {o_cnt:4d} trades  avg {o_avg:>+6.1f}%  WR {o_wr:5.1f}%  PnL Rs {o_pnl:>10,.0f}")

    # ---- CAGR ----
    if not equity_curve_df.empty and len(equity_curve_df) > 1:
        first_date = equity_curve_df["Date"].iloc[0]
        last_date  = equity_curve_df["Date"].iloc[-1]
        years = (last_date - first_date).days / 365.25
        if years > 0 and final_equity > 0:
            cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1
            print(f"\n  CAGR               : {cagr * 100:>11.2f}%  (over {years:.1f} years)")

    # ---- CALENDAR YEAR RETURNS ----
    if not equity_curve_df.empty:
        print(f"\n  CALENDAR YEAR RETURNS:")
        eq = equity_curve_df.copy()
        eq["Year"] = eq["Date"].dt.year
        years_list = sorted(eq["Year"].unique())
        print(f"    {'Year':<6} {'Start Equity':>14} {'End Equity':>14} {'Return':>10} {'Max DD':>10}")
        print(f"    {'─' * 58}")
        for yr in years_list:
            yr_data = eq[eq["Year"] == yr]
            yr_start = yr_data["Equity"].iloc[0]
            yr_end   = yr_data["Equity"].iloc[-1]
            yr_ret   = (yr_end - yr_start) / yr_start * 100
            yr_peak  = yr_data["Equity"].cummax()
            yr_dd    = ((yr_data["Equity"].values - yr_peak.values) / yr_peak.values * 100).min()
            bar = "+" * min(20, int(abs(yr_ret) / 5)) if yr_ret >= 0 else "-" * min(20, int(abs(yr_ret) / 5))
            print(f"    {yr:<6} Rs {yr_start:>12,.0f} Rs {yr_end:>12,.0f} {yr_ret:>+9.2f}% {yr_dd:>+9.2f}%  {bar}")

    print(f"\n  MONTHLY P&L BREAKDOWN:")
    net_abs = max(abs(net_pnl), 1)
    monthly = trades_df.groupby("Entry_Month")["PnL_Rs"].sum()
    for m, pnl in monthly.items():
        bar = ("+" if pnl >= 0 else "-") * min(40, int(abs(pnl) / net_abs * 200))
        print(f"    {m}  Rs {pnl:>12,.0f}  {bar}")

    print(f"\n  TOP 10 WINNING TRADES:")
    cols = ["Symbol", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price",
            "PnL_Rs", "PnL_Pct", "Hold_Days", "Signal_Type", "Exit_Reason"]
    print(trades_df.nlargest(10, "PnL_Rs")[cols].to_string(index=False))

    print(f"\n  TOP 10 LOSING TRADES:")
    print(trades_df.nsmallest(10, "PnL_Rs")[cols].to_string(index=False))

    print(sep)

# =============================================================================
# MAIN RUNNER
# =============================================================================
def run_backtest(
    use_all_nse: bool = False,
    tickers: list[str] | None = None,
    max_stocks: int | None = None,
):
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  NEAR HIGH + OLIVER KELL PORTFOLIO BACKTEST")
    print(f"  EMA Alignment + Near High ({int((1-NEAR_HIGH_UPPER)*100)}-{int((1-NEAR_HIGH_LOWER)*100)}% below) + Wedge_Pop | Crossback")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ref high")
    print(f"  Extension mult: {EXTENSION_MULT}x ATR  |  Min hold: {MIN_HOLD_DAYS} bars")
    print(f"  Lookback: {LOOKBACK_DAYS} days  |  Fetch: {FETCH_YEARS} years")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}  |  Max Positions: {MAX_POSITIONS}")
    print(f"  Per-Position (start): Rs {PER_POSITION_CAPITAL:,.0f}  (compounds with equity)")
    print(f"  Rank Method: {RANK_METHOD.upper()}  |  ROC Period: {ROC_PERIOD} days")
    print(f"  Entry: next-day open (T+1 fill)  |  Slippage: {SLIPPAGE_PCT:.2f}% on market-style fills")
    print(f"  Costs: STT {STT_PCT:.2f}% + exch {EXCHANGE_TXN_PCT:.4f}% + SEBI {SEBI_PCT:.4f}% "
          f"+ stamp {STAMP_DUTY_PCT:.3f}% (buy) + GST {GST_PCT:.0f}% + DP Rs {DP_CHARGE_PER_SELL:.0f} (sell)")
    print("=" * 70)

    # Decide universe
    print("\nBuilding stock universe...")
    if use_all_nse and tickers is None:
        universe = get_nse_universe()
        print(f"  Universe: ALL NSE-EQ stocks ({len(universe):,})")
    elif tickers is not None:
        universe = list(tickers)
        print(f"  Universe: {len(universe):,} stocks (from provided tickers list)")
    else:
        universe = list(NIFTY_500)
        print(f"  Universe: NIFTY 500 ({len(universe):,} stocks)")

    if max_stocks is not None:
        universe = universe[:max_stocks]

    total = len(universe)
    print(f"\nUniverse: {total:,} stocks\n")

    # Fetch Nifty reference data (for RS calculation)
    print("Fetching Nifty 50 data for RS calculation...")
    nifty_close = fetch_nifty_close(FETCH_YEARS)
    if nifty_close is None or nifty_close.empty:
        print("  WARNING: Nifty data unavailable, RS_Leader will be set to True for all stocks.")
        nifty_close = None
    else:
        print(f"  Nifty data: {len(nifty_close):,} bars")

    # =================================================================
    # PASS 1: Resolve cache / batch-fetch + compute indicators
    # =================================================================
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\n--- PASS 1: Loading data for {total:,} stocks ---\n")

    stock_data = {}   # symbol -> computed DataFrame
    to_fetch   = []   # symbols not found in any cache
    from_cache = 0
    t0         = time.time()

    for symbol in universe:
        primary = os.path.join(CACHE_DIR, f"{symbol}.pkl")
        search  = [primary] + [
            os.path.join(fb, f"{symbol}.pkl") for fb in CACHE_FALLBACK_DIRS
        ]
        df = None
        for path in search:
            if os.path.exists(path):
                age_h = (time.time() - os.path.getmtime(path)) / 3600
                if age_h < CACHE_MAX_AGE_HOURS:
                    try:
                        with open(path, "rb") as f:
                            df = pickle.load(f)
                        if path != primary and df is not None:
                            with open(primary, "wb") as f:
                                pickle.dump(df, f)
                        break
                    except Exception:
                        df = None

        if df is not None:
            d = compute_indicators(df.copy(), nifty_close)
            if d is not None:
                stock_data[symbol] = d
                from_cache += 1
        else:
            to_fetch.append(symbol)

    elapsed_cache = time.time() - t0
    print(f"  From cache   : {from_cache:,} stocks  ({elapsed_cache:.1f}s)")
    print(f"  To download  : {len(to_fetch):,} stocks via Yahoo Finance")

    if to_fetch:
        batches = [to_fetch[i:i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
        t1      = time.time()
        fetched = 0
        failed  = 0

        print(f"\n  Downloading {len(to_fetch):,} stocks in {len(batches)} batches of {BATCH_SIZE}...")
        for bi, batch in enumerate(batches, 1):
            batch_results = fetch_batch_yf(batch, years=FETCH_YEARS)

            for sym, df in batch_results.items():
                primary = os.path.join(CACHE_DIR, f"{sym}.pkl")
                try:
                    with open(primary, "wb") as f:
                        pickle.dump(df, f)
                except Exception:
                    pass

                d = compute_indicators(df.copy(), nifty_close)
                if d is not None:
                    stock_data[sym] = d
                    fetched += 1
                else:
                    failed += 1

            failed += len(batch) - len(batch_results)

            elapsed1 = time.time() - t1
            rate     = bi / elapsed1 if elapsed1 > 0 else 1
            eta_min  = (len(batches) - bi) / rate / 60
            print(f"  Batch [{bi:>4}/{len(batches)}]  loaded so far: {len(stock_data):>4}  ETA: {eta_min:.1f} min")

        elapsed_fetch = (time.time() - t1) / 60
        print(f"\n  Download done: {fetched:,} fetched, {failed:,} failed/empty in {elapsed_fetch:.1f} min")

    elapsed_1 = (time.time() - t0) / 60
    print(f"\n  Pass 1 done. {len(stock_data):,} stocks ready in {elapsed_1:.1f} min")

    if not stock_data:
        print("  No stock data available. Exiting.")
        return pd.DataFrame()

    # Nifty data for RS ranking
    nifty_rank = None
    if RANK_METHOD == "rs_nifty":
        print("\n  Loading Nifty data for RS ranking...")
        nifty_rank = nifty_close if nifty_close is not None and not nifty_close.empty else None
        if nifty_rank is None:
            print("  WARNING: Nifty data unavailable, falling back to momentum ranking")

    # Sector map and trend data
    sector_map    = {}
    sector_trends = {}
    if USE_SECTOR_FILTER:
        print("\nLoading sector map (Nifty 500 constituents)...")
        sector_map = get_sector_map()
        print("\nFetching sector index trends...")
        sector_trends = fetch_sector_trends(FETCH_YEARS)
        if not sector_trends:
            print("  WARNING: No sector trend data fetched. Trend filter will be skipped.")

    # =================================================================
    # PASS 2: Day-by-day portfolio simulation
    # =================================================================
    sector_status = "ON" if USE_SECTOR_FILTER else "OFF"
    print(f"\n--- PASS 2: Portfolio simulation ({len(stock_data):,} stocks) | Rank: {RANK_METHOD.upper()} | Sector filter: {sector_status} ---\n")

    t1 = time.time()
    all_trades, equity_curve, funnel = backtest_portfolio(stock_data, nifty_rank, sector_map, sector_trends)
    sim_elapsed = (time.time() - t1) / 60

    print(f"  Pass 2 done. {len(all_trades):,} trades in {sim_elapsed:.1f} min")
    print(f"  Equity curve: {len(equity_curve):,} data points\n")

    if funnel:
        print("  SIGNAL FUNNEL (why candidates did/didn't become trades):")
        print(f"    Raw signals found (Signal=True, not already held) : {funnel['signals_raw']:>7,}")
        print(f"    Lost to sector-trend filter (scan time)           : {funnel['lost_sector_trend']:>7,}")
        print(f"    Lost to sector-diversity cap (scan time)          : {funnel['lost_sector_diversity_scan']:>7,}")
        print(f"    Queued for next-day open                          : {funnel['queued']:>7,}")
        print(f"    Lost: stock didn't trade on the fill day          : {funnel['lost_no_data_next_day']:>7,}")
        print(f"    Lost: sector slot taken by another fill that day  : {funnel['lost_sector_diversity_fill']:>7,}")
        print(f"    Lost: insufficient capital / <1 share at fill     : {funnel['lost_capital']:>7,}")
        print(f"    Filled (became a trade)                           : {funnel['filled']:>7,}\n")

    if not all_trades:
        print("  No trades found. Exiting.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    trades_df["Entry_Date"] = pd.to_datetime(trades_df["Entry_Date"])
    trades_df["Exit_Date"]  = pd.to_datetime(trades_df["Exit_Date"])
    trades_df.sort_values("Entry_Date", inplace=True)

    equity_curve_df = pd.DataFrame(equity_curve)
    if not equity_curve_df.empty:
        equity_curve_df["Date"] = pd.to_datetime(equity_curve_df["Date"])

    performance_report(
        trades_df,
        equity_curve_df,
        label="NEAR HIGH + OLIVER KELL PORTFOLIO BACKTEST",
    )

    # Save results
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_trades = f"near_high_oliver_portfolio_trades_{ts}.csv"
    trades_df.to_csv(out_trades, index=False)
    print(f"  Trades saved to: {out_trades}")

    out_equity = f"near_high_oliver_portfolio_equity_{ts}.csv"
    equity_curve_df.to_csv(out_equity, index=False)
    print(f"  Equity curve saved to: {out_equity}")

    return trades_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Default: all NSE EQ stocks (~2500)
    run_backtest(use_all_nse=True, max_stocks=None)

    # Example: Nifty 500 stocks only (faster, focused on quality universe)
    # run_backtest(use_all_nse=False, tickers=NIFTY_500)

    # Example: specific tickers only
    # run_backtest(use_all_nse=False, tickers=["RELIANCE", "TCS", "INFY", "HDFCBANK"])

    # Example: first 100 stocks (for testing)
    # run_backtest(use_all_nse=True, max_stocks=100)
