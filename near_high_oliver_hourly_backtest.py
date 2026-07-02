"""
=============================================================================
NEAR HIGH + OLIVER KELL COMBINED BACKTEST  --   1-HOUR CANDLES  (Angel One)
=============================================================================
SECURITY  : This script needs your Angel One API key, client ID, PIN and
            TOTP secret (see CONFIGURATION below). Do NOT commit real
            values to a public/shared git repo -- fill them in locally,
            or set them as environment variables instead (see below).
            NOTE: earlier scripts in this repo's git history had real
            Angel One credentials hardcoded and committed -- if that
            account is still active, rotate its API key/PIN/TOTP secret.

DATA      : Angel One SmartAPI historical candle API (ONE_HOUR interval).
            Yahoo Finance only serves ~730 days of hourly data (a hard
            Yahoo-side limit) -- Angel One is used here instead because it
            retains years of intraday history. Angel's API is one symbol
            per request (no batching like yfinance) and caps each request
            to ~400 days for ONE_HOUR candles, so 6 years is fetched in
            ~6 chunked requests per stock, then stitched together.

UNIVERSE  : Nifty 500 by default (not full NSE ~2500). Angel's API has no
            multi-symbol batching, so full NSE would mean ~2375 x 6 =
            ~14,250 individual requests -- hours of fetching and much
            harder on Angel's rate limits. Nifty 500 (~500 x 6 = ~3,000
            requests) is a more practical starting universe; pass
            tickers=NIFTY_500[:N] or a custom list to run_backtest() to
            go smaller, or use_all_nse=True to attempt the full universe.

INDICATOR PERIODS: The original strategy's periods (EMA200, High_1Y=252
            bars, Mini_Base=7 bars, etc.) were designed for DAILY candles,
            where they represent real calendar time (EMA200 ~ 200 days).
            Ported naively to hourly bars, "200" would mean 200 HOURS
            (~32 days) -- a much faster, different-natured strategy.
            Instead, every period here is scaled by BARS_PER_DAY so the
            SAME calendar-time meaning is preserved (EMA200 still tracks
            ~200 days of trend); only entry/exit TIMING becomes intraday.
            BARS_PER_DAY defaults to 7 (NSE cash hours 9:15-15:30 with
            hourly candles: 9:15,10:15,11:15,12:15,13:15,14:15,15:15) --
            check the "Avg bars/day observed" diagnostic after your first
            fetch and adjust BARS_PER_DAY if your feed differs.

INTRADAY EXECUTION MODEL (per your instruction):
            - Signal is evaluated using the FIRST hourly candle of the day
              (9:15-10:15, i.e. the bar timestamped 09:15 assuming Angel
              stamps candles at interval START -- adjust SIGNAL_BAR_TIME
              if your feed stamps differently).
            - If a signal fires, the order is queued and filled at the
              NEXT candle's open (10:15-11:15 candle's open == the price
              at 10:15 AM) -- i.e. "buy during the day at the 10:15 AM
              candle", not next-day. No same-bar lookahead entry.
            - Exits (stop loss, target, Oliver EMA trailing stop,
              extension exit) are checked on EVERY hourly candle through
              the rest of that day and subsequent days -- a live bot
              checking intraday, exactly as you described. A position can
              now open and close within the SAME trading day.
            - Stop loss and target are still modeled as resting LIMIT
              orders (fill only if price actually trades at that level;
              a gap through the level leaves it unfilled) -- same
              philosophy as the daily backtest, just checked hourly.

COSTS     : Same slippage + NSE delivery cost model as the daily backtest
            (buy_cost()/sell_cost(), STT/exchange/SEBI/stamp/GST/DP) --
            reused unchanged, since these are per-trade broker/statutory
            charges independent of timeframe.

SIGNAL TYPES: 'NearHigh_Wedge_1Y', 'NearHigh_Wedge_5Y', 'NearHigh_Wedge_ATH',
              'NearHigh_Crossback_1Y', 'NearHigh_Crossback_5Y', 'NearHigh_Crossback_ATH'

RESULTS   : CSV saved with timestamp; full performance report printed,
            including a signal funnel and avg-bars/day diagnostic.
=============================================================================
"""

import os
import pickle
import time
import datetime
import warnings
import requests
from io import StringIO
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf
from SmartApi import SmartConnect
import pyotp

warnings.filterwarnings("ignore")

# =============================================================================
# ANGEL ONE CREDENTIALS -- fill these in, or set as environment variables.
# DO NOT commit real values to a shared/public repo.
# =============================================================================
ANGEL_API_KEY      = os.environ.get("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_CLIENT_PIN   = os.environ.get("ANGEL_CLIENT_PIN", "")
ANGEL_TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "")
NIFTY_TOKEN        = "99926000"          # Nifty 50 index token on NSE

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================
INITIAL_CAPITAL      = 100_000               # Rs 1 lakh
MAX_POSITIONS        = 3                     # max concurrent positions
STOP_LOSS_PCT        = 0.05                  # 5% stop loss
TARGET_ABOVE_HIGH_PCT = 0.10                 # 10% above ref high
NEAR_HIGH_LOWER      = 0.90                  # 10% below high (lower bound)
NEAR_HIGH_UPPER      = 0.95                  # 5% below high (upper bound)
FETCH_YEARS          = 6                     # years of hourly history to fetch
LOOKBACK_DAYS        = int(FETCH_YEARS * 365.25) - 30  # backtest window in days
RANK_METHOD          = "momentum"            # "roc" | "rs_nifty" | "near_ath" | "volume_surge" | "momentum"
ROC_PERIOD           = 90                    # DAYS for ROC/momentum (scaled to bars internally)
RS_RANK_PERIOD       = 90                    # DAYS for RS vs Nifty ranking
FRESH_HIGH_DAYS      = 100                   # DAYS: skip stock if it touched ref high within this window

# Oliver Kell specific config (all expressed in DAYS; scaled to hourly bars below)
MINI_BASE_DAYS       = 7                     # window for mini-base detection
MINI_BASE_MIN        = 4                     # minimum qualifying bars in base (bar-count, not scaled)
EXTENSION_MULT       = 3.0                   # ATR multiplier for extension exit (unitless ratio)
MIN_HOLD_DAYS        = 2                     # hold for at least this many DAYS before trailing stop
RS_LOOKBACK_DAYS     = 20                    # days for relative-strength calculation

# ---- Intraday execution ----
BARS_PER_DAY         = 7          # hourly candles/trading day (9:15,10:15,...,15:15). VERIFY after first fetch.
SIGNAL_BAR_TIME       = datetime.time(9, 15)  # candle used to evaluate signals (first bar of the day)

CACHE_DIR            = "cache_near_high_hourly"   # primary cache dir (per-symbol .pkl, resumable)
CACHE_MAX_AGE_HOURS  = 24                          # hourly data goes stale faster than daily; re-fetch daily
MIN_DATA_BARS        = 260 * BARS_PER_DAY          # minimum hourly bars needed (~260 trading days worth)
CHUNK_DAYS           = 400                         # Angel One's per-request cap for ONE_HOUR interval
REQUEST_SLEEP_SEC    = 0.35                        # pause between Angel API calls (rate-limit friendly)

# Derived: day-based periods -> hourly-bar-based periods (preserve calendar-time meaning)
EMA_SPANS = {
    "EMA5":   5   * BARS_PER_DAY,
    "EMA10":  10  * BARS_PER_DAY,
    "EMA20":  20  * BARS_PER_DAY,
    "EMA50":  50  * BARS_PER_DAY,
    "EMA150": 150 * BARS_PER_DAY,
    "EMA200": 200 * BARS_PER_DAY,
}
ATR_SPAN_BARS       = 14 * BARS_PER_DAY
HIGH_1Y_BARS        = 252 * BARS_PER_DAY
HIGH_5Y_BARS        = 1260 * BARS_PER_DAY
VOL_AVG_BARS        = 20 * BARS_PER_DAY
ROC_PERIOD_BARS     = ROC_PERIOD * BARS_PER_DAY
RS_LOOKBACK_BARS    = RS_LOOKBACK_DAYS * BARS_PER_DAY
MINI_BASE_BARS      = MINI_BASE_DAYS * BARS_PER_DAY
SWING_HI_BARS       = 10 * BARS_PER_DAY
FRESH_HIGH_BARS     = FRESH_HIGH_DAYS * BARS_PER_DAY
MIN_HOLD_BARS       = MIN_HOLD_DAYS * BARS_PER_DAY

# Derived: capital
PER_POSITION_CAPITAL = INITIAL_CAPITAL / MAX_POSITIONS  # ~33,333 per slot

# Sector filters (sector map + trend index data stay on DAILY/yfinance --
# a macro regime filter doesn't need hourly precision, and this avoids
# doubling the already-heavy Angel One request load)
USE_SECTOR_FILTER    = True
MAX_SAME_SECTOR      = 1
SECTOR_TREND_EMA     = 50

# =============================================================================
# SLIPPAGE & TRANSACTION COSTS (NSE equity DELIVERY trades, India)
# Same model as the daily backtest -- these are per-trade broker/statutory
# charges, independent of candle timeframe.
# =============================================================================
SLIPPAGE_PCT         = 0.05
BROKERAGE_PCT        = 0.0
BROKERAGE_FLAT       = 0.0
STT_PCT              = 0.1
EXCHANGE_TXN_PCT     = 0.00297
SEBI_PCT             = 0.0001
STAMP_DUTY_PCT       = 0.015
GST_PCT              = 18.0
DP_CHARGE_PER_SELL   = 20.0


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
    return industry

# =============================================================================
# NIFTY 500 UNIVERSE (Nifty50 + NiftyNext50 + Midcap150 + Smallcap250)
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
    "NMDC", "APOLLOTYRE", "INDHOTEL", "GODREJPROP", "PVRINOX",
    "ABBOTINDIA", "SANOFI", "GILLETTE", "ZOMATO", "JIOFIN", "PAYTM",
    "POLICYBZR", "KAYNES", "LLOYDSME", "SONATSOFTW", "STARHEALTH",
    "MAZDOCK", "COCHINSHIP", "GRSE", "DALBHARAT",
    # --- Nifty Smallcap 250 ---
    "IDBI", "IOB", "CENTRALBK", "UCOBANK", "MAHABANK", "J&KBANK",
    "KTKBANK", "KARURVYSYA", "CUB", "DCBBANK", "EQUITAS", "UJJIVAN",
    "CREDITACC", "SPANDANA", "HOMEFIRST", "APTUS", "REPCO", "CANARABNK",
    "CGCL", "CHOLAHLDNG", "BSE", "CAMS", "KFINTECH", "ANGELONE",
    "NYKAA", "CARTRADE", "EASEMYTRIP",
    "HUDCO", "IRCON", "RVNL", "IREDA", "SJVN", "NLCINDIA",
    "JSWENERGY", "POWERMECH", "KALPATPOWR", "KEC", "JYOTISTRUC",
    "BAJAJELEC", "BLUESTARCO", "AMBER", "ELGIEQUIP", "KIRLOSENG",
    "TRITURBINE", "GREAVESCOT", "MAHSEAMLES", "RATNAMANI", "GPPL",
    "GUJGASLTD", "IGL", "GSPL", "GAIL", "OIL", "MRPL", "CHENNPETRO", "HINDPETRO",
    "CASTROLIND", "GULFOILLUB", "GNFC", "GSFC", "CHAMBLFERT",
    "DEEPAKFERT", "RALLIS", "UPL", "BAYERCROP", "DHANUKA", "SHARDACROP",
    "ERIS", "GRANULES", "CAPLIPOINT", "STRIDES", "IPCA", "LAURUS",
    "THYROCARE", "POLYMED", "MEDANTA", "RAINBOW", "NH", "KIMS", "YATHARTH",
    "AKZOINDIA", "CENTURYPLY", "GREENPANEL", "CERA",
    "JKLAKSHMI", "HEIDELBERG", "STARCEMENT", "BIRLACORPN",
    "NATIONALUM", "HINDCOPPER", "MOIL", "WELCORP", "JINDALSAW",
    "FORCEMOT", "CEATLTD", "JKTYRE", "SUPRAJIT", "SKFINDIA", "FIVESTAR",
    "KALYANAJEW", "SENCO", "PCJEWELLER", "RAJESHEXPO",
    "TASTYBITE", "EIDPARRY", "DALMIASUGAR", "BALRAMCHIN", "RENUKA",
    "CCL", "UBL", "JYOTHYLAB", "HATSUN", "HERITGFOOD", "BIKAJI",
    "GODFRYPHLP", "VSTIND", "SHOPERSTOP", "VMART", "LUXIND", "RUPA", "BATA",
    "SWIGGY", "SPICEJET",
    "SOBHA", "SUNTECK", "MAHLIFE", "KOLTEPATIL", "ANANTRAJ", "RAYMOND",
    "COFORGE", "CYIENT", "ECLERX", "INTELLECT", "MASTEK",
    "HAPPSTMNDS", "NEWGEN", "ROUTE", "LATENTVIEW", "TANLA",
    "NETWORK18", "TV18BRDCST", "STAR", "SAREGAMA", "TIPS",
    "HFCL", "STLTECH", "RAILTEL", "TTML",
    "MAXFIN", "GODIGIT", "ISEC", "M&MFIN", "POONAWALLA",
    "BANDHANBNK", "GALAXYSURF", "CLEAN",
    "ICRA", "MMTC", "NFL", "RCF", "FACT", "RITES", "NBCC", "NCC", "BDL",
    "ITI", "PTC", "GODREJIND", "GODREJAGRO",
    "AARTI", "AARTIIND", "ABSLAMC", "ALOKINDS", "ANANDRATHI", "APARINDS",
    "ARE&M", "ARIES", "ASAHIINDIA", "ASTRAZEN", "BEML", "BLUEDART",
    "BORORENEW", "BSOFT", "CAMPUS", "CAPLIN", "CENTURYTEX", "CHALET",
    "CHAMBAL", "CHOICEIN", "CMSINFO", "CONCORDBIO", "CRAFTSMAN",
    "DBCORP", "DCMSHRIRAM", "DELTACORP", "DEVYANI", "EDELWEISS", "ELECON",
    "ENGINERSIN", "FDC", "FINCABLES", "FINPIPE", "FLUOROCHEM",
    "GARFIBRES", "GESHIP", "GHCL", "GMDCLTD", "GOCOLORS",
    "GOODYEAR", "GRPLTD", "GUJALKALI", "HBLPOWER", "HGS", "HIKAL",
    "HINDWAREAP", "IBULHSGFIN", "IFBIND", "INDIAMART", "INDIGOPNTS",
    "INOXWIND", "JAMNAAUTO", "JBM", "JMFINANCIL", "JSL", "JTEKTINDIA",
    "JUBLINGREA", "JUBLPHARMA", "JUSTDIAL", "KENNAMET", "KNRCON",
    "KPRMILL", "KRBL", "KRSNAA", "LAXMIMACH", "LEMONTREE", "MAPMYINDIA",
    "MAXIND", "MIDHANI", "MINDACORP", "MOTILALOFS", "NESCO", "NILKAMAL",
    "OLECTRA", "ORIENTELEC", "PIRAMAL", "PPLPHARMA",
    "PRINCEPIPE", "PRSMJOHNSN", "QUESS", "REDINGTON", "ROSSARI",
    "SAPPHIRE", "SHILPAMED", "SHYAMMETL", "SIS", "SOUTHBANK",
    "SUDARSCHEM", "SWANENERGY", "SYMPHONY", "TARSONS", "TEAMLEASE",
    "TECHNOE", "TEGA", "TIINDIA", "TRIDENT", "TTKPRESTIG", "UFLEX",
    "UTIAMC", "VAIBHAVGBL", "VARUN", "VIPIND", "VRLLOG", "WABAG",
    "WESTLIFE", "WOCKPHARMA", "WONDERLA", "ZENSARTECH",
]
NIFTY_500 = sorted(set(NIFTY_500))

# =============================================================================
# ANGEL ONE AUTHENTICATION
# =============================================================================
def _fix_totp(raw: str) -> str:
    s = raw.strip().upper().replace(" ", "").replace("-", "")
    pad = (8 - len(s) % 8) % 8
    return s + "=" * pad


def angel_login() -> SmartConnect:
    if not (ANGEL_API_KEY and ANGEL_CLIENT_ID and ANGEL_CLIENT_PIN and ANGEL_TOTP_SECRET):
        raise RuntimeError(
            "Angel One credentials missing. Set ANGEL_API_KEY, ANGEL_CLIENT_ID, "
            "ANGEL_CLIENT_PIN, ANGEL_TOTP_SECRET at the top of this file or as "
            "environment variables before running."
        )
    totp_fixed = _fix_totp(ANGEL_TOTP_SECRET)
    obj = SmartConnect(api_key=ANGEL_API_KEY)
    otp = pyotp.TOTP(totp_fixed).now()
    secs_left = 30 - datetime.datetime.now().second % 30
    print(f"  OTP: {otp}  (valid ~{secs_left}s)")
    data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_CLIENT_PIN, otp)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected login response: {data}")
    status = data.get("status")
    if status is True or str(status).lower() == "true":
        jwt = str(data.get("data", {}).get("jwtToken", ""))[:30]
        print(f"  JWT: {jwt}...")
        return obj
    raise RuntimeError(
        f"Login failed: {data.get('errorcode', '?')} | {data.get('message', str(data))}"
    )

# =============================================================================
# NSE SYMBOL MAP (Angel One scrip master)
# =============================================================================
def get_nse_symbol_map() -> dict:
    """Return {symbol -> token} for all NSE-EQ stocks, excluding ETFs."""
    cache_path = os.path.join(CACHE_DIR, "_symbol_map.pkl")
    if os.path.exists(cache_path):
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_h < 24:
            with open(cache_path, "rb") as fh:
                m = pickle.load(fh)
            m = {k: v for k, v in m.items() if not is_etf(k)}
            print(f"  Symbol map: {len(m):,} stocks (from cache, {age_h:.1f}h old)")
            return m

    print("  Downloading Angel One scrip master...")
    url = ("https://margincalculator.angelbroking.com"
           "/OpenAPI_File/files/OpenAPIScripMaster.json")
    resp = requests.get(url, timeout=60)
    df = pd.DataFrame(resp.json())
    nse = df[df["exch_seg"] == "NSE"]
    eq = nse[nse["symbol"].str.endswith("-EQ", na=False)].copy()
    eq["sym"] = eq["symbol"].str.replace("-EQ", "", regex=False).str.strip()
    raw_map = dict(zip(eq["sym"], eq["token"]))
    filtered = {k: v for k, v in raw_map.items() if not is_etf(k)}
    print(f"  NSE-EQ total: {len(raw_map):,}  after ETF exclusion: {len(filtered):,}")
    with open(cache_path, "wb") as fh:
        pickle.dump(filtered, fh)
    return filtered


def resolve_ticker(symbol: str, sym_map: dict) -> str | None:
    for candidate in [symbol, symbol.upper(), symbol + "-EQ"]:
        if candidate in sym_map:
            return sym_map[candidate]
    return None

# =============================================================================
# CHUNKED HOURLY OHLCV FETCH (Angel One)
# =============================================================================
def fetch_hourly_chunked(
    obj: SmartConnect,
    token: str,
    symbol: str,
    years: int = FETCH_YEARS,
    chunk_days: int = CHUNK_DAYS,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch ONE_HOUR OHLCV in ~400-day chunks (Angel One's per-request cap)."""
    rows = []
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=int(years * 365.25))
    cur = start

    while cur <= end:
        chunk_end = min(cur + pd.DateOffset(days=chunk_days - 1), end)
        for attempt in range(max_retries):
            try:
                r = obj.getCandleData({
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": "ONE_HOUR",
                    "fromdate": cur.strftime("%Y-%m-%d 09:00"),
                    "todate": chunk_end.strftime("%Y-%m-%d 15:30"),
                })
                if r.get("status") and r.get("data"):
                    rows.extend(r["data"])
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        cur = chunk_end + pd.DateOffset(days=1)
        time.sleep(REQUEST_SLEEP_SEC)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    # Angel returns ISO timestamps with +05:30 offset -> tz-aware. Strip the tz
    # label WITHOUT shifting the wall-clock time (tz_localize, not tz_convert --
    # tz_convert would roll every bar back by 5.5 hours into the wrong day/hour).
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    return df


def fetch_with_cache(obj: SmartConnect, token: str, symbol: str) -> pd.DataFrame:
    """Fetch hourly OHLCV using a per-stock pickle cache (resumable)."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
    if os.path.exists(cache_path):
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_h < CACHE_MAX_AGE_HOURS:
            with open(cache_path, "rb") as fh:
                df = pickle.load(fh)
            if df is not None and not df.empty:
                return df

    df = fetch_hourly_chunked(obj, token, symbol, years=FETCH_YEARS)
    with open(cache_path, "wb") as fh:
        pickle.dump(df, fh)
    return df

# =============================================================================
# NIFTY REFERENCE DATA (hourly, for RS calculation) -- Angel One
# =============================================================================
def load_nifty_hourly(obj: SmartConnect) -> pd.Series:
    cache_path = os.path.join(CACHE_DIR, "_NIFTY50.pkl")
    if os.path.exists(cache_path):
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_h < CACHE_MAX_AGE_HOURS:
            with open(cache_path, "rb") as fh:
                close = pickle.load(fh)
            if close is not None and not close.empty:
                return close

    nf = fetch_hourly_chunked(obj, NIFTY_TOKEN, "NIFTY50", years=FETCH_YEARS)
    close = nf["close"] if not nf.empty else pd.Series(dtype=float)
    with open(cache_path, "wb") as fh:
        pickle.dump(close, fh)
    return close

# =============================================================================
# SECTOR MAP + SECTOR TRENDS -- kept on DAILY / yfinance (macro regime filter
# doesn't need hourly precision; avoids extra Angel One request load)
# =============================================================================
def get_sector_map(cache_file: str = "sector_map_yf.pkl") -> dict:
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
            print("  WARNING: Sector CSV columns not as expected. Sector filter partial.")
    except Exception as e:
        print(f"  WARNING: Could not load sector map ({e}). Sector filters may be limited.")
    return sector_map


def fetch_sector_trends(years: int = FETCH_YEARS) -> dict:
    """Daily sector index trend (close > EMA50), via yfinance."""
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
                    close.index = close.index.tz_localize(None)
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
# INDICATOR COMPUTATION (hourly bars, periods scaled by BARS_PER_DAY)
# =============================================================================
def compute_indicators(df: pd.DataFrame, nifty_close: pd.Series | None = None) -> pd.DataFrame | None:
    if len(df) < MIN_DATA_BARS:
        return None

    d = df.copy()

    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    if nifty_close is not None and nifty_close.index.tz is not None:
        nifty_close = nifty_close.copy()
        nifty_close.index = nifty_close.index.tz_localize(None)

    # ---- EMAs on Close (spans in hourly bars) ----
    d["EMA5"]   = d["close"].ewm(span=EMA_SPANS["EMA5"],   adjust=False).mean()
    d["EMA10"]  = d["close"].ewm(span=EMA_SPANS["EMA10"],  adjust=False).mean()
    d["EMA20"]  = d["close"].ewm(span=EMA_SPANS["EMA20"],  adjust=False).mean()
    d["EMA50"]  = d["close"].ewm(span=EMA_SPANS["EMA50"],  adjust=False).mean()
    d["EMA150"] = d["close"].ewm(span=EMA_SPANS["EMA150"], adjust=False).mean()
    d["EMA200"] = d["close"].ewm(span=EMA_SPANS["EMA200"], adjust=False).mean()

    # ---- ATR (scaled span) ----
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(span=ATR_SPAN_BARS, adjust=False).mean()

    # ---- Key highs (on High column, scaled windows) ----
    d["High_1Y"] = d["high"].rolling(HIGH_1Y_BARS, min_periods=int(HIGH_1Y_BARS * 0.8)).max()
    d["High_5Y"] = d["high"].rolling(HIGH_5Y_BARS, min_periods=HIGH_1Y_BARS).max()
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

    # ---- Near-high conditions ----
    d["Near_1Y"]  = (d["close"] >= d["High_1Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_1Y"] * NEAR_HIGH_UPPER)
    d["Near_5Y"]  = (d["close"] >= d["High_5Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_5Y"] * NEAR_HIGH_UPPER)
    d["Near_ATH"] = (d["close"] >= d["ATH"]     * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["ATH"]      * NEAR_HIGH_UPPER)

    # Fresh-high filter (scaled window)
    touched_1Y  = (d["high"] >= d["High_1Y"].shift(1)).fillna(False)
    touched_5Y  = (d["high"] >= d["High_5Y"].shift(1)).fillna(False)
    touched_ATH = (d["high"] >= d["ATH"].shift(1)).fillna(False)

    d["Touched_1Y_Recent"]  = touched_1Y.rolling(FRESH_HIGH_BARS, min_periods=1).sum() > 0
    d["Touched_5Y_Recent"]  = touched_5Y.rolling(FRESH_HIGH_BARS, min_periods=1).sum() > 0
    d["Touched_ATH_Recent"] = touched_ATH.rolling(FRESH_HIGH_BARS, min_periods=1).sum() > 0

    d["Near_1Y"]  = d["Near_1Y"]  & ~d["Touched_1Y_Recent"]
    d["Near_5Y"]  = d["Near_5Y"]  & ~d["Touched_5Y_Recent"]
    d["Near_ATH"] = d["Near_ATH"] & ~d["Touched_ATH_Recent"]

    d["Near_High"] = d["Near_1Y"] | d["Near_5Y"] | d["Near_ATH"]

    # ---- Weekly EMAs (resample works the same regardless of bar granularity) ----
    weekly = d["close"].resample("W").last().ffill()
    w_ema10 = weekly.ewm(span=10, adjust=False).mean()
    w_ema20 = weekly.ewm(span=20, adjust=False).mean()
    d["W_EMA10"] = w_ema10.reindex(d.index, method="ffill")
    d["W_EMA20"] = w_ema20.reindex(d.index, method="ffill")
    d["Weekly_Uptrend"] = d["close"] > d["W_EMA20"]

    # ---- Volume indicators (scaled window) ----
    d["VolAvg"]       = d["volume"].rolling(VOL_AVG_BARS, min_periods=max(10, VOL_AVG_BARS // 2)).mean()
    d["Vol_Above_Avg"] = d["volume"] > d["VolAvg"]

    # ---- Relative Strength vs Nifty (scaled lookback) ----
    if nifty_close is not None and not nifty_close.empty:
        nifty_aligned = nifty_close.reindex(d.index, method="ffill")
        stock_ret  = d["close"].pct_change(RS_LOOKBACK_BARS)
        nifty_ret  = nifty_aligned.pct_change(RS_LOOKBACK_BARS)
        d["RS_Leader"] = stock_ret > nifty_ret
    else:
        d["RS_Leader"] = True

    # ---- Mini-Base (volatility contraction, scaled window) ----
    atr_declining = d["ATR"].diff() < 0
    d["Mini_Base"] = (
        atr_declining.rolling(MINI_BASE_BARS - 1, min_periods=MINI_BASE_BARS - 1).sum()
        >= MINI_BASE_MIN
    ).fillna(False)

    # ---- Swing High (scaled window, shifted by 1 to avoid lookahead) ----
    d["Swing_Hi"] = d["high"].rolling(SWING_HI_BARS).max().shift(1)

    # ---- Wedge Pop ----
    prev_mini = d["Mini_Base"].shift(1).fillna(False)
    d["Wedge_Pop"] = (
        (d["close"]  > d["Swing_Hi"]) &
        prev_mini &
        d["EMA_Aligned"] &
        d["Weekly_Uptrend"] &
        d["Vol_Above_Avg"] &
        d["RS_Leader"]
    )

    # ---- Crossback ----
    prev_low    = d["low"].shift(1)
    prev_ema20  = d["EMA20"].shift(1)
    touched_ema20 = (prev_low <= prev_ema20) & (d["close"].shift(1) >= prev_ema20)
    d["Crossback"] = (
        touched_ema20 &
        (d["close"] > d["EMA20"]) &
        d["Weekly_Uptrend"] &
        d["Vol_Above_Avg"] &
        d["RS_Leader"]
    )

    # ---- Extension exit indicator ----
    d["Extension"] = d["close"] > (d["EMA20"] + EXTENSION_MULT * d["ATR"])

    # ---- Combined Signal ----
    d["Oliver_Signal"] = d["Wedge_Pop"] | d["Crossback"]
    d["Signal"]        = d["EMA_Aligned"] & d["Near_High"] & d["Oliver_Signal"]

    # Ranking indicators (scaled ROC period)
    d["ROC_N"]         = d["close"].pct_change(ROC_PERIOD_BARS) * 100
    d["Momentum_N"]    = d["close"] / d["close"].shift(ROC_PERIOD_BARS)
    d["Vol_Surge"]     = d["volume"] / d["VolAvg"]
    d["Pct_From_ATH"]  = (d["ATH"] - d["close"]) / d["ATH"] * 100

    d.dropna(subset=["EMA200", "High_1Y"], inplace=True)
    return d if len(d) > 0 else None

# =============================================================================
# RANK SCORE / REFERENCE HIGH / SIGNAL LABEL (unchanged from daily version --
# these operate generically on whatever columns are present)
# =============================================================================
def rank_score(row, close_price, ref_high, nifty_close=None, ts=None):
    if RANK_METHOD == "roc":
        val = row.get("ROC_N", 0)
        return float(val) if pd.notna(val) else 0.0
    elif RANK_METHOD == "rs_nifty":
        stock_ret = float(row.get("ROC_N", 0)) if pd.notna(row.get("ROC_N", 0)) else 0.0
        if nifty_close is not None and not nifty_close.empty and ts is not None:
            try:
                nifty_at   = nifty_close.asof(ts)
                nifty_past = nifty_close.asof(ts - pd.Timedelta(hours=RS_LOOKBACK_BARS))
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
    else:  # "momentum"
        val = row.get("Momentum_N", 0)
        return float(val) if pd.notna(val) else 0.0


def get_ref_high(row) -> tuple[float, str]:
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
# INTRADAY PORTFOLIO BACKTEST (hourly bars)
# =============================================================================
def backtest_portfolio(
    stock_data: dict[str, pd.DataFrame],
    nifty_close=None,
    sector_map: dict | None = None,
    sector_trends: dict | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Bar-by-bar (hourly) portfolio simulation.

    Signals are evaluated only on the day's SIGNAL_BAR_TIME candle (default
    9:15, i.e. the first hourly bar); accepted candidates fill at the very
    next candle's open (10:15 AM). Exits are checked on every hourly bar.
    """
    end_dt   = pd.Timestamp.today().normalize()
    bt_start = (end_dt - pd.Timedelta(days=LOOKBACK_DAYS)).date()

    all_bars = set()
    stock_bt_data = {}
    for symbol, df in stock_data.items():
        bt_df = df[df.index.date >= bt_start].copy()
        if bt_df.empty:
            continue
        stock_bt_data[symbol] = bt_df
        all_bars.update(bt_df.index)

    if not all_bars:
        return [], [], {}

    all_bars = sorted(all_bars)

    # Diagnostic: average bars/day actually observed (verify BARS_PER_DAY)
    bars_by_date = Counter(ts.date() for ts in all_bars)
    if bars_by_date:
        avg_bars_day = sum(bars_by_date.values()) / len(bars_by_date)
        print(f"  Avg bars/day observed across universe: {avg_bars_day:.2f} "
              f"(BARS_PER_DAY is set to {BARS_PER_DAY} -- adjust config if these differ a lot)")

    cash = float(INITIAL_CAPITAL)
    open_positions = {}
    trades = []
    equity_curve = []
    peak_equity = float(INITIAL_CAPITAL)
    max_concurrent = 0
    pending_entries = []

    funnel = {
        "signals_raw": 0,
        "lost_sector_trend": 0,
        "lost_sector_diversity_scan": 0,
        "queued": 0,
        "lost_no_data_next_bar": 0,
        "lost_sector_diversity_fill": 0,
        "lost_capital": 0,
        "filled": 0,
    }

    stock_row_lookup = {}
    for symbol, bt_df in stock_bt_data.items():
        row_dict = {}
        for ts, row in bt_df.iterrows():
            row_dict[ts] = row
        stock_row_lookup[symbol] = row_dict

    # ---- Bar-by-bar simulation ----
    for dt in all_bars:
        # ================================================================
        # STEP 0: FILL PENDING ENTRIES (signal from the prior signal-bar,
        # filled at THIS bar's open -- "buy at 10:15 AM candle")
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
                    funnel["lost_no_data_next_bar"] += 1
                    continue

                sec = cand["sector"]
                if USE_SECTOR_FILTER and sectors_held_fill.get(sec, 0) >= MAX_SAME_SECTOR:
                    funnel["lost_sector_diversity_fill"] += 1
                    continue

                raw_open = float(row["open"])
                if raw_open <= 0:
                    funnel["lost_capital"] += 1
                    continue
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
        # STEP 1: CHECK ALL OPEN POSITIONS FOR EXIT CONDITIONS (every bar)
        # ================================================================
        symbols_to_close = []

        for symbol, pos in open_positions.items():
            if symbol not in stock_row_lookup:
                continue
            row = stock_row_lookup[symbol].get(dt)
            if row is None:
                continue

            pos["bars_held"] += 1
            bars_held = pos["bars_held"]
            exit_reason = None
            exit_price  = 0.0

            if row["low"] <= pos["stop_price"] <= row["high"]:
                exit_price  = pos["stop_price"]
                exit_reason = "Stop_Loss"
            elif row["high"] >= pos["target_price"]:
                exit_price  = pos["target_price"]
                exit_reason = "Target_Hit"
            elif bars_held > MIN_HOLD_BARS:
                trail_ema = row["EMA10"] if bars_held > MIN_HOLD_BARS else row["EMA20"]
                if row["close"] < trail_ema:
                    exit_price  = float(row["close"])
                    exit_reason = "Oliver_EMA_Stop"

            if exit_reason is None and bars_held > MIN_HOLD_BARS:
                if bool(row["Extension"]):
                    exit_price  = float(row["close"])
                    exit_reason = "Extension_Exit"

            if exit_reason:
                if exit_reason not in ("Stop_Loss", "Target_Hit"):
                    exit_price = exit_price * (1 - SLIPPAGE_PCT / 100)

                invested      = pos["allocated_capital"] + pos["entry_cost"]
                sell_turnover = exit_price * pos["shares"]
                exit_charges  = sell_cost(sell_turnover)
                net_proceeds  = sell_turnover - exit_charges
                pnl_rs        = net_proceeds - invested
                pnl_pct       = pnl_rs / invested * 100
                total_costs   = pos["entry_cost"] + exit_charges

                hold_hours = (dt - pos["entry_date"]).total_seconds() / 3600
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
                    "Hold_Bars":      bars_held,
                    "Hold_Hours":     round(hold_hours, 2),
                    "Hold_Days":      round(hold_hours / 24, 2),
                    "Exit_Reason":    exit_reason,
                    "Entry_Month":    pos["entry_date"].strftime("%Y-%m"),
                    "Allocated_Capital": round(pos["allocated_capital"], 2),
                    "Costs_Rs":       round(total_costs, 2),
                })

                cash += net_proceeds
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del open_positions[s]

        # ================================================================
        # STEP 2: SCAN FOR NEW ENTRY SIGNALS -- ONLY on the day's signal bar
        # (default 9:15 AM candle). Accepted candidates fill at the NEXT
        # bar's open (10:15 AM), not this same bar.
        # ================================================================
        if dt.time() == SIGNAL_BAR_TIME:
            available_slots = MAX_POSITIONS - len(open_positions)

            if available_slots > 0:
                sectors_held = Counter(pos["sector"] for pos in open_positions.values())

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
                        continue
                    row = row_dict.get(dt)
                    if row is None:
                        continue
                    if not row["Signal"]:
                        continue
                    funnel["signals_raw"] += 1

                    rh, rh_type = get_ref_high(row)
                    if pd.isna(rh) or rh <= 0:
                        continue

                    if bool(row["Wedge_Pop"]):
                        ok_type = "Wedge"
                    elif bool(row["Crossback"]):
                        ok_type = "Crossback"
                    else:
                        continue

                    close_price = float(row["close"])
                    score = rank_score(row, close_price, rh, nifty_close, pd.Timestamp(dt))

                    entry_candidates.append({
                        "symbol":        symbol,
                        "close":         close_price,
                        "ref_high":      rh,
                        "ref_high_type": rh_type,
                        "oliver_type":   ok_type,
                        "rank_score":    score,
                    })

                entry_candidates.sort(key=lambda x: x["rank_score"], reverse=True)

                slots_queued = 0
                for cand in entry_candidates:
                    if slots_queued >= available_slots:
                        break

                    symbol = cand["symbol"]
                    sec = (sector_map or {}).get(symbol, symbol)

                    if USE_SECTOR_FILTER:
                        if sec in sector_trends and sec not in trending_sectors:
                            funnel["lost_sector_trend"] += 1
                            continue
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
        # STEP 3: TRACK EQUITY CURVE (every bar)
        # ================================================================
        positions_value = 0.0
        for symbol, pos in open_positions.items():
            row = stock_row_lookup.get(symbol, {}).get(dt)
            if row is not None:
                positions_value += pos["shares"] * float(row["close"])
            else:
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
    if open_positions and all_bars:
        last_bar = all_bars[-1]
        for symbol, pos in list(open_positions.items()):
            row = stock_row_lookup.get(symbol, {}).get(last_bar)
            if row is not None:
                ep_exit = float(row["close"]) * (1 - SLIPPAGE_PCT / 100)
            else:
                ep_exit = pos["entry_price"]

            invested      = pos["allocated_capital"] + pos["entry_cost"]
            sell_turnover = ep_exit * pos["shares"]
            exit_charges  = sell_cost(sell_turnover)
            net_proceeds  = sell_turnover - exit_charges
            pnl_rs        = net_proceeds - invested
            pnl_pct       = pnl_rs / invested * 100
            total_costs   = pos["entry_cost"] + exit_charges
            hold_hours    = (last_bar - pos["entry_date"]).total_seconds() / 3600

            trades.append({
                "Symbol":         symbol,
                "Sector":         pos.get("sector", "Unknown"),
                "Entry_Date":     pos["entry_date"],
                "Exit_Date":      last_bar,
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
                "Hold_Bars":      pos["bars_held"],
                "Hold_Hours":     round(hold_hours, 2),
                "Hold_Days":      round(hold_hours / 24, 2),
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
    label: str = "NEAR HIGH + OLIVER KELL HOURLY INTRADAY BACKTEST",
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

    if not equity_curve_df.empty:
        max_dd = equity_curve_df["Drawdown_Pct"].min()
        final_equity = equity_curve_df["Equity"].iloc[-1]
    else:
        max_dd = 0.0
        final_equity = INITIAL_CAPITAL + net_pnl

    avg_hold_hours = trades_df["Hold_Hours"].mean()
    same_day_pct = (trades_df["Hold_Hours"] < 7).mean() * 100

    if not equity_curve_df.empty:
        max_concurrent = equity_curve_df["Open_Positions"].max()
        bars_with_positions = (equity_curve_df["Open_Positions"] > 0).sum()
        total_bars = len(equity_curve_df)
        capital_utilization = bars_with_positions / total_bars * 100 if total_bars > 0 else 0
        avg_positions = equity_curve_df["Open_Positions"].mean()
    else:
        max_concurrent = 0
        capital_utilization = 0
        avg_positions = 0

    print("  PORTFOLIO CONFIGURATION:")
    print(f"  Initial Capital    : Rs {INITIAL_CAPITAL:>14,.0f}")
    print(f"  Max Positions      : {MAX_POSITIONS:>12d}")
    print(f"  Starting Per-Position Size: Rs {PER_POSITION_CAPITAL:>8,.0f}  (compounds with equity)")
    print(f"  Signal bar         : {SIGNAL_BAR_TIME}  |  Fill: next bar's open  |  BARS_PER_DAY: {BARS_PER_DAY}")

    print("\n  PERFORMANCE SUMMARY:")
    print(f"  Final Equity       : Rs {final_equity:>14,.0f}")
    print(f"  Net P&L            : Rs {net_pnl:>14,.0f}")
    print(f"  Total Return       : {total_return:>11.2f}%")
    print(f"  Total Trades       : {total:>12d}")
    print(f"  Win Rate           : {win_rate:>11.2f}%")
    print(f"  Max Drawdown       : {max_dd:>11.2f}%")
    print(f"  Profit Factor      : {profit_factor:>12.2f}")
    print(f"  Avg Win            : {avg_win:>11.2f}%")
    print(f"  Avg Loss           : {avg_loss:>11.2f}%")
    print(f"  Avg Hold Time      : {avg_hold_hours:>10.1f} hours")
    print(f"  Same-day exits     : {same_day_pct:>11.1f}%  (held < 7 hours)")
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
        print("\n  SLIPPAGE & COST IMPACT:")
        print(f"  Slippage Assumption: {SLIPPAGE_PCT:.2f}%  (market-style fills only; limit fills get 0%)")
        print(f"  Total Costs Paid   : Rs {total_costs:>14,.0f}")
        print(f"  Costs % of Turnover: {cost_pct_turnover:>11.3f}%")
        print(f"  Avg Cost / Trade   : Rs {total_costs / total:>14,.0f}")
        print(f"  P&L Before Costs   : Rs {pnl_before_costs:>14,.0f}")
        print(f"  P&L After Costs    : Rs {net_pnl:>14,.0f}  (costs ate {total_costs / abs(pnl_before_costs) * 100 if pnl_before_costs != 0 else 0:.1f}% of gross P&L)")

    print("\n  PORTFOLIO-SPECIFIC STATS:")
    print(f"  Max Concurrent Pos : {max_concurrent:>12d}")
    print(f"  Avg Concurrent Pos : {avg_positions:>12.2f}")
    print(f"  Capital Utilization: {capital_utilization:>11.2f}%  (bars with >= 1 position)")
    print(f"  Hourly Bars        : {len(equity_curve_df) if not equity_curve_df.empty else 0:>12d}")

    print("\n  EXIT REASON BREAKDOWN:")
    for reason, cnt in trades_df["Exit_Reason"].value_counts().items():
        avg_p = trades_df[trades_df["Exit_Reason"] == reason]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Exit_Reason"] == reason) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {reason:<22}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    print("\n  NEAR-HIGH TYPE BREAKDOWN:")
    for ht, cnt in trades_df["Ref_High_Type"].value_counts().items():
        avg_p = trades_df[trades_df["Ref_High_Type"] == ht]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Ref_High_Type"] == ht) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {ht:<6}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    print("\n  OLIVER SIGNAL TYPE BREAKDOWN:")
    for ot, cnt in trades_df["Oliver_Type"].value_counts().items():
        avg_p = trades_df[trades_df["Oliver_Type"] == ot]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Oliver_Type"] == ot) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {ot:<12}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    if "Sector" in trades_df.columns:
        print("\n  SECTOR BREAKDOWN (sectors with >= 3 trades):")
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

    if not equity_curve_df.empty and len(equity_curve_df) > 1:
        first_date = equity_curve_df["Date"].iloc[0]
        last_date  = equity_curve_df["Date"].iloc[-1]
        years = (last_date - first_date).days / 365.25
        if years > 0 and final_equity > 0:
            cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1
            print(f"\n  CAGR               : {cagr * 100:>11.2f}%  (over {years:.1f} years)")

    if not equity_curve_df.empty:
        print("\n  CALENDAR YEAR RETURNS:")
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

    print("\n  MONTHLY P&L BREAKDOWN:")
    net_abs = max(abs(net_pnl), 1)
    monthly = trades_df.groupby("Entry_Month")["PnL_Rs"].sum()
    for m, pnl in monthly.items():
        bar = ("+" if pnl >= 0 else "-") * min(40, int(abs(pnl) / net_abs * 200))
        print(f"    {m}  Rs {pnl:>12,.0f}  {bar}")

    print("\n  TOP 10 WINNING TRADES:")
    cols = ["Symbol", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price",
            "PnL_Rs", "PnL_Pct", "Hold_Hours", "Signal_Type", "Exit_Reason"]
    print(trades_df.nlargest(10, "PnL_Rs")[cols].to_string(index=False))

    print("\n  TOP 10 LOSING TRADES:")
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
    print("  NEAR HIGH + OLIVER KELL -- 1-HOUR INTRADAY BACKTEST (Angel One)")
    print(f"  EMA Alignment + Near High ({int((1-NEAR_HIGH_UPPER)*100)}-{int((1-NEAR_HIGH_LOWER)*100)}% below) + Wedge_Pop | Crossback")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ref high")
    print(f"  Signal bar: {SIGNAL_BAR_TIME}  |  Fill: next hourly bar's open  |  Exits checked every hourly bar")
    print(f"  BARS_PER_DAY: {BARS_PER_DAY}  |  Min hold: {MIN_HOLD_DAYS} days ({MIN_HOLD_BARS} bars)")
    print(f"  Fetch: {FETCH_YEARS} years hourly via Angel One (chunked, ~{CHUNK_DAYS}-day requests)")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}  |  Max Positions: {MAX_POSITIONS}")
    print(f"  Rank Method: {RANK_METHOD.upper()}")
    print(f"  Slippage: {SLIPPAGE_PCT:.2f}% on market-style fills  |  Costs: STT/exch/SEBI/stamp/GST/DP (see config)")
    print("=" * 70)

    print("\nLogging in to Angel One...")
    obj = angel_login()
    print("  LOGIN SUCCESSFUL")

    print("\nLoading NSE symbol map...")
    sym_map = get_nse_symbol_map()

    if use_all_nse and tickers is None:
        universe = list(sym_map.keys())
        print(f"  Universe: ALL NSE-EQ stocks ({len(universe):,}) -- this will take a LONG time via Angel One")
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

    print("Fetching Nifty 50 hourly data for RS calculation...")
    nifty_close = load_nifty_hourly(obj)
    if nifty_close is None or nifty_close.empty:
        print("  WARNING: Nifty data unavailable, RS_Leader will be set to True for all stocks.")
        nifty_close = None
    else:
        print(f"  Nifty data: {len(nifty_close):,} hourly bars")

    # =================================================================
    # PASS 1: Resolve cache / fetch (chunked, resumable per-symbol) + indicators
    # =================================================================
    print(f"\n--- PASS 1: Loading hourly data for {total:,} stocks (Angel One, one request-set per stock) ---\n")

    stock_data = {}
    from_cache = 0
    fetched    = 0
    failed     = 0
    not_found  = 0
    t0 = time.time()

    for i, symbol in enumerate(universe, 1):
        token = resolve_ticker(symbol, sym_map)
        if token is None:
            not_found += 1
            continue

        cache_path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
        was_cached = os.path.exists(cache_path) and (
            time.time() - os.path.getmtime(cache_path) < CACHE_MAX_AGE_HOURS * 3600
        )

        df = fetch_with_cache(obj, token, symbol)
        if was_cached:
            from_cache += 1
        elif not df.empty:
            fetched += 1
        else:
            failed += 1

        if not df.empty:
            d = compute_indicators(df.copy(), nifty_close)
            if d is not None:
                stock_data[symbol] = d

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 1
            eta_min = (total - i) / rate / 60
            print(f"  [{i:>4}/{total}]  cache:{from_cache:>4}  fetched:{fetched:>4}  "
                  f"failed:{failed:>4}  ready:{len(stock_data):>4}  ETA:{eta_min:.1f}min", flush=True)

    elapsed_1 = (time.time() - t0) / 60
    print(f"\n  Pass 1 done. {len(stock_data):,} stocks ready in {elapsed_1:.1f} min "
          f"(from cache: {from_cache}, freshly fetched: {fetched}, failed: {failed}, not found: {not_found})")

    if not stock_data:
        print("  No stock data available. Exiting.")
        return pd.DataFrame()

    nifty_rank = None
    if RANK_METHOD == "rs_nifty":
        nifty_rank = nifty_close

    sector_map    = {}
    sector_trends = {}
    if USE_SECTOR_FILTER:
        print("\nLoading sector map (Nifty 500 constituents, daily/yfinance)...")
        sector_map = get_sector_map()
        print("\nFetching sector index trends (daily/yfinance)...")
        sector_trends = fetch_sector_trends(FETCH_YEARS)
        if not sector_trends:
            print("  WARNING: No sector trend data fetched. Trend filter will be skipped.")

    # =================================================================
    # PASS 2: Bar-by-bar (hourly) portfolio simulation
    # =================================================================
    sector_status = "ON" if USE_SECTOR_FILTER else "OFF"
    print(f"\n--- PASS 2: Intraday hourly simulation ({len(stock_data):,} stocks) | "
          f"Rank: {RANK_METHOD.upper()} | Sector filter: {sector_status} ---\n")

    t1 = time.time()
    all_trades, equity_curve, funnel = backtest_portfolio(stock_data, nifty_rank, sector_map, sector_trends)
    sim_elapsed = (time.time() - t1) / 60

    print(f"  Pass 2 done. {len(all_trades):,} trades in {sim_elapsed:.1f} min")
    print(f"  Equity curve: {len(equity_curve):,} hourly bars\n")

    if funnel:
        print("  SIGNAL FUNNEL (why candidates did/didn't become trades):")
        print(f"    Raw signals found (Signal=True, not already held) : {funnel['signals_raw']:>7,}")
        print(f"    Lost to sector-trend filter (scan time)           : {funnel['lost_sector_trend']:>7,}")
        print(f"    Lost to sector-diversity cap (scan time)          : {funnel['lost_sector_diversity_scan']:>7,}")
        print(f"    Queued for next-bar open (10:15 AM)               : {funnel['queued']:>7,}")
        print(f"    Lost: stock didn't trade on the fill bar          : {funnel['lost_no_data_next_bar']:>7,}")
        print(f"    Lost: sector slot taken by another fill that bar  : {funnel['lost_sector_diversity_fill']:>7,}")
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

    performance_report(trades_df, equity_curve_df)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_trades = f"near_high_oliver_hourly_trades_{ts}.csv"
    trades_df.to_csv(out_trades, index=False)
    print(f"  Trades saved to: {out_trades}")

    out_equity = f"near_high_oliver_hourly_equity_{ts}.csv"
    equity_curve_df.to_csv(out_equity, index=False)
    print(f"  Equity curve saved to: {out_equity}")

    return trades_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Default: Nifty 500 stocks (recommended -- ~500 x 6 chunks = ~3,000 Angel
    # One API calls; full NSE would be ~14,250 calls and much slower).
    run_backtest(use_all_nse=False, tickers=NIFTY_500)

    # Example: first 30 stocks only (fast smoke test before running the full universe)
    # run_backtest(use_all_nse=False, tickers=NIFTY_500[:30])

    # Example: specific tickers only
    # run_backtest(use_all_nse=False, tickers=["RELIANCE", "TCS", "INFY", "HDFCBANK"])

    # Example: full NSE universe (slow -- ~14,250 Angel One API calls)
    # run_backtest(use_all_nse=True)
