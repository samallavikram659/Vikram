"""
=============================================================================
NEAR HIGH + OLIVER KELL COMBINED BACKTEST  --  PORTFOLIO MODE
=============================================================================
UNIVERSE  : Nifty 500 stocks (default) or All NSE EQ-series (~2500)
            Excludes: ETFs, Liquid Funds, Bonds, Index Funds

PORTFOLIO : INITIAL_CAPITAL = Rs 1,00,000 (1 lakh)
            MAX_POSITIONS   = 3 (max 3 stocks held at any time)
            Equal allocation: capital / 3 per position (~33,333 each)
            Capital recycled on exit

SIMULATION: Two-pass approach
            Pass 1 - Fetch data + compute indicators for ALL stocks
            Pass 2 - Iterate day-by-day across all trading dates
              (a) Check all open positions for exits first
              (b) Then scan for new entry signals (if open < MAX_POSITIONS)
              (c) On ties, pick highest momentum (closest to reference high)

ENTRY     : EMA alignment (Close > EMA5 > EMA10 > EMA20 > EMA50 > EMA150 > EMA200)
          + Close within 5-10% BELOW the key high (1Y / 5Y / ATH)
          + Oliver Kell signal: Wedge_Pop OR Crossback

EXIT      : (checked in priority order)
            1. Stop loss 5% below entry price           (on bar Low)
            2. Target 10% above ref high                (on bar High)
            3. Oliver Trailing EMA stop (EMA20 early, EMA10 after MIN_HOLD_DAYS)
            4. Extension exit (Close > EMA20 + 3*ATR)

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

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from SmartApi import SmartConnect
import pyotp

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================
ANGEL_API_KEY        = "Tp7PJMIc"
ANGEL_CLIENT_ID      = "S63068620"
ANGEL_CLIENT_PIN     = "4098"
ANGEL_TOTP_SECRET    = "KVTELUFGR33YCPQUKO3ZSL4NMA"

INITIAL_CAPITAL      = 100_000               # Rs 1 lakh
MAX_POSITIONS        = 3                     # max concurrent positions
STOP_LOSS_PCT        = 0.05                  # 5% stop loss
TARGET_ABOVE_HIGH_PCT = 0.10                 # 10% above ref high
NEAR_HIGH_LOWER      = 0.90                  # 10% below high (lower bound)
NEAR_HIGH_UPPER      = 0.95                  # 5% below high (upper bound)
LOOKBACK_DAYS        = 365                   # backtest window in days
FETCH_YEARS          = 6                     # years of history to fetch
NIFTY_TOKEN          = "99926000"            # Nifty 50 token on NSE

# Oliver Kell specific config
MINI_BASE_BARS       = 7                     # window for mini-base detection
MINI_BASE_MIN        = 4                     # minimum qualifying bars in base
EXTENSION_MULT       = 3.0                   # ATR multiplier for extension exit
MIN_HOLD_DAYS        = 2                     # hold for at least this many bars before trailing stop
RS_LOOKBACK          = 20                    # days for relative-strength calculation

CACHE_DIR            = "cache_near_high"
MIN_DATA_DAYS        = 252                   # minimum bars needed

# Derived
PER_POSITION_CAPITAL = INITIAL_CAPITAL / MAX_POSITIONS  # ~33,333 per slot

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
# AUTHENTICATION
# =============================================================================
def _fix_totp(raw: str) -> str:
    s = raw.strip().upper().replace(" ", "").replace("-", "")
    pad = (8 - len(s) % 8) % 8
    return s + "=" * pad


def angel_login() -> SmartConnect:
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
# NSE SYMBOL MAP
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
    """Return Angel One token for a symbol, trying common suffixes."""
    for candidate in [symbol, symbol.upper(), symbol + "-EQ"]:
        if candidate in sym_map:
            return sym_map[candidate]
    return None

# =============================================================================
# CHUNKED OHLCV FETCH
# =============================================================================
def fetch_daily_ohlcv(
    obj: SmartConnect,
    token: str,
    symbol: str,
    from_date: str,
    to_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily OHLCV in 300-day chunks to handle API limits."""
    rows = []
    cur = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)

    while cur <= end:
        chunk_end = min(cur + pd.DateOffset(days=299), end)
        for attempt in range(max_retries):
            try:
                r = obj.getCandleData({
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": "ONE_DAY",
                    "fromdate": cur.strftime("%Y-%m-%d %H:%M"),
                    "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
                })
                if r.get("status") and r.get("data"):
                    rows.extend(r["data"])
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        cur = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.35)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    df.index = df.index.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


def fetch_with_cache(
    obj: SmartConnect,
    token: str,
    symbol: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV using per-stock pickle cache (< 1 day old = reuse)."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_1D.pkl")
    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days < 1:
            with open(cache_path, "rb") as fh:
                df = pickle.load(fh)
            if not df.empty:
                return df

    df = fetch_daily_ohlcv(obj, token, symbol, from_date, to_date)
    if not df.empty:
        with open(cache_path, "wb") as fh:
            pickle.dump(df, fh)
    return df

# =============================================================================
# NIFTY REFERENCE DATA (for RS calculation)
# =============================================================================
_nifty_cache: pd.Series | None = None

def load_nifty(obj: SmartConnect, from_date: str, to_date: str) -> pd.Series:
    global _nifty_cache
    if _nifty_cache is not None:
        return _nifty_cache
    nf = fetch_daily_ohlcv(obj, NIFTY_TOKEN, "NIFTY50", from_date, to_date)
    if nf.empty:
        _nifty_cache = pd.Series(dtype=float)
    else:
        _nifty_cache = nf["close"]
    return _nifty_cache

# =============================================================================
# INDICATOR COMPUTATION
# =============================================================================
def compute_indicators(df: pd.DataFrame, nifty_close: pd.Series | None = None) -> pd.DataFrame | None:
    if len(df) < MIN_DATA_DAYS:
        return None

    d = df.copy()

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

    # ---- Near-high conditions ----
    d["Near_1Y"]  = (d["close"] >= d["High_1Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_1Y"] * NEAR_HIGH_UPPER)
    d["Near_5Y"]  = (d["close"] >= d["High_5Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_5Y"] * NEAR_HIGH_UPPER)
    d["Near_ATH"] = (d["close"] >= d["ATH"]     * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["ATH"]      * NEAR_HIGH_UPPER)
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
    atr_series = d["ATR14"]
    mini_base_list = []
    for i in range(len(d)):
        if i < MINI_BASE_BARS:
            mini_base_list.append(False)
            continue
        window = atr_series.iloc[i - MINI_BASE_BARS + 1: i + 1].values
        # count bars where ATR is lower than the bar before it
        contracting = sum(1 for j in range(1, len(window)) if window[j] < window[j-1])
        mini_base_list.append(contracting >= MINI_BASE_MIN)
    d["Mini_Base"] = mini_base_list

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

    d.dropna(subset=["EMA200", "High_1Y"], inplace=True)
    return d if len(d) > 0 else None

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
def momentum_score(close: float, ref_high: float) -> float:
    """
    Higher score = closer to reference high = higher momentum.
    Returns close / ref_high ratio (0 to 1+).
    """
    if ref_high <= 0 or pd.isna(ref_high):
        return 0.0
    return close / ref_high

# =============================================================================
# PORTFOLIO-BASED BACKTEST
# =============================================================================
def backtest_portfolio(stock_data: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
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
        return [], []

    all_dates = sorted(all_dates)

    # ---- Portfolio state ----
    cash = float(INITIAL_CAPITAL)
    open_positions = {}  # symbol -> position dict
    trades = []
    equity_curve = []
    peak_equity = float(INITIAL_CAPITAL)
    max_concurrent = 0

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

            # 1. Hard stop loss (on Low)
            if row["low"] <= pos["stop_price"]:
                exit_price  = pos["stop_price"]
                exit_reason = "Stop_Loss"

            # 2. Target hit (on High)
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
                pnl_rs  = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

                trades.append({
                    "Symbol":         symbol,
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
                })

                # Return capital (allocated + pnl)
                cash += pos["allocated_capital"] + pnl_rs
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del open_positions[s]

        # ================================================================
        # STEP 2: SCAN FOR NEW ENTRY SIGNALS (if slots available)
        # ================================================================
        available_slots = MAX_POSITIONS - len(open_positions)

        if available_slots > 0:
            entry_candidates = []

            for symbol, row_dict in stock_row_lookup.items():
                if symbol in open_positions:
                    continue  # already holding this stock
                row = row_dict.get(dt)
                if row is None:
                    continue
                if not row["Signal"]:
                    continue

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
                score = momentum_score(close_price, rh)

                entry_candidates.append({
                    "symbol":      symbol,
                    "close":       close_price,
                    "ref_high":    rh,
                    "ref_high_type": rh_type,
                    "oliver_type": ok_type,
                    "momentum":    score,
                    "row":         row,
                })

            # Sort by highest momentum (closest to reference high)
            entry_candidates.sort(key=lambda x: x["momentum"], reverse=True)

            # Take up to available_slots
            for cand in entry_candidates[:available_slots]:
                ep = cand["close"]
                sp = round(ep * (1 - STOP_LOSS_PCT), 2)
                tp = round(cand["ref_high"] * (1 + TARGET_ABOVE_HIGH_PCT), 2)

                # Allocate capital for this position
                alloc = min(PER_POSITION_CAPITAL, cash)
                if alloc < 1.0 or ep <= 0:
                    continue  # not enough capital

                shares = int(alloc / ep)
                if shares < 1:
                    continue

                actual_cost = shares * ep
                if actual_cost > cash:
                    continue

                cash -= actual_cost

                open_positions[cand["symbol"]] = {
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
                }

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
                ep_exit = float(row["close"])
            else:
                ep_exit = pos["entry_price"]  # fallback

            pnl_rs  = (ep_exit - pos["entry_price"]) * pos["shares"]
            pnl_pct = (ep_exit - pos["entry_price"]) / pos["entry_price"] * 100

            trades.append({
                "Symbol":         symbol,
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
            })

    return trades, equity_curve

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
    print(f"  Per-Position Size  : Rs {PER_POSITION_CAPITAL:>14,.0f}")

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
    print(f"  Per-Position: Rs {PER_POSITION_CAPITAL:,.0f}")
    print("=" * 70)

    # Login
    print("\nLogging in to Angel One...")
    obj = angel_login()
    print("  LOGIN SUCCESSFUL\n")

    # Build symbol map
    print("Loading NSE symbol map...")
    sym_map = get_nse_symbol_map()

    # Decide universe
    if use_all_nse and tickers is None:
        universe = list(sym_map.items())
        print(f"  Universe: ALL NSE-EQ stocks ({len(universe):,})")
    elif tickers is not None:
        # Filter sym_map to only include stocks in the provided tickers list
        universe = []
        not_found = []
        for t in tickers:
            tok = resolve_ticker(t, sym_map)
            if tok:
                universe.append((t, tok))
            else:
                not_found.append(t)
        if not_found:
            print(f"  WARNING: {len(not_found)} tickers not found in symbol map: {not_found[:20]}{'...' if len(not_found) > 20 else ''}")
        print(f"  Universe: {len(universe):,} stocks (from provided tickers list)")
    else:
        # Default: use NIFTY_500
        universe = []
        not_found = []
        for t in NIFTY_500:
            tok = resolve_ticker(t, sym_map)
            if tok:
                universe.append((t, tok))
            else:
                not_found.append(t)
        if not_found:
            print(f"  WARNING: {len(not_found)} NIFTY_500 tickers not found: {not_found[:20]}{'...' if len(not_found) > 20 else ''}")
        print(f"  Universe: NIFTY 500 ({len(universe):,} stocks resolved)")

    if max_stocks is not None:
        universe = universe[:max_stocks]

    total = len(universe)
    print(f"\nUniverse: {total:,} stocks\n")

    # Date range for fetching
    today      = datetime.date.today()
    fetch_end  = today.strftime("%Y-%m-%d")
    fetch_start = (today - datetime.timedelta(days=365 * FETCH_YEARS)).strftime("%Y-%m-%d")

    # Fetch Nifty reference data (for RS calculation)
    print("Fetching Nifty 50 data for RS calculation...")
    nifty_close = load_nifty(obj, fetch_start, fetch_end)
    if nifty_close.empty:
        print("  WARNING: Nifty data unavailable, RS_Leader will be set to True for all stocks.")
    else:
        print(f"  Nifty data: {len(nifty_close):,} bars")

    # =================================================================
    # PASS 1: Fetch data + compute indicators for ALL stocks
    # =================================================================
    print(f"\n--- PASS 1: Fetching data & computing indicators for {total:,} stocks ---\n")
    stock_data = {}  # {symbol: DataFrame with indicators}
    processed  = 0
    skipped    = 0

    t0 = time.time()
    for i, (symbol, token) in enumerate(universe, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            rate    = i / max(elapsed, 1)
            eta_m   = (total - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:4d}/{total}] {symbol:<18}  loaded:{len(stock_data):4d}  "
                  f"ETA:{eta_m:.1f}min", flush=True)

        try:
            df_raw = fetch_with_cache(obj, token, symbol, fetch_start, fetch_end)
            if df_raw.empty or len(df_raw) < MIN_DATA_DAYS:
                skipped += 1
                continue

            df_ind = compute_indicators(df_raw, nifty_close if not nifty_close.empty else None)
            if df_ind is None:
                skipped += 1
                continue

            stock_data[symbol] = df_ind
            processed += 1

        except Exception as e:
            print(f"\n  ERROR on {symbol}: {e}")
            skipped += 1

    elapsed_m = (time.time() - t0) / 60
    print(f"\n  Pass 1 done. {processed:,} stocks loaded, {skipped:,} skipped in {elapsed_m:.1f} min")

    if not stock_data:
        print("  No stock data available. Exiting.")
        return pd.DataFrame()

    # =================================================================
    # PASS 2: Day-by-day portfolio simulation
    # =================================================================
    print(f"\n--- PASS 2: Portfolio simulation ({len(stock_data):,} stocks) ---\n")

    t1 = time.time()
    all_trades, equity_curve = backtest_portfolio(stock_data)
    sim_elapsed = (time.time() - t1) / 60

    print(f"  Pass 2 done. {len(all_trades):,} trades in {sim_elapsed:.1f} min")
    print(f"  Equity curve: {len(equity_curve):,} data points\n")

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
    # Default: Nifty 500 stocks only (faster, focused on quality universe)
    run_backtest(use_all_nse=False, tickers=NIFTY_500)

    # Example: all NSE EQ stocks (~2500)
    # run_backtest(use_all_nse=True, max_stocks=None)

    # Example: specific tickers only
    # run_backtest(use_all_nse=False, tickers=["RELIANCE", "TCS", "INFY", "HDFCBANK"])

    # Example: first 100 stocks (for testing)
    # run_backtest(use_all_nse=True, max_stocks=100)
