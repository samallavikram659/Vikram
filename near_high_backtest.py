"""
=============================================================================
NEAR HIGH BACKTEST - PORTFOLIO-BASED
=============================================================================
UNIVERSE  : Nifty 500 stocks (default) or All NSE EQ-series (~2500)
            Excludes: ETFs, Liquid Funds, Bonds, Index Funds

PORTFOLIO : INITIAL_CAPITAL = Rs 1,00,000
            MAX_POSITIONS   = 3 (equal allocation ~33,333 each)
            Capital recycled on exit

ENTRY     : EMA alignment (Close > EMA5 > EMA10 > EMA20 > EMA50 > EMA150 > EMA200)
            Close within 5-10% BELOW the key high (1Y / 5Y / ATH)
            Signal fires if ANY of the three key-high conditions are met

EXIT      : Stop loss 5% below entry price  OR  Target 10% above ref high
            Whichever is triggered first (checked bar-by-bar on Low / High)

SIGNAL    : 'NearHigh_1Y', 'NearHigh_5Y', 'NearHigh_ATH'

SIMULATION: Day-by-day across all stocks simultaneously
            - Exit checks first, then entry scans
            - If multiple entry signals on same day, pick highest momentum
              (closest to its reference high)

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

INITIAL_CAPITAL      = 100_000              # Rs 1 lakh
MAX_POSITIONS        = 3                    # max concurrent positions
STOP_LOSS_PCT        = 0.05                 # 5% stop loss
TARGET_ABOVE_HIGH_PCT = 0.10               # 10% above ref high
NEAR_HIGH_LOWER      = 0.90                # 10% below high (lower bound)
NEAR_HIGH_UPPER      = 0.95                # 5% below high (upper bound)
LOOKBACK_DAYS        = 365                 # backtest window in days
FETCH_YEARS          = 6                   # years of history to fetch
NIFTY_TOKEN          = "99926000"          # Nifty 50 token on NSE

RANK_METHOD          = "momentum"          # "roc" | "rs_nifty" | "near_ath" | "volume_surge" | "momentum"
ROC_PERIOD           = 90                  # days for ROC and momentum calculation
RS_RANK_PERIOD       = 90                  # days for RS vs Nifty ranking

CACHE_DIR            = "cache_near_high"
MIN_DATA_DAYS        = 252                 # minimum bars needed

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
    """Fetch daily OHLCV with no caching (raw fetch)."""
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
# INDICATOR COMPUTATION
# =============================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    if len(df) < MIN_DATA_DAYS:
        return None

    d = df.copy()

    # EMAs on Close
    d["EMA5"]   = d["close"].ewm(span=5,   adjust=False).mean()
    d["EMA10"]  = d["close"].ewm(span=10,  adjust=False).mean()
    d["EMA20"]  = d["close"].ewm(span=20,  adjust=False).mean()
    d["EMA50"]  = d["close"].ewm(span=50,  adjust=False).mean()
    d["EMA150"] = d["close"].ewm(span=150, adjust=False).mean()
    d["EMA200"] = d["close"].ewm(span=200, adjust=False).mean()

    # ATR14
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(span=14, adjust=False).mean()

    # Key highs (computed on High column)
    d["High_1Y"] = d["high"].rolling(252,  min_periods=200).max()
    d["High_5Y"] = d["high"].rolling(1260, min_periods=252).max()
    d["ATH"]     = d["high"].cummax()

    # EMA alignment
    d["EMA_Aligned"] = (
        (d["close"] > d["EMA5"])   &
        (d["EMA5"]  > d["EMA10"])  &
        (d["EMA10"] > d["EMA20"])  &
        (d["EMA20"] > d["EMA50"])  &
        (d["EMA50"] > d["EMA150"]) &
        (d["EMA150"]> d["EMA200"])
    )

    # Near-high conditions
    d["Near_1Y"]  = (d["close"] >= d["High_1Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_1Y"] * NEAR_HIGH_UPPER)
    d["Near_5Y"]  = (d["close"] >= d["High_5Y"] * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["High_5Y"] * NEAR_HIGH_UPPER)
    d["Near_ATH"] = (d["close"] >= d["ATH"]     * NEAR_HIGH_LOWER) & \
                    (d["close"] <= d["ATH"]      * NEAR_HIGH_UPPER)

    d["Near_High"] = d["Near_1Y"] | d["Near_5Y"] | d["Near_ATH"]
    d["Signal"]    = d["EMA_Aligned"] & d["Near_High"]

    # Ranking indicators
    d["ROC_90"]        = d["close"].pct_change(ROC_PERIOD) * 100
    d["Momentum_90"]   = d["close"] / d["close"].shift(ROC_PERIOD)
    d["Vol_Surge"]     = d["volume"] / d["volume"].rolling(20, min_periods=10).mean()
    d["Pct_From_ATH"]  = (d["ATH"] - d["close"]) / d["ATH"] * 100

    d.dropna(subset=["EMA200", "High_1Y"], inplace=True)
    return d if len(d) > 0 else None


# =============================================================================
# COMPUTE RS VS NIFTY FOR RANKING
# =============================================================================
_nifty_close_cache = None

def load_nifty_for_ranking(obj, from_date, to_date):
    global _nifty_close_cache
    if _nifty_close_cache is not None:
        return _nifty_close_cache
    nf = fetch_daily_ohlcv(obj, NIFTY_TOKEN, "NIFTY50", from_date, to_date)
    if nf.empty:
        _nifty_close_cache = pd.Series(dtype=float)
    else:
        _nifty_close_cache = nf["close"]
    return _nifty_close_cache


def rank_score(row, close_price, ref_high, nifty_close=None, day=None):
    if RANK_METHOD == "roc":
        val = row.get("ROC_90", 0)
        return float(val) if pd.notna(val) else 0.0

    elif RANK_METHOD == "rs_nifty":
        if nifty_close is not None and not nifty_close.empty and day is not None:
            stock_ret = float(row.get("ROC_90", 0))
            try:
                nifty_at = nifty_close.asof(day)
                nifty_past = nifty_close.asof(day - pd.Timedelta(days=RS_RANK_PERIOD))
                if pd.notna(nifty_at) and pd.notna(nifty_past) and nifty_past > 0:
                    nifty_ret = (nifty_at - nifty_past) / nifty_past * 100
                    return stock_ret - nifty_ret
            except Exception:
                pass
            return stock_ret
        return float(row.get("ROC_90", 0)) if pd.notna(row.get("ROC_90", 0)) else 0.0

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
    Prefers 1Y > 5Y > ATH in ascending-value order when multiple fire.
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
    # Use smallest triggered reference high
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def signal_type_label(ref_high_type: str) -> str:
    return f"NearHigh_{ref_high_type}"

# =============================================================================
# PORTFOLIO BACKTEST (DAY-BY-DAY ACROSS ALL STOCKS)
# =============================================================================
def portfolio_backtest(stock_data: dict[str, pd.DataFrame], nifty_close=None) -> tuple[list[dict], list[dict]]:
    """
    Run a portfolio-level day-by-day simulation across all stocks.

    Parameters
    ----------
    stock_data : dict mapping symbol -> DataFrame with indicators computed

    Returns
    -------
    trades : list of completed trade dicts
    equity_history : list of {date, equity, open_positions, invested, cash} dicts
    """
    today = datetime.date.today()
    bt_start = today - datetime.timedelta(days=LOOKBACK_DAYS)

    # -------------------------------------------------------------------------
    # Build a combined set of all trading dates within the backtest window
    # -------------------------------------------------------------------------
    all_dates = set()
    stock_bt_data = {}  # symbol -> DataFrame filtered to backtest period

    for symbol, df in stock_data.items():
        bt_df = df[df.index.date >= bt_start].copy()
        if bt_df.empty:
            continue
        stock_bt_data[symbol] = bt_df
        all_dates.update(bt_df.index.date)

    if not all_dates:
        return [], []

    trading_dates = sorted(all_dates)

    # -------------------------------------------------------------------------
    # Portfolio state
    # -------------------------------------------------------------------------
    cash = float(INITIAL_CAPITAL)
    open_positions = {}  # symbol -> position dict
    completed_trades = []
    equity_history = []
    max_concurrent = 0

    # For quick row lookup: pre-build {symbol -> {date -> row_dict}}
    symbol_date_rows = {}
    for symbol, bt_df in stock_bt_data.items():
        rows_by_date = {}
        for idx, row in bt_df.iterrows():
            rows_by_date[idx.date()] = row
        symbol_date_rows[symbol] = rows_by_date

    # -------------------------------------------------------------------------
    # Day-by-day simulation
    # -------------------------------------------------------------------------
    for day in trading_dates:

        # === STEP 1: CHECK EXITS ON ALL OPEN POSITIONS ===
        symbols_to_close = []
        for symbol, pos in open_positions.items():
            if symbol not in symbol_date_rows:
                continue
            row = symbol_date_rows[symbol].get(day)
            if row is None:
                continue  # stock not traded on this day

            pos["bars_held"] += 1
            exit_reason = None
            exit_price = 0.0

            # Check stop first (on Low), then target (on High)
            if row["low"] <= pos["stop_price"]:
                exit_price = pos["stop_price"]
                exit_reason = "Stop_Loss"
            elif row["high"] >= pos["target_price"]:
                exit_price = pos["target_price"]
                exit_reason = "Target_Hit"

            if exit_reason:
                pnl_rs = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
                proceeds = exit_price * pos["shares"]
                cash += proceeds

                completed_trades.append({
                    "Symbol":         symbol,
                    "Entry_Date":     pos["entry_date"],
                    "Exit_Date":      pd.Timestamp(day),
                    "Signal_Type":    signal_type_label(pos["ref_high_type"]),
                    "Entry_Price":    round(pos["entry_price"], 2),
                    "Exit_Price":     round(exit_price, 2),
                    "Ref_High":       round(pos["ref_high"], 2),
                    "Ref_High_Type":  pos["ref_high_type"],
                    "Target_Price":   round(pos["target_price"], 2),
                    "Stop_Price":     round(pos["stop_price"], 2),
                    "Shares":         pos["shares"],
                    "PnL_Rs":         round(pnl_rs, 2),
                    "PnL_Pct":        round(pnl_pct, 2),
                    "Hold_Days":      pos["bars_held"],
                    "Exit_Reason":    exit_reason,
                    "Entry_Month":    pos["entry_date"].strftime("%Y-%m"),
                    "Allocation":     round(pos["allocation"], 2),
                })
                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del open_positions[sym]

        # === STEP 2: SCAN FOR NEW ENTRIES (if slots available) ===
        available_slots = MAX_POSITIONS - len(open_positions)
        if available_slots > 0:
            candidates = []
            for symbol, date_rows in symbol_date_rows.items():
                if symbol in open_positions:
                    continue  # already holding this stock
                row = date_rows.get(day)
                if row is None:
                    continue
                if not row["Signal"]:
                    continue

                rh, rh_type = get_ref_high(row)
                if pd.isna(rh) or rh <= 0:
                    continue

                close_price = float(row["close"])
                score = rank_score(row, close_price, rh, nifty_close, pd.Timestamp(day))
                candidates.append({
                    "symbol": symbol,
                    "close": close_price,
                    "ref_high": rh,
                    "ref_high_type": rh_type,
                    "rank_score": score,
                    "row": row,
                })

            # Sort by ranking method (highest score first)
            candidates.sort(key=lambda x: x["rank_score"], reverse=True)

            # Take up to available_slots entries
            for cand in candidates[:available_slots]:
                ep = cand["close"]
                rh = cand["ref_high"]
                rh_type = cand["ref_high_type"]
                sp = round(ep * (1 - STOP_LOSS_PCT), 2)
                tp = round(rh * (1 + TARGET_ABOVE_HIGH_PCT), 2)

                allocation = PER_POSITION_CAPITAL
                if cash < allocation:
                    allocation = cash  # use whatever is left
                if allocation < ep:
                    continue  # not enough capital for even 1 share

                shares = int(allocation / ep)
                if shares <= 0:
                    continue

                cost = ep * shares
                cash -= cost

                open_positions[cand["symbol"]] = {
                    "entry_price": ep,
                    "stop_price": sp,
                    "target_price": tp,
                    "ref_high": rh,
                    "ref_high_type": rh_type,
                    "entry_date": pd.Timestamp(day),
                    "shares": shares,
                    "bars_held": 0,
                    "allocation": allocation,
                }

        # === STEP 3: TRACK EQUITY ===
        max_concurrent = max(max_concurrent, len(open_positions))

        # Mark-to-market: value open positions at today's close
        invested_value = 0.0
        for symbol, pos in open_positions.items():
            row = symbol_date_rows.get(symbol, {}).get(day)
            if row is not None:
                invested_value += float(row["close"]) * pos["shares"]
            else:
                # No data for this stock today, use entry price as fallback
                invested_value += pos["entry_price"] * pos["shares"]

        total_equity = cash + invested_value
        equity_history.append({
            "date": day,
            "equity": round(total_equity, 2),
            "cash": round(cash, 2),
            "invested": round(invested_value, 2),
            "open_positions": len(open_positions),
        })

    # -------------------------------------------------------------------------
    # Close any remaining open positions at end of data
    # -------------------------------------------------------------------------
    if open_positions:
        last_day = trading_dates[-1]
        for symbol, pos in open_positions.items():
            row = symbol_date_rows.get(symbol, {}).get(last_day)
            if row is not None:
                exit_price = float(row["close"])
            else:
                exit_price = pos["entry_price"]  # fallback

            pnl_rs = (exit_price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

            completed_trades.append({
                "Symbol":         symbol,
                "Entry_Date":     pos["entry_date"],
                "Exit_Date":      pd.Timestamp(last_day),
                "Signal_Type":    signal_type_label(pos["ref_high_type"]),
                "Entry_Price":    round(pos["entry_price"], 2),
                "Exit_Price":     round(exit_price, 2),
                "Ref_High":       round(pos["ref_high"], 2),
                "Ref_High_Type":  pos["ref_high_type"],
                "Target_Price":   round(pos["target_price"], 2),
                "Stop_Price":     round(pos["stop_price"], 2),
                "Shares":         pos["shares"],
                "PnL_Rs":         round(pnl_rs, 2),
                "PnL_Pct":        round(pnl_pct, 2),
                "Hold_Days":      pos["bars_held"],
                "Exit_Reason":    "End_Of_Data",
                "Entry_Month":    pos["entry_date"].strftime("%Y-%m"),
                "Allocation":     round(pos["allocation"], 2),
            })

    return completed_trades, equity_history

# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def performance_report(
    trades_df: pd.DataFrame,
    equity_history: list[dict],
    label: str = "NEAR HIGH PORTFOLIO BACKTEST",
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

    gross_profit = wins["PnL_Rs"].sum()
    gross_loss   = losses["PnL_Rs"].sum()
    net_pnl      = trades_df["PnL_Rs"].sum()
    total_return = net_pnl / INITIAL_CAPITAL * 100
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = wins["PnL_Pct"].mean()   if not wins.empty   else 0
    avg_loss = losses["PnL_Pct"].mean() if not losses.empty else 0
    best  = trades_df.loc[trades_df["PnL_Rs"].idxmax()]  if total > 0 else None
    worst = trades_df.loc[trades_df["PnL_Rs"].idxmin()]  if total > 0 else None

    # Equity curve drawdown (from day-by-day equity history)
    if equity_history:
        eq_df = pd.DataFrame(equity_history)
        eq_series = eq_df["equity"]
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak * 100
        max_dd = dd.min()
        final_equity = eq_series.iloc[-1]
    else:
        max_dd = 0
        final_equity = INITIAL_CAPITAL

    avg_hold = trades_df["Hold_Days"].mean()

    # Portfolio-specific stats
    if equity_history:
        eq_df = pd.DataFrame(equity_history)
        max_concurrent = eq_df["open_positions"].max()
        avg_positions = eq_df["open_positions"].mean()
        avg_invested = eq_df["invested"].mean()
        avg_utilization = avg_invested / INITIAL_CAPITAL * 100
        days_fully_invested = (eq_df["open_positions"] == MAX_POSITIONS).sum()
        days_no_positions = (eq_df["open_positions"] == 0).sum()
        total_days = len(eq_df)
    else:
        max_concurrent = 0
        avg_positions = 0
        avg_utilization = 0
        days_fully_invested = 0
        days_no_positions = 0
        total_days = 0

    print(f"  Initial Capital : Rs {INITIAL_CAPITAL:>14,.0f}")
    print(f"  Final Equity    : Rs {final_equity:>14,.0f}")
    print(f"  Max Positions   : {MAX_POSITIONS:>12d}")
    print(f"  Per-Position Cap: Rs {PER_POSITION_CAPITAL:>14,.0f}")
    print(f"  Total Trades    : {total:>12d}")
    print(f"  Win Rate        : {win_rate:>11.2f}%")
    print(f"  Net P&L         : Rs {net_pnl:>14,.0f}")
    print(f"  Total Return    : {total_return:>11.2f}%")
    print(f"  Max Drawdown    : {max_dd:>11.2f}%")
    print(f"  Profit Factor   : {profit_factor:>12.2f}")
    print(f"  Avg Win         : {avg_win:>11.2f}%")
    print(f"  Avg Loss        : {avg_loss:>11.2f}%")
    print(f"  Avg Hold Days   : {avg_hold:>12.1f}")
    if best is not None:
        print(f"  Best Trade      : {best['Symbol']}  Rs {best['PnL_Rs']:,.0f}  ({best['PnL_Pct']:.1f}%)")
    if worst is not None:
        print(f"  Worst Trade     : {worst['Symbol']}  Rs {worst['PnL_Rs']:,.0f}  ({worst['PnL_Pct']:.1f}%)")

    print(f"\n  PORTFOLIO STATS:")
    print(f"    Max Concurrent Positions : {max_concurrent}")
    print(f"    Avg Positions Open       : {avg_positions:.2f}")
    print(f"    Capital Utilization (avg): {avg_utilization:.1f}%")
    print(f"    Days Fully Invested      : {days_fully_invested} / {total_days}")
    print(f"    Days with No Positions   : {days_no_positions} / {total_days}")

    print(f"\n  EXIT REASON BREAKDOWN:")
    for reason, cnt in trades_df["Exit_Reason"].value_counts().items():
        avg_p = trades_df[trades_df["Exit_Reason"] == reason]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Exit_Reason"] == reason) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {reason:<20}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    print(f"\n  NEAR-HIGH TYPE BREAKDOWN:")
    for ht, cnt in trades_df["Ref_High_Type"].value_counts().items():
        avg_p = trades_df[trades_df["Ref_High_Type"] == ht]["PnL_Pct"].mean()
        wr    = (trades_df[(trades_df["Ref_High_Type"] == ht) & (trades_df["PnL_Rs"] > 0)].shape[0]
                 / cnt * 100)
        print(f"    {ht:<6}: {cnt:4d} trades  avg {avg_p:>+6.1f}%  WR {wr:5.1f}%")

    # ---- CAGR ----
    if equity_history and len(equity_history) > 1:
        eq_df_cagr = pd.DataFrame(equity_history)
        first_date = pd.Timestamp(eq_df_cagr["date"].iloc[0])
        last_date  = pd.Timestamp(eq_df_cagr["date"].iloc[-1])
        years = (last_date - first_date).days / 365.25
        if years > 0 and final_equity > 0:
            cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1
            print(f"\n  CAGR               : {cagr * 100:>11.2f}%  (over {years:.1f} years)")

    # ---- CALENDAR YEAR RETURNS ----
    if equity_history and len(equity_history) > 1:
        eq_yr = pd.DataFrame(equity_history)
        eq_yr["date"] = pd.to_datetime(eq_yr["date"])
        eq_yr["Year"] = eq_yr["date"].dt.year
        years_list = sorted(eq_yr["Year"].unique())
        print(f"\n  CALENDAR YEAR RETURNS:")
        print(f"    {'Year':<6} {'Start Equity':>14} {'End Equity':>14} {'Return':>10} {'Max DD':>10}")
        print(f"    {'─' * 58}")
        for yr in years_list:
            yr_data = eq_yr[eq_yr["Year"] == yr]
            yr_start = yr_data["equity"].iloc[0]
            yr_end   = yr_data["equity"].iloc[-1]
            yr_ret   = (yr_end - yr_start) / yr_start * 100
            yr_peak  = yr_data["equity"].cummax()
            yr_dd    = ((yr_data["equity"].values - yr_peak.values) / yr_peak.values * 100).min()
            bar = "+" * min(20, int(abs(yr_ret) / 5)) if yr_ret >= 0 else "-" * min(20, int(abs(yr_ret) / 5))
            print(f"    {yr:<6} Rs {yr_start:>12,.0f} Rs {yr_end:>12,.0f} {yr_ret:>+9.2f}% {yr_dd:>+9.2f}%  {bar}")

    print(f"\n  MONTHLY P&L BREAKDOWN:")
    monthly = trades_df.groupby("Entry_Month")["PnL_Rs"].sum()
    for m, pnl in monthly.items():
        bar = ("+" if pnl >= 0 else "-") * min(40, int(abs(pnl) / max(abs(net_pnl), 1) * 200))
        print(f"    {m}  Rs {pnl:>12,.0f}  {bar}")

    print(f"\n  TOP 10 WINNING TRADES:")
    cols = ["Symbol", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price",
            "PnL_Rs", "PnL_Pct", "Hold_Days", "Exit_Reason"]
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
    print("  NEAR HIGH BACKTEST - PORTFOLIO-BASED")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}  |  Max Positions: {MAX_POSITIONS}  |  Per Slot: Rs {PER_POSITION_CAPITAL:,.0f}")
    print(f"  EMA Alignment + Close within {int((1-NEAR_HIGH_UPPER)*100)}-{int((1-NEAR_HIGH_LOWER)*100)}% of Key High")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ref high")
    print(f"  Lookback: {LOOKBACK_DAYS} days  |  Fetch: {FETCH_YEARS} years")
    print(f"  Rank Method: {RANK_METHOD.upper()}  |  ROC Period: {ROC_PERIOD} days")
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

    # =========================================================================
    # FIRST PASS: Fetch data + compute indicators for ALL stocks
    # =========================================================================
    print("=" * 50)
    print("  PASS 1: Fetching data & computing indicators")
    print("=" * 50)

    stock_data = {}   # symbol -> DataFrame with indicators
    processed  = 0
    skipped    = 0

    t0 = time.time()
    for i, (symbol, token) in enumerate(universe, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            rate    = i / max(elapsed, 1)
            eta_m   = (total - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:4d}/{total}] {symbol:<18}  stocks_ready:{len(stock_data):4d}  "
                  f"ETA:{eta_m:.1f}min", flush=True)

        try:
            df_raw = fetch_with_cache(obj, token, symbol, fetch_start, fetch_end)
            if df_raw.empty or len(df_raw) < MIN_DATA_DAYS:
                skipped += 1
                continue

            df_ind = compute_indicators(df_raw)
            if df_ind is None:
                skipped += 1
                continue

            stock_data[symbol] = df_ind
            processed += 1

        except Exception as e:
            print(f"\n  ERROR on {symbol}: {e}")
            skipped += 1

    elapsed_m = (time.time() - t0) / 60
    print(f"\n  Pass 1 done. {processed:,} stocks ready, {skipped:,} skipped in {elapsed_m:.1f} min")

    if not stock_data:
        print("  No stock data available. Exiting.")
        return pd.DataFrame()

    # Fetch Nifty data for RS ranking (if needed)
    nifty_close = None
    if RANK_METHOD == "rs_nifty":
        print("\n  Fetching Nifty 50 data for RS ranking...")
        nifty_close = load_nifty_for_ranking(obj, fetch_start, fetch_end)
        if nifty_close.empty:
            print("  WARNING: Nifty data unavailable, falling back to momentum ranking")
            nifty_close = None
        else:
            print(f"  Nifty data: {len(nifty_close):,} bars")

    # =========================================================================
    # SECOND PASS: Day-by-day portfolio simulation
    # =========================================================================
    print("\n" + "=" * 50)
    print(f"  PASS 2: Portfolio simulation | Rank: {RANK_METHOD.upper()}")
    print("=" * 50)

    t1 = time.time()
    all_trades, equity_history = portfolio_backtest(stock_data, nifty_close)
    sim_elapsed = time.time() - t1
    print(f"  Simulation done in {sim_elapsed:.1f}s")
    print(f"  Total trades: {len(all_trades):,}\n")

    if not all_trades:
        print("  No trades found. Exiting.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    trades_df["Entry_Date"] = pd.to_datetime(trades_df["Entry_Date"])
    trades_df["Exit_Date"]  = pd.to_datetime(trades_df["Exit_Date"])
    trades_df.sort_values("Entry_Date", inplace=True)

    performance_report(trades_df, equity_history, label="NEAR HIGH PORTFOLIO BACKTEST")

    # Save results
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"near_high_portfolio_backtest_{ts}.csv"
    trades_df.to_csv(out, index=False)
    print(f"  Results saved to: {out}")

    # Save equity curve
    if equity_history:
        eq_out = f"near_high_equity_curve_{ts}.csv"
        pd.DataFrame(equity_history).to_csv(eq_out, index=False)
        print(f"  Equity curve saved to: {eq_out}")

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
