"""
=============================================================================
NSE MOMENTUM SCANNER - LIVE TRADING SIGNALS
=============================================================================
Scans entire NSE EQ-series universe (no ETFs, no BE, no surveillance stocks)

FILTERS:
  1. Price > SMA10 > SMA20 > SMA50 > SMA150 > SMA200  (perfect alignment)
  2. Price within 20% of ATH / 1Y High / 5Y High
  3. Shows WHICH high the stock is near (ATH/1Y/5Y)

MOMENTUM RANKING:
  - 20-day ROC (Rate of Change)
  - 60-day ROC
  - 90-day ROC
  - Composite momentum score (weighted average)
  - Relative strength rank across universe

OUTPUT:
  - Console table with top momentum stocks
  - CSV export with full details
  - Ready for live trading signals

=============================================================================
"""

import os
import pickle
import time
import datetime
import io
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from SmartApi import SmartConnect
import pyotp

# =============================================================================
# CREDENTIALS - Move to environment variables for production
# =============================================================================
API_KEY      = os.environ.get("ANGEL_API_KEY", "Tp7PJMIc")
CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "S63068620")
PASSWORD     = os.environ.get("ANGEL_PASSWORD", "4098")
TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "KVTELUFGR33YCPQUKO3ZSL4NMA")
# =============================================================================

# =============================================================================
# SCANNER CONFIG
# =============================================================================
CACHE_DIR         = "cache_scanner"
MIN_PRICE         = 20           # Minimum avg price (penny stock filter)
MIN_AVG_TURNOVER  = 10_00_000    # Min daily turnover Rs 10 lakh
FILTER_LOOKBACK   = 60           # Days for price/volume filter
NEAR_HIGH_PCT     = 0.20         # Within 20% of high
MIN_DATA_DAYS     = 252          # Need 1 year of data minimum

# Momentum periods
ROC_PERIODS = [20, 60, 90]

# Momentum weights for composite score
MOMENTUM_WEIGHTS = {20: 0.2, 60: 0.3, 90: 0.5}  # Favor longer-term momentum

# How many days of historical data to fetch
FETCH_YEARS = 6  # Need 5 years for 5Y high calculation
# =============================================================================

os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# COMPREHENSIVE FALLBACK UNIVERSE (EQ series only)
# =============================================================================
UNIVERSE_FALLBACK = [
    # Nifty 50
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL","BPCL",
    "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
    "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK",
    "INFY","ITC","JSWSTEEL","KOTAKBANK","LT","M&M","MARUTI",
    "NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE",
    "SBIN","SHREECEM","SUNPHARMA","TATAMOTORS","TATASTEEL",
    "TATACONSUM","TCS","TECHM","TITAN","ULTRACEMCO","WIPRO","ADANIGREEN",
    # Nifty Next 50
    "AMBUJACEM","BAJAJHLDNG","BANKBARODA","CANBK","DABUR","DLF",
    "GAIL","GODREJCP","HAVELLS","HINDPETRO","ICICIPRULI","INDHOTEL",
    "IOC","LUPIN","HDFCAMC","NAUKRI","OFSS","PAGEIND","PFC",
    "PIDILITIND","PNB","RECLTD","SAIL","SHRIRAMFIN","SIEMENS","SRF",
    "TATACOMM","TATACHEM","TORNTPHARM","TVSMOTOR","UBL","UNIONBANK",
    "UPL","VEDL","VBL","VOLTAS","ZEEL","ZOMATO","MUTHOOTFIN",
    "CHOLAFIN","INDUSTOWER","DMART","ADANITRANS","ADANIPOWER",
    "NYKAA","PAYTM","POLICYBZR","IRCTC","TRENT","RVNL","LICI",
    # Nifty Midcap 150
    "ABCAPITAL","ABFRL","AIAENG","ALKEM","APOLLOTYRE","ASHOKLEY",
    "ASTRAL","ATUL","AUBANK","AUROPHARMA","BALKRISHNA","BANDHANBNK",
    "BATAINDIA","BERGEPAINT","BHARATFORG","BHEL","BIOCON","CAMS",
    "CANFINHOME","CEATLTD","CHOLAHLDNG","COFORGE","COLPAL","CONCOR",
    "COROMANDEL","CROMPTON","CUMMINSIND","CYIENT","DEEPAKNTR",
    "DELTACORP","EMAMILTD","ESCORTS","EXIDEIND","FINEORG","FLUOROCHEM",
    "FORTIS","GLENMARK","GODREJPROP","GRANULES","GUJGASLTD",
    "IDFCFIRSTB","IPCALAB","IRFC","IGL","JINDALSTEL","JUBLFOOD",
    "KPITTECH","LALPATHLAB","LAURUSLABS","LICHSGFIN","LTIM","LTTS",
    "M&MFIN","MANAPPURAM","MARICO","METROPOLIS","MFSL","MPHASIS",
    "MRF","NATIONALUM","NCC","NMDC","NHPC","OBEROIRLTY","PERSISTENT",
    "PETRONET","PHOENIXLTD","POLYCAB","PVRINOX","RAMCOCEMENT","RBLBANK",
    "SCHAEFFLER","SKFINDIA","SOBHA","STARHEALTH","SUNTV","SUPREMEIND",
    "SYNGENE","TATAELXSI","TATAINVEST","THERMAX","TIMKEN","TORNTPOWER",
    "TRIDENT","VARUNBEV","VGUARD","WELCORP","WHIRLPOOL","WOCKPHARMA",
    "ZYDUSLIFE","HAPPSTMNDS","INOXWIND","SUZLON","IREDA","NBCC",
    "KEI","TITAGARH","CDSL","CGPOWER","DIXON","AMBER","MANYAVAR",
    "MOTILALOFS","MSTC","GRINDWELL","CARBORUNDUM","BOSCHLTD",
    # Nifty Smallcap 250
    "AARTIIND","ACCELYA","AFFLE","AJANTPHARM","ALKYLAMINE","AMARAJABAT",
    "ANGELONE","APARINDS","APTUS","ASAHIINDIA","ASTERDM","ATGL",
    "AVANTIFEED","BAJAJCON","BALRAMCHIN","BASF","BAYERCROP","BLUESTARCO",
    "BRIGADE","CAPLIPOINT","CASTROLIND","CENTURYPLY","CESC","CHAMBLFERT",
    "CLEAN","CRISIL","CUB","DHANUKA","EIL","EIDPARRY","ENDURANCE",
    "ENGINERSIN","EQUITAS","ESABINDIA","GALAXYSURF","GATEWAY","GHCL",
    "GICHSGFIN","GLAND","GNFC","GODAWARI","GODFRYPHLP","GPPL",
    "GREAVESCOT","GREENPANEL","GSFC","HBLPOWER","HERANBA","HINDCOPPER",
    "HOMEFIRST","HONAUT","IFCI","IIFL","INDIGOPNTS","IRCON","J&KBANK",
    "JKCEMENT","JKPAPER","JKLAKSHMI","JMFINANCIL","JUBLINGREA",
    "JUSTDIAL","KAJARIA","KANSAINER","KARURVYSYA","KFINTECH","KIMS",
    "KNRCON","KRBL","KSB","LATENTVIEW","LXCHEM","MANINFRA","MAXHEALTH",
    "MIDHANI","MOIL","NATCOPHARM","NAVINFLUOR","NILKAMAL","NUVAMA",
    "NUVOCO","OLECTRA","ORIENTELEC","PANAMAPET","PATELENG","PFIZER",
    "PIRAMALENT","PNBHOUSING","POLYPLEX","QUICKHEAL","RADICO","RAILTEL",
    "RAIN","REDINGTON","RELAXO","RITES","ROUTE","SAFARI","SANOFI",
    "SBICARD","SHANKARA","SHAREINDIA","SHILPAMED","SHOPERSTOP","SPANDANA",
    "STCINDIA","SUDARSCHEM","SUMICHEM","SUNFLAG","SUNTECK","SURYODAY",
    "SUVEN","TEJASNET","THYROCARE","TIMETECHNO","TVSHLTD","UCO",
    "UJJIVAN","UJJIVANSFB","UTIAMC","VAIBHAVGBL","VARROC","VINATIORGA",
    "VOLTAMP","VSTIND","WELSPUNIND","WESTLIFE","WONDERLA","YESBANK",
    "ZENSARTECH","DELHIVERY","INDIAMART","APLAPOLLO","BSOFT","JYOTHYLAB",
    "KOLTEPATIL","MATRIMONY","SAREGAMA","TANLA","TEAMLEASE","TIINDIA",
    "DATAPATTNS","NETWORK18","MINDACORP","TARSONS","THOUGHTWKS","SJVN",
    "PNCINFRA","JSWENERGY","RAJRATAN","SANSERA","SAPPHIRE","SHAKTIPUMP",
    "SNOWMAN","STLTECH","SWSOLAR","WABCOINDIA","WEBELSOLAR","JSWINFRA",
    "MAHINDCIE","SATIN","SEQUENT","VINDHYATEL","ANANTRAJ","PRESTIGE",
    "LODHA","ARVIND","RAYMOND","BIRLACORPN","DALBHARAT","HEIDELBERG",
    "DEVYANI","GOCOLORS","TCNSBRANDS","EDELWEISS","EMKAY","CENTRALBK",
    "INDIANB","BANKINDIA","MAHABANK","POWERINDIA","PRINCEPIPES","NSLNISP",
    "DCBBANK","EQUITASBNK","KTKBANK","CSBBANK","FINOLEXIND","APCOTEXIND",
    "NOCIL","ELGIEQUIP","EMCURE","EPIGRAL","ROLEXRINGS","RPGLIFE",
    "SKIPPER","ABBOTINDIA","GLAXO","PIIND","LUXIND","NIITLTD",
    "ANANDRATHI","BSE","ZENTEC","FORCEMOT","PCBL","HUDCO","GABRIEL",
    "PARADEEP","MAZDOCK","PGEL","COCHINSHIP","TECHNOE","JWL",
    "BIKAJI","CAMPUS","MEDANTA","NETWEB","SUVENPHAR","TCIEXP",
    "UCOBANK","VMART","DCMSHRIRAM","JINDALSAW","KPIL","KRSNAA",
    "SHYAMMETL","SONATSOFTW","OPTIEMUS","ORCHPHARMA","POWERMECH",
    "PRICOLLTD","WINDLAS","WPIL","HAL","BEL","MAZAGON","GRSE",
]

# Deduplicate
_s = set()
UNIVERSE_FALLBACK = [x for x in UNIVERSE_FALLBACK if not (x in _s or _s.add(x))]

# Symbol corrections
SYMBOL_MAP = {
    "BERGERPAINTS": "BERGEPAINT",
    "AUROBINDO": "AUROPHARMA",
    "JSPL": "JINDALSTEL",
    "REC": "RECLTD",
    "TVSMOTORS": "TVSMOTOR",
    "CEAT": "CEATLTD",
    "MCDOWELL-N": "UNITDSPR",
    "MOTHERSON": "MOTHERSUMI",
    "INFOEDGE": "NAUKRI",
}


# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================
def fix_totp(raw):
    s = raw.strip().upper().replace(" ", "").replace("-", "")
    return s + "=" * ((8 - len(s) % 8) % 8)


def validate_credentials():
    placeholders = {"your_api_key", "your_password", "your_totp_secret", ""}
    errors = []
    if API_KEY.strip() in placeholders:
        errors.append("API_KEY not set")
    if PASSWORD.strip() in placeholders:
        errors.append("PASSWORD not set")
    if TOTP_SECRET.strip() in placeholders:
        errors.append("TOTP_SECRET not set")
    if errors:
        print("\n" + "=" * 60)
        print("  CREDENTIALS NOT CONFIGURED")
        print("=" * 60)
        for e in errors:
            print(f"  x  {e}")
        raise SystemExit(1)
    import base64
    fixed = fix_totp(TOTP_SECRET)
    try:
        base64.b32decode(fixed, casefold=True)
    except Exception as e:
        print(f"  TOTP_SECRET invalid: {e}")
        raise SystemExit(1)
    return fixed


def login(totp_fixed):
    obj = SmartConnect(api_key=API_KEY)
    otp = pyotp.TOTP(totp_fixed).now()
    secs = 30 - datetime.datetime.now().second % 30
    print(f"  OTP: {otp}  (valid ~{secs}s)")
    data = obj.generateSession(CLIENT_ID, PASSWORD, otp)
    if not isinstance(data, dict):
        raise Exception(f"Unexpected response: {data}")
    status = data.get("status")
    if status is True or str(status).lower() == "true":
        print(f"  JWT: {str(data.get('data', {}).get('jwtToken', ''))[:30]}...")
        return obj
    raise Exception(f"Login failed: {data.get('errorcode', '?')} | {data.get('message', str(data))}")


# =============================================================================
# UNIVERSE DOWNLOAD
# =============================================================================
def get_universe():
    """Download NSE EQ-series stocks (excludes ETFs, BE, surveillance)"""
    cache_path = os.path.join(CACHE_DIR, "universe_list.pkl")

    # Check cache (refresh daily)
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 24:
            with open(cache_path, "rb") as f:
                syms = pickle.load(f)
            print(f"  Universe: {len(syms)} symbols (cache {age_hours:.1f}h old)")
            return syms
        print("  Universe cache >24h — refreshing...")

    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    equity_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    syms = []

    try:
        print("  Downloading NSE equity list...")
        resp = requests.get(equity_url, headers=headers, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 10000:
            df = pd.read_csv(io.StringIO(resp.text))

            # Filter ONLY EQ series (excludes BE, BZ, SM, etc.)
            if "SERIES" in df.columns:
                eq_mask = df["SERIES"] == "EQ"

                # Find symbol column
                sym_col = " SYMBOL" if " SYMBOL" in df.columns else "SYMBOL"
                syms = df[eq_mask][sym_col].dropna().str.strip().tolist()

            print(f"  NSE equity list: {len(syms)} EQ-series stocks (no ETF/BE/surveillance)")
    except Exception as e:
        print(f"  NSE download failed: {e}")

    if len(syms) < 100:
        print("  Using fallback universe list")
        syms = list(UNIVERSE_FALLBACK)

    # Apply symbol corrections
    syms = [SYMBOL_MAP.get(s, s) for s in syms]

    # Deduplicate
    seen = set()
    syms = [s for s in syms if not (s in seen or seen.add(s))]

    with open(cache_path, "wb") as f:
        pickle.dump(syms, f)
    print(f"  Universe saved: {len(syms)} symbols")
    return syms


# =============================================================================
# TOKEN MAP
# =============================================================================
def get_token_map():
    """Get Angel One symbol -> token mapping"""
    cache_path = os.path.join(CACHE_DIR, "scrip_master.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if len(cached) > 100 and age_h < 24:
            print(f"  Token map: {len(cached):,} tokens (cache {age_h:.1f}h old)")
            return cached
        os.remove(cache_path)

    import requests
    print("  Downloading scrip master...")
    resp = requests.get(
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
        timeout=30
    )
    df = pd.DataFrame(resp.json())
    nse = df[df["exch_seg"] == "NSE"]
    eq = nse[nse["symbol"].str.endswith("-EQ", na=False)].copy()
    eq["sym"] = eq["symbol"].str.replace("-EQ", "", regex=False).str.strip()
    token_map = eq.set_index("sym")["token"].to_dict()

    print(f"  {len(nse):,} NSE -> {len(eq):,} -EQ -> {len(token_map):,} tokens")

    with open(cache_path, "wb") as f:
        pickle.dump(token_map, f)
    return token_map


def find_token(token_map, symbol):
    """Find token for a symbol with fallback variations"""
    if symbol in token_map:
        return symbol, token_map[symbol]

    corrected = SYMBOL_MAP.get(symbol, symbol)
    if corrected in token_map:
        return corrected, token_map[corrected]

    # Try common variations
    variations = [
        symbol.replace("PAINTS", "PAINT"),
        symbol.replace("MOTORS", "MOTOR"),
        "AUROPHARMA" if symbol == "AUROBINDO" else None,
        "CEATLTD" if symbol == "CEAT" else None,
        "JINDALSTEL" if symbol == "JSPL" else None,
    ]
    for var in variations:
        if var and var in token_map:
            return var, token_map[var]

    return None, None


# =============================================================================
# OHLC DATA FETCH
# =============================================================================
def fetch_ohlc(obj, token, symbol, from_date, to_date):
    """Fetch OHLC data with caching"""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_1D.pkl")
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    # Check cache - refresh if stale (older than today)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        if not df.empty:
            last_date = str(df.index[-1].date())
            cache_age_days = (time.time() - os.path.getmtime(cache_path)) / 86400

            # Use cache if it has recent data (within 2 days) or was updated today
            if cache_age_days < 1 or last_date >= (datetime.date.today() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"):
                return df

    rows = []
    cur = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)

    while cur <= end:
        nxt = min(cur + pd.DateOffset(days=400), end)
        try:
            r = obj.getCandleData({
                "exchange": "NSE",
                "symboltoken": token,
                "interval": "ONE_DAY",
                "fromdate": cur.strftime("%Y-%m-%d %H:%M"),
                "todate": nxt.strftime("%Y-%m-%d %H:%M"),
            })
            if r.get("status") and r.get("data"):
                rows.extend(r["data"])
        except Exception:
            pass
        cur = nxt + pd.DateOffset(days=1)
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

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    return df


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
def compute_indicators(df):
    """Compute all SMAs, highs, and momentum indicators"""
    if len(df) < MIN_DATA_DAYS:
        return None

    # Quality filter
    recent = df.tail(FILTER_LOOKBACK)
    if recent["close"].mean() < MIN_PRICE:
        return None
    if "volume" in recent.columns:
        avg_turnover = (recent["close"] * recent["volume"]).mean()
        if avg_turnover < MIN_AVG_TURNOVER:
            return None

    df = df.copy()

    # SMAs
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma150"] = df["close"].rolling(150).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    # Highs
    df["ath"] = df["close"].expanding().max()  # All-time high
    df["high_1y"] = df["close"].rolling(252, min_periods=200).max()  # 1-year high
    df["high_5y"] = df["close"].rolling(252 * 5, min_periods=252).max()  # 5-year high

    # ROC (Rate of Change) for multiple periods
    for period in ROC_PERIODS:
        df[f"roc_{period}"] = df["close"].pct_change(period) * 100

    # Drop rows with NaN in critical columns
    df.dropna(subset=["sma200", "roc_90"], inplace=True)

    return df if len(df) > 0 else None


# =============================================================================
# STOCK ANALYSIS
# =============================================================================
def analyze_stock(symbol, df):
    """
    Analyze a single stock and return its metrics if it passes filters.
    Returns None if stock doesn't qualify.
    """
    if df is None or df.empty:
        return None

    row = df.iloc[-1]  # Latest data
    price = row["close"]

    # =========================================================================
    # FILTER 1: SMA Alignment (Price > SMA10 > SMA20 > SMA50 > SMA150 > SMA200)
    # =========================================================================
    sma10 = row["sma10"]
    sma20 = row["sma20"]
    sma50 = row["sma50"]
    sma150 = row["sma150"]
    sma200 = row["sma200"]

    if not (price > sma10 > sma20 > sma50 > sma150 > sma200):
        return None

    # =========================================================================
    # FILTER 2: Near High (within 20% of ATH / 1Y / 5Y high)
    # =========================================================================
    ath = row["ath"]
    high_1y = row["high_1y"]
    high_5y = row["high_5y"]

    # Calculate distance from each high
    dist_ath = (ath - price) / ath * 100 if ath > 0 else 999
    dist_1y = (high_1y - price) / high_1y * 100 if high_1y > 0 else 999
    dist_5y = (high_5y - price) / high_5y * 100 if high_5y > 0 else 999

    # Determine which high the stock is near
    near_high_type = None
    near_high_dist = None

    if dist_ath <= NEAR_HIGH_PCT * 100:
        near_high_type = "ATH"
        near_high_dist = dist_ath
    elif dist_1y <= NEAR_HIGH_PCT * 100:
        near_high_type = "1Y"
        near_high_dist = dist_1y
    elif dist_5y <= NEAR_HIGH_PCT * 100:
        near_high_type = "5Y"
        near_high_dist = dist_5y
    else:
        return None  # Not near any high

    # =========================================================================
    # MOMENTUM CALCULATIONS
    # =========================================================================
    roc_20 = row["roc_20"]
    roc_60 = row["roc_60"]
    roc_90 = row["roc_90"]

    # Composite momentum score (weighted average)
    composite_momentum = (
        MOMENTUM_WEIGHTS[20] * roc_20 +
        MOMENTUM_WEIGHTS[60] * roc_60 +
        MOMENTUM_WEIGHTS[90] * roc_90
    )

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "sma10": round(sma10, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma150": round(sma150, 2),
        "sma200": round(sma200, 2),
        "ath": round(ath, 2),
        "high_1y": round(high_1y, 2),
        "high_5y": round(high_5y, 2),
        "near_high_type": near_high_type,
        "dist_from_high": round(near_high_dist, 2),
        "roc_20": round(roc_20, 2),
        "roc_60": round(roc_60, 2),
        "roc_90": round(roc_90, 2),
        "composite_momentum": round(composite_momentum, 2),
        "last_date": str(df.index[-1].date()),
    }


# =============================================================================
# RELATIVE MOMENTUM RANKING
# =============================================================================
def calculate_relative_momentum(results_df):
    """
    Calculate relative momentum rank across universe.
    Rank 1 = highest momentum.
    """
    if results_df.empty:
        return results_df

    df = results_df.copy()

    # Rank by each ROC period (lower rank = higher momentum)
    df["rank_roc_20"] = df["roc_20"].rank(ascending=False, method="min").astype(int)
    df["rank_roc_60"] = df["roc_60"].rank(ascending=False, method="min").astype(int)
    df["rank_roc_90"] = df["roc_90"].rank(ascending=False, method="min").astype(int)
    df["rank_composite"] = df["composite_momentum"].rank(ascending=False, method="min").astype(int)

    # Final rank based on composite
    df = df.sort_values("rank_composite")
    df["final_rank"] = range(1, len(df) + 1)

    return df


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================
def print_results(df, top_n=30):
    """Print formatted results table"""
    if df.empty:
        print("\n  No stocks matched the criteria.")
        return

    sep = "=" * 140
    print(f"\n{sep}")
    print("  NSE MOMENTUM SCANNER - LIVE RESULTS")
    print(f"  Scan Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stocks Passing Filters: {len(df)}")
    print(sep)

    # Header
    print(f"\n  {'Rank':<5} {'Symbol':<15} {'Price':>10} {'High':>6} {'Dist%':>7} "
          f"{'ROC20':>8} {'ROC60':>8} {'ROC90':>8} {'Composite':>10} "
          f"{'SMA10':>9} {'SMA20':>9} {'SMA50':>9}")
    print("  " + "-" * 130)

    for idx, row in df.head(top_n).iterrows():
        print(f"  {row['final_rank']:<5} {row['symbol']:<15} "
              f"{row['price']:>10.2f} {row['near_high_type']:>6} {row['dist_from_high']:>6.1f}% "
              f"{row['roc_20']:>+7.1f}% {row['roc_60']:>+7.1f}% {row['roc_90']:>+7.1f}% "
              f"{row['composite_momentum']:>+9.1f}% "
              f"{row['sma10']:>9.2f} {row['sma20']:>9.2f} {row['sma50']:>9.2f}")

    if len(df) > top_n:
        print(f"\n  ... and {len(df) - top_n} more stocks (see CSV for full list)")

    print(sep)

    # Summary stats
    print("\n  MOMENTUM DISTRIBUTION:")
    print(f"    Avg ROC 20-day:  {df['roc_20'].mean():>+6.1f}%  (range: {df['roc_20'].min():>+.1f}% to {df['roc_20'].max():>+.1f}%)")
    print(f"    Avg ROC 60-day:  {df['roc_60'].mean():>+6.1f}%  (range: {df['roc_60'].min():>+.1f}% to {df['roc_60'].max():>+.1f}%)")
    print(f"    Avg ROC 90-day:  {df['roc_90'].mean():>+6.1f}%  (range: {df['roc_90'].min():>+.1f}% to {df['roc_90'].max():>+.1f}%)")

    print("\n  HIGH TYPE BREAKDOWN:")
    for ht, cnt in df["near_high_type"].value_counts().items():
        avg_mom = df[df["near_high_type"] == ht]["composite_momentum"].mean()
        print(f"    {ht:>5}: {cnt:>4} stocks  (avg composite momentum: {avg_mom:>+.1f}%)")


def export_results(df, filename="momentum_scan_results.csv"):
    """Export results to CSV"""
    if df.empty:
        return

    # Reorder columns for export
    cols = [
        "final_rank", "symbol", "price", "near_high_type", "dist_from_high",
        "roc_20", "roc_60", "roc_90", "composite_momentum",
        "rank_roc_20", "rank_roc_60", "rank_roc_90", "rank_composite",
        "sma10", "sma20", "sma50", "sma150", "sma200",
        "ath", "high_1y", "high_5y", "last_date"
    ]
    df[cols].to_csv(filename, index=False)
    print(f"\n  Results exported: {filename}")


# =============================================================================
# MAIN SCANNER
# =============================================================================
def run_scanner():
    """Main scanner function"""
    print("\n" + "=" * 70)
    print("  NSE MOMENTUM SCANNER - LIVE")
    print("  Filter: Price > SMA10 > SMA20 > SMA50 > SMA150 > SMA200")
    print("  Filter: Within 20% of ATH / 1Y High / 5Y High")
    print("  Ranking: Relative momentum (20d, 60d, 90d ROC)")
    print("=" * 70)

    # Validate and login
    totp_fixed = validate_credentials()

    print("\nLogging in...")
    print(f"  API_KEY   : {API_KEY[:6]}{'*' * 8}  CLIENT_ID: {CLIENT_ID}")
    try:
        obj = login(totp_fixed)
        print("  LOGIN SUCCESSFUL\n")
    except Exception as e:
        print(f"\n  LOGIN FAILED: {e}")
        raise SystemExit(1)

    # Get universe
    print("Loading universe...")
    symbols = get_universe()
    print(f"  Total EQ-series symbols: {len(symbols)}")

    # Get token map
    print("\nLoading token map...")
    token_map = get_token_map()

    # Match symbols to tokens
    print(f"\nMatching symbols to tokens...")
    matched = {}
    not_found = []
    for sym in symbols:
        used, tok = find_token(token_map, sym)
        if tok:
            matched[used] = tok
        else:
            not_found.append(sym)
    print(f"  Matched: {len(matched)}")
    if not_found:
        print(f"  Not found: {len(not_found)} (first 5: {not_found[:5]})")

    # Date range for fetch
    fetch_end = datetime.date.today().strftime("%Y-%m-%d")
    fetch_start = (datetime.date.today() - datetime.timedelta(days=365 * FETCH_YEARS)).strftime("%Y-%m-%d")

    # Scan all stocks
    total = len(matched)
    cached = sum(1 for s in matched if os.path.exists(os.path.join(CACHE_DIR, f"{s}_1D.pkl")))
    new_fetch = total - cached

    print(f"\nScanning {total} stocks...")
    print(f"  Cached: {cached} (instant)")
    print(f"  To fetch: {new_fetch} (ETA ~{new_fetch * 2.5 / 60:.0f} min)\n")

    results = []
    skipped = 0
    filtered_out = 0
    t0 = time.time()

    for i, (sym, tok) in enumerate(matched.items(), 1):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 1)
        eta_s = (total - i) / rate if rate > 0 else 0

        print(f"  [{i:4d}/{total}] {sym:<18} passed:{len(results):3d}  ETA:{eta_s/60:.1f}min  ", end="\r")

        # Fetch data
        df = fetch_ohlc(obj, tok, sym, fetch_start, fetch_end)
        if df is None or len(df) < MIN_DATA_DAYS:
            skipped += 1
            continue

        # Compute indicators
        df_ind = compute_indicators(df)
        if df_ind is None:
            filtered_out += 1
            continue

        # Analyze stock
        analysis = analyze_stock(sym, df_ind)
        if analysis:
            results.append(analysis)

    elapsed_m = (time.time() - t0) / 60
    print(f"\n\n  Scan complete in {elapsed_m:.1f} min")
    print(f"  Total scanned: {total}")
    print(f"  Insufficient data: {skipped}")
    print(f"  Filtered out (price/volume): {filtered_out}")
    print(f"  Passed all filters: {len(results)}")

    if not results:
        print("\n  No stocks matched all criteria.")
        return pd.DataFrame()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate relative momentum ranks
    results_df = calculate_relative_momentum(results_df)

    # Display and export
    print_results(results_df, top_n=30)
    export_results(results_df)

    return results_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    results = run_scanner()

    # Print top 10 trading signals
    if not results.empty:
        print("\n" + "=" * 70)
        print("  TOP 10 TRADING SIGNALS")
        print("=" * 70)
        for i, row in results.head(10).iterrows():
            print(f"\n  #{row['final_rank']} {row['symbol']}")
            print(f"      Price: Rs {row['price']:.2f}  |  Near {row['near_high_type']} high ({row['dist_from_high']:.1f}% away)")
            print(f"      Momentum: 20d={row['roc_20']:+.1f}%  60d={row['roc_60']:+.1f}%  90d={row['roc_90']:+.1f}%")
            print(f"      SMA Stack: {row['price']:.0f} > {row['sma10']:.0f} > {row['sma20']:.0f} > {row['sma50']:.0f} > {row['sma150']:.0f} > {row['sma200']:.0f}")
