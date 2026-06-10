"""
=============================================================================
NSE MOMENTUM BACKTEST  v9
=============================================================================
UNIVERSE  : All NSE EQ-series stocks with market cap > ~Rs 100 crore
            Source: NSE EQUITY_L.csv (all main board EQ stocks) matched
            with Angel One scrip master, then filtered by price/turnover.
            Falls back to comprehensive hardcoded list if download fails.
            Expected universe: 700-1000 liquid stocks.

ENTRY     : Price > SMA10 > SMA20 > SMA50 > SMA150 > SMA200
            Price within 20% of ATH / 1Y / 5Y high
            Top 3 by 90-day ROC

EXIT      : 8% stop loss | Target = ATH * 1.20
REBALANCE : Monthly (first trading day of each month)
POSITIONS : 3  (equal weight)
COSTS     : Full NSE equity delivery charges

FIRST RUN : ~20-40 min for new stocks not in cache
CACHED    : Instant on re-runs (~500 stocks already cached from earlier runs)
=============================================================================
"""

import os, pickle, time, datetime, io
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from SmartApi import SmartConnect
import pyotp

# =============================================================================
# CREDENTIALS
# =============================================================================
API_KEY      = "Tp7PJMIc"
CLIENT_ID    = "S63068620"
PASSWORD     = "4098"
TOTP_SECRET  = "KVTELUFGR33YCPQUKO3ZSL4NMA"
# =============================================================================

# =============================================================================
# CONFIG
# =============================================================================
INITIAL_CAPITAL   = 1_00_000
MAX_POSITIONS     = 3
STOP_LOSS_PCT     = 0.08        # change to 0.10 to match v5
ATH_EXTENSION_PCT = 0.10        # Phase 1: once price > ATH * 1.20, switch to trailing stop

# After price crosses ATH * 1.20, hold position and trail with 5% stop below peak.
# Lets big winners run much further than a fixed exit would allow.
TRAIL_STOP_PCT    = 0.20        # 5% trailing stop below peak once in Phase 2
NEAR_HIGH_PCT     = 0.10        # 10% matches v5 (tighter = better momentum filter)
MOMENTUM_PERIOD   = 30
BACKTEST_START    = "2016-01-01"  # extended -- delete stock pkl cache to re-fetch
BACKTEST_END      = "2026-12-31"
APPLY_COSTS       = True

# =============================================================================
# SLIPPAGE MODEL
# Simulates the gap between theoretical close price and actual execution price.
# Applied on every buy and sell. Midcap/smallcap stocks typically 0.1-0.3%.
# Large caps can be as low as 0.05%. Use 0.1% (0.001) as a conservative base.
# =============================================================================
SLIPPAGE_PCT = 0.001      # 0.1% per side  →  0.2% round trip per trade
MIN_PRICE         = 20          # avg close > Rs 20  (penny stock filter)
MIN_AVG_TURNOVER  = 10_00_000   # avg daily turnover > Rs 10 lakh (~Rs 100 cr market cap)
FILTER_LOOKBACK   = 60
CACHE_DIR         = "cache_momentum"
# =============================================================================

os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# COMPREHENSIVE FALLBACK UNIVERSE
# Used if NSE download fails. Covers Nifty 500 + Smallcap 250 + additional.
# ~700 unique stocks, all with market cap > ~Rs 100 crore.
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
    "ZYDUSLIFE","HAPPSTMNDS","MFSL","INOXWIND","SUZLON","IREDA","NBCC",
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
    "SKIPPER","ABBOTINDIA","GLAXO","PFIZER","SANOFI","ABBOTINDIA",
    # Stocks that appeared in backtest results / additional coverage
    "ANANDRATHI","BSE","ZENTEC","FORCEMOT","PCBL","HUDCO","GABRIEL",
    "PARADEEP","MAZDOCK","PGEL","COCHINSHIP","TARIL","TECHNOE","JWL",
    "GVT&D","IREDA","BIKAJI","CAMPUS","JLHL","MEDANTA","NETWEB",
    "SUVENPHAR","TCIEXP","UCOBANK","VMART","DCMSHRIRAM","JINDALSAW",
    "KPIL","KRSNAA","SAREGAMA","SHYAMMETL","SONATSOFTW","NMDC",
    "OPTIEMUS","ORCHPHARMA","PDSL","POWERMECH","PRICOLLTD","WINDLAS",
    "WPIL","WSL","LUXIND","PIIND","TMVL","STERLINV","NIITLTD",
]

# Deduplicate preserving order
_s = set()
UNIVERSE_FALLBACK = [x for x in UNIVERSE_FALLBACK if not (x in _s or _s.add(x))]


# =============================================================================
# HISTORICALLY DISTRESSED / CRASHED NSE STOCKS  (partial survivorship fix)
# These stocks were once prominent in NSE / Nifty 500 but crashed badly.
# Including them forces the backtest to take real losses on stocks that
# had bullish momentum before their collapse.
#
# NOTE ON SURVIVORSHIP BIAS:
# Fully delisted stocks (DHFL, Jet Airways, PC Jewellers, Gitanjali Gems,
# IL&FS, HDIL, Cox & Kings, Reliance Comm) CANNOT be included because
# Angel One API has no historical data for them post-delisting.
# This means the backtest is still optimistic by ~3-5% CAGR vs real trading.
#
# Stocks STILL LISTED but that crashed severely are included below.
# Their historical data IS available and their crashes will appear as losses.
# =============================================================================
HISTORICALLY_DISTRESSED = [
    "YESBANK",      # Nifty 50 component → crashed 95%+ (2018-2020)
    "VAKRANGEE",    # Nifty Midcap → accounting fraud, crashed 98%
    "PCJEWELLER",   # Nifty Midcap → suspended, effectively worthless
    "SUZLON",       # Nifty 500 → crashed, recovered partially
    "RCOM",         # Reliance Communications → near delisted
    "JPPOWER",      # JP Power → heavily distressed
    "JPASSOCIAT",   # JP Associates → heavily distressed
    "UNITECH",      # Nifty 500 → real estate fraud, near zero
    "VIDEOIND",     # Videocon Industries → bankrupt
    "ALOKTEXT",     # Alok Industries → bankrupt, recovered partially
    "BOMBTECH",     # Bombay Tech → crashed
    "IDEA",         # Vodafone Idea → near distress
    "RTNPOWER",     # Rattanindia Power → distressed
    "GTPL",         # GTPL Hathway
    "HDFCAMC",      # example of survivor (already in main list)
]
# We attempt to fetch these — the SL will handle most crashes naturally.
# Stocks where data is unavailable will simply be skipped.

SYMBOL_MAP = {
    "BERGERPAINTS":"BERGEPAINT","AUROBINDO":"AUROPHARMA",
    "JSPL":"JINDALSTEL","REC":"RECLTD","TVSMOTORS":"TVSMOTOR",
    "CEAT":"CEATLTD","MCDOWELL-N":"UNITDSPR","MOTHERSON":"MOTHERSUMI",
    "INFOEDGE":"NAUKRI","L&TFH":"L&TFH",
}


# =============================================================================
# DOWNLOAD UNIVERSE FROM NSE EQUITY LIST
# NSE's EQUITY_L.csv has ALL NSE main board stocks — no metadata rows,
# reliable CSV format. Filter by SERIES == "EQ" to exclude BE/SM/etc.
# =============================================================================
def get_universe(obj):
    cache_path = os.path.join(CACHE_DIR, "universe_list.pkl")
    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days < 7:
            with open(cache_path, "rb") as f: syms = pickle.load(f)
            print(f"  Universe: {len(syms)} symbols  (cache {age_days:.1f} days old)")
            return syms
        print("  Universe cache >7 days — refreshing...")

    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # NSE EQUITY_L.csv: all NSE listed stocks with series info
    # This is a clean CSV with no metadata rows — parses reliably
    equity_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    syms = []

    try:
        print(f"  Downloading NSE equity list...")
        resp = requests.get(equity_url, headers=headers, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 10000:
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"  Columns: {df.columns.tolist()[:5]}")
            # Filter EQ series only (main board, excludes BE/SM/BZ surveillance)
            if "SERIES" in df.columns and " SYMBOL" in df.columns:
                eq = df[df["SERIES"] == "EQ"][" SYMBOL"].dropna().str.strip()
                syms = eq.tolist()
            elif "SERIES" in df.columns and "SYMBOL" in df.columns:
                eq = df[df["SERIES"] == "EQ"]["SYMBOL"].dropna().str.strip()
                syms = eq.tolist()
            print(f"  NSE equity list: {len(syms)} EQ-series stocks")
    except Exception as e:
        print(f"  NSE equity list download failed: {e}")

    if len(syms) < 100:
        print("  Using comprehensive fallback universe list")
        syms = list(UNIVERSE_FALLBACK)

    # Apply symbol corrections
    syms = [SYMBOL_MAP.get(s, s) for s in syms]
    # Deduplicate
    seen = set(); syms = [s for s in syms if not (s in seen or seen.add(s))]

    with open(cache_path, "wb") as f: pickle.dump(syms, f)
    print(f"  Universe saved: {len(syms)} symbols")
    return syms


# =============================================================================
# TRANSACTION COSTS
# =============================================================================
def calc_cost(trade_value, side):
    if not APPLY_COSTS: return 0.0
    b = min(trade_value * 0.001, 20.0)
    e = trade_value * 0.0000335
    s = trade_value * 0.000001
    t = b + e + s + (b + e) * 0.18
    return t + (trade_value * 0.00015 if side == "buy" else trade_value * 0.001)


# =============================================================================
# TOTP + CREDENTIALS
# =============================================================================
def fix_totp(raw):
    s = raw.strip().upper().replace(" ","").replace("-","")
    return s + "=" * ((8 - len(s) % 8) % 8)

def validate_credentials():
    pl = {"your_api_key","your_password","your_totp_secret",""}
    err = []
    if API_KEY.strip()     in pl: err.append("API_KEY not set")
    if PASSWORD.strip()    in pl: err.append("PASSWORD not set")
    if TOTP_SECRET.strip() in pl: err.append("TOTP_SECRET not set")
    if err:
        print("\n"+"="*60); print("  CREDENTIALS NOT CONFIGURED"); print("="*60)
        for e in err: print(f"  x  {e}")
        raise SystemExit(1)
    import base64
    fixed = fix_totp(TOTP_SECRET)
    try: base64.b32decode(fixed, casefold=True)
    except Exception as e:
        print(f"  TOTP_SECRET invalid: {e}"); raise SystemExit(1)
    return fixed


# =============================================================================
# LOGIN
# =============================================================================
def login(totp_fixed):
    obj  = SmartConnect(api_key=API_KEY)
    otp  = pyotp.TOTP(totp_fixed).now()
    secs = 30 - datetime.datetime.now().second % 30
    print(f"  OTP: {otp}  (valid ~{secs}s)")
    data = obj.generateSession(CLIENT_ID, PASSWORD, otp)
    if not isinstance(data, dict): raise Exception(f"Unexpected: {data}")
    status = data.get("status")
    if status is True or str(status).lower() == "true":
        print(f"  JWT: {str(data.get('data',{}).get('jwtToken',''))[:30]}...")
        return obj
    raise Exception(f"Login rejected. {data.get('errorcode','?')} | {data.get('message',str(data))}")


# =============================================================================
# SCRIP MASTER
# =============================================================================
def get_token_map(obj):
    cp = os.path.join(CACHE_DIR,"scrip_master.pkl")
    if os.path.exists(cp):
        with open(cp,"rb") as f: cached = pickle.load(f)
        age_h = (time.time()-os.path.getmtime(cp))/3600
        if len(cached)>100 and age_h<24:
            print(f"  Token map: {len(cached):,} tokens (cache {age_h:.1f}h old)")
            return cached
        os.remove(cp)
    import requests
    print("  Downloading scrip master...")
    resp = requests.get(
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
        timeout=30)
    df  = pd.DataFrame(resp.json())
    nse = df[df["exch_seg"]=="NSE"]
    eq  = nse[nse["symbol"].str.endswith("-EQ",na=False)].copy()
    eq["sym"] = eq["symbol"].str.replace("-EQ","",regex=False).str.strip()
    tm  = eq.set_index("sym")["token"].to_dict()
    print(f"  {len(nse):,} NSE -> {len(eq):,} -EQ -> {len(tm):,} tokens")
    with open(cp,"wb") as f: pickle.dump(tm, f)
    return tm


# =============================================================================
# TOKEN LOOKUP
# =============================================================================
def find_token(token_map, symbol):
    if symbol in token_map: return symbol, token_map[symbol]
    c = SYMBOL_MAP.get(symbol, symbol)
    if c in token_map: return c, token_map[c]
    for var in [
        symbol.replace("PAINTS","PAINT"), symbol.replace("MOTORS","MOTOR"),
        "AUROPHARMA" if symbol=="AUROBINDO" else None,
        "CEATLTD"    if symbol=="CEAT"      else None,
        "JINDALSTEL"  if symbol=="JSPL"      else None,
    ]:
        if var and var in token_map: return var, token_map[var]
    return None, None


# =============================================================================
# OHLC FETCH
# =============================================================================
def fetch_ohlc(obj, token, symbol, from_date, to_date):
    cp = os.path.join(CACHE_DIR, f"{symbol}_1D.pkl")
    if os.path.exists(cp):
        with open(cp,"rb") as f: df = pickle.load(f)
        if not df.empty and str(df.index[-1].date()) >= to_date[:10]: return df
    rows = []
    cur = pd.Timestamp(from_date); end = pd.Timestamp(to_date)
    while cur <= end:
        nxt = min(cur+pd.DateOffset(days=400), end)
        try:
            r = obj.getCandleData({
                "exchange":"NSE","symboltoken":token,"interval":"ONE_DAY",
                "fromdate":cur.strftime("%Y-%m-%d %H:%M"),
                "todate":  nxt.strftime("%Y-%m-%d %H:%M"),
            })
            if r.get("status") and r.get("data"): rows.extend(r["data"])
        except Exception: pass
        cur = nxt+pd.DateOffset(days=1); time.sleep(0.35)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    df.index = df.index.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    with open(cp,"wb") as f: pickle.dump(df, f)
    return df


# =============================================================================
# INDICATORS + QUALITY FILTER
# =============================================================================
def compute_indicators(df):
    if len(df) < 210: return None
    recent = df.tail(FILTER_LOOKBACK)
    if recent["close"].mean() < MIN_PRICE: return None
    if "volume" in recent.columns:
        if (recent["close"]*recent["volume"]).mean() < MIN_AVG_TURNOVER: return None
    df = df.copy()
    df["sma20"]  = df["close"].rolling(20).mean()
    df["sma50"]  = df["close"].rolling(50).mean()
    df["sma150"] = df["close"].rolling(150).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["roc"]    = df["close"].pct_change(MOMENTUM_PERIOD)*100
    df["high_1y"]= df["close"].rolling(252).max()
    df["high_5y"]= df["close"].rolling(252*5).max()
    df["ath"]    = df["close"].expanding().max()
    df.dropna(subset=["sma200","roc"], inplace=True)
    return df if len(df)>0 else None


# =============================================================================
# ENTRY FILTER
# =============================================================================
def passes_entry(row):
    p = row["close"]
    return (
        p > row["sma20"] > row["sma50"] > row["sma150"] > row["sma200"]
        and p >= (1-NEAR_HIGH_PCT)*max(row["ath"],row["high_1y"],row["high_5y"])
    )


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def run_backtest(price_data):
    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    all_dates = [d for d in all_dates if BACKTEST_START <= str(d.date()) <= BACKTEST_END]
    month_starts = set(
        pd.Series(all_dates, index=pd.DatetimeIndex(all_dates))
        .resample("MS").first().dropna().tolist()
    )
    cash=float(INITIAL_CAPITAL); positions={}; trades=[]; equity=[]; total_costs=0.0

    def last_px(sym, date):
        sub=price_data[sym][price_data[sym].index<=date]
        return float(sub["close"].iloc[-1]) if not sub.empty else None

    def nav(date):
        v=cash
        for sym,pos in positions.items():
            px=last_px(sym,date)
            if px: v+=px*pos["shares"]
        return v

    print(f"\n{'='*62}")
    print(f"  BACKTEST  {BACKTEST_START}  ->  {BACKTEST_END}")
    print(f"  Universe  : {len(price_data)} stocks  |  Capital: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"  Positions : {MAX_POSITIONS}  |  Rebalance: Monthly")
    print(f"  Entry     : Price>SMA20>SMA50>SMA150>SMA200 + within {NEAR_HIGH_PCT*100:.0f}% of ATH")
    print(f"  Exit      : SL {STOP_LOSS_PCT*100:.0f}%  |  Target ATH x {1+ATH_EXTENSION_PCT:.2f}")
    print(f"{'='*62}")

    total_days=len(all_dates); last_pct=-1
    for idx,date in enumerate(all_dates):
        pct=int(idx/total_days*100)
        if pct%5==0 and pct!=last_pct:
            print(f"  {pct:3d}%  {str(date.date())}  Pos:{len(positions)}  Cash:Rs {cash:,.0f}    ",end="\r")
            last_pct=pct

        for sym in list(positions.keys()):
            pos=positions[sym]; px=last_px(sym,date)
            if px is None: continue

            exit_reason = None

            if not pos["trailing"]:
                # ── PHASE 1: fixed stop loss + target trigger ──────────────
                if px <= pos["stop"]:
                    exit_reason = "STOP_LOSS"
                elif px >= pos["target"]:
                    # Target hit → switch to trailing mode, do NOT exit yet
                    pos["trailing"]   = True
                    pos["peak_price"] = px
                    pos["trail_stop"] = round(px * (1 - TRAIL_STOP_PCT), 2)
            else:
                # ── PHASE 2: trailing stop (lets winners run) ──────────────
                if px > pos["peak_price"]:
                    pos["peak_price"] = px
                    pos["trail_stop"] = round(px * (1 - TRAIL_STOP_PCT), 2)
                if px <= pos["trail_stop"]:
                    exit_reason = "TRAILING_STOP"

            if exit_reason:
                ep = px * (1 - SLIPPAGE_PCT)
                sv = ep * pos["shares"]
                sc = calc_cost(sv, "sell")
                pg = (ep - pos["entry_price"]) * pos["shares"]
                pn = pg - sc - pos["buy_cost"]
                pp = pn / (pos["entry_price"] * pos["shares"]) * 100
                cash += sv - sc; total_costs += sc
                trades.append(dict(symbol=sym,entry_date=pos["entry_date"],exit_date=date,
                    entry_price=pos["entry_price"],exit_price=ep,shares=pos["shares"],
                    pnl_gross=pg,pnl_net=pn,pnl_pct=pp,total_cost=sc+pos["buy_cost"],
                    stop=pos["stop"],target=pos["target"],ath_entry=pos["ath_at_entry"],
                    trailing_peak=pos["peak_price"],
                    exit_reason=exit_reason))
                del positions[sym]

        if date in month_starts:
            slots=MAX_POSITIONS-len(positions)
            if slots>0:
                candidates=[]
                for sym,df in price_data.items():
                    if sym in positions: continue
                    sub=df[df.index<=date]
                    if sub.empty: continue
                    row=sub.iloc[-1]
                    if passes_entry(row):
                        candidates.append((sym,float(row["roc"]),float(row["close"]),float(row["ath"])))
                candidates.sort(key=lambda x:x[1],reverse=True)
                alloc=cash*0.99/slots
                for sym,roc,price,ath in candidates[:slots]:
                    ep=price*(1+SLIPPAGE_PCT)  # buy slippage: pay slightly more
                    shares=int(alloc/ep)
                    if shares<1: continue
                    bv=shares*ep; bc=calc_cost(bv,"buy")
                    if bv+bc>cash: continue
                    cash-=(bv+bc); total_costs+=bc
                    positions[sym]=dict(entry_date=date,entry_price=ep,shares=shares,
                        stop=round(ep*(1-STOP_LOSS_PCT),2),
                        target=round(ath*(1+ATH_EXTENSION_PCT),2),
                        ath_at_entry=ath,buy_cost=bc,
                        trailing=False,   # switches True once price > ATH*1.20
                        peak_price=None,  # highest price seen in trailing phase
                        trail_stop=None)  # 5% below peak_price

        equity.append({"date":date,"equity":nav(date)})

    last_date=all_dates[-1]
    for sym,pos in list(positions.items()):
        px=(last_px(sym,last_date) or pos["entry_price"])*(1-SLIPPAGE_PCT)
        sv=px*pos["shares"]; sc=calc_cost(sv,"sell")
        pg=(px-pos["entry_price"])*pos["shares"]
        pn=pg-sc-pos["buy_cost"]; pp=pn/(pos["entry_price"]*pos["shares"])*100
        cash+=sv-sc; total_costs+=sc
        trades.append(dict(symbol=sym,entry_date=pos["entry_date"],exit_date=last_date,
            entry_price=pos["entry_price"],exit_price=px,shares=pos["shares"],
            pnl_gross=pg,pnl_net=pn,pnl_pct=pp,total_cost=sc+pos["buy_cost"],
            stop=pos["stop"],target=pos["target"],ath_entry=pos["ath_at_entry"],
            trailing_peak=pos["peak_price"],
            exit_reason="END_OF_BACKTEST"))

    print(f"\n  100%  complete                                             ")
    eq_df=pd.DataFrame(equity).set_index("date"); eq_df["equity"]=eq_df["equity"].ffill()
    return pd.DataFrame(trades), eq_df, total_costs


# =============================================================================
# REPORT
# =============================================================================
def report(tr, eq, total_costs):
    if tr.empty: print("No trades."); return
    final=eq["equity"].iloc[-1]; tret=(final/INITIAL_CAPITAL-1)*100
    years=(eq.index[-1]-eq.index[0]).days/365.25
    cagr=((final/INITIAL_CAPITAL)**(1/years)-1)*100 if years>0 else 0
    mdd=((eq["equity"]-eq["equity"].cummax())/eq["equity"].cummax()*100).min()
    dr=eq["equity"].pct_change().dropna()
    sharpe=(dr.mean()/dr.std()*np.sqrt(252)) if dr.std()>0 else 0
    calmar=abs(cagr/mdd) if mdd else 0
    wins=tr[tr["pnl_net"]>0]; losses=tr[tr["pnl_net"]<=0]
    wr=len(wins)/len(tr)*100 if len(tr) else 0
    pf=abs(wins["pnl_net"].sum()/losses["pnl_net"].sum()) if losses["pnl_net"].sum() else float("inf")
    hold=(tr["exit_date"]-tr["entry_date"]).dt.days.mean()

    sep="="*62
    print(f"\n{sep}"); print("           PERFORMANCE REPORT  v10"); print(sep)
    print(f"  Period          : {eq.index[0].date()}  to  {eq.index[-1].date()}")
    print(f"  Initial Capital : Rs {INITIAL_CAPITAL:>14,.0f}")
    print(f"  Final Equity    : Rs {final:>14,.0f}")
    print(f"  Total Return    : {tret:>12.2f}%")
    print(f"  CAGR            : {cagr:>12.2f}%")
    print(f"  Max Drawdown    : {mdd:>12.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:>12.2f}")
    print(f"  Calmar Ratio    : {calmar:>12.2f}")
    print(f"  Total Costs     : Rs {total_costs:>12,.0f}")
    print("-"*62)
    print(f"  Total Trades    : {len(tr):>12}")
    print(f"  Win Rate        : {wr:>12.2f}%")
    if len(wins):   print(f"  Avg Win  (net)  : {wins['pnl_pct'].mean():>11.2f}%")
    if len(losses): print(f"  Avg Loss (net)  : {losses['pnl_pct'].mean():>11.2f}%")
    print(f"  Profit Factor   : {pf:>12.2f}")
    print(f"  Avg Hold Days   : {hold:>12.1f}")
    print("-"*62)
    for reason,cnt in tr["exit_reason"].value_counts().items():
        avg=tr[tr["exit_reason"]==reason]["pnl_pct"].mean()
        print(f"  {reason:<22}: {cnt:4d}  avg {avg:>+6.1f}%")
    print(sep)

    cols=["symbol","entry_date","exit_date","entry_price","exit_price","pnl_pct","exit_reason"]
    print("\n  TOP 10 WINNING TRADES:"); print(tr.nlargest(10,"pnl_net")[cols].to_string(index=False))
    print("\n  TOP 10 LOSING TRADES:");  print(tr.nsmallest(10,"pnl_net")[cols].to_string(index=False))

    eq_m=eq["equity"].resample("ME").last().dropna()
    print(f"\n  {'Date':<12}  {'Equity':>14}  {'MoM%':>7}  Chart"); print("  "+"-"*58)
    prev=float(INITIAL_CAPITAL)
    for dt,val in eq_m.items():
        ret=(val/prev-1)*100
        bar=("+"if ret>=0 else"-")*min(50,int(abs(ret)*2))
        print(f"  {str(dt.date()):<12}  Rs {val:>12,.0f}  {ret:>+6.1f}%  {bar}"); prev=val

    tr.to_csv("backtest_trades_v9.csv",index=False)
    eq.to_csv("backtest_equity_v9.csv")
    print(f"\n  Trades -> backtest_trades_v9.csv  |  Equity -> backtest_equity_v9.csv\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n"+"="*62)
    print("  NSE MOMENTUM BACKTEST  v10")
    print("  Universe : All NSE stocks  |  Market cap > ~Rs 100 crore")
    print("  Strategy : v5 rules on expanded universe  (no SMA10 filter)")
    print("="*62)

    totp_fixed = validate_credentials()

    cp = os.path.join(CACHE_DIR,"scrip_master.pkl")
    if os.path.exists(cp):
        with open(cp,"rb") as f: c = pickle.load(f)
        if len(c)<=100: os.remove(cp)

    print(f"\nLogging in...")
    print(f"  API_KEY   : {API_KEY[:6]}{'*'*8}  CLIENT_ID: {CLIENT_ID}")
    print(f"  TOTP      : {TOTP_SECRET[:4]}{'*'*8}  len={len(TOTP_SECRET)}")
    try:
        obj = login(totp_fixed); print("  LOGIN SUCCESSFUL\n")
    except Exception as e:
        print(f"\n  LOGIN FAILED: {e}"); raise SystemExit(1)

    # Get universe
    print("Loading universe...")
    symbols = get_universe(obj)
    print(f"  Total symbols in universe: {len(symbols)}")

    # Token map
    print("\nLoading token map...")
    token_map = get_token_map(obj)

    # Merge distressed stocks (partial survivorship bias fix)
    all_syms=list(dict.fromkeys(symbols+HISTORICALLY_DISTRESSED))
    print(f"\nMatching {len(all_syms)} symbols ({len(symbols)} universe + {len(HISTORICALLY_DISTRESSED)} historical distressed)...")
    matched={}; not_found=[]
    for sym in all_syms:
        used,tok = find_token(token_map, sym)
        if tok:  matched[used]=tok
        else:    not_found.append(sym)
    print(f"  Matched   : {len(matched)}")
    if not_found: print(f"  Not found : {len(not_found)}  {not_found[:6]}")

    # Fetch OHLC
    fetch_start=str((pd.Timestamp(BACKTEST_START)-pd.DateOffset(days=300)).date())
    fetch_end=BACKTEST_END
    total=len(matched)
    cached_n=sum(1 for s in matched if os.path.exists(os.path.join(CACHE_DIR,f"{s}_1D.pkl")))
    new_n=total-cached_n

    print(f"\nFetching OHLC for {total} symbols...")
    print(f"  Already cached : {cached_n}  (instant — reuses your existing cache)")
    print(f"  Need to fetch  : {new_n}   (ETA ~{new_n*2.5/60:.0f} min)\n")

    price_data={}; skipped=0; filtered=0; t0=time.time()
    for i,(sym,tok) in enumerate(matched.items(),1):
        elapsed=time.time()-t0; rate=i/max(elapsed,1)
        eta_s=(total-i)/rate if rate>0 else 0
        print(f"  [{i:3d}/{total}] {sym:<18}  loaded:{len(price_data):3d}  ETA:{eta_s/60:.1f}min  ",end="\r")
        df=fetch_ohlc(obj,tok,sym,fetch_start,fetch_end)
        if df is None or len(df)<210: skipped+=1; continue
        ind=compute_indicators(df)
        if ind is not None: price_data[sym]=ind
        else: filtered+=1

    elapsed_m=(time.time()-t0)/60
    print(f"\n  Done in {elapsed_m:.1f} min")
    print(f"  Loaded   : {len(price_data)} stocks in universe")
    if skipped:  print(f"  No data  : {skipped} (delisted / new listings)")
    if filtered: print(f"  Filtered : {filtered} (penny / illiquid / < Rs 100 cr market cap)")

    if not price_data: print("\n  ERROR: No data."); raise SystemExit(1)

    trades_df, equity_df, costs = run_backtest(price_data)
    report(trades_df, equity_df, costs)

if __name__ == "__main__":
    main()
