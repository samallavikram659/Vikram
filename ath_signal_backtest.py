"""
=============================================================================
ATH SIGNAL BACKTEST  -  PER-STOCK, FIXED CAPITAL  (Yahoo Finance edition)
=============================================================================
APPROACH  : Every stock that meets the strategy is traded independently.
            When signal fires: invest Rs PER_TRADE_CAPITAL (Rs 10,000).
            No position cap - all qualifying stocks traded simultaneously.
            Each stock can hold one open position at a time.

DATA      : Yahoo Finance (yfinance) — no login required.
            NSE universe from NSE public CSV (~2500 EQ stocks).
            Batch downloads 50 stocks per request (~2-3 min first run).

ENTRY     : EMA alignment (Close > EMA5 > EMA10 > EMA20 > EMA50 > EMA150 > EMA200)
            Close within ATH_NEAR_PCT (5%) BELOW All-Time High
            Fresh high filter: ATH not touched in last FRESH_HIGH_DAYS bars
            Oliver Kell signal: Wedge_Pop OR Crossback

EXIT      : (checked in priority order)
            1. Stop loss STOP_LOSS_PCT below entry price       (on bar Low)
            2. Target TARGET_ABOVE_HIGH_PCT above ATH          (on bar High)
            3. Oliver Trailing EMA stop (EMA20 early, EMA10 after MIN_HOLD_DAYS)
            4. Extension exit (Close > EMA20 + EXTENSION_MULT * ATR)

RESULTS   : Aggregate P&L, win rate, calendar year breakdown, per-stock stats
=============================================================================
"""

import os
import pickle
import time
import datetime
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
PER_TRADE_CAPITAL     = 10_000          # Rs per trade per stock
ATH_NEAR_PCT          = 0.05           # within 5% below ATH to qualify
STOP_LOSS_PCT         = 0.08           # 8% stop loss below entry
TARGET_ABOVE_HIGH_PCT = 0.15           # 15% above ATH as target
FRESH_HIGH_DAYS       = 200            # skip if ATH touched within this many bars
LOOKBACK_DAYS         = 2000           # backtest window in days (~5.5 years)
FETCH_YEARS           = 6              # years of history to fetch

# Oliver Kell config
MINI_BASE_BARS        = 7              # window for mini-base detection
MINI_BASE_MIN         = 4             # minimum contracting bars in base
EXTENSION_MULT        = 3.0           # ATR multiplier for extension exit
MIN_HOLD_DAYS         = 2             # bars before switching to tighter EMA stop
RS_LOOKBACK           = 20            # days for relative-strength calculation

CACHE_DIR             = "cache_ath_yf"         # primary cache dir for this script
CACHE_FALLBACK_DIRS   = ["cache_ath_signal", "cache_near_high"]  # check existing caches
CACHE_MAX_AGE_HOURS   = 168           # use cache up to 7 days old
MIN_DATA_DAYS         = 252           # minimum bars needed per stock
BATCH_SIZE            = 50            # stocks per yfinance batch request

# =============================================================================
# ETF / FUND EXCLUSION
# =============================================================================
ETF_PATTERNS = [
    "ETF", "FUND", "NIFTY", "SENSEX", "LIQUID", "GILT", "BOND",
    "INDEX", "BEES", "JUNIOR", "CPSE",
]

def is_etf(name: str) -> bool:
    return any(p in name.upper() for p in ETF_PATTERNS)

# =============================================================================
# NIFTY 500 FALLBACK UNIVERSE
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
    "AFFLE", "MANAPPURAM", "SHREECEM", "RAMCOCEM", "LAURUSLABS",
    "NMDC", "APOLLOTYRE", "INDHOTEL", "GODREJPROP", "PVRINOX",
    "ABBOTINDIA", "SANOFI", "GILLETTE", "ZOMATO", "JIOFIN", "PAYTM",
    "POLICYBZR", "KAYNES", "LLOYDSME", "SONATSOFTW", "STARHEALTH",
    "MAZDOCK", "COCHINSHIP", "GRSE", "DALBHARAT",
    # --- Nifty Smallcap 250 ---
    "IDBI", "IOB", "CENTRALBK", "UCOBANK", "MAHABANK", "J&KBANK",
    "KTKBANK", "KARURVYSYA", "CUB", "DCBBANK", "EQUITAS", "UJJIVAN",
    "CREDITACC", "SPANDANA", "HOMEFIRST", "APTUS", "REPCO",
    "CGCL", "CHOLAHLDNG", "BSE", "CAMS", "KFINTECH", "ANGELONE",
    "NYKAA", "CARTRADE", "EASEMYTRIP",
    "HUDCO", "IRCON", "RVNL", "IREDA", "SJVN", "NLCINDIA",
    "JSWENERGY", "POWERMECH", "KALPATPOWR", "KEC", "JYOTISTRUC",
    "BAJAJELEC", "BLUESTARCO", "AMBER", "ELGIEQUIP", "KIRLOSENG",
    "TRITURBINE", "GREAVESCOT", "MAHSEAMLES", "RATNAMANI", "GPPL",
    "GUJGASLTD", "IGL", "GSPL", "MRPL", "CHENNPETRO", "HINDPETRO",
    "CASTROLIND", "GULFOILLUB", "GNFC", "GSFC", "CHAMBLFERT",
    "DEEPAKFERT", "RALLIS", "UPL", "BAYERCROP", "DHANUKA", "SHARDACROP",
    "ERIS", "GRANULES", "CAPLIPOINT", "STRIDES", "LAURUS",
    "THYROCARE", "POLYMED", "MEDANTA", "RAINBOW", "NH", "KIMS", "YATHARTH",
    "KANSAINER", "CENTURYPLY", "GREENPANEL", "CERA",
    "JKLAKSHMI", "HEIDELBERG", "STARCEMENT", "BIRLACORPN",
    "NATIONALUM", "HINDCOPPER", "MOIL", "WELCORP", "JINDALSAW",
    "FORCEMOT", "CEATLTD", "JKTYRE", "SUPRAJIT", "SKFINDIA", "FIVESTAR",
    "KALYANAJEW", "SENCO", "PCJEWELLER", "RAJESHEXPO",
    "TASTYBITE", "EIDPARRY", "DALMIASUGAR", "BALRAMCHIN", "RENUKA",
    "CCL", "UBL", "JYOTHYLAB", "HATSUN", "HERITGFOOD", "BIKAJI",
    "GODFRYPHLP", "VSTIND", "SHOPERSTOP", "VMART", "LUXIND", "RUPA",
    "SWIGGY", "SPICEJET", "SOBHA", "SUNTECK", "MAHLIFE", "KOLTEPATIL",
    "ANANTRAJ", "RAYMOND", "COFORGE", "CYIENT", "ECLERX",
    "INTELLECT", "MASTEK", "HAPPSTMNDS", "NEWGEN", "ROUTE",
    "LATENTVIEW", "TANLA", "NETWORK18", "TV18BRDCST", "STAR", "SAREGAMA",
    "TIPS", "HFCL", "STLTECH", "TTML", "MAXFIN", "GODIGIT", "ISEC",
    "POONAWALLA", "M&MFIN", "MMTC", "NFL", "RCF", "FACT", "RITES",
    "NBCC", "NCC", "ITI", "PTC", "APLAPOLLO", "GODREJIND", "CLEAN",
    "GALAXYSURF", "ICRA", "APL",
    # --- Additional Smallcap / Midcap ---
    "AARTI", "AARTIIND", "ABSLAMC", "ALOKINDS", "ANANDRATHI", "APARINDS",
    "ASAHIINDIA", "ASTRAZEN", "BEML", "BLUEDART", "BSOFT", "CAMPUS",
    "CAPLIN", "CENTURYTEX", "CHALET", "CHAMBAL", "CHOICEIN", "CMSINFO",
    "CONCORDBIO", "CRAFTSMAN", "DBCORP", "DCMSHRIRAM", "DELTACORP",
    "DEVYANI", "EDELWEISS", "ELECON", "ENGINERSIN", "FDC", "FINCABLES",
    "FINPIPE", "FLUOROCHEM", "GARFIBRES", "GESHIP", "GHCL", "GMDCLTD",
    "GOCOLORS", "GODREJAGRO", "GOODYEAR", "GRPLTD", "GUJALKALI",
    "HBLPOWER", "HGS", "HIKAL", "HINDWAREAP", "IBULHSGFIN", "IFBIND",
    "INDIAMART", "INDIGOPNTS", "INOXWIND", "JAMNAAUTO", "JBM",
    "JMFINANCIL", "JSL", "JTEKTINDIA", "JUBLINGREA", "JUBLPHARMA",
    "JUSTDIAL", "KENNAMET", "KNRCON", "KPRMILL", "KRBL", "KRSNAA",
    "LAXMIMACH", "LEMONTREE", "MAPMYINDIA", "MAXIND", "MIDHANI",
    "MINDACORP", "MOTILALOFS", "NESCO", "NILKAMAL", "OLECTRA",
    "ORIENTELEC", "PIRAMAL", "PPLPHARMA", "PRINCEPIPE", "PRSMJOHNSN",
    "QUESS", "REDINGTON", "ROSSARI", "SAPPHIRE", "SHILPAMED", "SHYAMMETL",
    "SIS", "SOUTHBANK", "SUDARSCHEM", "SWANENERGY", "SYMPHONY", "TARSONS",
    "TEAMLEASE", "TECHNOE", "TEGA", "TIINDIA", "TRIDENT", "TTKPRESTIG",
    "UFLEX", "UTIAMC", "VAIBHAVGBL", "VARUN", "VIPIND", "VRLLOG",
    "WABAG", "WESTLIFE", "WOCKPHARMA", "WONDERLA", "ZENSARTECH",
]
NIFTY_500 = sorted(set(NIFTY_500))

# =============================================================================
# NSE UNIVERSE - Yahoo Finance
# =============================================================================
def get_nse_universe(cache_file: str = "nse_universe_yf.pkl") -> list:
    """Return list of NSE EQ symbols. Downloads from NSE public CSV with 24h cache."""
    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 24:
            with open(cache_file, "rb") as f:
                symbols = pickle.load(f)
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

        with open(cache_file, "wb") as f:
            pickle.dump(symbols, f)
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

            # Normalize column names to lowercase
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


def fetch_nifty_close(years: int = FETCH_YEARS) -> pd.Series | None:
    """Fetch Nifty 50 index close prices from Yahoo Finance."""
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    try:
        raw = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        close = raw["Close"].squeeze()
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_convert(None)
        close.name = "close"
        return close
    except Exception:
        return None


# =============================================================================
# INDICATORS
# =============================================================================
def compute_indicators(d: pd.DataFrame, nifty_close=None) -> pd.DataFrame | None:
    if len(d) < MIN_DATA_DAYS:
        return None

    # EMAs
    for span in [5, 10, 20, 50, 150, 200]:
        d[f"EMA{span}"] = d["close"].ewm(span=span, adjust=False).mean()

    # ATR14
    hl  = d["high"] - d["low"]
    hpc = (d["high"] - d["close"].shift(1)).abs()
    lpc = (d["low"]  - d["close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(span=14, adjust=False).mean()

    # EMA alignment
    d["EMA_Aligned"] = (
        (d["close"] > d["EMA5"])   &
        (d["EMA5"]  > d["EMA10"])  &
        (d["EMA10"] > d["EMA20"])  &
        (d["EMA20"] > d["EMA50"])  &
        (d["EMA50"] > d["EMA150"]) &
        (d["EMA150"]> d["EMA200"])
    )

    # ATH (rolling max of high over all available history)
    d["ATH"] = d["high"].cummax()

    # Near-ATH condition: within ATH_NEAR_PCT% below ATH
    d["Near_ATH"] = (
        (d["close"] >= d["ATH"] * (1 - ATH_NEAR_PCT)) &
        (d["close"] <= d["ATH"])
    )

    # Fresh high filter: exclude if ATH was touched within FRESH_HIGH_DAYS bars
    touched_ath = (d["high"] >= d["ATH"].shift(1)).fillna(False)
    d["Touched_ATH_Recent"] = touched_ath.rolling(FRESH_HIGH_DAYS, min_periods=1).sum() > 0
    d["Near_ATH"] = d["Near_ATH"] & ~d["Touched_ATH_Recent"]

    # Weekly EMAs (forward-filled to daily)
    weekly = d["close"].resample("W").last().ffill()
    d["W_EMA20"] = (
        weekly.ewm(span=20, adjust=False).mean()
        .reindex(d.index, method="ffill")
    )
    d["Weekly_Uptrend"] = d["close"] > d["W_EMA20"]

    # Volume
    d["Vol20_Avg"]     = d["volume"].rolling(20, min_periods=10).mean()
    d["Vol_Above_Avg"] = d["volume"] > d["Vol20_Avg"]

    # Relative strength vs Nifty
    if nifty_close is not None and not nifty_close.empty:
        nifty_aligned = nifty_close.reindex(d.index, method="ffill")
        d["RS_Leader"] = (
            d["close"].pct_change(RS_LOOKBACK) >
            nifty_aligned.pct_change(RS_LOOKBACK)
        )
    else:
        d["RS_Leader"] = True

    # Mini-base: vectorized ATR contraction detection
    atr_declining = d["ATR14"].diff() < 0
    d["Mini_Base"] = (
        atr_declining
        .rolling(MINI_BASE_BARS - 1, min_periods=MINI_BASE_BARS - 1)
        .sum() >= MINI_BASE_MIN
    ).fillna(False)

    # Swing high
    d["Swing_Hi10"] = d["high"].rolling(10).max().shift(1)

    # Wedge Pop
    prev_mini = d["Mini_Base"].shift(1).fillna(False)
    d["Wedge_Pop"] = (
        (d["close"]  > d["Swing_Hi10"]) &
        prev_mini &
        d["EMA_Aligned"] &
        d["Weekly_Uptrend"] &
        d["Vol_Above_Avg"] &
        d["RS_Leader"]
    )

    # Crossback (bounce off EMA20)
    prev_low   = d["low"].shift(1)
    prev_ema20 = d["EMA20"].shift(1)
    touched_ema20 = (prev_low <= prev_ema20) & (d["close"].shift(1) >= prev_ema20)
    d["Crossback"] = (
        touched_ema20 &
        (d["close"] > d["EMA20"]) &
        d["Weekly_Uptrend"] &
        d["Vol_Above_Avg"] &
        d["RS_Leader"]
    )

    # Extension level
    d["Extension"] = d["close"] > (d["EMA20"] + EXTENSION_MULT * d["ATR14"])

    # Combined signal
    d["Oliver_Signal"] = d["Wedge_Pop"] | d["Crossback"]
    d["Signal"]        = d["EMA_Aligned"] & d["Near_ATH"] & d["Oliver_Signal"]

    d.dropna(subset=["EMA200", "ATH"], inplace=True)
    return d if len(d) > 0 else None


# =============================================================================
# PER-STOCK SIMULATION
# =============================================================================
def simulate_stock(symbol: str, d: pd.DataFrame) -> list[dict]:
    """Simulate one stock independently. Returns list of completed trade dicts."""
    today    = datetime.date.today()
    bt_start = today - datetime.timedelta(days=LOOKBACK_DAYS)
    d = d[d.index.date >= bt_start]
    if d.empty:
        return []

    trades   = []
    position = None
    rows     = list(d.iterrows())

    for idx, (date, row) in enumerate(rows):
        # ---- EXIT CHECK ----
        if position is not None:
            position["hold_bars"] += 1
            stop   = position["stop_price"]
            tgt    = position["target_price"]
            held   = position["hold_bars"]
            shares = position["shares"]
            ep     = position["entry_price"]

            exit_price  = None
            exit_reason = None

            if row["low"] <= stop:
                exit_price  = stop
                exit_reason = "Stop_Loss"
            elif row["high"] >= tgt:
                exit_price  = tgt
                exit_reason = "Target_Hit"
            else:
                ema_trail = row["EMA20"] if held < MIN_HOLD_DAYS else row["EMA10"]
                if row["close"] < ema_trail:
                    exit_price  = row["close"]
                    exit_reason = "Oliver_EMA_Stop"
                elif row["Extension"]:
                    exit_price  = row["close"]
                    exit_reason = "Extension_Exit"

            if exit_price is not None:
                pnl_rs  = (exit_price - ep) * shares
                pnl_pct = (exit_price - ep) / ep * 100
                trades.append({
                    "Symbol"      : symbol,
                    "Entry_Date"  : position["entry_date"].date(),
                    "Exit_Date"   : date.date(),
                    "Entry_Price" : round(ep, 2),
                    "Exit_Price"  : round(exit_price, 2),
                    "Shares"      : shares,
                    "Invested_Rs" : round(position["invested"], 2),
                    "PnL_Rs"      : round(pnl_rs, 2),
                    "PnL_Pct"     : round(pnl_pct, 2),
                    "Hold_Days"   : held,
                    "Signal_Type" : position["signal_type"],
                    "Exit_Reason" : exit_reason,
                    "ATH_At_Entry": round(position["ath_at_entry"], 2),
                })
                position = None

        # ---- ENTRY CHECK ----
        if position is None and row["Signal"]:
            close  = row["close"]
            shares = int(PER_TRADE_CAPITAL / close)
            if shares == 0:
                continue

            ath    = row["ATH"]
            oliver = "Wedge" if row["Wedge_Pop"] else "Crossback"

            position = {
                "symbol"      : symbol,
                "entry_date"  : date,
                "entry_price" : close,
                "shares"      : shares,
                "invested"    : shares * close,
                "stop_price"  : close * (1 - STOP_LOSS_PCT),
                "target_price": ath * (1 + TARGET_ABOVE_HIGH_PCT),
                "ath_at_entry": ath,
                "signal_type" : f"ATH_{oliver}",
                "hold_bars"   : 0,
            }

    # Close any open position at end of data
    if position is not None:
        last_date, last_row = rows[-1]
        ep     = position["entry_price"]
        xp     = last_row["close"]
        shares = position["shares"]
        pnl_rs = (xp - ep) * shares
        trades.append({
            "Symbol"      : symbol,
            "Entry_Date"  : position["entry_date"].date(),
            "Exit_Date"   : last_date.date(),
            "Entry_Price" : round(ep, 2),
            "Exit_Price"  : round(xp, 2),
            "Shares"      : shares,
            "Invested_Rs" : round(position["invested"], 2),
            "PnL_Rs"      : round(pnl_rs, 2),
            "PnL_Pct"     : round((xp - ep) / ep * 100, 2),
            "Hold_Days"   : position["hold_bars"],
            "Signal_Type" : position["signal_type"],
            "Exit_Reason" : "End_Of_Data",
            "ATH_At_Entry": round(position["ath_at_entry"], 2),
        })

    return trades


# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def print_report(trades_df: pd.DataFrame):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_csv = f"ath_signal_trades_{ts}.csv"
    trades_df.to_csv(trades_csv, index=False)

    if trades_df.empty:
        print("  No trades generated.")
        return

    total  = len(trades_df)
    wins   = (trades_df["PnL_Rs"] > 0).sum()
    losses = (trades_df["PnL_Rs"] <= 0).sum()
    wr     = wins / total * 100

    gross_profit = trades_df.loc[trades_df["PnL_Rs"] > 0, "PnL_Rs"].sum()
    gross_loss   = trades_df.loc[trades_df["PnL_Rs"] <= 0, "PnL_Rs"].abs().sum()
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl    = trades_df["PnL_Rs"].sum()
    total_invest = trades_df["Invested_Rs"].sum()
    avg_win      = trades_df.loc[trades_df["PnL_Rs"] > 0, "PnL_Pct"].mean()
    avg_loss     = trades_df.loc[trades_df["PnL_Rs"] <= 0, "PnL_Pct"].mean()
    avg_hold     = trades_df["Hold_Days"].mean()

    best  = trades_df.loc[trades_df["PnL_Rs"].idxmax()]
    worst = trades_df.loc[trades_df["PnL_Rs"].idxmin()]

    trades_df["Year"] = pd.to_datetime(trades_df["Exit_Date"]).dt.year

    hdr = "=" * 70
    print(f"\n{hdr}")
    print("  ATH SIGNAL BACKTEST - PER-STOCK FIXED CAPITAL  (Yahoo Finance)")
    print(hdr)
    print(f"  CONFIGURATION:")
    print(f"  Per-Trade Capital  : Rs {PER_TRADE_CAPITAL:>10,}")
    print(f"  ATH Near %         : within {ATH_NEAR_PCT*100:.0f}% below ATH")
    print(f"  Stop Loss          : {STOP_LOSS_PCT*100:.0f}%")
    print(f"  Target             : {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ATH")
    print(f"  Fresh High Filter  : {FRESH_HIGH_DAYS} bars")
    print(f"  Lookback           : {LOOKBACK_DAYS} days")
    print()
    print(f"  PERFORMANCE SUMMARY:")
    print(f"  Total Trades       : {total:>12,}")
    print(f"  Win Rate           : {wr:>11.2f}%")
    print(f"  Profit Factor      : {pf:>12.2f}")
    print(f"  Total P&L          : Rs {total_pnl:>10,.0f}")
    print(f"  Total Invested     : Rs {total_invest:>10,.0f}")
    print(f"  Return on Invested : {total_pnl/total_invest*100 if total_invest>0 else 0:>11.2f}%")
    print(f"  Avg Win            : {avg_win:>11.2f}%")
    print(f"  Avg Loss           : {avg_loss:>11.2f}%")
    print(f"  Avg Hold Days      : {avg_hold:>11.1f}")
    print()
    print(f"  Best Trade   : {best['Symbol']:>10}  Rs {best['PnL_Rs']:>8,.0f}  ({best['PnL_Pct']:.1f}%)")
    print(f"  Worst Trade  : {worst['Symbol']:>10}  Rs {worst['PnL_Rs']:>8,.0f}  ({worst['PnL_Pct']:.1f}%)")

    print()
    print("  EXIT REASON BREAKDOWN:")
    for reason, grp in trades_df.groupby("Exit_Reason"):
        r_wr  = (grp["PnL_Rs"] > 0).mean() * 100
        r_avg = grp["PnL_Pct"].mean()
        sign  = "+" if r_avg >= 0 else ""
        print(f"    {reason:<22}: {len(grp):>4} trades  avg {sign}{r_avg:.1f}%  WR {r_wr:.1f}%")

    print()
    print("  SIGNAL TYPE BREAKDOWN:")
    for sig, grp in trades_df.groupby("Signal_Type"):
        s_wr  = (grp["PnL_Rs"] > 0).mean() * 100
        s_avg = grp["PnL_Pct"].mean()
        sign  = "+" if s_avg >= 0 else ""
        print(f"    {sig:<28}: {len(grp):>4} trades  avg {sign}{s_avg:.1f}%  WR {s_wr:.1f}%")

    print()
    print("  CALENDAR YEAR P&L:")
    print(f"    {'Year':<6} {'Trades':>7} {'Win%':>7} {'Total P&L':>14} {'Avg/Trade':>12}")
    print(f"    {'─'*50}")
    for yr, grp in trades_df.groupby("Year"):
        y_pnl = grp["PnL_Rs"].sum()
        y_wr  = (grp["PnL_Rs"] > 0).mean() * 100
        y_avg = grp["PnL_Rs"].mean()
        bar   = "+" * min(int(abs(y_pnl) / 5000), 20) if y_pnl >= 0 else "-" * min(int(abs(y_pnl) / 5000), 20)
        print(f"    {yr:<6} {len(grp):>7} {y_wr:>6.1f}%  Rs {y_pnl:>10,.0f}  Rs {y_avg:>8,.0f}  {bar}")

    print()
    print("  TOP 10 STOCKS BY TOTAL P&L:")
    per_stock = trades_df.groupby("Symbol").agg(
        Trades=("PnL_Rs", "count"),
        Total_PnL=("PnL_Rs", "sum"),
        WinRate=("PnL_Rs", lambda x: (x > 0).mean() * 100),
        Avg_Pct=("PnL_Pct", "mean"),
    ).sort_values("Total_PnL", ascending=False)

    print(f"    {'Symbol':<14} {'Trades':>6} {'WinRate':>8} {'Avg%':>7} {'Total P&L':>12}")
    print(f"    {'─'*52}")
    for sym, row in per_stock.head(10).iterrows():
        sign = "+" if row["Avg_Pct"] >= 0 else ""
        print(f"    {sym:<14} {int(row['Trades']):>6} {row['WinRate']:>7.1f}%  {sign}{row['Avg_Pct']:>5.1f}%  Rs {row['Total_PnL']:>8,.0f}")

    print()
    print("  BOTTOM 10 STOCKS BY TOTAL P&L:")
    print(f"    {'Symbol':<14} {'Trades':>6} {'WinRate':>8} {'Avg%':>7} {'Total P&L':>12}")
    print(f"    {'─'*52}")
    for sym, row in per_stock.tail(10).iterrows():
        sign = "+" if row["Avg_Pct"] >= 0 else ""
        print(f"    {sym:<14} {int(row['Trades']):>6} {row['WinRate']:>7.1f}%  {sign}{row['Avg_Pct']:>5.1f}%  Rs {row['Total_PnL']:>8,.0f}")

    print()
    print("  TOP 10 WINNING TRADES:")
    top10 = trades_df.nlargest(10, "PnL_Rs")[
        ["Symbol", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price",
         "PnL_Rs", "PnL_Pct", "Hold_Days", "Signal_Type", "Exit_Reason"]
    ]
    print(top10.to_string(index=False))

    print()
    print("  TOP 10 LOSING TRADES:")
    bot10 = trades_df.nsmallest(10, "PnL_Rs")[
        ["Symbol", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price",
         "PnL_Rs", "PnL_Pct", "Hold_Days", "Signal_Type", "Exit_Reason"]
    ]
    print(bot10.to_string(index=False))

    print()
    print(f"  Trades saved to: {trades_csv}")
    print(hdr)


# =============================================================================
# MAIN BACKTEST RUNNER
# =============================================================================
def run_backtest(use_all_nse: bool = True, tickers: list | None = None,
                 max_stocks: int | None = None):
    ts_start = time.time()

    print("=" * 70)
    print("  ATH SIGNAL BACKTEST  -  PER-STOCK, FIXED CAPITAL  (Yahoo Finance)")
    print(f"  EMA Alignment + Within {ATH_NEAR_PCT*100:.0f}% of ATH + Wedge_Pop | Crossback")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ATH")
    print(f"  Fresh High Filter: {FRESH_HIGH_DAYS} bars  |  Extension mult: {EXTENSION_MULT}x ATR")
    print(f"  Lookback: {LOOKBACK_DAYS} days  |  Fetch: {FETCH_YEARS} years")
    print(f"  Per-Trade Capital: Rs {PER_TRADE_CAPITAL:,}")
    print(f"  Batch size: {BATCH_SIZE} stocks/request  (no login required)")
    print("=" * 70)

    # Build universe
    print("\nBuilding stock universe...")
    if use_all_nse:
        universe = get_nse_universe()
        print(f"  Universe: ALL NSE-EQ stocks ({len(universe):,})")
    else:
        universe = list(tickers or NIFTY_500)
        print(f"  Universe: {len(universe):,} stocks (from provided list)")

    if max_stocks:
        universe = universe[:max_stocks]
        print(f"  Limited to first {max_stocks} stocks")

    # Fetch Nifty 50 for RS calculation
    print("\nFetching Nifty 50 index data for RS calculation...")
    nifty_close = fetch_nifty_close(FETCH_YEARS)
    if nifty_close is not None:
        print(f"  Nifty data: {len(nifty_close):,} bars")
    else:
        print("  WARNING: Nifty fetch failed. RS filter disabled.")

    # ---- PASS 1: Resolve cache / batch-fetch ----
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\n--- PASS 1: Loading data for {len(universe):,} stocks ---\n")

    stock_data   = {}   # symbol -> computed DataFrame
    to_fetch     = []   # symbols not found in any cache
    from_cache   = 0
    t0           = time.time()

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
                        # Copy fallback hit to primary cache
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

    # Batch-download what's not cached
    if to_fetch:
        batches  = [to_fetch[i:i+BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
        t1       = time.time()
        fetched  = 0
        failed   = 0

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
            loaded   = len(stock_data)
            print(
                f"  Batch [{bi:>4}/{len(batches)}]  "
                f"loaded so far: {loaded:>4}  "
                f"ETA: {eta_min:.1f} min"
            )

        elapsed_fetch = (time.time() - t1) / 60
        print(
            f"\n  Download done: {fetched:,} fetched, {failed:,} failed/empty "
            f"in {elapsed_fetch:.1f} min"
        )

    elapsed_1 = (time.time() - ts_start) / 60
    print(f"\n  Pass 1 done. {len(stock_data):,} stocks ready in {elapsed_1:.1f} min")

    if not stock_data:
        print("  No data loaded. Exiting.")
        return

    # ---- PASS 2: Per-stock simulation ----
    print(f"\n--- PASS 2: Simulating {len(stock_data):,} stocks independently ---\n")
    t2         = time.time()
    all_trades = []

    for i, (symbol, d) in enumerate(stock_data.items(), 1):
        trades = simulate_stock(symbol, d)
        all_trades.extend(trades)

        if i % 200 == 0 or i == len(stock_data):
            elapsed2 = time.time() - t2
            rate2    = i / elapsed2 if elapsed2 > 0 else 1
            eta2     = (len(stock_data) - i) / rate2 / 60
            print(
                f"  [{i:>5}/{len(stock_data)}] simulated  |  "
                f"{len(all_trades):,} trades so far  ETA: {eta2:.1f} min"
            )

    elapsed_2 = (time.time() - t2) / 60
    print(f"\n  Pass 2 done. {len(all_trades):,} trades in {elapsed_2:.1f} min")

    if not all_trades:
        print("  No trades generated.")
        return

    trades_df = pd.DataFrame(all_trades)
    print_report(trades_df)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # ── All NSE EQ stocks (~2500) — no login required ─────────────────────────
    # First run: ~2-3 min to batch-download via Yahoo Finance
    # Subsequent runs: < 30 sec from cache (7-day TTL)
    run_backtest(use_all_nse=True)

    # ── Nifty 500 only (~475 stocks) — quick backtest ─────────────────────────
    # run_backtest(use_all_nse=False, tickers=NIFTY_500)

    # ── Quick smoke test ──────────────────────────────────────────────────────
    # run_backtest(use_all_nse=True, max_stocks=100)
