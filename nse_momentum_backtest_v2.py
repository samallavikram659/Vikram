"""
=============================================================================
NSE MOMENTUM BACKTEST - NIFTY 500 UNIVERSE
=============================================================================
Uses SAME filters as the live scanner but runs historical backtest

UNIVERSE  : Nifty 500 stocks only (uses Nifty 500 list from NSE)
            Uses cached data from scanner runs

ENTRY     : Price > SMA10 > SMA20 > SMA50 > SMA150 > SMA200
            Price within 20% of ATH / 1Y High / 5Y High
            Top 3 by momentum (single ROC period - editable)

EXIT      : 8% stop loss | 20% profit target
REBALANCE : Monthly (first trading day of each month)
POSITIONS : 3 (equal weight)
COSTS     : Full NSE equity delivery charges + slippage

=============================================================================
"""

import os
import pickle
import time
import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# BACKTEST CONFIG
# =============================================================================
CACHE_DIR         = "cache_scanner"      # Use same cache as scanner
INITIAL_CAPITAL   = 1_00_000             # Rs 1 lakh starting capital
MAX_POSITIONS     = 3                     # Hold 3 stocks at a time
STOP_LOSS_PCT     = 0.08                  # 8% stop loss
TARGET_PCT        = 0.20                  # 20% profit target
NEAR_HIGH_PCT     = 0.20                  # Within 20% of high
MIN_PRICE         = 20                    # Minimum price filter
MIN_AVG_TURNOVER  = 10_00_000             # Min daily turnover Rs 10 lakh
FILTER_LOOKBACK   = 60                    # Days for quality filter
MIN_DATA_DAYS     = 252                   # Need 1 year of data

# Momentum settings - EDITABLE: Change ROC_PERIOD to use different momentum period
# Options: 20, 60, 90, or any other period you want
ROC_PERIOD = 60  # <-- EDIT THIS VALUE to change momentum period (days)

# Backtest period
BACKTEST_START    = "2020-01-01"
BACKTEST_END      = "2026-12-31"

# Costs
APPLY_COSTS       = True
SLIPPAGE_PCT      = 0.001                 # 0.1% slippage per side
# =============================================================================


# =============================================================================
# NIFTY 500 STOCKS LIST (as of 2024)
# This is the universe filter - only these stocks will be backtested
# =============================================================================
NIFTY_500_STOCKS = [
    # Nifty 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "BHARTIARTL",
    "SBIN", "BAJFINANCE", "KOTAKBANK", "ITC", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC", "ADANIENT",
    "POWERGRID", "DMART", "M&M", "BAJAJFINSV", "NESTLEIND", "TATAMOTORS", "JSWSTEEL",
    "ADANIPORTS", "TATASTEEL", "TECHM", "COALINDIA", "HINDALCO", "INDUSINDBK", "GRASIM",
    "BPCL", "CIPLA", "DRREDDY", "BRITANNIA", "DIVISLAB", "EICHERMOT", "APOLLOHOSP",
    "TATACONSUM", "HEROMOTOCO", "SBILIFE", "HDFCLIFE", "UPL", "LTIM",
    # Nifty Next 50
    "ADANIGREEN", "ADANITRANS", "AMBUJACEM", "AUROPHARMA", "BAJAJHLDNG", "BANKBARODA",
    "BERGEPAINT", "BOSCHLTD", "CADILAHC", "CHOLAFIN", "COLPAL", "DABUR", "DLF",
    "GAIL", "GODREJCP", "HAVELLS", "HDFCAMC", "HINDPETRO", "ICICIPRULI", "ICICIGI",
    "INDIGO", "IOC", "JUBLFOOD", "LTF", "LICI", "LUPIN", "MARICO", "MCDOWELL-N",
    "MOTHERSON", "MUTHOOTFIN", "NAUKRI", "NHPC", "NMDC", "PAGEIND", "PGHH", "PIDILITIND",
    "PNB", "SAIL", "SBI_CARD", "SBICARD", "SHREECEM", "SIEMENS", "SRF", "TATAPOWER",
    "TORNTPHARM", "TRENT", "VEDL", "VBL", "ZOMATO", "ZYDUSLIFE",
    # Nifty Midcap 150
    "AARTIIND", "ABB", "ABCAPITAL", "ABFRL", "ACC", "ADANIPOWER", "AJANTPHARM",
    "ALKEM", "ANGELONE", "APLLTD", "ASHOKLEY", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA",
    "BAJAJ-AUTO", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BEL", "BHEL", "BIOCON",
    "CANFINHOME", "CANBK", "CARBORUNIV", "CGPOWER", "CHAMBLFERT", "CLEAN", "COFORGE",
    "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND", "DEEPAKFERT", "DEEPAKNTR",
    "DELHIVERY", "DEVYANI", "DIXON", "ELGIEQUIP", "EMAMILTD", "ENDURANCE", "ENGINERSIN",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "FINCABLES", "FLUOROCHEM", "FSL", "GICRE",
    "GLAXO", "GMRAIRPORT", "GNFC", "GODREJIND", "GODREJPROP", "GRANULES", "GSFC",
    "GSPL", "GUJGASLTD", "HAL", "HONASA", "HONAUT", "IBREALEST", "IDFCFIRSTB", "IEX",
    "IIFL", "INDHOTEL", "INDUSTOWER", "INTELLECT", "IOB", "IPCALAB", "IRB", "IRCTC",
    "IRFC", "IGL", "JKCEMENT", "JKLAKSHMI", "JSWENERGY", "JSWINFRA", "JINDALSTEL",
    "JSL", "JUBLINGREA", "KAJARIACER", "KALYANKJIL", "KANSAINER", "KEI", "KIOCL",
    "KPITTECH", "KRBL", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LINDEINDIA",
    "LODHA", "LTI", "LTTS", "M&MFIN", "MAHINDCIE", "MANAPPURAM", "MASFIN", "MAXHEALTH",
    "MFSL", "MGL", "MINDTREE", "MPHASIS", "MRF", "NAM-INDIA", "NATIONALUM", "NAVINFLUOR",
    "NIACL", "NCC", "NESCO", "NHPC", "NLCINDIA", "OBEROIRLTY", "OIL", "OFSS",
    "PATANJALI", "PAYTM", "PERSISTENT", "PETRONET", "PFC", "PFIZER", "PHOENIXLTD",
    "PIIND", "PEL", "POLICYBZR", "POLYCAB", "POONAWALLA", "PRESTIGE", "PRINCEPIPE",
    "PVRINOX", "RADICO", "RAJESHEXPO", "RAMCOCEM", "RATNAMANI", "RAYMOND", "RBA",
    "RECLTD", "REDINGTON", "RELAXO", "RKFORGE", "ROUTE", "SANOFI", "SAPPHIRE",
    "SCHAEFFLER", "SHILPAMED", "SHOPERSTOP", "SJVN", "SKFINDIA", "SONACOMS", "SONATSOFTW",
    "STARHEALTH", "SUMICHEM", "SUNCLAYLTD", "SUNDARMFIN", "SUNDRMFAST", "SUPREMEIND",
    "SUVENPHAR", "SWANENERGY", "SYNGENE", "TATACOMM", "TATAELXSI", "TATAINVEST",
    "TCIEXP", "TEJASNET", "THERMAX", "TIINDIA", "TIMKEN", "TMB", "TORNTPOWER",
    "TRIDENT", "TRIVENI", "TTML", "TUBE", "TV18BRDCST", "TVSMOTOR", "UBL", "UNIONBANK",
    "UNOMINDA", "UPL", "UTIAMC", "VAIBHAVGBL", "VAKRANGEE", "VARROC", "VEDL",
    "VINATIORGA", "VOLTAS", "VGUARD", "VSTIND", "WELCORP", "WELSPUNIND", "WESTLIFE",
    "WHIRLPOOL", "WINDMACHIN", "WOCKPHARMA", "YESBANK", "ZEEL", "ZENSAR", "ZFCVINDIA",
    # Nifty Smallcap 250 (partial - major ones)
    "3MINDIA", "AARTIDRUGS", "AAVAS", "ACE", "ADVENZYMES", "AEGISLOG", "AETHER",
    "AFFLE", "AIAENG", "AJMERA", "AKZOINDIA", "ALKYLAMINE", "ALLCARGO", "ALOKINDS",
    "AMARAJABAT", "AMBER", "ANANTRAJ", "ANDHRAPET", "ANURAS", "APARINDS", "APLAPOLLO",
    "APOLLOPIPE", "APTECHT", "APTUS", "ARCHIDPLY", "ARVINDFASN", "ASAHIINDIA", "ASHIANA",
    "ASKAUTOLTD", "ASTEC", "ASTRAZEN", "ATFL", "ATUL", "AVANTIFEED", "AXISCADES",
    "BALAMINES", "BASF", "BBTC", "BCG", "BEML", "BIRLACORPN", "BORORENEW", "BLS",
    "BLUESTARCO", "BOMDYEING", "BRIGADE", "BSE", "BSOFT", "CAMPUS", "CANFINHOME",
    "CAPACITE", "CARERATING", "CASTROLIND", "CCL", "CDSL", "CENTURYPLY", "CERA",
    "CHALET", "CHEMCON", "CMSINFO", "COCHINSHIP", "CONTROLPRI", "CREDITACC", "CRISIL",
    "CYIENT", "DATAPATTNS", "DCAL", "DCBBANK", "DCMSHRIRAM", "DELTACORP", "DHAMPURSUG",
    "DODLA", "DREAMFOLKS", "ECLERX", "EDELWEISS", "EIDPARRY", "ELECON", "ELECTCAST",
    "ELIN", "EPL", "EQUITAS", "EQUITASBNK", "ERIS", "ESABINDIA", "EVEREADY",
    "EXPLEOSOL", "FACT", "FAIRCHEM", "FAZE3Q", "FDC", "FINPIPE", "FIVESTAR",
    "GABRIEL", "GALAXYSURF", "GARFIBRES", "GATEWAY", "GENUSPOWER", "GILLETTE",
    "GLAND", "GLENMARK", "GLOBUSSPR", "GLS", "GMDCLTD", "GMMPFAUDLR", "GODFRYPHLP",
    "GOKEX", "GOLDIAM", "GPIL", "GPPL", "GREAVESCOT", "GREENPANEL", "GREENLAM",
    "GRINDWELL", "GRSE", "GTLINFRA", "GUFICBIO", "HAPPSTMNDS", "HARSHA", "HATSUN",
    "HBLPOWER", "HCG", "HDFCBANK", "HEIDELBERG", "HEMIPROP", "HERITGFOOD", "HFCL",
    "HGS", "HIKAL", "HIL", "HLEGLAS", "HMT", "HOMEFIRST", "HSCL", "HUDCO",
    "IBULHSGFIN", "ICRA", "IDBI", "IDFC", "IGARASHI", "IGPL", "IMAGICAA", "IMFA",
    "INDIAGLYCO", "INDIAMART", "INDIGO", "INDNIPPON", "INFIBEAM", "INGERRAND",
    "INOXGREEN", "INOXLEISUR", "INOXWIND", "INSECTIND", "INTELLECT", "IONEXCHANG",
    "IRCON", "ISEC", "ITI", "J&KBANK", "JAMNAAUTO", "JAYNECOIND", "JBCHEPHARM",
    "JCHAC", "JISLJALEQS", "JKPAPER", "JKTYRE", "JMFINANCIL", "JMC", "JPASSOCIAT",
    "JPPOWER", "JSLHISAR", "JUBILANT", "JUSTDIAL", "JYOTHYLAB", "KABRAEXTRU",
    "KAJARIACER", "KALPATPOWR", "KARDA", "KDDL", "KEC", "KENNAMET", "KESORAMIND",
    "KEYFINSERV", "KINGFA", "KIRIINDUS", "KIRLOSENG", "KIRLOSBROS", "KNRCON",
    "KOLTEPATIL", "KOPRAN", "KPRMILL", "KRBL", "KSCL", "KTKNEON", "LAOPALA",
    "LATENTVIEW", "LAXMIMACH", "LEMONTREE", "LGBBROSLTD", "LINCOLN", "LUXIND",
    "MAHABANK", "MAHLIFE", "MAHLOG", "MAHSCOOTER", "MAHSEAMLES", "MANINFRA",
    "MANKIND", "MAPMYINDIA", "MARATHON", "MARKSANS", "MASTEK", "MAYURUNIQ",
    "MAZAGON", "MEDANTA", "MEDPLUS", "METROPOLIS", "MHRIL", "MIDHANI", "MINDACORP",
    "MIRZAINT", "MMTC", "MOIL", "MOLDTKPAC", "MONTECARLO", "MOREPENLAB", "MOTILALOFS",
    "MSTCLTD", "MTARTECH", "MUKANDLTD", "MUNJALSHOW", "NATCOPHARM", "NBCC", "NCC",
    "NCLIND", "NDL", "NDTV", "NELCO", "NETWORK18", "NEWGEN", "NFL", "NILKAMAL",
    "NIITMTS", "NITINSPIN", "NOCIL", "NPST", "NRBBEARING", "NUCLEUS", "NURECA",
    "NUVAMA", "OLECTRA", "OMAXE", "ONMOBILE", "OPTIEMUS", "ORCHPHARMA", "ORIENTELEC",
    "ORIENTPPR", "ORIENTREF", "PAISALO", "PARADEEP", "PARAS", "PATEL", "PCBL",
    "PDSL", "PENIND", "PFIZER", "PFS", "PGHL", "PGINVIT", "PNCINFRA", "PNBHOUSING",
    "POLYPLEX", "POWERINDIA", "PPLPHARMA", "PRAXIS", "PREMEXPLN", "PRICOLLTD",
    "PRSMJOHNSN", "PSB", "PSPPROJECT", "PURVA", "QUESS", "RADICO", "RAIN",
    "RAJRATAN", "RALLIS", "RAMKYLAM", "RANEHOLDIN", "RATEGAIN", "RATNAMANI",
    "RAYMOND", "RBL", "RBLBANK", "RCF", "RECLTD", "REDTAPE", "REFEX", "RELAXO",
    "RENAISSANCE", "RESPONIND", "RHI", "RHIM", "RITES", "RKEC", "ROLEXRINGS",
    "ROSSARI", "ROSSELLIND", "ROUTE", "RPGLIFE", "RPSGVENT", "RSWM", "RTNINDIA",
    "RTNPOWER", "RVNL", "S&SPOWER", "SADHNANIQ", "SAFARI", "SAGCEM", "SALASAR",
    "SANDHAR", "SANDUMA", "SANGHIIND", "SANGHVIMOV", "SANOFI", "SARDAEN", "SASKEN",
    "SATIN", "SATINDLTD", "SCI", "SEQUENT", "SHAKTIPUMP", "SHALBY", "SHALPAINTS",
    "SHANKARA", "SHARDACROP", "SHAREINDIA", "SHK", "SHOPERSTOP", "SHREECEM",
    "SHRIRAMCIT", "SHRIRAMPPS", "SIL", "SINTERCOM", "SIS", "SJVN", "SKIPPER",
    "SMCGLOBAL", "SNOWMAN", "SOBHA", "SOLARA", "SOLARINDS", "SONACOMS", "SOUTHBANK",
    "SPANDANA", "SPARC", "SPENCERS", "SPIC", "SRHHYPOLTD", "SRTRANSFIN", "STAR",
    "STCINDIA", "STLTECH", "STYRENIX", "SUBROS", "SUDARSCHEM", "SUMIT", "SUNDARMHLD",
    "SUNFLAG", "SUPERHOUSE", "SUPRAJIT", "SUPREMEENG", "SURYAROSNI", "SUULD",
    "SUVENPHAR", "SUVEN", "SUZLON", "SVPGLOB", "SWARAJENG", "SYMPHONY", "SYRMA",
    "TANLA", "TARSONS", "TATACHEM", "TATACOFFEE", "TATAMETALI", "TATASPONGE",
    "TATATECH", "TBZ", "TCNSBRANDS", "TCIEXP", "TCPLPACK", "TDPOWERSYS", "TEAMLEASE",
    "TECHNOE", "TEGA", "TEXMOPIPES", "TEXRAIL", "THANGAMAYL", "THERMAX", "THYROCARE",
    "TI", "TIDEWATER", "TIIL", "TIMETECHNO", "TINPLATE", "TITAN", "TNPL", "TNTELE",
    "TCFC", "TRANSPEK", "TRENT", "TRF", "TRIL", "TRITURBINE", "TRIVENI", "TTKHLTCARE",
    "TTKPRESTIG", "TV18BRDCST", "TVSSRICHAK", "TVTODAY", "TWL", "UBL", "UCOBANK",
    "UFLEX", "ULTRACEMCO", "UNICHEMLAB", "UNIPARTS", "UNIONBANK", "UNIVASTU",
    "UNIVCABLES", "UNOMINDA", "USHA", "UTIAMC", "UTKARSHBNK", "UTTAMSUGAR", "VADILALIND",
    "VAIBHAVGBL", "VALIANTORG", "VASCONEQ", "VBL", "VEDL", "VENKEYS", "VENUSPIPES",
    "VERANDA", "VESUVIUS", "VGUARD", "VIDHIING", "VIJAYA", "VIKAS", "VIKASECO",
    "VINATIORGA", "VIPIND", "VIPULLTD", "VISHNU", "VLSFINANCE", "VOLTAMP", "VOLTAS",
    "VRLLOG", "VSTIND", "WABAG", "WABCOINDIA", "WANBURY", "WELCORP", "WELENT",
    "WELSPUNLIV", "WESTLIFE", "WEWIN", "WHEELS", "WHIRLPOOL", "WINDMACHIN", "WIPRO",
    "WOCKPHARMA", "WONDERLA", "WPIL", "XCHANGING", "XPROINDIA", "YASHO", "YESBANK",
    "ZEEL", "ZENSAR", "ZENTEC", "ZEEMEDIA", "ZENSARTECH", "ZFCVINDIA", "ZODIAC",
    "ZOMATO", "ZODJRDMKJ", "ZYDUSLIFE", "ZYDUSWELL",
]
# =============================================================================


# =============================================================================
# TRANSACTION COSTS (NSE Delivery)
# =============================================================================
def calc_cost(trade_value, side):
    """Calculate NSE delivery transaction costs"""
    if not APPLY_COSTS:
        return 0.0
    brokerage = min(trade_value * 0.001, 20.0)  # 0.1% or Rs 20 max
    exchange_charges = trade_value * 0.0000335
    sebi_charges = trade_value * 0.000001
    gst = (brokerage + exchange_charges) * 0.18
    stt = trade_value * 0.001 if side == "sell" else trade_value * 0.00015
    return brokerage + exchange_charges + sebi_charges + gst + stt


# =============================================================================
# LOAD CACHED DATA
# =============================================================================
def load_all_cached_data():
    """Load cached stock data for Nifty 500 stocks only"""
    if not os.path.exists(CACHE_DIR):
        print(f"  ERROR: Cache directory '{CACHE_DIR}' not found!")
        print("  Please run the scanner first to download stock data.")
        raise SystemExit(1)

    # Find all cached stock files
    stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith("_1D.pkl")]
    print(f"  Found {len(stock_files):,} total cached stock files")

    # Filter for Nifty 500 stocks only
    nifty500_set = set(NIFTY_500_STOCKS)
    nifty500_files = [f for f in stock_files if f.replace("_1D.pkl", "") in nifty500_set]
    print(f"  Filtering to Nifty 500 universe: {len(nifty500_files)} stocks found in cache")

    if len(nifty500_files) < 50:
        print("  WARNING: Very few Nifty 500 stocks cached. Run scanner first!")

    price_data = {}
    skipped = 0

    for i, filename in enumerate(nifty500_files, 1):
        symbol = filename.replace("_1D.pkl", "")

        if i % 100 == 0:
            print(f"  Loading: {i:,}/{len(nifty500_files):,}  ({symbol})  ", end="\r")

        try:
            filepath = os.path.join(CACHE_DIR, filename)
            with open(filepath, "rb") as f:
                df = pickle.load(f)

            if df is None or df.empty or len(df) < MIN_DATA_DAYS:
                skipped += 1
                continue

            # Compute indicators
            df = compute_indicators(df)
            if df is not None and len(df) > 0:
                price_data[symbol] = df

        except Exception as e:
            skipped += 1
            continue

    print(f"\n  Loaded: {len(price_data):,} Nifty 500 stocks with sufficient data")
    print(f"  Skipped: {skipped:,} (insufficient data or errors)")

    return price_data


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
    df["ath"] = df["close"].expanding().max()
    df["high_1y"] = df["close"].rolling(252, min_periods=200).max()
    df["high_5y"] = df["close"].rolling(252 * 5, min_periods=252).max()

    # Single ROC period (configurable via ROC_PERIOD variable)
    df["momentum"] = df["close"].pct_change(ROC_PERIOD) * 100

    df.dropna(subset=["sma200", "momentum"], inplace=True)

    return df if len(df) > 0 else None


# =============================================================================
# ENTRY FILTER (Same as scanner)
# =============================================================================
def passes_entry(row):
    """Check if stock passes entry criteria"""
    price = row["close"]
    sma10 = row["sma10"]
    sma20 = row["sma20"]
    sma50 = row["sma50"]
    sma150 = row["sma150"]
    sma200 = row["sma200"]

    # SMA alignment check
    if not (price > sma10 > sma20 > sma50 > sma150 > sma200):
        return False, None

    # Near high check
    ath = row["ath"]
    high_1y = row["high_1y"]
    high_5y = row["high_5y"]

    dist_ath = (ath - price) / ath if ath > 0 else 999
    dist_1y = (high_1y - price) / high_1y if high_1y > 0 else 999
    dist_5y = (high_5y - price) / high_5y if high_5y > 0 else 999

    if dist_ath <= NEAR_HIGH_PCT:
        return True, "ATH"
    elif dist_1y <= NEAR_HIGH_PCT:
        return True, "1Y"
    elif dist_5y <= NEAR_HIGH_PCT:
        return True, "5Y"

    return False, None


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def run_backtest(price_data):
    """Run the momentum backtest"""
    # Get all trading dates
    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    all_dates = [d for d in all_dates if BACKTEST_START <= str(d.date()) <= BACKTEST_END]

    if not all_dates:
        print("  ERROR: No trading dates in backtest period!")
        return pd.DataFrame(), pd.DataFrame(), 0

    # Identify month starts for rebalancing
    month_starts = set(
        pd.Series(all_dates, index=pd.DatetimeIndex(all_dates))
        .resample("MS").first().dropna().tolist()
    )

    # Initialize
    cash = float(INITIAL_CAPITAL)
    positions = {}  # {symbol: {entry_date, entry_price, shares, stop, target, buy_cost, high_type}}
    trades = []
    equity = []
    total_costs = 0.0

    def get_price(sym, date):
        """Get last available price for symbol on or before date"""
        if sym not in price_data:
            return None
        sub = price_data[sym][price_data[sym].index <= date]
        return float(sub["close"].iloc[-1]) if not sub.empty else None

    def calc_nav(date):
        """Calculate total portfolio value"""
        nav = cash
        for sym, pos in positions.items():
            px = get_price(sym, date)
            if px:
                nav += px * pos["shares"]
        return nav

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST: {BACKTEST_START} -> {BACKTEST_END}")
    print(f"  Universe: {len(price_data):,} Nifty 500 stocks")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"  Momentum: {ROC_PERIOD}-day ROC | Positions: {MAX_POSITIONS}")
    print(f"  Stop Loss: {STOP_LOSS_PCT*100:.0f}% | Target: {TARGET_PCT*100:.0f}%")
    print(f"{'=' * 70}\n")

    total_days = len(all_dates)
    last_pct = -1

    for idx, date in enumerate(all_dates):
        pct = int(idx / total_days * 100)
        if pct % 5 == 0 and pct != last_pct:
            print(f"  {pct:3d}%  {str(date.date())}  Pos:{len(positions)}  Cash:Rs {cash:,.0f}  NAV:Rs {calc_nav(date):,.0f}    ", end="\r")
            last_pct = pct

        # =====================================================================
        # CHECK EXITS (Stop Loss / Target)
        # =====================================================================
        for sym in list(positions.keys()):
            pos = positions[sym]
            px = get_price(sym, date)
            if px is None:
                continue

            exit_reason = None

            # Stop loss check (8%)
            if px <= pos["stop"]:
                exit_reason = "STOP_LOSS"
            # Target check (20%)
            elif px >= pos["target"]:
                exit_reason = "TARGET_HIT"

            if exit_reason:
                exit_price = px * (1 - SLIPPAGE_PCT)  # Sell slippage
                sell_value = exit_price * pos["shares"]
                sell_cost = calc_cost(sell_value, "sell")
                pnl_gross = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_net = pnl_gross - sell_cost - pos["buy_cost"]
                pnl_pct = pnl_net / (pos["entry_price"] * pos["shares"]) * 100

                cash += sell_value - sell_cost
                total_costs += sell_cost

                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "shares": pos["shares"],
                    "pnl_gross": pnl_gross,
                    "pnl_net": pnl_net,
                    "pnl_pct": pnl_pct,
                    "total_cost": sell_cost + pos["buy_cost"],
                    "stop": pos["stop"],
                    "target": pos["target"],
                    "high_type": pos["high_type"],
                    "exit_reason": exit_reason,
                })

                del positions[sym]

        # =====================================================================
        # CHECK ENTRIES (Monthly rebalancing)
        # =====================================================================
        if date in month_starts:
            slots = MAX_POSITIONS - len(positions)
            if slots > 0:
                # Find all candidates
                candidates = []
                for sym, df in price_data.items():
                    if sym in positions:
                        continue
                    sub = df[df.index <= date]
                    if sub.empty:
                        continue
                    row = sub.iloc[-1]
                    passes, high_type = passes_entry(row)
                    if passes:
                        candidates.append({
                            "symbol": sym,
                            "momentum": float(row["momentum"]),  # Single ROC period
                            "price": float(row["close"]),
                            "high_type": high_type,
                        })

                # Sort by composite momentum (highest first)
                candidates.sort(key=lambda x: x["momentum"], reverse=True)

                # Take top N candidates
                allocation = cash * 0.99 / slots if slots > 0 else 0

                for cand in candidates[:slots]:
                    sym = cand["symbol"]
                    price = cand["price"]
                    entry_price = price * (1 + SLIPPAGE_PCT)  # Buy slippage

                    shares = int(allocation / entry_price)
                    if shares < 1:
                        continue

                    buy_value = shares * entry_price
                    buy_cost = calc_cost(buy_value, "buy")

                    if buy_value + buy_cost > cash:
                        continue

                    cash -= (buy_value + buy_cost)
                    total_costs += buy_cost

                    positions[sym] = {
                        "entry_date": date,
                        "entry_price": entry_price,
                        "shares": shares,
                        "stop": round(entry_price * (1 - STOP_LOSS_PCT), 2),
                        "target": round(entry_price * (1 + TARGET_PCT), 2),
                        "buy_cost": buy_cost,
                        "high_type": cand["high_type"],
                    }

        # Record daily equity
        equity.append({"date": date, "equity": calc_nav(date)})

    # =========================================================================
    # CLOSE REMAINING POSITIONS
    # =========================================================================
    last_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        px = get_price(sym, last_date)
        if px is None:
            px = pos["entry_price"]

        exit_price = px * (1 - SLIPPAGE_PCT)
        sell_value = exit_price * pos["shares"]
        sell_cost = calc_cost(sell_value, "sell")
        pnl_gross = (exit_price - pos["entry_price"]) * pos["shares"]
        pnl_net = pnl_gross - sell_cost - pos["buy_cost"]
        pnl_pct = pnl_net / (pos["entry_price"] * pos["shares"]) * 100

        cash += sell_value - sell_cost
        total_costs += sell_cost

        trades.append({
            "symbol": sym,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "shares": pos["shares"],
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "pnl_pct": pnl_pct,
            "total_cost": sell_cost + pos["buy_cost"],
            "stop": pos["stop"],
            "target": pos["target"],
            "high_type": pos["high_type"],
            "exit_reason": "END_OF_BACKTEST",
        })

    print(f"\n  100% complete                                                      ")

    # Create DataFrames
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity).set_index("date")
    equity_df["equity"] = equity_df["equity"].ffill()

    return trades_df, equity_df, total_costs


# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def report(trades_df, equity_df, total_costs):
    """Generate performance report"""
    if trades_df.empty:
        print("\n  No trades executed.")
        return

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Drawdown
    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Sharpe ratio
    daily_returns = equity_df["equity"].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # Calmar ratio
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

    # Win/Loss stats
    wins = trades_df[trades_df["pnl_net"] > 0]
    losses = trades_df[trades_df["pnl_net"] <= 0]
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    profit_factor = abs(wins["pnl_net"].sum() / losses["pnl_net"].sum()) if losses["pnl_net"].sum() != 0 else float("inf")
    avg_hold = (trades_df["exit_date"] - trades_df["entry_date"]).dt.days.mean()

    sep = "=" * 70
    print(f"\n{sep}")
    print("  BACKTEST PERFORMANCE REPORT")
    print(sep)
    print(f"  Period          : {equity_df.index[0].date()}  to  {equity_df.index[-1].date()}")
    print(f"  Initial Capital : Rs {INITIAL_CAPITAL:>14,.0f}")
    print(f"  Final Equity    : Rs {final_equity:>14,.0f}")
    print(f"  Total Return    : {total_return:>12.2f}%")
    print(f"  CAGR            : {cagr:>12.2f}%")
    print(f"  Max Drawdown    : {max_dd:>12.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:>12.2f}")
    print(f"  Calmar Ratio    : {calmar:>12.2f}")
    print(f"  Total Costs     : Rs {total_costs:>12,.0f}")
    print("-" * 70)
    print(f"  Total Trades    : {len(trades_df):>12}")
    print(f"  Win Rate        : {win_rate:>12.2f}%")
    if len(wins) > 0:
        print(f"  Avg Win (net)   : {wins['pnl_pct'].mean():>11.2f}%")
    if len(losses) > 0:
        print(f"  Avg Loss (net)  : {losses['pnl_pct'].mean():>11.2f}%")
    print(f"  Profit Factor   : {profit_factor:>12.2f}")
    print(f"  Avg Hold Days   : {avg_hold:>12.1f}")
    print("-" * 70)

    # Exit reason breakdown
    print("\n  EXIT REASON BREAKDOWN:")
    for reason, cnt in trades_df["exit_reason"].value_counts().items():
        avg_pnl = trades_df[trades_df["exit_reason"] == reason]["pnl_pct"].mean()
        print(f"    {reason:<20}: {cnt:4d} trades  avg {avg_pnl:>+6.1f}%")

    # High type breakdown
    print("\n  ENTRY HIGH TYPE BREAKDOWN:")
    for ht, cnt in trades_df["high_type"].value_counts().items():
        avg_pnl = trades_df[trades_df["high_type"] == ht]["pnl_pct"].mean()
        print(f"    {ht:<5}: {cnt:4d} trades  avg {avg_pnl:>+6.1f}%")

    print(sep)

    # Top trades
    cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "exit_reason"]
    print("\n  TOP 10 WINNING TRADES:")
    print(trades_df.nlargest(10, "pnl_net")[cols].to_string(index=False))
    print("\n  TOP 10 LOSING TRADES:")
    print(trades_df.nsmallest(10, "pnl_net")[cols].to_string(index=False))

    # Monthly equity
    eq_monthly = equity_df["equity"].resample("ME").last().dropna()
    print(f"\n  {'Date':<12}  {'Equity':>14}  {'MoM%':>7}  Chart")
    print("  " + "-" * 58)
    prev = float(INITIAL_CAPITAL)
    for dt, val in eq_monthly.items():
        ret = (val / prev - 1) * 100
        bar = ("+" if ret >= 0 else "-") * min(50, int(abs(ret) * 2))
        print(f"  {str(dt.date()):<12}  Rs {val:>12,.0f}  {ret:>+6.1f}%  {bar}")
        prev = val

    # Export
    trades_df.to_csv("backtest_trades.csv", index=False)
    equity_df.to_csv("backtest_equity.csv")

    try:
        trades_df.to_excel("backtest_trades.xlsx", index=False, sheet_name="Trades")
        equity_df.to_excel("backtest_equity.xlsx", sheet_name="Equity")
        print(f"\n  Exported: backtest_trades.csv/.xlsx | backtest_equity.csv/.xlsx")
    except:
        print(f"\n  Exported: backtest_trades.csv | backtest_equity.csv")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  NSE MOMENTUM BACKTEST - NIFTY 500 UNIVERSE")
    print("  Strategy: SMA alignment + Near High + Momentum")
    print(f"  Momentum Period: {ROC_PERIOD}-day ROC  (edit ROC_PERIOD to change)")
    print(f"  Stop Loss: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_PCT*100:.0f}%  |  Positions: {MAX_POSITIONS}")
    print("=" * 70)

    # Load cached data from scanner (Nifty 500 only)
    print("\nLoading Nifty 500 stock data from cache...")
    price_data = load_all_cached_data()

    if not price_data:
        print("\n  ERROR: No stock data found. Run the scanner first!")
        raise SystemExit(1)

    # Run backtest
    trades_df, equity_df, total_costs = run_backtest(price_data)

    # Generate report
    report(trades_df, equity_df, total_costs)


if __name__ == "__main__":
    main()
