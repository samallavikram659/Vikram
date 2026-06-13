"""
=============================================================================
NSE MOMENTUM BACKTEST - ALL NSE STOCKS (NO ETF/LIQUID/INDICES)
=============================================================================
UNIVERSE  : All NSE EQ-series stocks (~2500 from Angel One scrip master)
            Excludes: ETFs, Liquid Funds, Bonds, Index Funds, BE series

ENTRY     : Price > SMA10 > SMA20 > SMA50 > SMA150 > SMA200
            Price within X% of ATH / 1Y High / 5Y High
            Top 3 by momentum (single ROC period - editable)

EXIT      : Stop loss | Profit target
REBALANCE : Monthly (first trading day of each month)
POSITIONS : 3 (equal weight)
COSTS     : Full NSE equity delivery charges + slippage

NOTE      : First run fetches data from Angel One (~6-10 hours)
            Subsequent runs use cached data (instant)
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

from SmartApi import SmartConnect
import pyotp

# =============================================================================
# CREDENTIALS - UPDATE THESE
# =============================================================================
API_KEY      = os.environ.get("ANGEL_API_KEY", "Tp7PJMIc")
CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "S63068620")
PASSWORD     = os.environ.get("ANGEL_PASSWORD", "4098")
TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "KVTELUFGR33YCPQUKO3ZSL4NMA")
# =============================================================================

# =============================================================================
# BACKTEST CONFIG - EDIT THESE VALUES
# =============================================================================
CACHE_DIR         = "cache_scanner"      # Cache folder for stock data
INITIAL_CAPITAL   = 1_00_000             # Rs 1 lakh starting capital
MAX_POSITIONS     = 3                     # Hold 3 stocks at a time
STOP_LOSS_PCT     = 0.05                  # 5% stop loss
TARGET_PCT        = 0.10                  # 10% profit target
NEAR_HIGH_PCT     = 0.05                  # Within 5% of high
MIN_PRICE         = 20                    # Minimum price filter
MIN_AVG_TURNOVER  = 10_00_000             # Min daily turnover Rs 10 lakh
FILTER_LOOKBACK   = 60                    # Days for quality filter
MIN_DATA_DAYS     = 252                   # Need 1 year of data

# Momentum settings - EDITABLE
ROC_PERIOD = 20  # <-- EDIT THIS VALUE (20, 60, 90, etc.)

# How many years of data to fetch
FETCH_YEARS = 11  # Fetches from 2015 onwards

# Backtest period
BACKTEST_START    = "2016-01-01"
BACKTEST_END      = "2026-12-31"

# Costs
APPLY_COSTS       = True
SLIPPAGE_PCT      = 0.001                 # 0.1% slippage per side
# =============================================================================

os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# EXCLUSION PATTERNS FOR ETFs, LIQUID FUNDS, BONDS, INDICES
# =============================================================================
ETF_LIQUID_PATTERNS = [
    "ETF", "BEES", "LIQUID", "GOLD", "SILVER", "NIFTY", "BANK", "GILT",
    "GSEC", "BOND", "CPSE", "PSU", "INFRA", "NEXT50", "MIDCAP", "SMALLCAP",
    "CONSUMPTION", "DIVIDEND", "GROWTH", "VALUE", "MOMENTUM", "QUALITY",
    "LOWVOL", "ALPHA", "EQUAL", "SHARIAH", "ESG", "HEALTHCARE", "IT",
    "PRIVATE", "SETF", "NETF", "IETF", "CASE", "ADD", "PLUS", "SHRI",
    "LIQUIDBETF", "LIQUIDCASE", "LIQUIDPLUS", "CASHIETF", "LIQUIDADD",
    "HDFCLIQUID", "LIQUIDSHRI", "GROWWLIQID", "LIQUID1", "AONELIQUID",
    "EBBETF", "GILT5YBEES", "GSEC10YEAR", "GSEC5IETF",
]

def is_etf_or_liquid(symbol):
    sym_upper = symbol.upper()
    for pattern in ETF_LIQUID_PATTERNS:
        if pattern in sym_upper:
            return True
    return False


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
# GET ALL NSE STOCKS FROM ANGEL ONE SCRIP MASTER
# =============================================================================
def get_all_nse_stocks():
    cache_path = os.path.join(CACHE_DIR, "all_nse_stocks.pkl")

    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 24:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            filtered = {k: v for k, v in data.items() if not is_etf_or_liquid(k)}
            print(f"  Loaded {len(filtered):,} NSE stocks from cache ({age_hours:.1f}h old)")
            return filtered
        print("  Cache >24h old — refreshing...")

    import requests
    print("  Downloading Angel One scrip master (all NSE stocks)...")

    try:
        resp = requests.get(
            "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
            timeout=60
        )
        df = pd.DataFrame(resp.json())

        nse = df[df["exch_seg"] == "NSE"].copy()
        print(f"    Total NSE instruments: {len(nse):,}")

        eq = nse[nse["symbol"].str.endswith("-EQ", na=False)].copy()
        print(f"    NSE-EQ instruments: {len(eq):,}")

        eq["sym"] = eq["symbol"].str.replace("-EQ", "", regex=False).str.strip()
        stock_map = dict(zip(eq["sym"], eq["token"]))

        stock_map_filtered = {k: v for k, v in stock_map.items() if not is_etf_or_liquid(k)}
        excluded = len(stock_map) - len(stock_map_filtered)
        print(f"    After excluding ETFs/liquid funds: {len(stock_map_filtered):,} stocks")

        with open(cache_path, "wb") as f:
            pickle.dump(stock_map, f)

        return stock_map_filtered

    except Exception as e:
        print(f"  ERROR downloading scrip master: {e}")
        raise SystemExit(1)


# =============================================================================
# OHLC DATA FETCH WITH RETRY LOGIC
# =============================================================================
def fetch_ohlc(obj, token, symbol, from_date, to_date, max_retries=3):
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_1D.pkl")

    if os.path.exists(cache_path):
        cache_age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if cache_age_days < 1:
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
            if not df.empty:
                return df

    rows = []
    cur = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)

    while cur <= end:
        nxt = min(cur + pd.DateOffset(days=400), end)

        for attempt in range(max_retries):
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
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        cur = nxt + pd.DateOffset(days=1)
        time.sleep(0.4)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    df.index = df.index.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)

    if not df.empty:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

    return df


# =============================================================================
# TRANSACTION COSTS (NSE Delivery)
# =============================================================================
def calc_cost(trade_value, side):
    if not APPLY_COSTS:
        return 0.0
    brokerage = min(trade_value * 0.001, 20.0)
    exchange_charges = trade_value * 0.0000335
    sebi_charges = trade_value * 0.000001
    gst = (brokerage + exchange_charges) * 0.18
    stt = trade_value * 0.001 if side == "sell" else trade_value * 0.00015
    return brokerage + exchange_charges + sebi_charges + gst + stt


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
def compute_indicators(df):
    if len(df) < MIN_DATA_DAYS:
        return None

    recent = df.tail(FILTER_LOOKBACK)
    if recent["close"].mean() < MIN_PRICE:
        return None
    if "volume" in recent.columns:
        avg_turnover = (recent["close"] * recent["volume"]).mean()
        if avg_turnover < MIN_AVG_TURNOVER:
            return None

    df = df.copy()

    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma150"] = df["close"].rolling(150).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    df["ath"] = df["close"].expanding().max()
    df["high_1y"] = df["close"].rolling(252, min_periods=200).max()
    df["high_5y"] = df["close"].rolling(252 * 5, min_periods=252).max()

    df["momentum"] = df["close"].pct_change(ROC_PERIOD) * 100

    df.dropna(subset=["sma200", "momentum"], inplace=True)

    return df if len(df) > 0 else None


# =============================================================================
# LOAD OR FETCH DATA
# =============================================================================
def load_or_fetch_data():
    """Load cached data or fetch from Angel One if cache is empty"""

    # Check if we have cached data
    stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith("_1D.pkl")]

    if len(stock_files) >= 100:
        # Use cached data
        print(f"  Found {len(stock_files):,} cached stock files - using cache")
        return load_cached_data()
    else:
        # Need to fetch data
        print(f"  Only {len(stock_files)} cached files - fetching from Angel One...")
        return fetch_all_data()


def load_cached_data():
    """Load data from cache"""
    stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith("_1D.pkl")]

    filtered_files = []
    excluded_count = 0
    for f in stock_files:
        symbol = f.replace("_1D.pkl", "")
        if is_etf_or_liquid(symbol):
            excluded_count += 1
        else:
            filtered_files.append(f)

    print(f"  Excluded {excluded_count} ETFs/Liquid/Index funds")
    print(f"  Loading {len(filtered_files):,} regular stocks...")

    price_data = {}
    skipped = 0

    for i, filename in enumerate(filtered_files, 1):
        symbol = filename.replace("_1D.pkl", "")

        if i % 500 == 0:
            print(f"  Loading: {i:,}/{len(filtered_files):,}  ({symbol})  ", end="\r")

        try:
            filepath = os.path.join(CACHE_DIR, filename)
            with open(filepath, "rb") as f:
                df = pickle.load(f)

            if df is None or df.empty or len(df) < MIN_DATA_DAYS:
                skipped += 1
                continue

            df = compute_indicators(df)
            if df is not None and len(df) > 0:
                price_data[symbol] = df

        except:
            skipped += 1
            continue

    print(f"\n  Loaded: {len(price_data):,} stocks with sufficient data")
    print(f"  Skipped: {skipped:,} (insufficient data or errors)")

    return price_data


def fetch_all_data():
    """Fetch data from Angel One API"""

    # Login
    totp_fixed = validate_credentials()
    print("\nLogging in to Angel One...")
    print(f"  API_KEY   : {API_KEY[:6]}{'*' * 8}  CLIENT_ID: {CLIENT_ID}")

    try:
        obj = login(totp_fixed)
        print("  LOGIN SUCCESSFUL\n")
    except Exception as e:
        print(f"\n  LOGIN FAILED: {e}")
        raise SystemExit(1)

    # Get stock list
    print("Loading ALL NSE stocks from Angel One scrip master...")
    all_stocks = get_all_nse_stocks()
    total = len(all_stocks)
    print(f"  Total NSE-EQ stocks to fetch: {total:,}")

    # Date range
    fetch_end = datetime.date.today().strftime("%Y-%m-%d")
    fetch_start = (datetime.date.today() - datetime.timedelta(days=365 * FETCH_YEARS)).strftime("%Y-%m-%d")

    # Count cached
    cached = sum(1 for s in all_stocks if os.path.exists(os.path.join(CACHE_DIR, f"{s}_1D.pkl")))
    new_fetch = total - cached

    print(f"\nFetching {total:,} stocks...")
    print(f"  Already cached: {cached:,} (instant)")
    print(f"  Need to fetch: {new_fetch:,} (ETA ~{new_fetch * 2.5 / 60:.0f} min)\n")

    price_data = {}
    skipped = 0
    filtered_out = 0
    t0 = time.time()

    for i, (sym, tok) in enumerate(all_stocks.items(), 1):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 1)
        eta_s = (total - i) / rate if rate > 0 else 0

        print(f"  [{i:4d}/{total}] {sym:<18} loaded:{len(price_data):3d}  ETA:{eta_s/60:.1f}min  ", end="\r")

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

        price_data[sym] = df_ind

    elapsed_m = (time.time() - t0) / 60
    print(f"\n\n  Fetch complete in {elapsed_m:.1f} min")
    print(f"  Total fetched: {total:,}")
    print(f"  Insufficient data: {skipped:,}")
    print(f"  Filtered out (price/volume): {filtered_out:,}")
    print(f"  Ready for backtest: {len(price_data)}")

    return price_data


# =============================================================================
# ENTRY FILTER
# =============================================================================
def passes_entry(row):
    price = row["close"]
    sma10 = row["sma10"]
    sma20 = row["sma20"]
    sma50 = row["sma50"]
    sma150 = row["sma150"]
    sma200 = row["sma200"]

    if not (price > sma10 > sma20 > sma50 > sma150 > sma200):
        return False, None

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
    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    all_dates = [d for d in all_dates if BACKTEST_START <= str(d.date()) <= BACKTEST_END]

    if not all_dates:
        print("  ERROR: No trading dates in backtest period!")
        return pd.DataFrame(), pd.DataFrame(), 0

    month_starts = set(
        pd.Series(all_dates, index=pd.DatetimeIndex(all_dates))
        .resample("MS").first().dropna().tolist()
    )

    cash = float(INITIAL_CAPITAL)
    positions = {}
    trades = []
    equity = []
    total_costs = 0.0

    def get_price(sym, date):
        if sym not in price_data:
            return None
        sub = price_data[sym][price_data[sym].index <= date]
        return float(sub["close"].iloc[-1]) if not sub.empty else None

    def calc_nav(date):
        nav = cash
        for sym, pos in positions.items():
            px = get_price(sym, date)
            if px:
                nav += px * pos["shares"]
        return nav

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST: {BACKTEST_START} -> {BACKTEST_END}")
    print(f"  Universe: {len(price_data):,} stocks (excl. ETF/Liquid/Index)")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"  Momentum: {ROC_PERIOD}-day ROC | Positions: {MAX_POSITIONS}")
    print(f"  Stop Loss: {STOP_LOSS_PCT*100:.0f}% | Target: {TARGET_PCT*100:.0f}%")
    print(f"  Near High: {NEAR_HIGH_PCT*100:.0f}%")
    print(f"{'=' * 70}\n")

    total_days = len(all_dates)
    last_pct = -1

    for idx, date in enumerate(all_dates):
        pct = int(idx / total_days * 100)
        if pct % 5 == 0 and pct != last_pct:
            print(f"  {pct:3d}%  {str(date.date())}  Pos:{len(positions)}  Cash:Rs {cash:,.0f}  NAV:Rs {calc_nav(date):,.0f}    ", end="\r")
            last_pct = pct

        # CHECK EXITS
        for sym in list(positions.keys()):
            pos = positions[sym]
            px = get_price(sym, date)
            if px is None:
                continue

            exit_reason = None

            if px <= pos["stop"]:
                exit_reason = "STOP_LOSS"
            elif px >= pos["target"]:
                exit_reason = "TARGET_HIT"

            if exit_reason:
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

        # CHECK ENTRIES
        if date in month_starts:
            slots = MAX_POSITIONS - len(positions)
            if slots > 0:
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
                            "momentum": float(row["momentum"]),
                            "price": float(row["close"]),
                            "high_type": high_type,
                        })

                candidates.sort(key=lambda x: x["momentum"], reverse=True)
                allocation = cash * 0.99 / slots if slots > 0 else 0

                for cand in candidates[:slots]:
                    sym = cand["symbol"]
                    price = cand["price"]
                    entry_price = price * (1 + SLIPPAGE_PCT)

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

        equity.append({"date": date, "equity": calc_nav(date)})

    # CLOSE REMAINING POSITIONS
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

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity).set_index("date")
    equity_df["equity"] = equity_df["equity"].ffill()

    return trades_df, equity_df, total_costs


# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def report(trades_df, equity_df, total_costs):
    if trades_df.empty:
        print("\n  No trades executed.")
        return

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    daily_returns = equity_df["equity"].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

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
    print(f"  Momentum Period : {ROC_PERIOD}-day ROC")
    print(f"  Near High       : {NEAR_HIGH_PCT*100:.0f}%")
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

    print("\n  EXIT REASON BREAKDOWN:")
    for reason, cnt in trades_df["exit_reason"].value_counts().items():
        avg_pnl = trades_df[trades_df["exit_reason"] == reason]["pnl_pct"].mean()
        print(f"    {reason:<20}: {cnt:4d} trades  avg {avg_pnl:>+6.1f}%")

    print("\n  ENTRY HIGH TYPE BREAKDOWN:")
    for ht, cnt in trades_df["high_type"].value_counts().items():
        avg_pnl = trades_df[trades_df["high_type"] == ht]["pnl_pct"].mean()
        print(f"    {ht:<5}: {cnt:4d} trades  avg {avg_pnl:>+6.1f}%")

    print(sep)

    cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "exit_reason"]
    print("\n  TOP 10 WINNING TRADES:")
    print(trades_df.nlargest(10, "pnl_net")[cols].to_string(index=False))
    print("\n  TOP 10 LOSING TRADES:")
    print(trades_df.nsmallest(10, "pnl_net")[cols].to_string(index=False))

    eq_monthly = equity_df["equity"].resample("ME").last().dropna()
    print(f"\n  {'Date':<12}  {'Equity':>14}  {'MoM%':>7}  Chart")
    print("  " + "-" * 58)
    prev = float(INITIAL_CAPITAL)
    for dt, val in eq_monthly.items():
        ret = (val / prev - 1) * 100
        bar = ("+" if ret >= 0 else "-") * min(50, int(abs(ret) * 2))
        print(f"  {str(dt.date()):<12}  Rs {val:>12,.0f}  {ret:>+6.1f}%  {bar}")
        prev = val

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
    print("  NSE MOMENTUM BACKTEST - ALL STOCKS (NO ETF/LIQUID/INDEX)")
    print("  Strategy: SMA alignment + Near High + Momentum")
    print(f"  Momentum: {ROC_PERIOD}-day ROC  (edit ROC_PERIOD to change)")
    print(f"  Stop Loss: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_PCT*100:.0f}%  |  Positions: {MAX_POSITIONS}")
    print(f"  Near High: {NEAR_HIGH_PCT*100:.0f}%")
    print("=" * 70)

    print("\nLoading stock data...")
    price_data = load_or_fetch_data()

    if not price_data:
        print("\n  ERROR: No stock data available!")
        raise SystemExit(1)

    trades_df, equity_df, total_costs = run_backtest(price_data)
    report(trades_df, equity_df, total_costs)


if __name__ == "__main__":
    main()
