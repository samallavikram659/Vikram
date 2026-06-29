"""
=============================================================================
ATH SIGNAL BACKTEST  -  PER-STOCK, FIXED CAPITAL
=============================================================================
APPROACH  : Every stock that meets the strategy is traded independently.
            When signal fires: invest Rs PER_TRADE_CAPITAL (Rs 10,000).
            No position cap - all qualifying stocks traded simultaneously.
            Each stock can hold one open position at a time.

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

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from SmartApi import SmartConnect
import pyotp

# =============================================================================
# CONFIGURATION
# =============================================================================
ANGEL_API_KEY        = "Tp7PJMIc"
ANGEL_CLIENT_ID      = "S63068620"
ANGEL_CLIENT_PIN     = "4098"
ANGEL_TOTP_SECRET    = "KVTELUFGR33YCPQUKO3ZSL4NMA"

PER_TRADE_CAPITAL    = 10_000           # Rs per trade per stock
ATH_NEAR_PCT         = 0.05            # within 5% below ATH to qualify
STOP_LOSS_PCT        = 0.08            # 8% stop loss below entry
TARGET_ABOVE_HIGH_PCT = 0.15           # 15% above ATH as target
FRESH_HIGH_DAYS      = 200             # skip if ATH touched within this many bars
LOOKBACK_DAYS        = 2000            # backtest window in days (~5.5 years)
FETCH_YEARS          = 6               # years of history to fetch
NIFTY_TOKEN          = "99926000"      # Nifty 50 token on NSE

# Oliver Kell config
MINI_BASE_BARS       = 7               # window for mini-base detection
MINI_BASE_MIN        = 4               # minimum contracting bars in base
EXTENSION_MULT       = 3.0             # ATR multiplier for extension exit
MIN_HOLD_DAYS        = 2               # bars before switching to tighter EMA stop
RS_LOOKBACK          = 20              # days for relative-strength calculation

CACHE_DIR            = "cache_ath_signal"
MIN_DATA_DAYS        = 252             # minimum bars needed per stock

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
# ANGEL ONE - LOGIN
# =============================================================================
def angel_login():
    totp_code = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
    remaining  = 30 - (int(time.time()) % 30)
    print(f"  OTP: {totp_code}  (valid ~{remaining}s)")
    obj = SmartConnect(api_key=ANGEL_API_KEY)
    data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_CLIENT_PIN, totp_code)
    if data.get("status") is False:
        raise RuntimeError(f"Login failed: {data.get('message')}")
    jwt = data["data"]["jwtToken"]
    print(f"  JWT: {jwt[:30]}...")
    print("  LOGIN SUCCESSFUL")
    return obj

# =============================================================================
# NSE SYMBOL MAP
# =============================================================================
def load_symbol_map(smart, cache_path="nse_symbol_map.pkl"):
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 24:
            with open(cache_path, "rb") as f:
                sym_map = pickle.load(f)
            print(f"  Symbol map: {len(sym_map):,} stocks (from cache, {age_hours:.1f}h old)")
            return sym_map

    import requests
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    sym_map = {}
    for item in data:
        if item.get("exch_seg") != "NSE":
            continue
        sym = item.get("symbol", "")
        if not sym.endswith("-EQ"):
            continue
        name   = sym[:-3]
        token  = item.get("token", "")
        if not is_etf(name):
            sym_map[name] = token

    with open(cache_path, "wb") as f:
        pickle.dump(sym_map, f)
    print(f"  Symbol map: {len(sym_map):,} stocks loaded")
    return sym_map

# =============================================================================
# FETCH OHLCV
# =============================================================================
def fetch_ohlcv(smart, token: str, years: int = FETCH_YEARS) -> pd.DataFrame | None:
    today    = datetime.date.today()
    end_dt   = datetime.datetime.combine(today, datetime.time(23, 59))
    start_dt = datetime.datetime.combine(
        today - datetime.timedelta(days=int(years * 365.25)), datetime.time(0, 0)
    )

    def _fmt(dt): return dt.strftime("%Y-%m-%d %H:%M")

    # Split into 1-year chunks (API limit)
    chunks, cur = [], start_dt
    while cur < end_dt:
        nxt = min(cur + datetime.timedelta(days=365), end_dt)
        chunks.append((_fmt(cur), _fmt(nxt)))
        cur = nxt + datetime.timedelta(minutes=1)

    rows = []
    for from_date, to_date in chunks:
        for attempt in range(3):
            try:
                resp = smart.getCandleData({
                    "exchange": "NSE", "symboltoken": token,
                    "interval": "ONE_DAY",
                    "fromdate": from_date, "todate": to_date,
                })
                if resp and resp.get("data"):
                    rows.extend(resp["data"])
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(1)
        time.sleep(0.3)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index.date <= today]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df if len(df) >= MIN_DATA_DAYS else None

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
        (d["close"] > d["EMA5"]) &
        (d["EMA5"]  > d["EMA10"]) &
        (d["EMA10"] > d["EMA20"]) &
        (d["EMA20"] > d["EMA50"]) &
        (d["EMA50"] > d["EMA150"]) &
        (d["EMA150"]> d["EMA200"])
    )

    # ATH (rolling max of high over all available history)
    d["ATH"] = d["high"].cummax()

    # Near-ATH condition: within ATH_NEAR_PCT% below ATH
    d["Near_ATH"] = (d["close"] >= d["ATH"] * (1 - ATH_NEAR_PCT)) & \
                    (d["close"] <= d["ATH"])

    # Fresh high filter: exclude if ATH was touched within FRESH_HIGH_DAYS bars
    touched_ath = (d["high"] >= d["ATH"].shift(1)).fillna(False)
    d["Touched_ATH_Recent"] = touched_ath.rolling(FRESH_HIGH_DAYS, min_periods=1).sum() > 0
    d["Near_ATH"] = d["Near_ATH"] & ~d["Touched_ATH_Recent"]

    # Weekly EMAs (forward-filled to daily)
    weekly = d["close"].resample("W").last().ffill()
    d["W_EMA20"] = weekly.ewm(span=20, adjust=False).mean().reindex(d.index, method="ffill")
    d["Weekly_Uptrend"] = d["close"] > d["W_EMA20"]

    # Volume
    d["Vol20_Avg"]     = d["volume"].rolling(20, min_periods=10).mean()
    d["Vol_Above_Avg"] = d["volume"] > d["Vol20_Avg"]

    # Relative strength vs Nifty
    if nifty_close is not None and not nifty_close.empty:
        nifty_aligned = nifty_close.reindex(d.index, method="ffill")
        d["RS_Leader"] = d["close"].pct_change(RS_LOOKBACK) > nifty_aligned.pct_change(RS_LOOKBACK)
    else:
        d["RS_Leader"] = True

    # Mini-base (ATR contraction)
    atr_series = d["ATR14"].values
    mini_base = [False] * len(d)
    for i in range(MINI_BASE_BARS, len(d)):
        window = atr_series[i - MINI_BASE_BARS + 1: i + 1]
        contracting = sum(1 for j in range(1, len(window)) if window[j] < window[j-1])
        mini_base[i] = contracting >= MINI_BASE_MIN
    d["Mini_Base"] = mini_base

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
    n        = len(rows)

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

            # 1. Stop loss (checked on bar Low)
            if row["low"] <= stop:
                exit_price  = stop
                exit_reason = "Stop_Loss"

            # 2. Target (checked on bar High)
            elif row["high"] >= tgt:
                exit_price  = tgt
                exit_reason = "Target_Hit"

            # 3. Oliver EMA trailing stop
            else:
                ema_trail = row["EMA20"] if held < MIN_HOLD_DAYS else row["EMA10"]
                if row["close"] < ema_trail:
                    exit_price  = row["close"]
                    exit_reason = "Oliver_EMA_Stop"

                # 4. Extension exit
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

    # Calendar year P&L
    trades_df["Year"] = pd.to_datetime(trades_df["Exit_Date"]).dt.year

    hdr = "=" * 70
    print(f"\n{hdr}")
    print("  ATH SIGNAL BACKTEST - PER-STOCK FIXED CAPITAL")
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

    # Exit reason breakdown
    print()
    print("  EXIT REASON BREAKDOWN:")
    for reason, grp in trades_df.groupby("Exit_Reason"):
        r_wr  = (grp["PnL_Rs"] > 0).mean() * 100
        r_avg = grp["PnL_Pct"].mean()
        sign  = "+" if r_avg >= 0 else ""
        print(f"    {reason:<22}: {len(grp):>4} trades  avg {sign}{r_avg:.1f}%  WR {r_wr:.1f}%")

    # Signal type breakdown
    print()
    print("  SIGNAL TYPE BREAKDOWN:")
    for sig, grp in trades_df.groupby("Signal_Type"):
        s_wr  = (grp["PnL_Rs"] > 0).mean() * 100
        s_avg = grp["PnL_Pct"].mean()
        sign  = "+" if s_avg >= 0 else ""
        print(f"    {sig:<28}: {len(grp):>4} trades  avg {sign}{s_avg:.1f}%  WR {s_wr:.1f}%")

    # Calendar year P&L
    print()
    print("  CALENDAR YEAR P&L:")
    print(f"    {'Year':<6} {'Trades':>7} {'Win%':>7} {'Total P&L':>14} {'Avg/Trade':>12}")
    print(f"    {'─'*50}")
    for yr, grp in trades_df.groupby("Year"):
        y_pnl = grp["PnL_Rs"].sum()
        y_wr  = (grp["PnL_Rs"] > 0).mean() * 100
        y_avg = grp["PnL_Rs"].mean()
        sign  = "+" if y_pnl >= 0 else ""
        bar   = "+" * min(int(abs(y_pnl) / 5000), 20) if y_pnl >= 0 else "-" * min(int(abs(y_pnl) / 5000), 20)
        print(f"    {yr:<6} {len(grp):>7} {y_wr:>6.1f}%  Rs {y_pnl:>10,.0f}  Rs {y_avg:>8,.0f}  {bar}")

    # Top 10 stocks by total P&L
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

    # Bottom 10 stocks by total P&L
    print()
    print("  BOTTOM 10 STOCKS BY TOTAL P&L:")
    print(f"    {'Symbol':<14} {'Trades':>6} {'WinRate':>8} {'Avg%':>7} {'Total P&L':>12}")
    print(f"    {'─'*52}")
    for sym, row in per_stock.tail(10).iterrows():
        sign = "+" if row["Avg_Pct"] >= 0 else ""
        print(f"    {sym:<14} {int(row['Trades']):>6} {row['WinRate']:>7.1f}%  {sign}{row['Avg_Pct']:>5.1f}%  Rs {row['Total_PnL']:>8,.0f}")

    # Top 10 individual winning trades
    print()
    print("  TOP 10 WINNING TRADES:")
    top10 = trades_df.nlargest(10, "PnL_Rs")[
        ["Symbol", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price",
         "PnL_Rs", "PnL_Pct", "Hold_Days", "Signal_Type", "Exit_Reason"]
    ]
    print(top10.to_string(index=False))

    # Top 10 losing trades
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
def run_backtest(use_all_nse: bool = True, tickers: list[str] | None = None,
                 max_stocks: int | None = None):
    ts_start = time.time()

    print("=" * 70)
    print("  ATH SIGNAL BACKTEST  -  PER-STOCK, FIXED CAPITAL")
    print(f"  EMA Alignment + Within {ATH_NEAR_PCT*100:.0f}% of ATH + Wedge_Pop | Crossback")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ATH")
    print(f"  Fresh High Filter: {FRESH_HIGH_DAYS} bars  |  Extension mult: {EXTENSION_MULT}x ATR")
    print(f"  Lookback: {LOOKBACK_DAYS} days  |  Fetch: {FETCH_YEARS} years")
    print(f"  Per-Trade Capital: Rs {PER_TRADE_CAPITAL:,}")
    print("=" * 70)

    print("\nLogging in to Angel One...")
    smart = angel_login()

    print("\nLoading NSE symbol map...")
    sym_map = load_symbol_map(smart)

    if use_all_nse:
        universe = list(sym_map.keys())
        print(f"  Universe: ALL NSE-EQ stocks ({len(universe):,})")
    else:
        missing = [t for t in (tickers or []) if t not in sym_map]
        if missing:
            print(f"  WARNING: {len(missing)} tickers not found: {missing[:20]}...")
        universe = [t for t in (tickers or []) if t in sym_map]
        print(f"  Universe: {len(universe):,} stocks (from provided list)")

    if max_stocks:
        universe = universe[:max_stocks]

    print(f"\nUniverse: {len(universe):,} stocks")

    # Fetch Nifty for RS calculation
    print("\nFetching Nifty 50 data for RS calculation...")
    nifty_df = fetch_ohlcv(smart, NIFTY_TOKEN, years=FETCH_YEARS)
    nifty_close = nifty_df["close"] if nifty_df is not None else None
    if nifty_close is not None:
        print(f"  Nifty data: {len(nifty_close):,} bars")

    # ---- PASS 1: Fetch & compute indicators ----
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\n--- PASS 1: Fetching & computing indicators for {len(universe):,} stocks ---\n")

    stock_data = {}
    skipped    = 0
    t0         = time.time()

    for i, symbol in enumerate(universe, 1):
        cache_file = os.path.join(CACHE_DIR, f"{symbol}.pkl")
        df = None

        # Try cache
        if os.path.exists(cache_file):
            age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_h < 12:
                try:
                    with open(cache_file, "rb") as f:
                        df = pickle.load(f)
                except Exception:
                    df = None

        # Fetch from API if not cached
        if df is None:
            df = fetch_ohlcv(smart, sym_map[symbol])
            if df is not None:
                with open(cache_file, "wb") as f:
                    pickle.dump(df, f)

        if df is None:
            skipped += 1
        else:
            d = compute_indicators(df.copy(), nifty_close)
            if d is not None:
                stock_data[symbol] = d
            else:
                skipped += 1

        # Progress
        if i % 10 == 0 or i == len(universe):
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 1
            eta_min = (len(universe) - i) / rate / 60
            loaded  = len(stock_data)
            print(f"  [{i:>5}/{len(universe)}] {symbol:<20} loaded:{loaded:>4}  ETA:{eta_min:.1f}min")

    elapsed_1 = (time.time() - ts_start) / 60
    print(f"\n  Pass 1 done. {len(stock_data):,} stocks loaded, {skipped} skipped in {elapsed_1:.1f} min")

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
            print(f"  [{i:>5}/{len(stock_data)}] simulated  |  {len(all_trades):,} trades so far  ETA:{eta2:.1f}min")

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
    # All NSE EQ stocks (~2500)
    run_backtest(use_all_nse=True, max_stocks=None)

    # Nifty 500 only (faster)
    # run_backtest(use_all_nse=False, tickers=NIFTY_500)

    # Quick test: first 100 stocks
    # run_backtest(use_all_nse=True, max_stocks=100)
