"""
=============================================================================
NEAR HIGH BACKTEST - STANDALONE (NO OLIVER KELL)
=============================================================================
UNIVERSE  : All NSE EQ-series stocks (~2500 from Angel One scrip master)
            Excludes: ETFs, Liquid Funds, Bonds, Index Funds

ENTRY     : EMA alignment (Close > EMA5 > EMA10 > EMA20 > EMA50 > EMA150 > EMA200)
            Close within 5-10% BELOW the key high (1Y / 5Y / ATH)
            Signal fires if ANY of the three key-high conditions are met

EXIT      : Stop loss 5% below entry price  OR  Target 10% above ref high
            Whichever is triggered first (checked bar-by-bar on Low / High)

SIGNAL    : 'NearHigh_1Y', 'NearHigh_5Y', 'NearHigh_ATH'

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

INITIAL_CAPITAL      = 1_000_000          # Rs 10 lakh
POSITION_SIZE_PCT    = 0.15               # 15% per trade
STOP_LOSS_PCT        = 0.05               # 5% stop loss
TARGET_ABOVE_HIGH_PCT = 0.10              # 10% above ref high
NEAR_HIGH_LOWER      = 0.90              # 10% below high (lower bound)
NEAR_HIGH_UPPER      = 0.95              # 5% below high (upper bound)
LOOKBACK_DAYS        = 365               # backtest window in days
FETCH_YEARS          = 6                 # years of history to fetch
NIFTY_TOKEN          = "99926000"        # Nifty 50 token on NSE

CACHE_DIR            = "cache_near_high"
MIN_DATA_DAYS        = 252               # minimum bars needed

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

    d.dropna(subset=["EMA200", "High_1Y"], inplace=True)
    return d if len(d) > 0 else None

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
# SINGLE-STOCK BACKTEST
# =============================================================================
def backtest_stock(symbol: str, df: pd.DataFrame) -> list[dict]:
    """Run bar-by-bar simulation for a single stock. Returns list of trade dicts."""
    trades = []

    today = datetime.date.today()
    bt_start = today - datetime.timedelta(days=LOOKBACK_DAYS)
    bt_df = df[df.index.date >= bt_start].copy()
    if bt_df.empty:
        return trades

    in_trade    = False
    entry_price = 0.0
    stop_price  = 0.0
    target_price = 0.0
    ref_high    = 0.0
    ref_high_type = ""
    entry_date  = None
    entry_idx   = 0
    bars_held   = 0

    rows = bt_df.reset_index()    # columns: datetime, open, high, low, close, ...

    for i, row in rows.iterrows():
        if in_trade:
            bars_held += 1
            exit_reason = None
            exit_price  = 0.0

            # Check stop first (on Low), then target (on High)
            if row["low"] <= stop_price:
                exit_price  = stop_price
                exit_reason = "Stop_Loss"
            elif row["high"] >= target_price:
                exit_price  = target_price
                exit_reason = "Target_Hit"

            if exit_reason:
                shares  = int(INITIAL_CAPITAL * POSITION_SIZE_PCT / entry_price)
                pnl_rs  = (exit_price - entry_price) * shares
                pnl_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    "Symbol":         symbol,
                    "Entry_Date":     entry_date,
                    "Exit_Date":      row["datetime"],
                    "Signal_Type":    signal_type_label(ref_high_type),
                    "Entry_Price":    round(entry_price, 2),
                    "Exit_Price":     round(exit_price, 2),
                    "Ref_High":       round(ref_high, 2),
                    "Ref_High_Type":  ref_high_type,
                    "Target_Price":   round(target_price, 2),
                    "Stop_Price":     round(stop_price, 2),
                    "Shares":         shares,
                    "PnL_Rs":         round(pnl_rs, 2),
                    "PnL_Pct":        round(pnl_pct, 2),
                    "Hold_Days":      bars_held,
                    "Exit_Reason":    exit_reason,
                    "Entry_Month":    entry_date.strftime("%Y-%m"),
                })
                in_trade = False
                bars_held = 0

        else:
            # Look for entry signal on this bar
            if not row["Signal"]:
                continue

            rh, rh_type = get_ref_high(row)
            if pd.isna(rh) or rh <= 0:
                continue

            ep = float(row["close"])
            sp = round(ep * (1 - STOP_LOSS_PCT), 2)
            tp = round(rh * (1 + TARGET_ABOVE_HIGH_PCT), 2)

            in_trade      = True
            entry_price   = ep
            stop_price    = sp
            target_price  = tp
            ref_high      = rh
            ref_high_type = rh_type
            entry_date    = row["datetime"]
            bars_held     = 0

    # Close open trade at end of data
    if in_trade and len(rows) > 0:
        last = rows.iloc[-1]
        ep_exit = float(last["close"])
        shares  = int(INITIAL_CAPITAL * POSITION_SIZE_PCT / entry_price)
        pnl_rs  = (ep_exit - entry_price) * shares
        pnl_pct = (ep_exit - entry_price) / entry_price * 100
        trades.append({
            "Symbol":         symbol,
            "Entry_Date":     entry_date,
            "Exit_Date":      last["datetime"],
            "Signal_Type":    signal_type_label(ref_high_type),
            "Entry_Price":    round(entry_price, 2),
            "Exit_Price":     round(ep_exit, 2),
            "Ref_High":       round(ref_high, 2),
            "Ref_High_Type":  ref_high_type,
            "Target_Price":   round(target_price, 2),
            "Stop_Price":     round(stop_price, 2),
            "Shares":         shares,
            "PnL_Rs":         round(pnl_rs, 2),
            "PnL_Pct":        round(pnl_pct, 2),
            "Hold_Days":      bars_held,
            "Exit_Reason":    "End_Of_Data",
            "Entry_Month":    entry_date.strftime("%Y-%m"),
        })

    return trades

# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def performance_report(trades_df: pd.DataFrame, label: str = "NEAR HIGH BACKTEST"):
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

    # Simple equity curve for drawdown
    eq = INITIAL_CAPITAL + trades_df.sort_values("Exit_Date")["PnL_Rs"].cumsum()
    peak = eq.cummax()
    dd   = (eq - peak) / peak * 100
    max_dd = dd.min() if not dd.empty else 0

    avg_hold = trades_df["Hold_Days"].mean()

    print(f"  Initial Capital : Rs {INITIAL_CAPITAL:>14,.0f}")
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
    use_all_nse: bool = True,
    tickers: list[str] | None = None,
    max_stocks: int | None = None,
):
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  NEAR HIGH BACKTEST - STANDALONE (NO OLIVER KELL)")
    print(f"  EMA Alignment + Close within {int((1-NEAR_HIGH_UPPER)*100)}-{int((1-NEAR_HIGH_LOWER)*100)}% of Key High")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%  |  Target: {TARGET_ABOVE_HIGH_PCT*100:.0f}% above ref high")
    print(f"  Lookback: {LOOKBACK_DAYS} days  |  Fetch: {FETCH_YEARS} years")
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
    else:
        universe = []
        for t in (tickers or []):
            tok = resolve_ticker(t, sym_map)
            if tok:
                universe.append((t, tok))
            else:
                print(f"  WARNING: {t} not found in symbol map")

    if max_stocks is not None:
        universe = universe[:max_stocks]

    total = len(universe)
    print(f"\nUniverse: {total:,} stocks\n")

    # Date range for fetching
    today     = datetime.date.today()
    fetch_end  = today.strftime("%Y-%m-%d")
    fetch_start = (today - datetime.timedelta(days=365 * FETCH_YEARS)).strftime("%Y-%m-%d")

    all_trades = []
    processed  = 0
    skipped    = 0

    t0 = time.time()
    for i, (symbol, token) in enumerate(universe, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            rate    = i / max(elapsed, 1)
            eta_m   = (total - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:4d}/{total}] {symbol:<18}  trades_so_far:{len(all_trades):4d}  "
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

            stock_trades = backtest_stock(symbol, df_ind)
            all_trades.extend(stock_trades)
            processed += 1

        except Exception as e:
            print(f"\n  ERROR on {symbol}: {e}")
            skipped += 1

    elapsed_m = (time.time() - t0) / 60
    print(f"\n  Done. {processed:,} stocks processed, {skipped:,} skipped in {elapsed_m:.1f} min")
    print(f"  Total trades: {len(all_trades):,}\n")

    if not all_trades:
        print("  No trades found. Exiting.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    trades_df["Entry_Date"] = pd.to_datetime(trades_df["Entry_Date"])
    trades_df["Exit_Date"]  = pd.to_datetime(trades_df["Exit_Date"])
    trades_df.sort_values("Entry_Date", inplace=True)

    performance_report(trades_df, label="NEAR HIGH BACKTEST - STANDALONE")

    # Save results
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"near_high_backtest_results_{ts}.csv"
    trades_df.to_csv(out, index=False)
    print(f"  Results saved to: {out}")

    return trades_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Default: all NSE EQ stocks
    run_backtest(use_all_nse=True, max_stocks=None)

    # Example: specific tickers only
    # run_backtest(use_all_nse=False, tickers=["RELIANCE", "TCS", "INFY", "HDFCBANK"])

    # Example: first 100 stocks (for testing)
    # run_backtest(use_all_nse=True, max_stocks=100)
