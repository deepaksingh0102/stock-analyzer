"""
Market data + analyzer engine (pure Python / pandas, no UI code).

Ported from single-stock-analyzer.html. This module is the "processing"
layer: fetching, the full indicator set (EMA/SMA/RSI/MACD/VWAP/ATR/
Bollinger/ADX/OBV), candlestick-pattern detection, the 150-minute
breakout check, trend votes, the hard-gate verdict classifier, the
Direction Radar, live pressure checks, the 45-minute direction forecast,
and mechanical trade planning.

Conventions
-----------
* Price frames are pandas DataFrames with columns
  ``Open, High, Low, Close, Volume`` and a DatetimeIndex.
* ``NaN`` (``float("nan")``) marks "not available", mirroring the
  JavaScript engine.
* Functions that need "today's regular-session bars" use
  :func:`session_bars` (bars sharing the last bar's calendar date).
"""
import math
import zlib

import numpy as np
import pandas as pd

# How far back to download. We always pull 2 years so the indicators
# (e.g. 50-day average) are valid, then show only the window you pick.
HISTORY_PERIOD = "2y"
# Intraday pull for the 5-minute engine (covers ~5 trading days).
INTRADAY_PERIOD = "5d"

WINDOWS = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}

NAN = float("nan")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if df.index.tz is not None:
        # Drop the timezone but KEEP exchange-local wall-clock times, so
        # session grouping stays on the exchange's calendar date.
        df.index = df.index.tz_localize(None)
    return df


def _secret(name: str):
    """Read a Streamlit Cloud secret when available (None otherwise).

    Kept inside data.py (not app.py) so it works whichever app.py version
    is deployed; the import is lazy so offline tests still run.
    """
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def alpaca_is_configured() -> bool:
    """True once real Alpaca keys are set (config.py or Streamlit Secrets)."""
    import config
    key = _secret("APCA_API_KEY_ID") or config.APCA_API_KEY_ID
    sec = _secret("APCA_API_SECRET_KEY") or config.APCA_API_SECRET_KEY
    return (key not in ("", "Dummy", "PASTE_YOUR_KEY_HERE")
            and sec not in ("", "Dummy", "PASTE_YOUR_SECRET_HERE"))


def _alpaca_headers() -> dict:
    import config
    return {"APCA-API-KEY-ID": _secret("APCA_API_KEY_ID") or config.APCA_API_KEY_ID,
            "APCA-API-SECRET-KEY": _secret("APCA_API_SECRET_KEY") or config.APCA_API_SECRET_KEY,
            "Accept": "application/json"}


def _alpaca_feed() -> str:
    import config
    return _secret("ALPACA_FEED") or config.ALPACA_FEED


def fetch_alpaca_bars(symbol: str, timeframe: str = "5Min",
                      days: int = 7, limit: int = 1000, feed: str | None = None) -> pd.DataFrame:
    """Bars from Alpaca (data.alpaca.markets) — the HTML's data source.

    Mirrors the HTML's fetchCandles: timeframe/start/end/limit/feed params,
    APCA-API-KEY-ID / APCA-API-SECRET-KEY headers.
    """
    import config
    import requests
    from datetime import datetime, timedelta, timezone

    feed = feed or _alpaca_feed()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {"timeframe": timeframe, "start": start.isoformat(),
              "end": end.isoformat(), "limit": limit, "feed": feed,
              "adjustment": "raw"}
    r = requests.get("https://data.alpaca.markets/v2/stocks/" + symbol + "/bars",
                     headers=_alpaca_headers(), params=params, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Alpaca bars ({timeframe}) HTTP {r.status_code}: {r.text[:200]}")
    bars = r.json().get("bars") or []
    if not bars:
        raise ValueError(f"No Alpaca bars for '{symbol}' ({timeframe}).")
    df = pd.DataFrame(
        [{"time": pd.Timestamp(b["t"]), "Open": b["o"], "High": b["h"],
          "Low": b["l"], "Close": b["c"], "Volume": b["v"]} for b in bars]
    ).set_index("time").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_alpaca_quote(symbol: str, feed: str | None = None) -> dict | None:
    """Latest Alpaca quote: bid/ask price AND size (powers order flow + spread)."""
    import config
    import requests

    feed = feed or _alpaca_feed()
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
        headers=_alpaca_headers(), params={"feed": feed}, timeout=15)
    if not r.ok:
        raise RuntimeError(f"Alpaca quote HTTP {r.status_code}: {r.text[:200]}")
    q = r.json().get("quote") or {}
    if not q.get("bp") or not q.get("ap"):
        return None
    return {"bid": float(q["bp"]), "ask": float(q["ap"]),
            "bidSize": q.get("bs"), "askSize": q.get("as")}


def fetch_prices(symbol: str) -> pd.DataFrame:
    """Daily prices: Alpaca when keys are set (HTML parity), else Yahoo Finance."""
    if alpaca_is_configured():
        return fetch_alpaca_bars(symbol, timeframe="1Day", days=730, limit=1000)

    import yfinance as yf  # imported here so demo mode works without it

    df = yf.Ticker(symbol).history(period=HISTORY_PERIOD, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(
            f"No data found for '{symbol}'. Check the symbol "
            "(e.g. AAPL, MSFT, RELIANCE.NS) or your internet connection."
        )
    return _clean_ohlc(df)


def fetch_intraday(symbol: str, period: str = INTRADAY_PERIOD) -> pd.DataFrame:
    """5-minute bars (regular session): Alpaca when keys are set, else Yahoo."""
    if alpaca_is_configured():
        return fetch_alpaca_bars(symbol, timeframe="5Min", days=7, limit=1000)

    import yfinance as yf

    df = yf.Ticker(symbol).history(period=period, interval="5m", auto_adjust=True,
                                   prepost=False)
    if df is None or df.empty:
        raise ValueError(f"No intraday data found for '{symbol}'.")
    return _clean_ohlc(df)


def fetch_1m(symbol: str) -> pd.DataFrame:
    """1-minute bars (last ~2 days) for the fast-momentum check."""
    if alpaca_is_configured():
        return fetch_alpaca_bars(symbol, timeframe="1Min", days=3, limit=1000)

    import yfinance as yf

    df = yf.Ticker(symbol).history(period="2d", interval="1m", auto_adjust=True,
                                   prepost=False)
    if df is None or df.empty:
        return df
    return _clean_ohlc(df)


def fetch_quote(symbol: str) -> tuple[dict | None, str]:
    """Latest bid/ask (+sizes when Alpaca is configured).

    Returns (quote, note): note carries the real reason the Alpaca quote
    failed (bad keys, wrong feed, non-US symbol, ...), so the UI can show
    it instead of a generic message.
    """
    if alpaca_is_configured():
        try:
            q = fetch_alpaca_quote(symbol)
            if q:
                return q, ""
            return None, "Alpaca returned an empty quote."
        except Exception as e:
            return None, f"Alpaca quote failed: {e}"
    try:
        import yfinance as yf
        fi = yf.Ticker(symbol).fast_info
        bid, ask = fi.get("bid"), fi.get("ask")
        if bid and ask and bid > 0 and ask > 0:
            return {"bid": float(bid), "ask": float(ask)}, ""
    except Exception:
        pass
    return None, ""


def demo_prices(symbol: str, days: int = 504) -> pd.DataFrame:
    """Realistic-looking fake daily prices, so the app works offline."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    days = len(dates)  # some pandas versions return one fewer date on weekends
    returns = rng.normal(0.0005, 0.018, days)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, days)))
    volume = rng.integers(5_000_000, 50_000_000, days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def demo_intraday(symbol: str, days: int = 3, bars_per_day: int = 78) -> pd.DataFrame:
    """Fake 5-minute bars (regular-session hours) for offline testing."""
    rng = np.random.default_rng(zlib.crc32((symbol + ":5m").encode()))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    frames = []
    price = 100.0
    for day in dates:
        idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                            periods=bars_per_day, freq="5min")
        rets = rng.normal(0.0002, 0.004, bars_per_day)
        closes = price * np.exp(np.cumsum(rets))
        opens = np.r_[closes[0] / (1 + rets[0]), closes[:-1]]
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, bars_per_day)))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, bars_per_day)))
        vols = rng.integers(10_000, 200_000, bars_per_day)
        frames.append(pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
            index=idx))
        price = closes[-1]
    return pd.concat(frames)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _f2(x: float) -> str:
    return ("+" if x >= 0 else "") + f"{x:.2f}"


def session_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Bars sharing the last bar's calendar date (today's regular session)."""
    if df is None or df.empty:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    last_date = df.index[-1].date()
    return df[[d.date() == last_date for d in df.index]]


def _bar_dicts(df: pd.DataFrame) -> list:
    out = []
    for ts, row in df.iterrows():
        out.append({
            "time": ts,
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
            "volume": float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Core indicators (HTML engine ports)
# ---------------------------------------------------------------------------
def sma_last(values, n: int) -> float:
    v = [float(x) for x in values]
    if len(v) < n:
        return NAN
    return sum(v[-n:]) / n


def ema_last(values, n: int) -> float:
    """EMA seeded with the first value, exactly like the HTML calcEMA."""
    v = [float(x) for x in values]
    if not v:
        return NAN
    k = 2 / (n + 1)
    x = v[0]
    for i in range(1, len(v)):
        x = v[i] * k + x * (1 - k)
    return x


def ema_series(values, n: int) -> list:
    v = [float(x) for x in values]
    if not v:
        return []
    k = 2 / (n + 1)
    out = [v[0]]
    for i in range(1, len(v)):
        out.append(v[i] * k + out[-1] * (1 - k))
    return out


def rsi_last(values, n: int = 14) -> float:
    """RSI as in the HTML calcRSI: simple average of gains/losses over n."""
    v = [float(x) for x in values]
    if len(v) < n + 1:
        return NAN
    g = l = 0.0
    for i in range(len(v) - n, len(v)):
        d = v[i] - v[i - 1]
        if d >= 0:
            g += d
        else:
            l -= d
    ag, al = g / n, l / n
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def rsi_series(values, n: int = 14) -> list:
    """Wilder-smoothed RSI series (HTML rsiSeries), NaN until warmed up."""
    v = [float(x) for x in values]
    out = [NAN] * len(v)
    if len(v) < n + 1:
        return out
    g = l = 0.0
    for i in range(1, n + 1):
        d = v[i] - v[i - 1]
        if d >= 0:
            g += d
        else:
            l -= d
    g /= n
    l /= n
    out[n] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    for i in range(n + 1, len(v)):
        d = v[i] - v[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        l = (l * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out


def macd_last(values) -> tuple:
    """MACD(12,26,9) exactly like the HTML calcMACD. Returns (macd, signal)."""
    v = [float(x) for x in values]
    if not v:
        return (NAN, NAN)
    macd = ema_last(v, 12) - ema_last(v, 26)
    seq = []
    for i in range(max(26, len(v) - 20), len(v)):
        sub = v[: i + 1]
        seq.append(ema_last(sub, 12) - ema_last(sub, 26))
    return (macd, ema_last(seq, 9))


def vwap_last(df: pd.DataFrame, n: int = 78) -> float:
    """Session VWAP over the last n bars (78 = one 6.5h session of 5m bars)."""
    x = df.tail(n)
    pv = ((x["High"] + x["Low"] + x["Close"]) / 3 * x["Volume"].fillna(0)).sum()
    vol = x["Volume"].fillna(0).sum()
    return pv / vol if vol else NAN


def avg_volume(df: pd.DataFrame, n: int = 20) -> float:
    """Mean volume of the last n bars *excluding* the current one."""
    x = df["Volume"].fillna(0).tolist()
    window = x[-n - 1: -1]
    return sum(window) / len(window) if window else NAN


def atr_last(df: pd.DataFrame, n: int = 14) -> float:
    """ATR as a simple average of the last n true ranges (HTML calcATR)."""
    h = df["High"].astype(float).tolist()
    l = df["Low"].astype(float).tolist()
    c = df["Close"].astype(float).tolist()
    if len(c) < n + 1:
        return NAN
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
           for i in range(1, len(c))]
    recent = trs[-n:]
    return sum(recent) / len(recent)


def bollinger(values, n: int = 20, k: float = 2.0) -> dict:
    """Bollinger Bands with population std, like the HTML calcBollinger."""
    v = [float(x) for x in values]
    if len(v) < n:
        return {"mid": NAN, "upper": NAN, "lower": NAN, "sd": NAN, "width": NAN}
    sl = v[-n:]
    mid = sum(sl) / n
    sd = math.sqrt(sum((x - mid) ** 2 for x in sl) / n)
    upper, lower = mid + k * sd, mid - k * sd
    return {"mid": mid, "upper": upper, "lower": lower, "sd": sd,
            "width": (upper - lower) / mid if mid else NAN}


def adx(df: pd.DataFrame, n: int = 14) -> dict:
    """ADX with Wilder's smoothing, exactly like the HTML calcADX.

    Measures trend STRENGTH, not direction. Guide: <20 no real trend,
    20-25 developing, >25 conviction, >40 very strong / possibly late.
    """
    bad = {"adx": NAN, "plusDI": NAN, "minusDI": NAN}
    h = df["High"].astype(float).tolist()
    l = df["Low"].astype(float).tolist()
    c = df["Close"].astype(float).tolist()
    if len(c) < 2 * n + 1:
        return bad
    tr, pdm, mdm = [], [], []
    for i in range(1, len(c)):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    sTR, sP, sM = sum(tr[:n]), sum(pdm[:n]), sum(mdm[:n])
    dxs = []
    pDI = mDI = NAN
    for i in range(n - 1, len(tr)):
        if i >= n:
            sTR = sTR - sTR / n + tr[i]
            sP = sP - sP / n + pdm[i]
            sM = sM - sM / n + mdm[i]
        pDI = 100 * sP / sTR if sTR else 0.0
        mDI = 100 * sM / sTR if sTR else 0.0
        s = pDI + mDI
        dxs.append(100 * abs(pDI - mDI) / s if s else 0.0)
    if len(dxs) < n:
        return bad
    a = sum(dxs[:n]) / n
    for i in range(n, len(dxs)):
        a = (a * (n - 1) + dxs[i]) / n
    return {"adx": a, "plusDI": pDI, "minusDI": mDI}


def adx_label(a: float) -> str:
    if not math.isfinite(a):
        return "n/a"
    if a < 20:
        return "No real trend — chop / noise"
    if a < 25:
        return "Weak / developing trend"
    if a < 40:
        return "Trending with conviction"
    return "Very strong trend (can be late-stage)"


# ---------------------------------------------------------------------------
# Candlestick patterns (last 150 minutes of 5-min bars)
# ---------------------------------------------------------------------------
def _body(c): return abs(c["close"] - c["open"])
def _rng(c): return (c["high"] - c["low"]) or 1e-9
def _is_bull(c): return c["close"] > c["open"]
def _is_bear(c): return c["close"] < c["open"]
def _upper_wick(c): return c["high"] - max(c["open"], c["close"])
def _lower_wick(c): return min(c["open"], c["close"]) - c["low"]
def _midpoint(c): return (c["open"] + c["close"]) / 2
def _near(a, b, p=0.0015):
    return abs(a - b) / max(abs(a), abs(b), 1) <= p


def _prior_trend(c, i, n=6):
    x = c[max(0, i - n): i]
    if len(x) < 3:
        return "flat"
    ch = (x[-1]["close"] - x[0]["close"]) / x[0]["close"]
    return "up" if ch > 0.004 else ("down" if ch < -0.004 else "flat")


def detect_patterns(df: pd.DataFrame) -> list:
    """Score every candlestick pattern in the last 30 five-minute bars."""
    c = _bar_dicts(df.tail(30))
    out = []
    if len(c) < 3:
        return out
    avg_vol = sum(x["volume"] for x in c[:-1]) / max(1, len(c) - 1)

    def push(label, direction, strength, i, context):
        rv = c[i]["volume"] / avg_vol if avg_vol > 0 else 1.0
        recency = (i + 1) / len(c)
        score = min(1.0, strength * 0.72 + recency * 0.18
                    + max(0.0, min(0.10, (rv - 1) * 0.07)))
        out.append({"label": label, "dir": direction, "strength": strength,
                    "score": score, "index": i, "time": c[i]["time"],
                    "context": context, "relVol": rv})

    for i in range(len(c)):
        x, p = c[i], c[i - 1] if i > 0 else None
        p2 = c[i - 2] if i > 1 else None
        t = _prior_trend(c, i)
        if t == "down" and _lower_wick(x) >= _body(x) * 2 \
                and _upper_wick(x) <= max(_body(x), _rng(x) * 0.15):
            push("Hammer", 1, 0.72, i, "Lower-price rejection after decline")
        if t == "down" and _upper_wick(x) >= _body(x) * 2 \
                and _lower_wick(x) <= max(_body(x), _rng(x) * 0.15):
            push("Inverted Hammer", 1, 0.62, i, "Potential bullish reversal after decline")
        if p and _is_bear(p) and _is_bull(x) and x["open"] <= p["close"] and x["close"] >= p["open"]:
            push("Bullish Engulfing", 1, 0.82, i, "Bullish body engulfs prior bearish body")
        if p and _is_bull(p) and _is_bear(x) and x["open"] >= p["close"] and x["close"] <= p["open"]:
            push("Bearish Engulfing", -1, 0.82, i, "Bearish body engulfs prior bullish body")
        if p and _is_bear(p) and _is_bull(x) and x["open"] < p["close"] \
                and x["close"] > _midpoint(p) and x["close"] < p["open"]:
            push("Piercing Pattern", 1, 0.70, i, "Bullish recovery above midpoint of prior bearish candle")
        if p2 and _is_bear(p2) and _body(p2) / _rng(p2) > 0.5 and _body(p) / _rng(p) < 0.35 \
                and _is_bull(x) and x["close"] > _midpoint(p2):
            push("Morning Star", 1, 0.86, i, "Three-candle bullish reversal")
        if p2 and _is_bull(p2) and _body(p2) / _rng(p2) > 0.5 and _body(p) / _rng(p) < 0.35 \
                and _is_bear(x) and x["close"] < _midpoint(p2):
            push("Evening Star", -1, 0.86, i, "Three-candle bearish reversal")
        if p2 and _is_bull(p2) and _is_bull(p) and _is_bull(x) \
                and p2["close"] < p["close"] < x["close"]:
            push("Three White Soldiers", 1, 0.88, i, "Three advancing bullish candles")
        if p2 and _is_bear(p2) and _is_bear(p) and _is_bear(x) \
                and p2["close"] > p["close"] > x["close"]:
            push("Three Black Crows", -1, 0.88, i, "Three declining bearish candles")
        if p and _is_bear(p) and _is_bull(x) \
                and max(x["open"], x["close"]) < max(p["open"], p["close"]) \
                and min(x["open"], x["close"]) > min(p["open"], p["close"]):
            push("Bullish Harami", 1, 0.58, i, "Selling pressure may be slowing")
        if p and _is_bull(p) and _is_bear(x) \
                and max(x["open"], x["close"]) < max(p["open"], p["close"]) \
                and min(x["open"], x["close"]) > min(p["open"], p["close"]):
            push("Bearish Harami", -1, 0.58, i, "Buying pressure may be slowing")
        if p and t == "down" and _near(x["low"], p["low"]):
            push("Tweezer Bottom", 1, 0.64, i, "Repeated rejection of similar low")
        if p and t == "up" and _near(x["high"], p["high"]):
            push("Tweezer Top", -1, 0.64, i, "Repeated rejection of similar high")
        if t == "up" and _upper_wick(x) >= _body(x) * 2 \
                and _lower_wick(x) <= max(_body(x), _rng(x) * 0.15):
            push("Shooting Star", -1, 0.72, i, "Upper-price rejection after rise")
        if t == "up" and _lower_wick(x) >= _body(x) * 2 \
                and _upper_wick(x) <= max(_body(x), _rng(x) * 0.15):
            push("Hanging Man", -1, 0.62, i, "Potential exhaustion after rise")
        if _body(x) / _rng(x) <= 0.10 and _lower_wick(x) > _rng(x) * 0.6 \
                and _upper_wick(x) < _rng(x) * 0.1:
            push("Dragonfly Doji", 1, 0.60, i, "Strong lower rejection")
        if _body(x) / _rng(x) <= 0.10 and _upper_wick(x) > _rng(x) * 0.6 \
                and _lower_wick(x) < _rng(x) * 0.1:
            push("Gravestone Doji", -1, 0.60, i, "Strong upper rejection")
        if _is_bull(x) and _body(x) / _rng(x) >= 0.85 \
                and _upper_wick(x) / _rng(x) <= 0.08 and _lower_wick(x) / _rng(x) <= 0.08:
            push("Bullish Marubozu", 1, 0.78, i, "Strong bullish momentum")
        if _is_bear(x) and _body(x) / _rng(x) >= 0.85 \
                and _upper_wick(x) / _rng(x) <= 0.08 and _lower_wick(x) / _rng(x) <= 0.08:
            push("Bearish Marubozu", -1, 0.78, i, "Strong bearish momentum")
        if p and _is_bull(p) and _is_bear(x) and x["open"] > p["close"] \
                and x["close"] < _midpoint(p) and x["close"] > p["open"]:
            push("Dark Cloud Cover", -1, 0.70, i, "Bearish reversal into prior bullish body")
        if _body(x) / _rng(x) <= 0.10:
            push("Doji", 0, 0.40, i, "Indecision")
        if p and x["high"] < p["high"] and x["low"] > p["low"]:
            push("Inside Bar", 0, 0.45, i, "Range compression")
        if p and x["high"] > p["high"] and x["low"] < p["low"]:
            push("Outside Bar", 1 if _is_bull(x) else -1, 0.62, i, "Range expansion")
        if p and x["low"] > p["high"]:
            push("Gap Up", 1, 0.55, i, "Bullish price gap")
        if p and x["high"] < p["low"]:
            push("Gap Down", -1, 0.55, i, "Bearish price gap")
    return sorted(out, key=lambda d: d["score"], reverse=True)


def detect_pattern(df: pd.DataFrame) -> dict | None:
    """Best directional pattern, else the best pattern, else None."""
    allp = detect_patterns(df)
    return next((p for p in allp if p["dir"] != 0), allp[0] if allp else None)


def check_breakout(df: pd.DataFrame, direction: int) -> dict:
    """150-minute breakout/breakdown check (session bars only)."""
    today = _bar_dicts(session_bars(df))
    if not today:
        return {"confirmed": False, "label": "No regular-session data yet today",
                "rangeHigh": NAN, "rangeLow": NAN}
    if len(today) < 2:
        return {"confirmed": False, "label": "Not enough bars yet this session",
                "rangeHigh": NAN, "rangeLow": NAN}
    current = today[-1]
    if len(today) >= 31:
        prior = today[-31:-1]
        high = max(b["high"] for b in prior)
        low = min(b["low"] for b in prior)
        if direction == 1 and current["close"] > high:
            return {"confirmed": True, "label": f"Broke today's 150m high ${high:.2f}",
                    "rangeHigh": high, "rangeLow": low}
        if direction == -1 and current["close"] < low:
            return {"confirmed": True, "label": f"Broke today's 150m low ${low:.2f}",
                    "rangeHigh": high, "rangeLow": low}
        base = (f"Below today's 150m high ${high:.2f} (no fresh breakout in the last 150m)"
                if direction == 1 else
                f"Above today's 150m low ${low:.2f} (no fresh breakdown in the last 150m)")
        return {"confirmed": False, "label": base, "rangeHigh": high, "rangeLow": low}
    # Session under 150 minutes old: opening-range breakout instead.
    prior = today[:-1]
    high = max(b["high"] for b in prior)
    low = min(b["low"] for b in prior)
    if direction == 1 and current["close"] > high:
        return {"confirmed": True,
                "label": f"Broke today's opening range high ${high:.2f} (session <150m old)",
                "rangeHigh": high, "rangeLow": low}
    if direction == -1 and current["close"] < low:
        return {"confirmed": True,
                "label": f"Broke today's opening range low ${low:.2f} (session <150m old)",
                "rangeHigh": high, "rangeLow": low}
    return {"confirmed": False,
            "label": f"Within today's range so far (${low:.2f}-${high:.2f}); session <150 minutes old",
            "rangeHigh": high, "rangeLow": low}


# ---------------------------------------------------------------------------
# Trend votes
# ---------------------------------------------------------------------------
def current_trend(df: pd.DataFrame, latest_price: float | None = None,
                  latest_volume: float | None = None) -> dict:
    """Intraday (5-min) trend: 6 indicator votes + a volume kicker."""
    closes = df["Close"].astype(float).tolist()
    if latest_price is not None and math.isfinite(latest_price):
        closes.append(float(latest_price))
    price = closes[-1] if closes else NAN
    ema20, ema50, sma20 = ema_last(closes, 20), ema_last(closes, 50), sma_last(closes, 20)
    rsi = rsi_last(closes, 14)
    macd, sig = macd_last(closes)
    vwap = vwap_last(df)
    av = avg_volume(df, 20)
    vols = df["Volume"].fillna(0).tolist()
    lv = latest_volume if latest_volume is not None else (vols[-1] if vols else 0.0)
    vol_ratio = lv / av if math.isfinite(av) and av > 0 else NAN

    bull = bear = 0
    bull += 1 if price > ema20 else 0
    bear += 1 if not price > ema20 else 0
    bull += 1 if ema20 > ema50 else 0
    bear += 1 if not ema20 > ema50 else 0
    bull += 1 if price > sma20 else 0
    bear += 1 if not price > sma20 else 0
    if math.isfinite(vwap):
        bull += 1 if price > vwap else 0
        bear += 1 if not price > vwap else 0
    if math.isfinite(rsi):
        if rsi > 52:
            bull += 1
        elif rsi < 48:
            bear += 1
    if math.isfinite(sig):
        bull += 1 if macd > sig else 0
        bear += 1 if not macd > sig else 0
    if math.isfinite(vol_ratio) and vol_ratio >= 1.2:
        if bull > bear:
            bull += 0.5
        elif bear > bull:
            bear += 0.5
    bias = "Bullish" if bull >= bear + 2 else ("Bearish" if bear >= bull + 2 else "Mixed")
    return {"bias": bias, "price": price, "ema20": ema20, "ema50": ema50,
            "sma20": sma20, "rsi": rsi, "macd": macd, "macdSignal": sig,
            "vwap": vwap, "volRatio": vol_ratio, "bull": bull, "bear": bear}


def daily_trend(df: pd.DataFrame) -> dict:
    """Higher-timeframe (daily) trend votes."""
    if df is None or len(df) < 55:
        return {"bias": "Mixed", "price": NAN, "note": "Not enough daily history"}
    closes = df["Close"].astype(float).tolist()
    price = closes[-1]
    ema20, ema50 = ema_last(closes, 20), ema_last(closes, 50)
    ema200 = ema_last(closes, 200) if len(closes) >= 200 else NAN
    rsi = rsi_last(closes, 14)
    bull = bear = 0
    bull += 1 if price > ema20 else 0
    bear += 1 if not price > ema20 else 0
    bull += 1 if ema20 > ema50 else 0
    bear += 1 if not ema20 > ema50 else 0
    if math.isfinite(ema200):
        bull += 1 if price > ema200 else 0
        bear += 1 if not price > ema200 else 0
    if math.isfinite(rsi):
        if rsi > 52:
            bull += 1
        elif rsi < 48:
            bear += 1
    bias = "Bullish" if bull >= bear + 2 else ("Bearish" if bear >= bull + 2 else "Mixed")
    return {"bias": bias, "price": price, "ema20": ema20, "ema50": ema50,
            "ema200": ema200, "rsi": rsi, "bull": bull, "bear": bear}


# ---------------------------------------------------------------------------
# Liquidity, spread, relative strength, OBV, order flow
# ---------------------------------------------------------------------------
def avg_dollar_volume(df: pd.DataFrame, n: int = 20) -> float:
    if df is None or len(df) < n:
        return NAN
    x = df.tail(n)
    return float((x["Close"] * x["Volume"].fillna(0)).sum() / len(x))


def spread_info(quote: dict | None) -> dict | None:
    if not quote or not math.isfinite(quote.get("bid", NAN)) \
            or not math.isfinite(quote.get("ask", NAN)):
        return None
    bid, ask = quote["bid"], quote["ask"]
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2
    return {"bid": bid, "ask": ask, "mid": mid, "spread": ask - bid,
            "pct": (ask - bid) / mid * 100}


def relative_strength(df: pd.DataFrame, entry: float, bench_df: pd.DataFrame | None,
                      bench_price: float | None, bench_ticker: str) -> dict:
    """% change since today's open: stock vs. benchmark (raw, not beta-adjusted)."""
    if bench_df is None or bench_df.empty:
        return {"available": False, "benchTicker": bench_ticker,
                "note": f"Benchmark {bench_ticker} data unavailable"}
    s_today, b_today = session_bars(df), session_bars(bench_df)
    if len(s_today) == 0 or len(b_today) == 0:
        return {"available": False, "benchTicker": bench_ticker,
                "note": "No regular-session bars yet for the stock or the benchmark"}
    if s_today.index[0].date() != b_today.index[0].date():
        return {"available": False, "benchTicker": bench_ticker,
                "note": "Stock and benchmark session dates don't match"}
    stock_open = float(s_today["Open"].iloc[0])
    bench_open = float(b_today["Open"].iloc[0])
    if not (stock_open > 0 and bench_open > 0 and math.isfinite(entry)
            and bench_price is not None and math.isfinite(bench_price)):
        return {"available": False, "benchTicker": bench_ticker,
                "note": "Missing open/price data"}
    stock_chg = (entry - stock_open) / stock_open * 100
    bench_chg = (bench_price - bench_open) / bench_open * 100
    stock60 = bench60 = NAN
    if len(s_today) >= 13 and len(b_today) >= 13:
        ref_utc = s_today.index[-13].tz_convert("UTC") \
            if s_today.index.tz is not None else s_today.index[-13]
        b_utc = b_today.index.tz_convert("UTC") \
            if b_today.index.tz is not None else b_today.index
        match = b_utc[b_utc == ref_utc]
        if len(match):
            b_ref = float(b_today.loc[b_today.index[b_utc == ref_utc][0], "Close"])
            ref_close = float(s_today["Close"].iloc[-13])
            if ref_close > 0 and b_ref > 0:
                stock60 = (entry - ref_close) / ref_close * 100
                bench60 = (bench_price - b_ref) / b_ref * 100
        else:
            # Cross-timezone sessions (e.g. NSE vs SPY) never share bar
            # timestamps — fall back to positional matching ("~60 min ago").
            b_ref = float(b_today["Close"].iloc[-13])
            ref_close = float(s_today["Close"].iloc[-13])
            if ref_close > 0 and b_ref > 0:
                stock60 = (entry - ref_close) / ref_close * 100
                bench60 = (bench_price - b_ref) / b_ref * 100
    rs60 = stock60 - bench60 if math.isfinite(stock60) and math.isfinite(bench60) else NAN
    return {"available": True, "benchTicker": bench_ticker,
            "stockChg": stock_chg, "benchChg": bench_chg,
            "rsOpen": stock_chg - bench_chg,
            "stock60": stock60, "bench60": bench60, "rs60": rs60}


def obv_session(df: pd.DataFrame) -> dict:
    """Session OBV as a net-accumulation ratio (-1..+1), whole session + last hour."""
    bars = session_bars(df)
    if len(bars) < 8:
        return {"available": False,
                "note": f"Only {len(bars)} regular-session bar(s) so far — need 8+ for an OBV read"}
    closes = bars["Close"].astype(float).tolist()
    opens = bars["Open"].astype(float).tolist()
    vols = bars["Volume"].fillna(0).astype(float).tolist()
    obv = [_sign(closes[0] - opens[0]) * vols[0]]
    for i in range(1, len(bars)):
        obv.append(obv[i - 1] + _sign(closes[i] - closes[i - 1]) * vols[i])
    total_vol = sum(vols)
    if not total_vol > 0:
        return {"available": False, "note": "No volume data this session"}
    session_ratio = obv[-1] / total_vol
    n = min(12, len(bars) - 1)
    base = len(bars) - 1 - n
    recent_vol = sum(vols[base + 1:])
    recent_ratio = (obv[-1] - obv[base]) / recent_vol if recent_vol > 0 else NAN
    price_recent = (closes[-1] - closes[base]) / closes[base] if closes[base] else NAN
    composite = 0.5 * session_ratio + 0.5 * (recent_ratio if math.isfinite(recent_ratio) else 0)
    state = ("Accumulation" if composite > 0.10 else
             "Distribution" if composite < -0.10 else "Neutral / mixed")
    divergence = "None"
    if math.isfinite(price_recent) and math.isfinite(recent_ratio):
        if price_recent > 0.001 and recent_ratio < -0.10:
            divergence = "Bearish — price rising on net selling volume"
        elif price_recent < -0.001 and recent_ratio > 0.10:
            divergence = "Bullish — price falling on net buying volume"
    return {"available": True, "bars": len(bars), "sessionRatio": session_ratio,
            "recentRatio": recent_ratio, "recentBars": n, "priceRecent": price_recent,
            "state": state, "divergence": divergence}


def order_flow(quote: dict | None = None) -> dict:
    """Bid-size vs. ask-size imbalance in [-1, +1], like the HTML's orderFlow.

    +1 = all size on the bid (buy-side pressure), -1 = all on the ask.
    Sizes come from the Alpaca latest quote; without Alpaca keys configured
    this is unavailable (same as the HTML when quotes are plan-gated).
    """
    bs = (quote or {}).get("bidSize")
    as_ = (quote or {}).get("askSize")
    if bs is None or as_ is None:
        return {"snapshot": NAN, "primary": NAN, "source": "n/a", "label": "n/a",
                "note": "Quote size data unavailable — add your Alpaca keys under "
                        "Settings → Secrets (Streamlit Cloud) or in config.py (local) "
                        "to enable the order-flow read.",
                "bidSize": NAN, "askSize": NAN}
    bs, as_ = float(bs), float(as_)
    if not math.isfinite(bs) or not math.isfinite(as_) or bs + as_ <= 0:
        return {"snapshot": NAN, "primary": NAN, "source": "n/a", "label": "n/a",
                "note": "Quote sizes invalid.", "bidSize": NAN, "askSize": NAN}
    imb = (bs - as_) / (bs + as_)
    label = ("Buy-side pressure (more size on the bid)" if imb >= 0.2 else
             "Sell-side pressure (more size on the ask)" if imb <= -0.2 else "Balanced")
    return {"snapshot": imb, "primary": imb, "source": "Alpaca latest quote",
            "label": label, "note": "", "bidSize": bs, "askSize": as_}


# ---------------------------------------------------------------------------
# Volatility setup
# ---------------------------------------------------------------------------
def detect_squeeze(df: pd.DataFrame) -> dict:
    """Bollinger squeeze: band width in its trailing distribution + ATR contraction."""
    closes = df["Close"].astype(float).tolist()
    if len(closes) < 50:
        return {"squeeze": False, "note": "Not enough bars for a squeeze read yet"}
    bb = bollinger(closes, 20, 2)
    widths = []
    for i in range(20, len(closes)):
        w = bollinger(closes[: i + 1], 20, 2)["width"]
        if math.isfinite(w):
            widths.append(w)
    if len(widths) < 15 or not math.isfinite(bb["width"]):
        return {"squeeze": False, "note": "Not enough width history yet"}
    sw = sorted(widths)
    rank = sum(1 for w in sw if w <= bb["width"]) / len(sw)
    atr_now = atr_last(df, 14)
    atr_past = atr_last(df.iloc[:-10], 14) if len(df) > 24 else NAN
    contracting = (math.isfinite(atr_now) and math.isfinite(atr_past) and atr_past > 0
                   and atr_now < atr_past * 0.85)
    return {"squeeze": rank <= 0.20 and contracting, "widthPercentile": rank,
            "bb": bb, "atrNow": atr_now, "atrPast": atr_past,
            "atrContracting": contracting}


def mean_reversion_note(df: pd.DataFrame) -> dict | None:
    """Counter-trend risk flag (kept SEPARATE from the trend verdict)."""
    closes = df["Close"].astype(float).tolist()
    if len(closes) < 20:
        return None
    bb = bollinger(closes, 20, 2)
    price = closes[-1]
    rsi = rsi_last(closes, 14)
    if not math.isfinite(bb["upper"]) or not math.isfinite(rsi):
        return None
    if price >= bb["upper"] * 0.995 and rsi >= 70:
        return {"side": "upside",
                "detail": f"Price is at/above the upper Bollinger Band (${bb['upper']:.2f}) "
                          f"and RSI is {rsi:.1f} (overbought). Classic mean-reversion risk — "
                          "a pullback here wouldn't be surprising even if the trend read is Bullish."}
    if price <= bb["lower"] * 1.005 and rsi <= 30:
        return {"side": "downside",
                "detail": f"Price is at/below the lower Bollinger Band (${bb['lower']:.2f}) "
                          f"and RSI is {rsi:.1f} (oversold). Classic mean-reversion risk — "
                          "a bounce here wouldn't be surprising even if the trend read is Bearish."}
    return None


def paired_confirmations(trend: dict, direction: int) -> dict:
    """VWAP+Volume and EMA+Volume confirmations, made explicit."""
    if direction == 0:
        return {"vwapVolume": "n/a — no directional read",
                "emaVolume": "n/a — no directional read"}

    def label(agree, vol):
        if agree is None or vol is None:
            return "n/a"
        if agree and vol:
            return "Confirmed (aligned + volume support)"
        if agree:
            return "Direction aligned, but no volume support"
        return "Not confirmed"

    vol_support = trend["volRatio"] >= 1.2 if math.isfinite(trend.get("volRatio", NAN)) else None
    vwap_agree = ((trend["price"] > trend["vwap"]) if direction == 1
                  else (trend["price"] < trend["vwap"])) \
        if math.isfinite(trend.get("vwap", NAN)) else None
    ema_agree = ((trend["ema20"] > trend["ema50"]) if direction == 1
                 else (trend["ema20"] < trend["ema50"])) \
        if math.isfinite(trend.get("ema20", NAN)) and math.isfinite(trend.get("ema50", NAN)) else None
    return {"vwapVolume": label(vwap_agree, vol_support),
            "emaVolume": label(ema_agree, vol_support)}


# ---------------------------------------------------------------------------
# Risk: levels, stops/targets, sizing
# ---------------------------------------------------------------------------
def next_levels(entry: float, breakout: dict, bb: dict, vwap: float,
                daily: dict, atr: float) -> dict:
    """Nearby structural reference levels — NOT a prediction."""
    resistances, supports = [], []

    def add(arr, label, value):
        if value is not None and math.isfinite(value):
            arr.append({"label": label, "value": value})

    add(resistances, "150m range high", breakout.get("rangeHigh", NAN))
    add(supports, "150m range low", breakout.get("rangeLow", NAN))
    if bb:
        add(resistances, "Upper Bollinger Band", bb.get("upper", NAN))
        add(supports, "Lower Bollinger Band", bb.get("lower", NAN))
    if math.isfinite(vwap):
        (resistances if vwap > entry else supports).append({"label": "VWAP", "value": vwap})
    if math.isfinite(daily.get("ema20", NAN)):
        (resistances if daily["ema20"] > entry else supports).append(
            {"label": "Daily EMA20", "value": daily["ema20"]})
    if math.isfinite(daily.get("ema50", NAN)):
        (resistances if daily["ema50"] > entry else supports).append(
            {"label": "Daily EMA50", "value": daily["ema50"]})
    up = sorted([r for r in resistances if r["value"] > entry], key=lambda r: r["value"])
    dn = sorted([s for s in supports if s["value"] < entry], key=lambda s: s["value"],
                reverse=True)
    return {"nearestResistance": up[0] if up else None,
            "nearestSupport": dn[0] if dn else None,
            "atrUp1": entry + atr, "atrUp2": entry + 2 * atr,
            "atrDown1": entry - atr, "atrDown2": entry - 2 * atr}


def risk_plan(df: pd.DataFrame, entry: float, direction: int, atr: float,
              breakout: dict) -> dict:
    """Stop-loss (ATR + structure), measured-move and 2R targets."""
    lookback = _bar_dicts(df.tail(10))
    swing_low = min(b["low"] for b in lookback)
    swing_high = max(b["high"] for b in lookback)
    atr_stop = (entry - 1.5 * atr) if direction == 1 else (entry + 1.5 * atr) \
        if math.isfinite(atr) else NAN
    struct_stop = (swing_low - (0.1 * atr if math.isfinite(atr) else 0)) if direction == 1 \
        else (swing_high + (0.1 * atr if math.isfinite(atr) else 0))
    if math.isfinite(atr_stop):
        stop = min(atr_stop, struct_stop) if direction == 1 else max(atr_stop, struct_stop)
    else:
        stop = struct_stop
    stop_dist = abs(entry - stop)
    rh, rl = breakout.get("rangeHigh", NAN), breakout.get("rangeLow", NAN)
    range_size = (rh - rl) if math.isfinite(rh) and math.isfinite(rl) else NAN
    measured = (entry + range_size) if direction == 1 else (entry - range_size) \
        if math.isfinite(range_size) else NAN
    target_2r = entry + 2 * stop_dist if direction == 1 else entry - 2 * stop_dist
    rr = abs(measured - entry) / stop_dist \
        if (stop_dist > 0 and math.isfinite(measured)) else NAN
    return {"stop": stop, "stopDist": stop_dist, "measuredTarget": measured,
            "target2R": target_2r, "rr": rr}


def position_size(account_size: float, risk_pct: float, stop_dist: float,
                  entry: float) -> dict:
    risk_amount = account_size * (risk_pct / 100)
    shares = math.floor(risk_amount / stop_dist) if stop_dist > 0 else 0
    position_value = shares * entry
    return {"riskAmount": risk_amount, "shares": shares,
            "positionValue": position_value,
            "pctOfAccount": position_value / account_size * 100 if account_size > 0 else NAN}


# ---------------------------------------------------------------------------
# Verdict: hard gates + confirmation scorecard
# ---------------------------------------------------------------------------
def build_scorecard(direction: int, ext: dict, thresholds: dict, strong: bool) -> dict:
    """Soft signals that raise/lower conviction after the hard gates pass."""
    items = []
    d = "bullish" if direction == 1 else "bearish"

    def add(name, status, detail):
        items.append({"name": name, "status": status, "detail": detail})

    a = ext.get("adx") or {}
    if math.isfinite(a.get("adx", NAN)):
        di_ok = a["plusDI"] > a["minusDI"] if direction == 1 else a["minusDI"] > a["plusDI"]
        if not di_ok:
            add("ADX / DI", "contradict",
                f"ADX {a['adx']:.1f} but directional lines (+DI {a['plusDI']:.1f} / "
                f"−DI {a['minusDI']:.1f}) point against the {d} read")
        elif a["adx"] >= 25:
            add("ADX / DI", "support",
                f"ADX {a['adx']:.1f} — trend has conviction and DI lines agree")
        else:
            add("ADX / DI", "neutral",
                f"ADX {a['adx']:.1f} — trend is developing, not yet strong")
    else:
        add("ADX / DI", "na", "Not enough bars")

    r, r2 = ext.get("rs"), ext.get("rs2")
    if r and r.get("available"):
        tol = thresholds["rs_tol"]
        aligned = lambda x: x >= tol if direction == 1 else x <= -tol
        against = lambda x: x <= -tol if direction == 1 else x >= tol
        fmt = lambda x: ("+" if x >= 0 else "") + f"{x:.2f}%"
        has2 = bool(r2 and r2.get("available"))
        detail = f"{fmt(r['rsOpen'])} vs {r['benchTicker']}" + \
                 (f", {fmt(r2['rsOpen'])} vs {r2['benchTicker']}" if has2 else "") + \
                 " since open"
        if has2 and against(r2["rsOpen"]):
            add("Relative strength", "contradict",
                f"{detail} — {'lagging' if direction == 1 else 'outperforming'} "
                f"{r2['benchTicker']}, against the {d} read")
        elif aligned(r["rsOpen"]) and (not has2 or aligned(r2["rsOpen"])):
            add("Relative strength", "support",
                f"{detail} — {'outperforming' if direction == 1 else 'underperforming'} "
                f"{'both benchmarks' if has2 else 'the benchmark'}")
        else:
            add("Relative strength", "neutral",
                f"{detail} — roughly in line with the benchmark{'s' if has2 else ''}")
    else:
        add("Relative strength", "na", r.get("note") if r else "Unavailable")

    f = ext.get("flow") or {}
    if math.isfinite(f.get("primary", NAN)):
        pct = f["primary"] * 100
        if (direction == 1 and f["primary"] >= 0.2) or (direction == -1 and f["primary"] <= -0.2):
            add("Order flow", "support", f"Size imbalance {pct:.0f}% — aligned with the {d} read")
        elif (direction == 1 and f["primary"] <= -0.2) or (direction == -1 and f["primary"] >= 0.2):
            add("Order flow", "contradict", f"Size imbalance {pct:.0f}% — pressure is against the {d} read")
        else:
            add("Order flow", "neutral", f"Size imbalance {pct:.0f}% — balanced")
    else:
        add("Order flow", "na", f.get("note") or "Quote size data unavailable")

    o = ext.get("obv")
    if o and o.get("available"):
        want = "Accumulation" if direction == 1 else "Distribution"
        against_state = "Distribution" if direction == 1 else "Accumulation"
        div_against = o["divergence"].startswith("Bearish") if direction == 1 \
            else o["divergence"].startswith("Bullish")
        if div_against:
            add("OBV", "contradict", o["divergence"])
        elif o["state"] == want:
            add("OBV", "support", f"{o['state']} building across the session")
        elif o["state"] == against_state:
            add("OBV", "contradict", f"{o['state']} — cumulative volume is against the {d} read")
        else:
            add("OBV", "neutral", "Cumulative volume is neutral / mixed")
    else:
        add("OBV", "na", o.get("note") if o else "Unavailable")

    supports = sum(1 for i in items if i["status"] == "support")
    contradicts = sum(1 for i in items if i["status"] == "contradict")
    net = supports - contradicts + (1 if strong else 0)
    conviction = "High" if net >= 3 else ("Moderate" if net >= 1 else "Low")
    return {"items": items, "supports": supports, "contradicts": contradicts,
            "conviction": conviction}


def classify_verdict(pattern: dict | None, breakout: dict, trend: dict, daily: dict,
                     liquidity: dict, spread: dict | None, thresholds: dict,
                     ext: dict | None = None) -> dict:
    """Hard gates first; a directional verdict only if everything agrees."""
    ext = ext or {}

    def side(reason):
        # A Sideways verdict is itself a low-conviction directional call:
        # the hard gates refused to agree on a direction, so there is no
        # directional edge to size up. Show "Low" instead of a blank.
        return {"verdict": "Sideways", "dir": 0, "reason": reason,
                "scorecard": None, "conviction": "Low"}

    if daily["bias"] == "Mixed":
        return side("Daily (higher-timeframe) trend has no clear majority — "
                    "no higher-timeframe conviction to lean on.")
    if trend["bias"] == "Mixed":
        return side("Intraday trend indicators are split with no majority.")
    if trend["bias"] != daily["bias"]:
        return side(f"Intraday trend is {trend['bias']} but the daily trend is "
                    f"{daily['bias']} — no higher-timeframe confirmation.")
    if pattern and pattern["dir"] != 0 and (
            (trend["bias"] == "Bullish" and pattern["dir"] == -1)
            or (trend["bias"] == "Bearish" and pattern["dir"] == 1)):
        return side("The latest candlestick pattern contradicts the trend direction.")
    if liquidity and math.isfinite(liquidity.get("value", NAN)) \
            and liquidity["value"] < thresholds["min_dollar_vol"]:
        return side(f"Average dollar volume (${liquidity['value']:,.0f}/day) is below your "
                    f"${thresholds['min_dollar_vol']:,.0f} liquidity floor.")
    if spread and math.isfinite(spread.get("pct", NAN)) \
            and spread["pct"] > thresholds["max_spread_pct"]:
        return side(f"Bid-ask spread ({spread['pct']:.3f}%) is wider than your "
                    f"{thresholds['max_spread_pct']}% limit.")
    a = ext.get("adx") or {}
    if math.isfinite(a.get("adx", NAN)) and a["adx"] < thresholds["min_adx"]:
        return side(f"Trend indicators lean {trend['bias']}, but ADX is only {a['adx']:.1f} "
                    f"(below your {thresholds['min_adx']} minimum) — that's drifting/choppy "
                    "price action, not a real trend, so the other indicators are likely noise.")

    direction = 1 if trend["bias"] == "Bullish" else -1
    r = ext.get("rs")
    if r and r.get("available"):
        if direction == 1 and r["rsOpen"] < -thresholds["rs_tol"]:
            return side(f"Trend reads Bullish, but the stock is lagging the benchmark by "
                        f"{abs(r['rsOpen']):.2f}% — weak bullish signal.")
        if direction == -1 and r["rsOpen"] > thresholds["rs_tol"]:
            return side(f"Trend reads Bearish, but the stock is outperforming the benchmark by "
                        f"{r['rsOpen']:.2f}% — weak bearish signal.")

    strong = bool(pattern and pattern["dir"] == direction and breakout.get("confirmed"))
    scorecard = build_scorecard(direction, ext, thresholds, strong)
    reason = ("Daily trend, intraday trend, candlestick pattern, and 150-minute breakout "
              "all agree — highest-confidence "
              f"{trend['bias'].lower()} read." if strong else
              f"Daily and intraday trend agree ({trend['bias']})"
              + (", and price has confirmed the 150-minute breakout/breakdown."
                 if breakout.get("confirmed") else
                 ", though the 150-minute breakout/breakdown isn't confirmed yet — "
                 "lower-conviction than a full agreement."))
    reason += (f" Of the four context checks (ADX, relative strength, order flow, OBV), "
               f"{scorecard['supports']} support and {scorecard['contradicts']} contradict.")
    return {"verdict": trend["bias"], "dir": direction, "reason": reason,
            "scorecard": scorecard, "conviction": scorecard["conviction"]}


# ---------------------------------------------------------------------------
# Direction Radar — a LEADING read on what is changing right now
# ---------------------------------------------------------------------------
def swing_points(bars: list, w: int = 2) -> dict:
    H, L = [], []
    for i in range(w, len(bars) - w):
        hi = all(bars[i]["high"] > bars[i - j]["high"] and bars[i]["high"] > bars[i + j]["high"]
                 for j in range(1, w + 1))
        lo = all(bars[i]["low"] < bars[i - j]["low"] and bars[i]["low"] < bars[i + j]["low"]
                 for j in range(1, w + 1))
        if hi:
            H.append(bars[i]["high"])
        if lo:
            L.append(bars[i]["low"])
    return {"H": H, "L": L}


def compute_radar(df: pd.DataFrame, atr: float,
                  latest_price: float | None = None) -> dict | None:
    """Momentum/pressure gauge, -100..+100. Not a forecast.

    Without a live tick tape the 'Live tape' component is skipped and the
    remaining weights are renormalized, exactly as the HTML does when the
    tape isn't streaming yet.
    """
    if df is None or len(df) < 40:
        return None
    bars = _bar_dicts(df.tail(300))
    c = [b["close"] for b in bars]
    n = len(c)
    px = c[-1]
    atr = atr if atr and atr > 0 else px * 0.001
    comps = []

    def add(name, w, s, detail):
        if s is not None and math.isfinite(s):
            comps.append({"name": name, "w": w, "s": _clamp(s), "detail": detail})

    th = math.tanh
    e5, e13 = ema_series(c, 5), ema_series(c, 13)
    add("Fast momentum (EMA 5/13)", 0.18,
        0.5 * th((e5[n - 1] - e13[n - 1]) / atr * 1.5)
        + 0.5 * th((e5[n - 1] - e5[n - 4]) / atr * 2),
        f"EMA5 {_f2(e5[n - 1] - e13[n - 1])} vs EMA13 · 15m slope {_f2(e5[n - 1] - e5[n - 4])}")

    e12, e26 = ema_series(c, 12), ema_series(c, 26)
    ml = [a - b for a, b in zip(e12, e26)]
    sg = ema_series(ml, 9)
    hs = [a - b for a, b in zip(ml, sg)]
    add("MACD histogram trend", 0.14,
        0.7 * th((hs[n - 1] - hs[n - 4]) / atr * 6) + 0.3 * th(hs[n - 1] / atr * 6),
        f"hist {_f2(hs[n - 1])} · 15m change {_f2(hs[n - 1] - hs[n - 4])}")

    rs = rsi_series(c)
    add("RSI level & slope", 0.10,
        0.5 * th((rs[n - 1] - 50) / 12) + 0.5 * th((rs[n - 1] - rs[n - 4]) / 6),
        f"RSI {rs[n - 1]:.0f} · 15m change {_f2(rs[n - 1] - rs[n - 4])}")

    add("Price velocity", 0.14,
        0.6 * th((px - c[n - 7]) / atr / 2) + 0.4 * th((px - c[n - 13]) / atr / 3),
        f"{_f2((px - c[n - 7]) / c[n - 7] * 100)}% over 30m · "
        f"{_f2((px - c[n - 13]) / c[n - 13] * 100)}% over 60m")

    sw = swing_points(bars[-40:])
    H, L = sw["H"][-2:], sw["L"][-2:]
    last_h, last_l = sw["H"][-1] if sw["H"] else NAN, sw["L"][-1] if sw["L"] else NAN
    ss, brk = 0.0, ""
    if len(H) == 2:
        ss += 0.5 if H[1] > H[0] else -0.5
    if len(L) == 2:
        ss += 0.5 if L[1] > L[0] else -0.5
    if math.isfinite(last_l) and px < last_l:
        ss, brk = -1.0, f" · BROKE swing low ${last_l:.2f}"
    elif math.isfinite(last_h) and px > last_h:
        ss, brk = 1.0, f" · BROKE swing high ${last_h:.2f}"
    add("Swing structure", 0.20, ss,
        f"{(H[1] > H[0] and 'higher' or 'lower') + ' high' if len(H) == 2 else '—'}, "
        f"{(L[1] > L[0] and 'higher' or 'lower') + ' low' if len(L) == 2 else '—'}{brk}")

    td = _bar_dicts(session_bars(df))
    pv = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in td)
    vv = sum(b["volume"] for b in td)
    if vv > 0:
        vwap = pv / vv
        add("Position vs VWAP", 0.08, th((px - vwap) / atr * 1.2),
            f"{_f2(px - vwap)} vs VWAP ${vwap:.2f}")

    up = dn = 0.0
    for i in range(len(bars) - 6, len(bars)):
        d = bars[i]["close"] - bars[i - 1]["close"]
        v = bars[i]["volume"]
        if d > 0:
            up += v
        elif d < 0:
            dn += v
    if up + dn > 0:
        add("Volume pressure (30m)", 0.12, (up - dn) / (up + dn),
            f"{_f2((up - dn) / (up + dn) * 100)}% net up-volume, last 6 bars")

    W = sum(x["w"] for x in comps)
    if not W:
        return None
    score = round(max(-100, min(100, sum(x["w"] * x["s"] for x in comps) / W * 130)))
    return {"score": score, "comps": comps, "px": px, "tape": False,
            "lastH": last_h, "lastL": last_l,
            "up": sum(1 for x in comps if x["s"] > 0.15),
            "dn": sum(1 for x in comps if x["s"] < -0.15)}


# ---------------------------------------------------------------------------
# Live pressure checks (tape + 1-min frame)
# ---------------------------------------------------------------------------
def compute_pressure(df: pd.DataFrame, df_1m: pd.DataFrame | None,
                     atr: float) -> dict:
    """Absorption/velocity need a live tick tape (unavailable here);
    the 1-minute fast-momentum check runs on bars."""
    out = {"available": False, "note": "", "checks": [], "lean": 0.0,
           "leanLabel": "no tape", "summary": ""}
    checks = []

    def add(name, state, detail):
        checks.append({"name": name, "state": state, "detail": detail})

    add("Absorption (5 min)", "",
        "No live tick tape in this app — needs a streaming trade feed.")
    add("Tape velocity", "",
        "No live tick tape in this app — needs a streaming trade feed.")

    fast_lean, fast_ok = 0, False
    if df_1m is not None and len(df_1m) >= 20:
        c1 = df_1m["Close"].astype(float).tolist()
        n1 = len(c1)
        e5 = ema_series(c1, 5)
        a1 = atr_last(df_1m, 14)
        slope = (e5[n1 - 1] - e5[n1 - 6]) / a1 \
            if math.isfinite(a1) and a1 > 0 else NAN
        fast_lean = 1 if slope > 0.5 else (-1 if slope < -0.5 else 0)
        fast_ok = True
        div = ""
        td = session_bars(df)
        if len(td):
            px, d_hi, d_lo = c1[n1 - 1], float(td["High"].max()), float(td["Low"].min())
            if px >= d_hi * 0.999 and fast_lean <= 0:
                div = " — new session high without fast-momentum confirmation (fade risk)"
            elif px <= d_lo * 1.001 and fast_lean >= 0:
                div = " — new session low without fast-momentum confirmation (bounce risk)"
        side = ("buyers own the fast frame" if fast_lean > 0 else
                "sellers own the fast frame" if fast_lean < 0 else "fast frame flat")
        add("Fast momentum (1-min)", "warn" if div else ("ok" if fast_lean else ""),
            f"1-min EMA5 slope {slope:+.2f}× ATR → {side}{div}")
    else:
        add("Fast momentum (1-min)", "", "1-min bars unavailable (need 20+ bars)")

    out["checks"] = checks
    out["available"] = fast_ok
    if not fast_ok:
        out["note"] = "Waiting for 1-min bars…"
        return out
    out["lean"] = float(fast_lean)
    out["leanLabel"] = ("buyers in control" if fast_lean >= 1 else
                        "sellers in control" if fast_lean <= -1 else "balanced / mixed")
    out["summary"] = f"composite tape lean {fast_lean:+.2f} — {out['leanLabel']} " \
                     "(1-min frame only; no live tick tape)"
    return out


# ---------------------------------------------------------------------------
# Intraday direction forecast — weighted ensemble, next ~45 minutes
# ---------------------------------------------------------------------------
FORECAST_HORIZON_MIN = 45


def compute_forecast(radar: dict | None, pressure: dict, flow: dict,
                     obv: dict, rs: dict, df: pd.DataFrame, atr: float,
                     px: float) -> dict | None:
    """UP / DOWN / SIDEWAYS with agreement-based probabilities.

    Probabilities come from signal agreement, not a backtested model —
    treat 60% as "leaning", not "likely".
    """
    if not radar:
        return None
    atr = atr if atr and atr > 0 else max(px * 0.001, 1e-9)
    td = _bar_dicts(session_bars(df))
    feats = []

    def addF(name, w, v, detail):
        if v is not None and math.isfinite(v):
            feats.append({"name": name, "w": w, "v": _clamp(v), "detail": detail})

    addF("Radar momentum", 0.25, radar["score"] / 100,
         f"score {radar['score']:+d} / 100")
    tape_lean = pressure["lean"] if pressure.get("available") else 0.0
    tape_d = pressure["summary"] if pressure.get("available") else "no live tape"
    addF("Live tape lean", 0.15, tape_lean, tape_d)
    addF("Order-flow imbalance", 0.10,
         flow.get("primary", NAN),
         "n/a" if not math.isfinite(flow.get("primary", NAN))
         else f"bid/ask size {(flow['primary'] * 100):+.0f}%")
    addF("OBV trajectory (1h)", 0.10,
         obv.get("recentRatio", NAN) if obv.get("available") else NAN,
         "n/a" if not obv.get("available")
         else f"{obv['state']}, last-hour net {(obv['recentRatio'] * 100):+.0f}% of volume")
    rs60 = rs.get("rs60", NAN) if rs.get("available") else NAN
    addF("Relative strength (60m)", 0.10,
         math.tanh(rs60 / 0.75) if math.isfinite(rs60) else NAN,
         "n/a" if not math.isfinite(rs60)
         else f"{rs60:+.2f}% vs {rs['benchTicker']} (60m)")

    vwap = NAN
    if td:
        pv = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in td)
        vv = sum(b["volume"] for b in td)
        vwap = pv / vv if vv > 0 else NAN
    addF("Position vs VWAP", 0.10,
         math.tanh((px - vwap) / atr * 1.2) if math.isfinite(vwap) else NAN,
         "n/a" if not math.isfinite(vwap) else f"{_f2(px - vwap)} vs VWAP ${vwap:.2f}")

    rp_v, rp_d = 0.0, "n/a"
    if td:
        hi = max(b["high"] for b in td)
        lo = min(b["low"] for b in td)
        if hi > lo:
            rp = (px - lo) / (hi - lo)
            if radar["score"] >= 20 and rp > 0.75:
                rp_v, rp_d = 0.8, f"{rp * 100:.0f}% of day range with strong momentum — continuation"
            elif radar["score"] <= -20 and rp < 0.25:
                rp_v, rp_d = -0.8, f"{rp * 100:.0f}% of day range with strong momentum — continuation"
            elif abs(radar["score"]) < 20 and rp > 0.85:
                rp_v, rp_d = -0.5, f"stretched to {rp * 100:.0f}% of range, momentum flat — fade risk"
            elif abs(radar["score"]) < 20 and rp < 0.15:
                rp_v, rp_d = 0.5, f"stretched to {rp * 100:.0f}% of range, momentum flat — bounce risk"
            else:
                rp_d = f"mid-range ({rp * 100:.0f}% of ${lo:.2f}–${hi:.2f})"
    addF("Range position", 0.10, rp_v, rp_d)

    gap_v, gap_d = 0.0, "no meaningful gap"
    if td:
        all_bars = _bar_dicts(df)
        idx = next((i for i, b in enumerate(all_bars) if b["time"] == td[0]["time"]), None)
        pc = all_bars[idx - 1]["close"] if idx else NAN
        if math.isfinite(pc) and pc > 0:
            g = (td[0]["open"] - pc) / pc * 100
            if g >= 1 and px > pc:
                gap_v, gap_d = -0.4 * min(1, g / 2), \
                    f"gapped up {_f2(g)}%, unfilled — gravity toward ${pc:.2f}"
            elif g <= -1 and px < pc:
                gap_v, gap_d = 0.4 * min(1, -g / 2), \
                    f"gapped down {_f2(g)}%, unfilled — gravity toward ${pc:.2f}"
            elif abs(g) >= 1:
                gap_d = f"{_f2(g)}% gap already filled"
    addF("Overnight gap pull", 0.10, gap_v, gap_d)

    W = sum(f["w"] for f in feats) or 1
    score = sum(f["w"] * f["v"] for f in feats) / W
    direction = 1 if score >= 0.12 else (-1 if score <= -0.12 else 0)
    label = "UP" if direction == 1 else ("DOWN" if direction == -1 else "SIDEWAYS")
    agree = sum(f["w"] for f in feats if f["v"] * direction > 0.05) / W
    conf = min(0.92, 0.50 + abs(score) * 0.70 + (agree - 0.5) * 0.30)
    if direction == 1:
        p_up, p_down, p_side = conf, (1 - conf) * 0.3, (1 - conf) * 0.7
    elif direction == -1:
        p_down, p_up, p_side = conf, (1 - conf) * 0.3, (1 - conf) * 0.7
    else:
        p_side, p_up, p_down = conf, (1 - conf) / 2, (1 - conf) / 2
    drivers = sorted(feats, key=lambda f: abs(f["w"] * f["v"]), reverse=True)[:3]
    drivers = [{"name": d["name"], "dir": _sign(d["v"]), "detail": d["detail"]}
               for d in drivers]
    return {"dir": direction, "label": label, "conf": conf,
            "pUp": p_up, "pDown": p_down, "pSide": p_side,
            "score": score, "feats": feats, "drivers": drivers,
            "expMove": atr * 0.6, "px": px}


# ---------------------------------------------------------------------------
# Provisional trade plan (radar-led, checklist-gated, reduced size)
# ---------------------------------------------------------------------------
def provisional_plan(radar: dict | None, df: pd.DataFrame, atr: float,
                     ctx: dict, hist: list | None = None) -> dict | None:
    """Checklist-gated provisional plan. Returns None when blocked or neutral."""
    if not radar or not ctx:
        return None
    score, px = radar["score"], radar["px"]
    bars = _bar_dicts(df)
    atr = atr if atr and atr > 0 else px * 0.001
    direction = 1 if score >= 20 else (-1 if score <= -20 else 0)
    side = "LONG" if direction == 1 else "SHORT"
    res = ctx["result"]

    rb = bars[-31:-1]
    range_high = max(b["high"] for b in rb)
    range_low = min(b["low"] for b in rb)
    bo = {"rangeHigh": range_high, "rangeLow": range_low}
    long_trig = radar["lastH"] if math.isfinite(radar["lastH"]) else range_high
    short_trig = radar["lastL"] if math.isfinite(radar["lastL"]) else range_low

    def trig(d, level):
        pl = risk_plan(df, level, d, atr, bo)
        sz = position_size(ctx["account_size"], ctx["risk_pct"] * 0.5,
                           pl["stopDist"], level)
        word = "above" if d == 1 else "below"
        return {"dir": d, "level": level, "stop": pl["stop"],
                "target2R": pl["target2R"], "shares": sz["shares"],
                "text": f"{'Long' if d == 1 else 'Short'} on a 5-min close {word} "
                        f"${level:.2f} → stop ${pl['stop']:.2f}, "
                        f"2R target ${pl['target2R']:.2f}, ~{sz['shares']:,} sh"}

    triggers = {"long": trig(1, long_trig) if math.isfinite(long_trig) else None,
                "short": trig(-1, short_trig) if math.isfinite(short_trig) else None}

    # Hard gates block the plan completely.
    hard = []
    if ctx.get("liquidity") and not ctx["liquidity"].get("ok", True):
        hard.append("liquidity below your floor")
    if ctx.get("spread") and not ctx["spread"].get("ok", True):
        hard.append("bid-ask spread wider than your limit")
    if hard:
        return {"blocked": True, "reason": "Blocked by hard gate: " + " and ".join(hard),
                "triggers": triggers}

    if direction == 0:
        return {"blocked": False, "neutral": True, "score": score,
                "reason": f"Radar is neutral ({score:+d}) — no provisional direction. "
                          "Use the conditional trigger levels.",
                "triggers": triggers}

    checks = []

    def add(name, state, detail):
        checks.append({"name": name, "state": state, "detail": detail})

    add("Radar strength", "ok" if direction * score >= 45 else "warn",
        f"{score:+d} ({'strong' if direction * score >= 45 else 'leaning'}, need ±45 for full pass)")

    if hist and len(hist) >= 2:
        span = hist[-1][0] - hist[0][0]
        frac = sum(1 for _, s in hist if s * direction >= 20) / len(hist)
        add("Radar persistence",
            "ok" if frac >= 0.8 else ("warn" if frac >= 0.5 else "fail"),
            f"{frac * 100:.0f}% of the sampled window on this side (need 80%)")
    else:
        add("Radar persistence", "warn", "building — not enough score history yet")

    lvl = radar["lastH"] if direction == 1 else radar["lastL"]
    if math.isfinite(lvl):
        brk = px > lvl if direction == 1 else px < lvl
        add("Swing level broken", "ok" if brk else "fail",
            f"{'price is past' if brk else 'price has not passed'} "
            f"{'swing high' if direction == 1 else 'swing low'} ${lvl:.2f}")

    td = _bar_dicts(session_bars(df))
    pv = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in td)
    vv = sum(b["volume"] for b in td)
    if vv > 0:
        vw = pv / vv
        ok = px > vw if direction == 1 else px < vw
        add(f"Price {'above' if direction == 1 else 'below'} VWAP",
            "ok" if ok else "fail", f"${px:.2f} vs VWAP ${vw:.2f}")

    closes = [b["close"] for b in bars]
    n = len(closes)
    e20, e50 = ema_series(closes, 20), ema_series(closes, 50)
    ok = (px > e20[n - 1] and e20[n - 1] > e50[n - 1]) if direction == 1 \
        else (px < e20[n - 1] and e20[n - 1] < e50[n - 1])
    part = (px > e20[n - 1]) if direction == 1 else (px < e20[n - 1])
    add("5-min EMA 20/50 structure", "ok" if ok else ("warn" if part else "fail"),
        f"price ${px:.2f} · EMA20 ${e20[n - 1]:.2f} · EMA50 ${e50[n - 1]:.2f}")

    e13 = ema_series(closes, 13)
    last3 = closes[-3:]
    ok3 = all(x > e13[n - 1] if direction == 1 else x < e13[n - 1] for x in last3)
    add("Last 3 closes vs EMA13", "ok" if ok3 else "warn",
        "all 3 closed on the right side of EMA13" if ok3
        else "not all 3 closes on the right side of EMA13 — move not yet established")

    a = adx(df, 14)
    if math.isfinite(a["adx"]):
        di_ok = a["plusDI"] > a["minusDI"] if direction == 1 else a["minusDI"] > a["plusDI"]
        add("ADX trend strength",
            "ok" if (a["adx"] >= ctx["thresholds"]["min_adx"] and di_ok)
            else ("warn" if a["adx"] >= ctx["thresholds"]["min_adx"] else "fail"),
            f"ADX {a['adx']:.1f} (min {ctx['thresholds']['min_adx']}) · "
            f"+DI {a['plusDI']:.0f} / −DI {a['minusDI']:.0f}"
            + ("" if di_ok else " — DI disagrees"))
    else:
        add("ADX trend strength", "", "Not enough bars")

    rs = ctx.get("rs")
    if rs and rs.get("available"):
        v = direction * rs["rsOpen"]
        add(f"Relative strength vs {rs['benchTicker']}",
            "ok" if v >= 0 else ("warn" if v >= -ctx["thresholds"]["rs_tol"] else "fail"),
            f"{rs['rsOpen']:+.2f}% since open")
    else:
        add("Relative strength", "", (rs or {}).get("note", "n/a"))

    vols = [b["volume"] for b in bars]
    recent = sum(vols[-3:]) / 3 if len(vols) >= 3 else NAN
    base = sum(vols[-23:-3]) / 20 if len(vols) >= 23 else NAN
    rel = recent / base if base and base > 0 else NAN
    vp = next((x for x in radar["comps"] if x["name"].startswith("Volume pressure")), None)
    flow_ok = bool(vp and vp["s"] * direction > 0.15)
    flow_bad = bool(vp and vp["s"] * direction < -0.15)
    if math.isfinite(rel):
        add("Volume confirmation",
            "fail" if (flow_bad or rel < 0.8)
            else ("ok" if (rel >= 1.2 and flow_ok) else "warn"),
            f"relative volume {rel:.2f}× (need 1.2×) · "
            f"flow {'agrees' if flow_ok else ('opposes' if flow_bad else 'neutral')}")
    else:
        add("Volume confirmation", "", "n/a")

    pr = ctx.get("pressure")
    if pr and pr.get("available"):
        align = pr["lean"] * direction
        add("Live tape pressure (1-min frame)",
            "fail" if align <= -0.25 else ("ok" if align >= 0.25 else "warn"),
            f"{pr['summary']} — "
            f"{'tape opposes the radar direction' if align <= -0.25 else
               ('tape agrees with the radar direction' if align >= 0.25 else 'tape is mixed / flat')}")
    else:
        add("Live tape pressure", "", "n/a — no live tape")

    db = ctx["daily"]["bias"]
    aligned_daily = (db == "Bullish" and direction == 1) or (db == "Bearish" and direction == -1)
    add("Daily trend alignment",
        "ok" if aligned_daily else ("warn" if db == "Mixed" else "fail"),
        f"daily is {db}"
        + ("" if db == "Mixed" else (" — with the trade" if aligned_daily
                                    else " — this is a counter-trend trade")))

    ext = abs(px - e20[n - 1]) / atr if atr > 0 else NAN
    add("Not overextended",
        "ok" if ext <= 2 else ("warn" if ext <= 3 else "fail") if math.isfinite(ext) else "",
        f"{ext:.1f}× ATR from EMA20 (>2× = chasing; wait for a pullback)"
        if math.isfinite(ext) else "n/a")

    plan = risk_plan(df, px, direction, atr, bo)
    rr = plan["rr"]
    add("Reward : risk ≥ 1.5",
        "warn" if not math.isfinite(rr) else ("ok" if rr >= 1.5 else ("warn" if rr >= 1 else "fail")),
        f"{rr:.2f} : 1 to measured move" if math.isfinite(rr) else "n/a — range unavailable")

    pts = sum(1 if c["state"] == "ok" else 0.5 if c["state"] == "warn" else 0 for c in checks)
    scored = [c for c in checks if c["state"] in ("ok", "warn", "fail")]
    pct = pts / len(scored) if scored else 0
    if pct >= 0.75:
        tier, mult = "Moderate", 0.5
    elif pct >= 0.55:
        tier, mult = "Low", 0.25
    else:
        tier, mult = "Unconfirmed", 0.0
    counter_daily = db != "Mixed" and not aligned_daily
    contradicts = res["dir"] == -direction
    if mult > 0 and (counter_daily or contradicts):
        mult = min(mult, 0.25) / (2 if (counter_daily and contradicts) else 1)

    sizing = None
    if mult > 0:
        sz = position_size(ctx["account_size"], ctx["risk_pct"] * mult, plan["stopDist"], px)
        sizing = {"side": side, "entry": px, "stop": plan["stop"],
                  "stopDist": plan["stopDist"], "measuredTarget": plan["measuredTarget"],
                  "target2R": plan["target2R"], "riskPct": ctx["risk_pct"] * mult,
                  "riskAmount": sz["riskAmount"], "shares": sz["shares"],
                  "positionValue": sz["positionValue"],
                  "pctOfAccount": sz["pctOfAccount"]}
    return {"blocked": False, "neutral": False, "score": score, "dir": direction,
            "side": side, "checks": checks, "tier": tier, "mult": mult,
            "sizing": sizing, "triggers": triggers,
            "contradictsVerdict": contradicts, "counterDaily": counter_daily,
            "passed": sum(1 for c in checks if c["state"] == "ok"),
            "total": len(scored), "pct": pct}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"min_dollar_vol": 10_000_000, "max_spread_pct": 0.50,
                      "min_adx": 20.0, "rs_tol": 0.25}


def analyze_symbol(symbol: str, df_daily: pd.DataFrame, df_5m: pd.DataFrame | None,
                   df_1m: pd.DataFrame | None = None,
                   bench_5m: pd.DataFrame | None = None,
                   bench2_5m: pd.DataFrame | None = None,
                   bench_ticker: str = "SPY", bench2_ticker: str | None = None,
                   quote: dict | None = None, account_size: float = 0.0,
                   risk_pct: float = 0.0,
                   thresholds: dict | None = None) -> dict:
    """Run the whole analyzer engine. Returns every section the UI needs."""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    out: dict = {"symbol": symbol, "ok": False}
    if df_5m is None or len(df_5m) < 31:
        out["error"] = (f"Not enough 5-min bars for {symbol} "
                        "(need 31+). Market may be closed or symbol may be invalid.")
        return out

    pattern = detect_pattern(df_5m)
    breakout = check_breakout(df_5m, pattern["dir"] if pattern else 0)
    trend = current_trend(df_5m)
    daily = daily_trend(df_daily)
    liq_value = avg_dollar_volume(df_daily)
    liquidity = {"value": liq_value,
                 "ok": math.isfinite(liq_value) and liq_value >= th["min_dollar_vol"]}
    spread_raw = spread_info(quote)
    spread = {**spread_raw, "ok": spread_raw["pct"] <= th["max_spread_pct"]} \
        if spread_raw else None
    entry = trend["price"] if math.isfinite(trend["price"]) else float(df_5m["Close"].iloc[-1])
    atr = atr_last(df_5m, 14)
    closes_5m = df_5m["Close"].astype(float).tolist()
    bb = bollinger(closes_5m, 20, 2)
    levels = next_levels(entry, breakout, bb, trend["vwap"], daily, atr)

    td = session_bars(df_5m)
    session_open = float(td["Open"].iloc[0]) if len(td) else NAN
    change_open = (entry - session_open) / session_open * 100 \
        if math.isfinite(session_open) and session_open > 0 else NAN

    adx_5m = adx(df_5m, 14)
    adx_daily = adx(df_daily, 14) if df_daily is not None and len(df_daily) >= 29 else \
        {"adx": NAN, "plusDI": NAN, "minusDI": NAN}
    bench_price = float(bench_5m["Close"].iloc[-1]) if bench_5m is not None \
        and len(bench_5m) else None
    rs = relative_strength(df_5m, entry, bench_5m, bench_price, bench_ticker) \
        if bench_ticker != symbol else \
        {"available": False, "benchTicker": bench_ticker,
         "note": "Ticker is the benchmark itself"}
    if bench2_ticker:
        b2_price = float(bench2_5m["Close"].iloc[-1]) if bench2_5m is not None \
            and len(bench2_5m) else None
        rs2 = relative_strength(df_5m, entry, bench2_5m, b2_price, bench2_ticker)
    else:
        rs2 = {"available": False, "benchTicker": None, "note": "No secondary benchmark set"}
    flow = order_flow(quote)
    obv = obv_session(df_5m)

    result = classify_verdict(pattern, breakout, trend, daily, liquidity, spread, th,
                              {"adx": adx_5m, "rs": rs, "rs2": rs2,
                               "flow": flow, "obv": obv})
    squeeze = detect_squeeze(df_5m)
    mean_rev = mean_reversion_note(df_5m)
    paired = paired_confirmations(trend, result["dir"])

    radar = compute_radar(df_5m, atr)
    pressure = compute_pressure(df_5m, df_1m, atr)
    forecast = compute_forecast(radar, pressure, flow, obv, rs, df_5m, atr, entry)

    trade_plan = None
    if result["dir"] != 0:
        plan = risk_plan(df_5m, entry, result["dir"], atr, breakout)
        size = position_size(account_size, risk_pct, plan["stopDist"], entry)
        trade_plan = {"plan": plan, "size": size}

    ctx = {"account_size": account_size, "risk_pct": risk_pct,
           "thresholds": th, "result": result, "daily": daily, "rs": rs,
           "liquidity": liquidity, "spread": spread, "pressure": pressure}
    provisional = provisional_plan(radar, df_5m, atr, ctx)

    out.update({
        "ok": True, "pattern": pattern, "breakout": breakout, "trend": trend,
        "daily": daily, "liquidity": liquidity, "spread": spread,
        "entry": entry, "atr": atr, "bb": bb, "levels": levels,
        "sessionOpen": session_open, "changeSinceOpen": change_open,
        "adx5m": adx_5m, "adxDaily": adx_daily, "rs": rs, "rs2": rs2,
        "flow": flow, "obv": obv, "verdict": result, "squeeze": squeeze,
        "meanRev": mean_rev, "paired": paired, "radar": radar,
        "pressure": pressure, "forecast": forecast, "tradePlan": trade_plan,
        "provisional": provisional, "benchTicker": bench_ticker,
        "bench2Ticker": bench2_ticker,
    })
    return out


# ---------------------------------------------------------------------------
# Legacy helpers (kept for the watchlist + older callers)
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns. Add your own calculations here."""
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI (14), Wilder smoothing — matches the HTML rsiSeries.
    rs = rsi_series(df["Close"].tolist(), 14)
    df["RSI"] = rs

    # MACD(12,26,9).
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() \
        - df["Close"].ewm(span=26, adjust=False).mean()
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2).
    mid = df["Close"].rolling(20).mean()
    sd = df["Close"].rolling(20).std(ddof=0)
    df["BB_UP"], df["BB_MID"], df["BB_LO"] = mid + 2 * sd, mid, mid - 2 * sd

    # ATR (14): typical daily price range, used for stop-loss / target sizing.
    prev_close = df["Close"].shift()
    true_range = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(),
         (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def trend_label(df: pd.DataFrame) -> str:
    """Plain-English trend from the moving averages."""
    last = df.iloc[-1]
    if pd.isna(last["SMA50"]):
        return "Not enough data"
    if last["Close"] > last["SMA20"] > last["SMA50"]:
        return "Uptrend"
    if last["Close"] < last["SMA20"] < last["SMA50"]:
        return "Downtrend"
    return "Sideways"


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the top of the page."""
    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
    return {
        "price": float(last),
        "change": float(last - prev),
        "change_pct": float((last / prev - 1) * 100),
        "rsi": float(df["RSI"].iloc[-1]),
        "trend": trend_label(df),
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
    }
