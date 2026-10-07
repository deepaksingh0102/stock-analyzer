"""
Entry / exit signals — computed locally by the analyzer engine
(ported from single-stock-analyzer.html).

Market data comes from Alpaca when APCA_API_KEY_ID / APCA_API_SECRET_KEY
are set in config.py (the HTML's API configuration), otherwise from
Yahoo Finance. Set the keys in config.py to use the HTML's data source.
"""
from dataclasses import dataclass

import pandas as pd

import config
import data as engine


@dataclass
class Signal:
    action: str                      # "BUY", "SELL", "HOLD" or "ERROR"
    entry: float | None = None       # suggested entry price
    target: float | None = None      # suggested exit / take-profit price
    stop_loss: float | None = None   # suggested exit if the trade goes wrong
    confidence: float | None = None  # 0-1, from the verdict's conviction
    note: str = ""
    source: str = "analyzer"         # "analyzer" or "demo"
    verdict: str = "Sideways"        # analyzer verdict: Bullish / Bearish / Sideways
    conviction: str | None = None    # High / Moderate / Low (analyzer only)
    direction: int = 0               # +1 / -1 / 0


def alpaca_is_configured() -> bool:
    """True once real Alpaca keys are pasted into config.py."""
    return engine.alpaca_is_configured()


def get_signal(symbol: str, df_daily: pd.DataFrame,
               df_5m: pd.DataFrame | None = None,
               bench_5m: pd.DataFrame | None = None,
               account_size: float = 0.0, risk_pct: float = 0.0,
               res: dict | None = None, quote: dict | None = None) -> Signal:
    """Signal for one symbol, from the analyzer's hard-gate verdict."""
    return analyzer_signal(symbol, df_daily, df_5m, bench_5m,
                           account_size=account_size, risk_pct=risk_pct,
                           res=res, quote=quote)


# ---------------------------------------------------------------------------
# Analyzer engine (local computation)
# ---------------------------------------------------------------------------
_CONVICTION_TO_CONFIDENCE = {"High": 0.75, "Moderate": 0.60, "Low": 0.45}


def analyzer_signal(symbol: str, df_daily: pd.DataFrame,
                    df_5m: pd.DataFrame | None = None,
                    bench_5m: pd.DataFrame | None = None,
                    account_size: float = 0.0, risk_pct: float = 0.0,
                    res: dict | None = None, quote: dict | None = None) -> Signal:
    """Verdict -> signal. Falls back to the demo rule when no intraday data.

    Pass a precomputed ``res`` (from ``data.analyze_symbol``) and ``quote``
    to avoid running the analysis twice.
    """
    if df_5m is None or len(df_5m) < 31:
        return demo_signal(df_daily)
    if res is None:
        if quote is None:
            quote, _ = engine.fetch_quote(symbol)
        res = engine.analyze_symbol(
            symbol, df_daily, df_5m, bench_5m=bench_5m,
            bench_ticker=config.BENCHMARK, quote=quote,
            account_size=account_size or config.ACCOUNT_SIZE,
            risk_pct=risk_pct or config.RISK_PCT,
            thresholds={"min_dollar_vol": config.MIN_DOLLAR_VOL,
                        "max_spread_pct": config.MAX_SPREAD_PCT,
                        "min_adx": config.MIN_ADX,
                        "rs_tol": config.RS_TOL_PCT},
        )
    if not res.get("ok"):
        return Signal(action="ERROR", note=res.get("error", "Analysis failed."),
                      source="analyzer")
    v = res["verdict"]
    action = "BUY" if v["dir"] == 1 else ("SELL" if v["dir"] == -1 else "HOLD")
    entry = target = stop = None
    if res.get("tradePlan"):
        plan = res["tradePlan"]["plan"]
        entry = res["entry"]
        target = plan["measuredTarget"]
        stop = plan["stop"]
    return Signal(
        action=action, entry=entry, target=target, stop_loss=stop,
        confidence=_CONVICTION_TO_CONFIDENCE.get(v["conviction"], 0.50),
        note=v["reason"], source="analyzer",
        verdict=v["verdict"], conviction=v["conviction"], direction=v["dir"],
    )


def daily_signal(df_daily: pd.DataFrame) -> Signal:
    """Lightweight daily-only signal for the watchlist (no intraday fetch)."""
    d = engine.daily_trend(df_daily)
    action = {"Bullish": "BUY", "Bearish": "SELL"}.get(d["bias"], "HOLD")
    price = d["price"]
    atr = engine.atr_last(df_daily, 14)
    stop = target = None
    if action != "HOLD" and pd.notna(atr):
        sgn = 1 if action == "BUY" else -1
        stop, target = price - sgn * 2 * atr, price + sgn * 3 * atr
    return Signal(action=action, entry=price if action != "HOLD" else None,
                  target=target, stop_loss=stop, confidence=0.50,
                  note=f"Daily trend: {d['bias']} ({d['bull']} bull / {d['bear']} bear votes).",
                  source="analyzer", verdict=d["bias"], direction=1 if action == "BUY"
                  else (-1 if action == "SELL" else 0))


# ---------------------------------------------------------------------------
# Demo rule (used when the analyzer can't run: no intraday data)
# ---------------------------------------------------------------------------
def demo_signal(df: pd.DataFrame) -> Signal:
    """20/50-day moving-average rule with ATR-based exits. For illustration only."""
    last = df.iloc[-1]
    if pd.isna(last["SMA50"]) or pd.isna(last["ATR"]):
        return Signal(action="HOLD", note="Not enough history for a signal.",
                      source="demo")

    price, atr = float(last["Close"]), float(last["ATR"])
    if last["SMA20"] > last["SMA50"]:
        return Signal(
            action="BUY", entry=price, target=price + 3 * atr, stop_loss=price - 2 * atr,
            note="Demo rule: 20-day average above 50-day average (uptrend).",
            source="demo", verdict="Bullish", direction=1,
        )
    return Signal(
        action="SELL", entry=price, target=price - 3 * atr, stop_loss=price + 2 * atr,
        note="Demo rule: 20-day average below 50-day average (downtrend).",
        source="demo", verdict="Bearish", direction=-1,
    )
