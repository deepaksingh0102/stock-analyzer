"""
Quick self-check.   Run:   python test_app.py          (no internet needed)
                    or:    python test_app.py --live   (also checks real Yahoo data)
"""
import math
import sys

import pandas as pd

import config
import data
import signals
from charts import price_chart, radar_history_figure, forecast_figure


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        sys.exit(1)


def isna(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# 1. Data + indicators
# ---------------------------------------------------------------------------
df = data.add_indicators(data.demo_prices("AAPL"))
check("demo prices generated", len(df) >= 500)
check("indicators calculated",
      df[["SMA20", "SMA50", "EMA20", "EMA50", "RSI", "ATR", "MACD", "BB_UP"]].iloc[-1].notna().all())
check("RSI in 0-100", df["RSI"].dropna().between(0, 100).all())
s = data.summary(df)
check("summary works", s["trend"] in ("Uptrend", "Downtrend", "Sideways"))

df5 = data.demo_intraday("AAPL")
check("demo intraday generated", len(df5) >= 200)

closes = df5["Close"].tolist()
check("sma_last", abs(data.sma_last(closes, 20) - sum(closes[-20:]) / 20) < 1e-9)
check("ema_last finite", math.isfinite(data.ema_last(closes, 20)))
check("rsi_last in 0-100", 0 <= data.rsi_last(closes) <= 100)
macd, sigline = data.macd_last(closes)
check("macd finite", math.isfinite(macd) and math.isfinite(sigline))
check("vwap in bar range",
      df5.tail(78)["Low"].min() <= data.vwap_last(df5) <= df5.tail(78)["High"].max())
check("atr positive", data.atr_last(df5) > 0)
bb = data.bollinger(closes)
check("bollinger ordered", bb["lower"] < bb["mid"] < bb["upper"])
a = data.adx(df5)
check("adx in 0-100", 0 <= a["adx"] <= 100 and a["plusDI"] >= 0 and a["minusDI"] >= 0)
obv = data.obv_session(df5)
check("obv session", obv["available"] and -1 <= obv["sessionRatio"] <= 1)
check("squeeze runs", isinstance(data.detect_squeeze(df5)["squeeze"], bool))

# ---------------------------------------------------------------------------
# 2. Patterns + breakout
# ---------------------------------------------------------------------------
pat_df = pd.DataFrame(
    {"Open": [100.0, 99.0, 98.0], "High": [101.0, 100.0, 102.0],
     "Low": [98.0, 97.0, 97.5], "Close": [99.0, 98.0, 101.0],
     "Volume": [1000.0, 1000.0, 3000.0]},
    index=pd.bdate_range("2026-01-05", periods=3))
pat = data.detect_pattern(pat_df)
check("bullish engulfing detected", pat is not None and pat["label"] == "Bullish Engulfing"
      and pat["dir"] == 1)

idx = pd.date_range("2026-10-05 09:30", periods=35, freq="5min")
bo_df = pd.DataFrame(
    {"Open": [99.5] * 35, "High": [100.0] * 34 + [101.5],
     "Low": [99.0] * 35, "Close": [99.5] * 34 + [101.0],
     "Volume": [1000.0] * 35}, index=idx)
bo = data.check_breakout(bo_df, 1)
check("150m breakout confirmed", bo["confirmed"] is True)
bo2 = data.check_breakout(bo_df, -1)
check("no false breakdown", bo2["confirmed"] is False)

# ---------------------------------------------------------------------------
# 3. Trend votes
# ---------------------------------------------------------------------------
trend = data.current_trend(df5)
check("intraday trend votes", trend["bias"] in ("Bullish", "Bearish", "Mixed")
      and math.isfinite(trend["bull"]))
daily = data.daily_trend(df)
check("daily trend votes", daily["bias"] in ("Bullish", "Bearish", "Mixed"))

# ---------------------------------------------------------------------------
# 4. Verdict hard gates
# ---------------------------------------------------------------------------
th = data.DEFAULT_THRESHOLDS
liq = {"value": 50_000_000.0, "ok": True}
base_ext = {"adx": {"adx": 30.0, "plusDI": 25.0, "minusDI": 15.0}}
bull_trend = {"bias": "Bullish"}
bull_daily = {"bias": "Bullish"}
bo_none = {"confirmed": False, "rangeHigh": 110.0, "rangeLow": 100.0}

v = data.classify_verdict(None, bo_none, {"bias": "Mixed"}, bull_daily, liq, None, th, base_ext)
check("mixed intraday -> Sideways", v["verdict"] == "Sideways" and v["dir"] == 0)
v = data.classify_verdict(None, bo_none, bull_trend, {"bias": "Mixed"}, liq, None, th, base_ext)
check("mixed daily -> Sideways", v["verdict"] == "Sideways")
v = data.classify_verdict(None, bo_none, bull_trend, {"bias": "Bearish"}, liq, None, th, base_ext)
check("timeframe conflict -> Sideways", v["verdict"] == "Sideways")
v = data.classify_verdict(None, bo_none, bull_trend, bull_daily,
                          {"value": 1.0, "ok": False}, None, th, base_ext)
check("illiquid -> Sideways", v["verdict"] == "Sideways")
weak_adx = {"adx": {"adx": 10.0, "plusDI": 20.0, "minusDI": 12.0}}
v = data.classify_verdict(None, bo_none, bull_trend, bull_daily, liq, None, th, weak_adx)
check("low ADX -> Sideways", v["verdict"] == "Sideways")
v = data.classify_verdict(None, bo_none, bull_trend, bull_daily, liq, None, th, base_ext)
check("full agreement -> Bullish",
      v["verdict"] == "Bullish" and v["dir"] == 1 and v["conviction"] in ("High", "Moderate", "Low"))
check("scorecard built", v["scorecard"] is not None and len(v["scorecard"]["items"]) == 4)

# ---------------------------------------------------------------------------
# 5. Radar + forecast
# ---------------------------------------------------------------------------
radar = data.compute_radar(df5, data.atr_last(df5))
check("radar scores", radar is not None and -100 <= radar["score"] <= 100
      and len(radar["comps"]) >= 5)
pressure = data.compute_pressure(df5, None, data.atr_last(df5))
check("pressure degrades gracefully", isinstance(pressure["checks"], list))
fc = data.compute_forecast(radar, pressure, data.order_flow(),
                           {"available": False}, {"available": False},
                           df5, data.atr_last(df5), radar["px"])
check("forecast probabilities sum to 1",
      fc is not None and abs(fc["pUp"] + fc["pSide"] + fc["pDown"] - 1.0) < 1e-9
      and fc["label"] in ("UP", "DOWN", "SIDEWAYS"))
check("forecast drivers", len(fc["drivers"]) == 3)

# ---------------------------------------------------------------------------
# 6. Risk plan + sizing
# ---------------------------------------------------------------------------
plan = data.risk_plan(df5, 100.0, 1, 2.0, bo_none)
check("long stop below entry, targets above",
      plan["stop"] < 100.0 < plan["measuredTarget"] and plan["target2R"] > 100.0)
sz = data.position_size(100_000, 1.0, 2.0, 100.0)
check("position sizing math",
      sz["riskAmount"] == 1000.0 and sz["shares"] == 500 and sz["positionValue"] == 50000.0)

# ---------------------------------------------------------------------------
# 7. Signals: analyzer engine (offline) + demo fallback
# ---------------------------------------------------------------------------
check("alpaca not configured with placeholder keys", not signals.alpaca_is_configured())
sig = signals.get_signal("AAPL", df, df5)
check("analyzer signal", sig.action in ("BUY", "SELL", "HOLD") and sig.source == "analyzer"
      and sig.verdict in ("Bullish", "Bearish", "Sideways"))
sig_demo = signals.get_signal("AAPL", df)  # no intraday -> demo rule
check("demo fallback without intraday", sig_demo.source == "demo")
sig_wl = signals.daily_signal(df)
check("watchlist daily signal", sig_wl.action in ("BUY", "SELL", "HOLD"))

# ---------------------------------------------------------------------------
# 8. Alpaca config flag + order flow (quote-size path)
# ---------------------------------------------------------------------------
orig_keys = (config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY = "test-id-123", "test-secret-123"
check("alpaca configured when keys are set", signals.alpaca_is_configured())
check("data layer agrees", data.alpaca_is_configured())
config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY = orig_keys
check("alpaca unconfigured after restore", not signals.alpaca_is_configured())

fl = data.order_flow({"bid": 100.0, "ask": 100.05, "bidSize": 300, "askSize": 100})
check("order-flow imbalance math",
      abs(fl["primary"] - 0.5) < 1e-9 and "Buy-side" in fl["label"])
fl = data.order_flow({"bid": 100.0, "ask": 100.05, "bidSize": 100, "askSize": 300})
check("order-flow sell pressure", fl["primary"] < -0.2 and "Sell-side" in fl["label"])
fl = data.order_flow()
check("order flow unavailable without sizes", math.isnan(fl["primary"]))
sp = data.spread_info({"bid": 100.0, "ask": 100.10})
check("spread info", abs(sp["pct"] - 0.09995) < 1e-3)
check("spread none without quote", data.spread_info(None) is None)

# ---------------------------------------------------------------------------
# 9. Charts
# ---------------------------------------------------------------------------
fig = price_chart(df.tail(126), "AAPL", signals.demo_signal(df))
check("price chart builds", hasattr(fig, "data") and len(fig.data) > 0)
check("radar history chart builds",
      hasattr(radar_history_figure([(pd.Timestamp.now(), 10)]), "data"))
check("forecast chart builds", hasattr(forecast_figure(fc), "data"))

# ---------------------------------------------------------------------------
# 10. Optional: real market data
# ---------------------------------------------------------------------------
if "--live" in sys.argv:
    live = data.add_indicators(data.fetch_prices("AAPL"))
    check(f"live Yahoo data (AAPL last close {live['Close'].iloc[-1]:.2f})", len(live) > 100)
    live5 = data.fetch_intraday("AAPL")
    check(f"live intraday bars ({len(live5)})", len(live5) >= 31)

print("\nAll checks passed. Start the app with:  streamlit run app.py")
