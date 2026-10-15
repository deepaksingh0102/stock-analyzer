"""
Stock Analyzer — Streamlit app.   Run with:   streamlit run app.py

Wraps the analyzer engine in data.py (ported from single-stock-analyzer.html):
hard-gate verdict, confirmation scorecard, Direction Radar, pressure checks,
45-minute direction forecast, and mechanical trade planning.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import time

import pandas as pd
import streamlit as st

import config
import data
import signals
from charts import intraday_chart, radar_history_figure, forecast_figure

# Streamlit Cloud secrets (App Settings → Secrets) override config.py,
# so real API keys never have to be committed to the repo.
try:
    if "APCA_API_KEY_ID" in st.secrets:
        config.APCA_API_KEY_ID = st.secrets["APCA_API_KEY_ID"]
    if "APCA_API_SECRET_KEY" in st.secrets:
        config.APCA_API_SECRET_KEY = st.secrets["APCA_API_SECRET_KEY"]
    if "ALPACA_FEED" in st.secrets:
        config.ALPACA_FEED = st.secrets["ALPACA_FEED"]
except Exception:
    pass

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")


ET = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    """Naive wall-clock time in US Eastern (market time).

    Streamlit Cloud servers run on UTC, so plain datetime.now() labels
    charts ~4-5 hours ahead of market time. All displayed times use ET,
    matching the main chart's axis. Kept naive (tzinfo stripped) so it
    stays comparable with history entries stamped before this fix.
    """
    return datetime.now(ET).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Cached data loading. The `bucket` parameter is a time window derived from
# the refresh interval: while live tracking is on, each interval gets a
# fresh cache entry, so auto-refresh always fetches new data. When live
# tracking is off, bucket=0 and the cache behaves normally.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def load_daily(symbol: str, use_demo: bool, bucket: int = 0) -> pd.DataFrame:
    raw = data.demo_prices(symbol) if use_demo else data.fetch_prices(symbol)
    return data.add_indicators(raw)


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def load_intraday(symbol: str, use_demo: bool, bucket: int = 0) -> pd.DataFrame:
    """Raw 1-minute intraday feed. The chart plots it directly, and the engine
    reads the same 1-minute bars — what you see is what the verdict uses.
    No separate loader — this feed IS the 1-minute feed."""
    return data.demo_intraday(symbol) if use_demo else data.fetch_intraday(symbol)


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def load_quote(symbol: str, use_demo: bool, bucket: int = 0):
    if use_demo:
        return None, ""
    return data.fetch_quote(symbol)


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def load_trades(symbol: str, use_demo: bool, bucket: int = 0):
    """Recent trade prints for absorption/velocity. None = not attempted
    (no Alpaca keys); [] = attempted but nothing returned."""
    if use_demo or not data.alpaca_is_configured():
        return None
    try:
        return data.fetch_alpaca_trades(symbol)
    except Exception:
        return []


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def load_sp500_movers(use_demo: bool, bucket: int = 0):
    """Top 10 S&P 500 gainers/losers. {"error": msg} on failure."""
    if use_demo:
        return data.demo_movers()
    try:
        return data.sp500_movers()
    except Exception as exc:
        return {"error": str(exc)[:200]}


def _style_movers(df: pd.DataFrame):
    """Format movers table: $ price, ±% change, green/red change values."""
    sty = df.style.format({"Price": "${:,.2f}", "Change %": "{:+.2f}%"})
    def _color(v):
        try:
            return "color: #16a34a" if v > 0 else ("color: #dc2626" if v < 0 else "")
        except TypeError:
            return ""
    try:
        return sty.map(_color, subset=["Change %"])
    except Exception:
        return sty


def _bucket(live: bool, refresh_secs: int, bucket_secs: int | None = None) -> int:
    # Time bucket (live tracking) + manual-refresh nonce. Bumping the nonce
    # forces fresh fetches without clearing the whole cache or rerunning.
    # bucket_secs lets slow-moving data refresh less often than the live
    # tick: daily bars and movers don't need a 30s refresh, so they get
    # their own slower bucket instead of re-fetching every cycle.
    base = int(time.time() // (bucket_secs or refresh_secs)) if live else 0
    try:
        nonce = st.session_state.get("refresh_nonce", 0)
    except Exception:
        nonce = 0
    return base + nonce


def fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.2f}"


def status_icon(state: str) -> str:
    return {"ok": "✔", "warn": "~", "fail": "✘", "support": "✔",
            "contradict": "✘", "neutral": "–", "na": "·"}.get(state, "")


# ---------------------------------------------------------------------------
# Sidebar: search + analyzer settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Search")
    symbol = st.text_input("Stock symbol", value=config.DEFAULT_SYMBOL,
                           help="e.g. AAPL, MSFT, TSLA, RELIANCE.NS").strip().upper()

    # Benchmarks are fixed (SPY primary) rather than sidebar inputs.
    bench = config.BENCHMARK
    bench2 = config.BENCHMARK2

    st.header("Verdict gates")
    min_dollar_vol = st.number_input("Min avg $ volume/day", value=float(config.MIN_DOLLAR_VOL),
                                     step=1_000_000.0, format="%.0f")
    max_spread_pct = st.number_input("Max spread %", value=float(config.MAX_SPREAD_PCT),
                                     step=0.05, format="%.2f")
    min_adx = st.number_input("Min ADX", value=float(config.MIN_ADX), step=1.0, format="%.0f")
    rs_tol = st.number_input("Relative-strength tolerance %", value=float(config.RS_TOL_PCT),
                             step=0.05, format="%.2f")

    st.header("Risk sizing")
    account_size = st.number_input("Account size $", value=float(config.ACCOUNT_SIZE),
                                   step=5000.0, format="%.0f")
    risk_pct = st.number_input("Risk per trade %", value=float(config.RISK_PCT),
                               step=0.25, format="%.2f")

    st.header("Watchlist")
    watchlist_text = st.text_area("Symbols (comma separated)", value=config.DEFAULT_WATCHLIST)

    st.header("Settings")
    live = st.toggle("Live tracking (auto-refresh)", value=False,
                     help="When on, the analysis re-runs automatically at the interval below.")
    refresh_secs = st.number_input("Refresh every (seconds)", min_value=30, max_value=3600,
                                   value=max(config.AUTO_REFRESH_SECONDS, 30), step=5,
                                   help="How often live tracking re-runs the analysis. "
                                        "Below 30s the server runs out of threads/memory "
                                        "(5s refresh caused a 'can't start new thread' crash).")
    if st.button("🔄 Refresh now"):
        # The click itself reruns the app; just bump the nonce so the
        # loaders fetch fresh data. (Previously this cleared the whole
        # cache and called st.rerun(), which could leave the page blank
        # while fragments were active.)
        st.session_state["refresh_nonce"] = st.session_state.get("refresh_nonce", 0) + 1
    chart_tf = st.selectbox("Chart timeframe",
                            ["5 minutes", "15 minutes", "1 hour", "2 hours",
                             "3 hours", "4 hours", "Since market open"],
                            index=6,
                            help="How much recent history the chart shows. Candles are "
                                 "always 1-minute. The Direction Radar, signal, and forecast "
                                 "read the same 1-minute bars as the chart.")
    use_demo = st.toggle("Demo data (offline / no internet)", value=False)

    if signals.alpaca_is_configured():
        st.success("Alpaca: connected (keys in config.py)")
    else:
        st.info("Alpaca: not configured — using Yahoo Finance.\n"
                "Paste your keys in config.py for the HTML's data feed.")


st.title("📈 Stock Analyzer")
tab_detail, tab_movers, tab_watch = st.tabs(["Chart & Signal", "Top Movers", "Watchlist"])


def _thresholds():
    return {"min_dollar_vol": min_dollar_vol, "max_spread_pct": max_spread_pct,
            "min_adx": min_adx, "rs_tol": rs_tol}


# Sidebar "Chart timeframe" labels -> trailing minutes shown on the chart
# (None = full session since market open). Candles are always 1-minute;
# the selector controls the window, not the candle size.
CHART_TF_TRAILING = {"5 minutes": 5, "15 minutes": 15, "1 hour": 60,
                     "2 hours": 120, "3 hours": 180, "4 hours": 240,
                     "Since market open": None}


# ---------------------------------------------------------------------------
# Tab 1: one symbol in detail
# ---------------------------------------------------------------------------
@st.fragment(run_every=refresh_secs if live else None)
def detail_view(symbol: str, use_demo: bool,
                live: bool, refresh_secs: int, chart_tf: str = "Since market open"):
    if not symbol:
        st.info("Type a stock symbol in the sidebar to begin.")
        return
    if live:
        st.caption(f"🟢 Live tracking on — refreshing every {refresh_secs}s · "
                   f"last updated {_now_et().strftime('%H:%M:%S')} ET · "
                   "auto-refresh pauses while this tab is in the background")
    # Tiered freshness: not all data needs the same refresh rate. The quote
    # and trade tape are tiny and live-feeling (30s); 1-min bars only
    # move every minute (60s); daily bars barely change intraday (5 min).
    # This cuts most of the per-refresh fetching without losing freshness
    # where it matters.
    bucket = _bucket(live, refresh_secs)
    bucket_mid = _bucket(live, refresh_secs, bucket_secs=60)
    bucket_slow = _bucket(live, refresh_secs, bucket_secs=300)
    try:
        with st.spinner(f"Loading {symbol}…"):
            df_daily = load_daily(symbol, use_demo, bucket_slow)
            # The raw feed is 1-minute bars (chart plots them directly, new
            # candle every minute). The engine reads the same 1-minute bars,
            # so the RSI/MACD/EMA values behind the verdict are exactly the
            # ones plotted on the chart.
            df_1m = load_intraday(symbol, use_demo, bucket_mid)
            bench_1m = (load_intraday(bench, use_demo, bucket_mid)
                        if bench != symbol else None)
            bench2_1m = (load_intraday(bench2, use_demo, bucket_mid)
                         if bench2 else None)
            quote, quote_note = load_quote(symbol, use_demo, bucket)
            trades = load_trades(symbol, use_demo, bucket)
    except Exception as exc:
        st.error(f"Could not load **{symbol}**: {exc}")
        st.caption("Tip: turn on **Demo data** in the sidebar to test without internet.")
        return

    res = data.analyze_symbol(
        symbol, df_daily, df_1m, bench_1m, bench2_1m,
        bench_ticker=bench, bench2_ticker=bench2 or None, quote=quote,
        account_size=account_size, risk_pct=risk_pct, trades=trades,
        thresholds=_thresholds())
    if not res.get("ok"):
        st.error(res.get("error", "Analysis failed."))
        return

    v, sig = res["verdict"], signals.analyzer_signal(
        symbol, df_daily, df_1m, bench_1m,
        account_size=account_size, risk_pct=risk_pct, res=res, quote=quote)

    # Radar score history: backfilled from today's 1-minute bars (5-min steps)
    # so it covers the session even when the tab was just opened, then live
    # points append every 30s on top. Keyed by symbol+date so each session
    # starts fresh.
    today_str = df_1m.index[-1].date().isoformat() if len(df_1m) else "nodate"
    hist_key = f"radar_hist_{symbol}_{today_str}"
    hist = st.session_state.setdefault(hist_key, [])
    if f"{hist_key}_seeded" not in st.session_state and res.get("ok"):
        seed_pts = []
        sess_idx = df_1m.index[df_1m.index.date == df_1m.index[-1].date()]
        atr_v = res.get("atr")
        for i in range(0, len(sess_idx), 5):
            sub = df_1m.loc[:sess_idx[i]]
            r = data.compute_radar(sub, atr_v)
            if r:
                seed_pts.append((sess_idx[i].to_pydatetime(), r["score"]))
        # Keep seeds strictly before any live point already recorded.
        hist.extend(seed_pts)
        hist.sort(key=lambda p: p[0])
        st.session_state[f"{hist_key}_seeded"] = True
    if res["radar"]:
        now = _now_et()
        if not hist or (now - hist[-1][0]).total_seconds() >= 30:
            hist.append((now, res["radar"]["score"]))
            del hist[:-360]

    # Headline numbers (3+3 so they stay readable on phones)
    h1, h2, h3 = st.columns(3)
    h4, h5, h6 = st.columns(3)
    chg = res["changeSinceOpen"]
    # Show the live quote mid-price when available — the bar-close price only
    # moves when a new 1-min bar completes, which makes the headline look frozen.
    si = data.spread_info(quote) if quote else None
    live_px = si["mid"] if si else None
    px_label = f"{symbol} price · live" if live_px else f"{symbol} price"
    h1.metric(px_label, fmt(live_px if live_px else res["entry"]),
              None if pd.isna(chg) else f"{chg:+.2f}% since open")
    h2.metric("Verdict", v["verdict"])
    h3.metric("Conviction", v["conviction"] or "—")
    h4.metric("RSI (14)", f"{res['trend']['rsi']:.0f}" if pd.notna(res["trend"]["rsi"]) else "—")
    h5.metric("ADX", f"{res['adx1m']['adx']:.0f}" if pd.notna(res["adx1m"]["adx"]) else "—")
    h6.metric("Signal", sig.action)

    # Fragile-verdict notification: when 2+ inputs sit near their decision
    # thresholds (measured against this session's typical 1-minute noise),
    # warn that conviction could flip on the next bar.
    frag = v.get("fragility") or []
    if v["verdict"] != "Sideways" and len(frag) >= 2:
        st.warning("**Fragile verdict** — conviction could change on the next bar:\n\n" +
                   "\n".join(f"- {f}" for f in frag))

    # Why: reason + scorecard
    with st.expander("Why this verdict", expanded=True):
        st.write(v["reason"])
        if v["scorecard"]:
            sc = v["scorecard"]
            rows = [{"Check": f"{status_icon(i['status'])} {i['name']}",
                     "Detail": i["detail"]} for i in sc["items"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(f"Conviction: **{sc['conviction']}** "
                       f"({sc['supports']} support / {sc['contradicts']} contradict)")

    # Intraday chart: 1-minute candles + EMA9/21 + VWAP + opening-range
    # high/low. The trailing window (sidebar) controls how much recent
    # history is shown; the engine above reads the same 1-minute bars.
    trailing = CHART_TF_TRAILING.get(chart_tf)
    frame = data.intraday_frame(df_1m, "session", trailing_minutes=trailing)
    or_ = data.opening_range(df_1m)
    pivots = data.prior_day_pivots(df_daily)
    if frame.empty:
        st.info("Not enough intraday data to draw the chart.")
    else:
        st.plotly_chart(intraday_chart(
            frame, symbol, sig,
            or_high=or_["high"] if or_ else None,
            or_low=or_["low"] if or_ else None,
            window_label=(f"Last {chart_tf}" if trailing is not None
                          else "Since market open"),
            pivots=pivots),
            use_container_width=True,
            key=f"intraday_{symbol}_{chart_tf.replace(' ', '_')}")
        st.caption(data.frame_caption(frame, "1 minute"))
        # Plain-English takeaway for novices
        last = frame.iloc[-1]
        px = live_px if live_px else res["entry"]
        st.info(data.plain_takeaway(
            px, last["EMA9"], last["EMA21"], last["VWAP"], last["RSI"],
            or_["high"] if or_ else None, or_["low"] if or_ else None))
        with st.expander("How to read this chart"):
            window_desc = (f"the last {chart_tf}" if trailing is not None
                           else "today's full session")
            st.write(
                "- **Candles** — each one is a 1-minute "
                "slice of time. "
                "Green = price went up in that minute, red = it went down.\n"
                f"- **Window** — the chart shows {window_desc}. "
                "Change it with Chart timeframe in the sidebar.\n"
                "- **One timeframe** — the chart, the Direction Radar, the signal, "
                "and the forecast all read the same 1-minute bars, so the "
                "indicators you see are the indicators the verdict uses.\n"
                "- **Orange & blue lines (EMA 9 / EMA 21)** — the average price over the last "
                "9 / 21 slices. Price above both = uptrend; below both = downtrend.\n"
                "- **Dark line (Close)** — the closing price at each point in time. Above "
                "the orange/blue EMA lines = upside; below them = downside.\n"
                "- **Purple dashed line (VWAP)** — the day's average price, weighted by volume. "
                "Big traders judge themselves against it: above it = buyers winning.\n"
                "- **Green / red dashed lines** — the highest and lowest price of the first "
                "30 minutes (the \"opening range\"). Breaking above or below it often sets "
                "the tone for the rest of the day.\n"
                "- **Volume bars** — how much traded in each slice. Tall bars = strong participation.\n"
                "- **RSI (0–100)** — momentum energy. Above 70 = overbought (buyers may be tired); "
                "below 30 = oversold (sellers may be tired).\n"
                "- **MACD** — momentum direction. Blue line crossing above orange = momentum turning "
                "up; crossing below = turning down.")

    # ---- Direction Radar ----
    st.subheader("Direction Radar — what's changing right now")
    r = res["radar"]
    if r:
        label = ("STRONG UP" if r["score"] >= 45 else "LEANING UP" if r["score"] >= 20
                 else "STRONG DOWN" if r["score"] <= -45 else
                 "LEANING DOWN" if r["score"] <= -20 else "NEUTRAL / CHOP")
        st.metric("Radar score", f"{r['score']:+d}", label,
                  help=f"{r['up']} up · {r['dn']} down · "
                       f"{len(r['comps']) - r['up'] - r['dn']} flat")
        st.plotly_chart(radar_history_figure(hist), use_container_width=True)
        comp_rows = [{"Component": c["name"], "Lean": f"{c['s']:+.2f}",
                      "Detail": c["detail"]} for c in r["comps"]]
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
        # Divergence alert (radar vs slow trend votes)
        tb = res["trend"]["bias"]
        if tb == "Bullish" and r["score"] <= -20:
            st.warning("Divergence: 1-min trend votes read Bullish, but momentum has turned "
                       "down. The slow indicators haven't caught up yet.")
        elif tb == "Bearish" and r["score"] >= 20:
            st.warning("Divergence: 1-min trend votes read Bearish, but momentum has turned "
                       "up. The slow indicators haven't caught up yet.")
        st.caption("Reading: ±20 = leaning, ±45 = strong. Momentum/pressure gauge, not a "
                   "forecast — it can't see news.")
    else:
        st.info("Not enough 1-min bars for the radar yet (need 70+).")

    # ---- Live pressure checks ----
    st.subheader("Live Pressure Checks")
    pr = res["pressure"]
    if pr["available"]:
        st.caption(pr["summary"])
        st.dataframe(pd.DataFrame(
            [{"Check": f"{status_icon(c['state'])} {c['name']}", "Detail": c["detail"]}
             for c in pr["checks"]]), hide_index=True, use_container_width=True)
    else:
        st.info(pr["note"] or "Pressure checks need the session tape / 1-min bars.")
        if pr["checks"]:
            st.dataframe(pd.DataFrame(
                [{"Check": c["name"], "Detail": c["detail"]} for c in pr["checks"]]),
                hide_index=True, use_container_width=True)
    st.caption("Absorption and tape-velocity need a live tick feed (not available in this "
               "app); the 1-minute fast-momentum check runs on bars.")

    # ---- 45-minute forecast ----
    st.subheader(f"Direction Forecast — next ~{config.FORECAST_HORIZON_MIN} minutes")
    fc = res["forecast"]
    if fc:
        st.markdown(f"**Next ~{config.FORECAST_HORIZON_MIN} min: {fc['label']} "
                    f"({fc['conf']:.0%} confidence)**")
        st.plotly_chart(forecast_figure(fc), use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric("Expected move", f"±${fc['expMove']:.2f}",
                  f"±{fc['expMove'] / fc['px'] * 100:.2f}% (~0.6× ATR)")
        m2.metric("Reference price", f"${fc['px']:.2f}")
        st.write("**What's driving it**")
        st.dataframe(pd.DataFrame(
            [{"Driver": f"{'▲' if d['dir'] > 0 else ('▼' if d['dir'] < 0 else '–')} {d['name']}",
              "Detail": d["detail"]} for d in fc["drivers"]]),
            hide_index=True, use_container_width=True)
        st.caption("Probabilities come from signal agreement, not a backtested model — "
                   "60% means “leaning”, not “likely”. News or a market-wide move can "
                   "override everything here instantly.")
    else:
        st.info("Forecast needs the radar (40+ five-minute bars).")

    # ---- Trend & pattern ----
    st.subheader("Trend & Pattern")
    t, dly, pat, bo = res["trend"], res["daily"], res["pattern"], res["breakout"]
    tp_rows = [
        ("Last price", fmt(res["entry"])),
        ("Intraday (1-min) trend", f"{t['bias']} ({t['bull']} bullish / {t['bear']} bearish votes)"),
        ("Daily trend", f"{dly['bias']}"
         + (f" ({dly['bull']} bull / {dly['bear']} bear votes)" if "bull" in dly else "")),
        ("Candlestick pattern (150m)",
         f"{pat['label']} — {pat['context']}" if pat else "None detected"),
        ("150-min breakout/breakdown", bo["label"]),
        ("RSI(14) intraday", f"{t['rsi']:.1f}" if pd.notna(t["rsi"]) else "n/a"),
        ("RSI(14) daily", f"{dly['rsi']:.1f}" if pd.notna(dly.get("rsi", float("nan"))) else "n/a"),
        ("VWAP + Volume confirmation", res["paired"]["vwapVolume"]),
        ("EMA + Volume confirmation", res["paired"]["emaVolume"]),
    ]
    st.dataframe(pd.DataFrame(tp_rows, columns=["Field", "Value"]), hide_index=True,
                 use_container_width=True)

    # ---- Relative strength ----
    st.subheader(f"Relative Strength vs {bench}" + (f" + {bench2}" if bench2 else ""))
    rs, rs2 = res["rs"], res["rs2"]
    if rs["available"]:
        st.dataframe(pd.DataFrame([
            (f"{symbol} since open", f"{rs['stockChg']:+.2f}%"),
            (f"{rs['benchTicker']} since open (gates verdict)", f"{rs['benchChg']:+.2f}%"),
            (f"Relative to {rs['benchTicker']}", f"{rs['rsOpen']:+.2f}%"),
            (f"Relative, last ~60 min",
             f"{rs['rs60']:+.2f}%" if pd.notna(rs["rs60"]) else "n/a"),
        ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    else:
        st.info(rs["note"])
    if rs2["available"]:
        st.caption(f"Secondary {rs2['benchTicker']}: {rs2['rsOpen']:+.2f}% since open "
                   f"(soft check — conviction only).")
    st.caption("Raw % comparison, not beta-adjusted.")

    # ---- ADX ----
    st.subheader("Trend Strength (ADX)")
    a5, ad = res["adx1m"], res["adxDaily"]
    if pd.notna(a5["adx"]):
        st.dataframe(pd.DataFrame([
            ("ADX(14), 1-min", f"{a5['adx']:.1f} — {data.adx_label(a5['adx'])}"),
            ("+DI / −DI (1-min)",
             f"{a5['plusDI']:.1f} / {a5['minusDI']:.1f} "
             f"({'buyers' if a5['plusDI'] > a5['minusDI'] else 'sellers'} directing)"),
            ("ADX(14), daily",
             f"{ad['adx']:.1f} — {data.adx_label(ad['adx'])}" if pd.notna(ad["adx"]) else "n/a"),
        ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    else:
        st.info("Not enough 1-min bars for ADX (needs 29+).")

    # ---- Order flow / OBV ----
    st.subheader("Order Flow & Volume Accumulation")
    fl = res["flow"]
    if fl["note"]:
        st.info(fl["note"])
    if quote_note and pd.isna(fl["primary"]):
        st.warning("Quote diagnostic: " + quote_note)
    if pd.notna(fl["primary"]):
        c1, c2, c3 = st.columns(3)
        c1.metric("Bid/ask size imbalance", f"{fl['primary']:+.1%}")
        c2.metric("Bid size", f"{fl['bidSize']:,.0f}")
        c3.metric("Ask size", f"{fl['askSize']:,.0f}")
        st.caption(fl["label"])
    ob = res["obv"]
    if ob["available"]:
        st.dataframe(pd.DataFrame([
            ("State", ob["state"]),
            ("Session net accumulation",
             f"{ob['sessionRatio'] * 100:+.1f}% of session volume ({ob['bars']} bars)"),
            (f"Last ~{ob['recentBars'] * 5} min net",
             f"{ob['recentRatio'] * 100:+.1f}% of volume, price "
             f"{ob['priceRecent'] * 100:+.1f}%"),
            ("Price / OBV divergence", ob["divergence"]),
        ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    else:
        st.info(ob["note"])

    # ---- Next levels / liquidity / volatility ----
    st.subheader("Next-Level Watch (reference levels — not a prediction)")
    lv = res["levels"]
    st.dataframe(pd.DataFrame([
        ("Nearest resistance above",
         f"${lv['nearestResistance']['value']:.2f} ({lv['nearestResistance']['label']})"
         if lv["nearestResistance"] else "None identified nearby"),
        ("Nearest support below",
         f"${lv['nearestSupport']['value']:.2f} ({lv['nearestSupport']['label']})"
         if lv["nearestSupport"] else "None identified nearby"),
        ("Typical move (±1 ATR)", f"${lv['atrDown1']:.2f} – ${lv['atrUp1']:.2f}"
         if pd.notna(res["atr"]) else "n/a"),
        ("Wider move (±2 ATR)", f"${lv['atrDown2']:.2f} – ${lv['atrUp2']:.2f}"
         if pd.notna(res["atr"]) else "n/a"),
    ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)

    st.subheader("Liquidity & Execution Checks")
    liq, sp = res["liquidity"], res["spread"]
    st.dataframe(pd.DataFrame([
        ("Avg $ volume (20-day)",
         f"${liq['value']:,.0f}/day" if pd.notna(liq["value"]) else "n/a"),
        ("Liquidity floor met", "Yes" if liq["ok"] else "No"),
        ("Bid / Ask", f"${sp['bid']:.2f} / ${sp['ask']:.2f}" if sp else "n/a"),
        ("Spread %", f"{sp['pct']:.3f}%" if sp else "n/a"),
    ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)

    st.subheader("Volatility Setup (Bollinger + ATR)")
    sq = res["squeeze"]
    if sq.get("bb"):
        st.dataframe(pd.DataFrame([
            ("Bollinger Bands (20, 2σ)",
             f"${sq['bb']['lower']:.2f} – ${sq['bb']['mid']:.2f} – ${sq['bb']['upper']:.2f}"),
            ("Band width percentile",
             f"{sq['widthPercentile'] * 100:.0f}th (0 = tightest seen)"
             if pd.notna(sq["widthPercentile"]) else "n/a"),
            ("ATR now vs ~10 bars ago",
             f"${sq['atrNow']:.3f} vs ${sq['atrPast']:.3f} "
             f"({'contracting' if sq['atrContracting'] else 'not contracting'})"
             if pd.notna(sq["atrNow"]) and pd.notna(sq["atrPast"]) else "n/a"),
            ("Squeeze active", "Yes — breakout setup may be forming" if sq["squeeze"] else "No"),
        ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    else:
        st.info(sq.get("note", "n/a"))

    if res["meanRev"]:
        st.warning("**Counter-trend note (mean-reversion risk):** " + res["meanRev"]["detail"] +
                   " This is a separate, competing read from the trend verdict — not a vote "
                   "inside it.")

    # ---- Trade plan ----
    st.subheader("Trade Plan (mechanical — sanity-check before using)")
    if res["tradePlan"]:
        pl, sz = res["tradePlan"]["plan"], res["tradePlan"]["size"]
        st.dataframe(pd.DataFrame([
            ("ATR(14), 1-min", f"${res['atr']:.3f}" if pd.notna(res["atr"]) else "n/a"),
            ("Entry (reference)", f"${res['entry']:.2f}"),
            ("Stop-loss", f"${pl['stop']:.2f} ({pl['stopDist']:.2f} away)"),
            ("Measured-move target",
             f"${pl['measuredTarget']:.2f}" if pd.notna(pl["measuredTarget"]) else "n/a"),
            ("2R reference target", f"${pl['target2R']:.2f}"),
            ("Risk:Reward", f"{pl['rr']:.2f} : 1" if pd.notna(pl["rr"]) else "n/a"),
            ("Risk amount", f"${sz['riskAmount']:.2f} ({risk_pct}% of ${account_size:,.0f})"),
            ("Suggested share size", f"{sz['shares']:,} shares"),
            ("Position value",
             f"${sz['positionValue']:,.2f} ({sz['pctOfAccount']:.1f}% of account)"
             if pd.notna(sz["pctOfAccount"]) else f"${sz['positionValue']:,.2f}"),
        ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    else:
        st.info("No confirmed directional verdict — no mechanical plan. "
                "See the provisional plan below.")

    # ---- Provisional trade plan ----
    st.subheader("Provisional Trade Plan (radar-led · not a confirmed verdict)")
    pp = res["provisional"]
    if not pp:
        st.info("Provisional plan needs the radar.")
    elif pp.get("blocked"):
        st.error(pp["reason"])
    else:
        if pp.get("neutral"):
            st.info(pp["reason"])
        else:
            st.write(f"Radar points **{'UP' if pp['dir'] == 1 else 'DOWN'}** → candidate "
                     f"**{pp['side']}**. Checklist: **{pp['passed']} of {pp['total']}** passed "
                     f"(weighted {pp['pct']:.0%}) → **{pp['tier']} confidence**.")
            if pp["contradictsVerdict"]:
                st.warning("Radar points the opposite way from the main verdict "
                           f"({v['verdict']}) — treat this as a reversal attempt.")
            st.dataframe(pd.DataFrame(
                [{"Check": f"{status_icon(c['state'])} {c['name']}", "Detail": c["detail"]}
                 for c in pp["checks"]]), hide_index=True, use_container_width=True)
            if pp["sizing"]:
                s = pp["sizing"]
                st.dataframe(pd.DataFrame([
                    ("Provisional side", s["side"]),
                    ("Entry (reference)", f"${s['entry']:.2f}"),
                    ("Stop-loss", f"${s['stop']:.2f} ({s['stopDist']:.2f} away)"),
                    ("Measured-move target",
                     f"${s['measuredTarget']:.2f}" if pd.notna(s["measuredTarget"]) else "n/a"),
                    ("2R reference target", f"${s['target2R']:.2f}"),
                    ("Risk this trade",
                     f"${s['riskAmount']:.2f} ({s['riskPct']:.2f}% — reduced size)"),
                    ("Suggested share size", f"{s['shares']:,} shares"),
                    ("Position value", f"${s['positionValue']:,.2f}"),
                ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
                st.caption("Invalidate if the radar score crosses back through 0 or price "
                           "trades through the stop. Size is cut because the slow gates "
                           "have not confirmed.")
            else:
                st.error("Too few confirmations — no sizing is produced. "
                         "Wait for the trigger levels instead of anticipating.")
        tr = pp["triggers"]
        if tr["long"]:
            st.caption("Conditional trigger (long): " + tr["long"]["text"])
        if tr["short"]:
            st.caption("Conditional trigger (short): " + tr["short"]["text"])

    # Raw data
    with st.expander("Data table"):
        table = df_1m.tail(120)[["Open", "High", "Low", "Close", "Volume"]]
        st.dataframe(table.iloc[::-1].round(2), use_container_width=True)
        st.download_button("Download CSV (1-min)", table.to_csv().encode(),
                           f"{symbol}_1m.csv", "text/csv")

    src = "demo data" if use_demo else "Yahoo Finance"
    st.caption(f"Data as of {df_1m.index[-1]} · source: {src}"
               + (" · live tracking on" if live else ""))


# ---------------------------------------------------------------------------
# Tab 2: watchlist overview
# ---------------------------------------------------------------------------
@st.fragment(run_every=refresh_secs if live else None)
# ---------------------------------------------------------------------------
# Tab 2: S&P 500 top movers
# ---------------------------------------------------------------------------
def movers_view(use_demo: bool, live: bool, refresh_secs: int):
    st.subheader("S&P 500 — Top Movers of the Day")
    # Daily closes: 5-min refresh is plenty.
    bucket = _bucket(live, refresh_secs, bucket_secs=300)
    with st.spinner("Loading S&P 500 movers…"):
        m = load_sp500_movers(use_demo, bucket)
    if m.get("error"):
        st.error(f"Could not load S&P 500 movers: {m['error']}")
        st.caption("Check your internet connection and try again.")
        return
    st.caption(f"As of {m['asof']} · source: {m.get('source', 'Yahoo Finance')}")
    st.markdown("**🟢 Top 10 Gainers**")
    st.dataframe(_style_movers(m["gainers"]), hide_index=True,
                 use_container_width=True)
    st.markdown("**🔴 Top 10 Losers**")
    st.dataframe(_style_movers(m["losers"]), hide_index=True,
                 use_container_width=True)


def watchlist_view(watchlist_text: str, use_demo: bool,
                   live: bool, refresh_secs: int):
    symbols = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
    if not symbols:
        st.info("Add symbols to the watchlist in the sidebar.")
        return
    # Daily bars only: 5-min refresh is plenty.
    bucket = _bucket(live, refresh_secs, bucket_secs=300)

    rows = []
    for sym in symbols:
        try:
            df = load_daily(sym, use_demo, bucket)
            s, sig = data.summary(df), signals.daily_signal(df)
            rows.append({"Symbol": sym, "Price": round(s["price"], 2),
                         "Change %": round(s["change_pct"], 2), "Trend": s["trend"],
                         "RSI": round(s["rsi"]), "Daily signal": sig.action,
                         "Entry": sig.entry, "Exit / target": sig.target,
                         "Stop loss": sig.stop_loss})
        except Exception as exc:
            rows.append({"Symbol": sym, "Signal": "ERROR", "Trend": str(exc)[:60]})

    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
    st.caption("Tip: type any of these symbols in the sidebar search to see its chart. "
               "Watchlist signals use the daily trend (lighter than the full analyzer).")


with tab_detail:
    detail_view(symbol, use_demo, live, refresh_secs, chart_tf)
with tab_movers:
    movers_view(use_demo, live, refresh_secs)
with tab_watch:
    watchlist_view(watchlist_text, use_demo, live, refresh_secs)
