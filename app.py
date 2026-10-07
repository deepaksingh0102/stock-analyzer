"""
Stock Analyzer — Streamlit app.   Run with:   streamlit run app.py

Wraps the analyzer engine in data.py (ported from single-stock-analyzer.html):
hard-gate verdict, confirmation scorecard, Direction Radar, pressure checks,
45-minute direction forecast, and mechanical trade planning.
"""
from datetime import datetime
import time

import pandas as pd
import streamlit as st

import config
import data
import signals
from charts import price_chart, radar_history_figure, forecast_figure

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

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading. The `bucket` parameter is a time window derived from
# the refresh interval: while live tracking is on, each interval gets a
# fresh cache entry, so auto-refresh always fetches new data. When live
# tracking is off, bucket=0 and the cache behaves normally.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_daily(symbol: str, use_demo: bool, bucket: int = 0) -> pd.DataFrame:
    raw = data.demo_prices(symbol) if use_demo else data.fetch_prices(symbol)
    return data.add_indicators(raw)


@st.cache_data(ttl=600, show_spinner=False)
def load_intraday(symbol: str, use_demo: bool, bucket: int = 0) -> pd.DataFrame:
    return data.demo_intraday(symbol) if use_demo else data.fetch_intraday(symbol)


@st.cache_data(ttl=600, show_spinner=False)
def load_1m(symbol: str, use_demo: bool, bucket: int = 0):
    if use_demo:
        return None
    try:
        return data.fetch_1m(symbol)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_quote(symbol: str, use_demo: bool, bucket: int = 0):
    if use_demo:
        return None, ""
    return data.fetch_quote(symbol)


def _bucket(live: bool, refresh_secs: int) -> int:
    return int(time.time() // refresh_secs) if live else 0


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
    window_name = st.selectbox("Chart window", list(data.WINDOWS), index=2)

    st.header("Benchmarks")
    bench = st.text_input("Primary benchmark", value=config.BENCHMARK,
                          help="Hard gate: a bullish read lagging this is downgraded.").strip().upper() or "SPY"
    bench2 = st.text_input("Secondary benchmark (optional)", value=config.BENCHMARK2,
                           help="Soft check: affects conviction only.").strip().upper()

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
    refresh_secs = st.number_input("Refresh every (seconds)", min_value=10, max_value=3600,
                                   value=config.AUTO_REFRESH_SECONDS, step=10,
                                   help="How often live tracking re-runs the analysis.")
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()
    use_demo = st.toggle("Demo data (offline / no internet)", value=False)

    if signals.alpaca_is_configured():
        st.success("Alpaca: connected (keys in config.py)")
    else:
        st.info("Alpaca: not configured — using Yahoo Finance.\n"
                "Paste your keys in config.py for the HTML's data feed.")


st.title("📈 Stock Analyzer")
tab_detail, tab_watch = st.tabs(["Chart & Signal", "Watchlist"])


def _thresholds():
    return {"min_dollar_vol": min_dollar_vol, "max_spread_pct": max_spread_pct,
            "min_adx": min_adx, "rs_tol": rs_tol}


# ---------------------------------------------------------------------------
# Tab 1: one symbol in detail
# ---------------------------------------------------------------------------
@st.fragment(run_every=refresh_secs if live else None)
def detail_view(symbol: str, window_name: str, use_demo: bool,
                live: bool, refresh_secs: int):
    if not symbol:
        st.info("Type a stock symbol in the sidebar to begin.")
        return
    if live:
        st.caption(f"🟢 Live tracking on — refreshing every {refresh_secs}s · "
                   f"last updated {datetime.now().strftime('%H:%M:%S')}")
    bucket = _bucket(live, refresh_secs)
    try:
        with st.spinner(f"Loading {symbol}…"):
            df_daily = load_daily(symbol, use_demo, bucket)
            df_5m = load_intraday(symbol, use_demo, bucket)
            df_1m = load_1m(symbol, use_demo, bucket)
            bench_5m = load_intraday(bench, use_demo, bucket) if bench != symbol else None
            bench2_5m = load_intraday(bench2, use_demo, bucket) if bench2 else None
            quote, quote_note = load_quote(symbol, use_demo, bucket)
    except Exception as exc:
        st.error(f"Could not load **{symbol}**: {exc}")
        st.caption("Tip: turn on **Demo data** in the sidebar to test without internet.")
        return

    res = data.analyze_symbol(
        symbol, df_daily, df_5m, df_1m, bench_5m, bench2_5m,
        bench_ticker=bench, bench2_ticker=bench2 or None, quote=quote,
        account_size=account_size, risk_pct=risk_pct, thresholds=_thresholds())
    if not res.get("ok"):
        st.error(res.get("error", "Analysis failed."))
        return

    v, sig = res["verdict"], signals.analyzer_signal(
        symbol, df_daily, df_5m, bench_5m,
        account_size=account_size, risk_pct=risk_pct, res=res, quote=quote)

    # Radar score history (session state, like the HTML's rolling history)
    hist_key = f"radar_hist_{symbol}"
    hist = st.session_state.setdefault(hist_key, [])
    if res["radar"]:
        now = datetime.now()
        if not hist or (now - hist[-1][0]).total_seconds() >= 30:
            hist.append((now, res["radar"]["score"]))
            del hist[:-360]

    # Headline numbers
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    chg = res["changeSinceOpen"]
    c1.metric(f"{symbol} price", fmt(res["entry"]),
              None if pd.isna(chg) else f"{chg:+.2f}% since open")
    c2.metric("Verdict", v["verdict"])
    c3.metric("Conviction", v["conviction"] or "—")
    c4.metric("RSI (14)", f"{res['trend']['rsi']:.0f}" if pd.notna(res["trend"]["rsi"]) else "—")
    c5.metric("ADX", f"{res['adx5m']['adx']:.0f}" if pd.notna(res["adx5m"]["adx"]) else "—")
    c6.metric("Signal", sig.action)

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

    # Chart
    shown = df_daily.tail(data.WINDOWS[window_name])
    st.plotly_chart(price_chart(shown, symbol, sig), use_container_width=True)

    # ---- Direction Radar ----
    st.subheader("Direction Radar — what's changing right now")
    r = res["radar"]
    if r:
        rc1, rc2 = st.columns([1, 2])
        label = ("STRONG UP" if r["score"] >= 45 else "LEANING UP" if r["score"] >= 20
                 else "STRONG DOWN" if r["score"] <= -45 else
                 "LEANING DOWN" if r["score"] <= -20 else "NEUTRAL / CHOP")
        rc1.metric("Radar score", f"{r['score']:+d}", label)
        rc1.caption(f"{r['up']} up · {r['dn']} down · "
                    f"{len(r['comps']) - r['up'] - r['dn']} flat")
        with rc2:
            st.plotly_chart(radar_history_figure(hist), use_container_width=True)
        comp_rows = [{"Component": c["name"], "Lean": f"{c['s']:+.2f}",
                      "Detail": c["detail"]} for c in r["comps"]]
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
        # Divergence alert (radar vs slow trend votes)
        tb = res["trend"]["bias"]
        if tb == "Bullish" and r["score"] <= -20:
            st.warning("Divergence: 5-min trend votes read Bullish, but momentum has turned "
                       "down. The slow indicators haven't caught up yet.")
        elif tb == "Bearish" and r["score"] >= 20:
            st.warning("Divergence: 5-min trend votes read Bearish, but momentum has turned "
                       "up. The slow indicators haven't caught up yet.")
        st.caption("Reading: ±20 = leaning, ±45 = strong. Momentum/pressure gauge, not a "
                   "forecast — it can't see news.")
    else:
        st.info("Not enough 5-min bars for the radar yet (need 40+).")

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
        fc["horizon_min"] = config.FORECAST_HORIZON_MIN
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
        ("Intraday (5-min) trend", f"{t['bias']} ({t['bull']} bullish / {t['bear']} bearish votes)"),
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
    a5, ad = res["adx5m"], res["adxDaily"]
    if pd.notna(a5["adx"]):
        st.dataframe(pd.DataFrame([
            ("ADX(14), 5-min", f"{a5['adx']:.1f} — {data.adx_label(a5['adx'])}"),
            ("+DI / −DI (5-min)",
             f"{a5['plusDI']:.1f} / {a5['minusDI']:.1f} "
             f"({'buyers' if a5['plusDI'] > a5['minusDI'] else 'sellers'} directing)"),
            ("ADX(14), daily",
             f"{ad['adx']:.1f} — {data.adx_label(ad['adx'])}" if pd.notna(ad["adx"]) else "n/a"),
        ], columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    else:
        st.info("Not enough 5-min bars for ADX (needs 29+).")

    # ---- Order flow / OBV ----
    st.subheader("Order Flow & Volume Accumulation")
    fl = res["flow"]
    st.info(fl["note"])
    if quote_note and pd.isna(fl["primary"]):
        st.warning("Quote diagnostic: " + quote_note)
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
            ("ATR(14), 5-min", f"${res['atr']:.3f}" if pd.notna(res["atr"]) else "n/a"),
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
        table = df_5m.tail(78)[["Open", "High", "Low", "Close", "Volume"]]
        st.dataframe(table.iloc[::-1].round(2), use_container_width=True)
        st.download_button("Download CSV (5-min)", table.to_csv().encode(),
                           f"{symbol}_5m.csv", "text/csv")

    src = "demo data" if use_demo else "Yahoo Finance"
    st.caption(f"Data as of {df_5m.index[-1]} · source: {src}"
               + (" · live tracking on" if live else ""))


# ---------------------------------------------------------------------------
# Tab 2: watchlist overview
# ---------------------------------------------------------------------------
@st.fragment(run_every=refresh_secs if live else None)
def watchlist_view(watchlist_text: str, use_demo: bool,
                   live: bool, refresh_secs: int):
    symbols = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
    if not symbols:
        st.info("Add symbols to the watchlist in the sidebar.")
        return
    bucket = _bucket(live, refresh_secs)

    rows = []
    for sym in symbols:
        try:
            df = load_daily(sym, use_demo, bucket)
            s, sig = data.summary(df), signals.daily_signal(df)
            rows.append({"Symbol": sym, "Price": round(s["price"], 2),
                         "Change %": round(s["change_pct"], 2), "Trend": s["trend"],
                         "RSI": round(s["rsi"]), "Signal": sig.action,
                         "Entry": sig.entry, "Exit / target": sig.target,
                         "Stop loss": sig.stop_loss})
        except Exception as exc:
            rows.append({"Symbol": sym, "Signal": "ERROR", "Trend": str(exc)[:60]})

    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
    st.caption("Tip: type any of these symbols in the sidebar search to see its chart. "
               "Watchlist signals use the daily trend (lighter than the full analyzer).")


with tab_detail:
    detail_view(symbol, window_name, use_demo, live, refresh_secs)
with tab_watch:
    watchlist_view(watchlist_text, use_demo, live, refresh_secs)
