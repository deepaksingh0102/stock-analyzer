"""
============================================================
  YOUR SETTINGS  —  the only file you need to edit
============================================================
Mirrored from single-stock-analyzer.html.

1. Paste your Alpaca API key + secret below. In the HTML these live as
   APCA_API_KEY_ID / APCA_API_SECRET_KEY near the top of the script
   (currently "Dummy" placeholders there too).
   Until you set real keys, the app fetches data from Yahoo Finance.
2. Adjust the analyzer inputs to taste — defaults match the HTML.

Keys stay in this file only: they are sent to Alpaca (data.alpaca.markets)
and nowhere else.
"""

# --- Alpaca: market-data API (from single-stock-analyzer.html) ----------------
APCA_API_KEY_ID = "Dummy"            # <-- REPLACE with your Alpaca key ID
APCA_API_SECRET_KEY = "Dummy"        # <-- REPLACE with your Alpaca secret key
ALPACA_FEED = "sip"                  # "sip" (Algo Trader Plus) or "iex" (free-tier fallback)

# --- Analyzer inputs (same defaults as single-stock-analyzer.html) -------------
BENCHMARK = "SPY"          # primary benchmark (hard gate for the verdict)
BENCHMARK2 = ""            # optional secondary benchmark, e.g. "QQQ" (soft check)
MIN_DOLLAR_VOL = 5_000_000   # avg $ volume/day floor (verdict -> Sideways below)
MAX_SPREAD_PCT = 0.15        # max bid-ask spread % (verdict -> Sideways above)
MIN_ADX = 20.0               # minimum ADX trend strength for a directional verdict
RS_TOL_PCT = 0.25            # relative-strength tolerance vs benchmark (%)
ACCOUNT_SIZE = 10_000        # $ account size for position sizing
RISK_PCT = 1.0               # % of account risked per trade idea
FORECAST_HORIZON_MIN = 45    # direction forecast horizon (minutes)

# --- App defaults (optional to change) ------------------------------------------
DEFAULT_SYMBOL = "AAPL"
DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, TSLA"
AUTO_REFRESH_SECONDS = 60          # how often "Live tracking" refreshes data
