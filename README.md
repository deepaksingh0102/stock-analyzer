# Stock Analyzer

Intraday stock analyzer ported from `single-stock-analyzer.html`: hard-gate
verdict, Direction Radar, pressure checks, 45-minute direction forecast and
mechanical trade planning. Built with Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

- **Default:** Yahoo Finance (no keys needed).
- **Alpaca:** paste `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` into `config.py`
  to use the Alpaca data feed (enables real order-flow sizes too).

## Share it with friends (free)

1. Push these files to a **public** GitHub repo (keep the `"Dummy"`
   placeholders in `config.py` — never commit real keys).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub → **New app** → pick the repo, branch and `app.py` → **Deploy**.
3. Share the app URL. (Free apps sleep after inactivity and wake on visit.)

To use your own Alpaca keys in the deployed app without committing them,
add them under the app's **Settings → Secrets** as TOML:

```toml
APCA_API_KEY_ID = "your-key-id"
APCA_API_SECRET_KEY = "your-secret-key"
```

## Quick temporary share (no deploy)

```bash
ngrok http 8501
```

Share the ngrok URL — it works until you stop it (your Mac must stay on).

## Self-check

```bash
python test_app.py            # offline tests
python test_app.py --live     # also checks real Yahoo data
```
