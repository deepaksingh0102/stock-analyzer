"""Chart building (Plotly)."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import Signal


def intraday_chart(df, symbol: str, signal: Signal | None = None,
                   or_high: float | None = None, or_low: float | None = None,
                   window_label: str = "") -> go.Figure:
    """Intraday candles + EMA9/21, VWAP, opening-range lines, volume, MACD, RSI."""
    import math
    title = f"{symbol} — {window_label}" if window_label else f"{symbol} intraday"
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.12, 0.165, 0.165], vertical_spacing=0.05,
        subplot_titles=(title, "Volume", "MACD", "RSI (momentum)"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
    ), row=1, col=1)
    if "EMA9" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA 9",
                                 line=dict(width=1.5, color="#f59e0b")), row=1, col=1)
    if "EMA21" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA 21",
                                 line=dict(width=1.5, color="#3b82f6")), row=1, col=1)
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                                 line=dict(width=1.5, color="#a855f7", dash="dash")),
                      row=1, col=1)

    # Opening-range high/low (first 30 min of the session)
    if or_high is not None and math.isfinite(or_high):
        fig.add_hline(y=or_high, line_dash="dash", line_color="#16a34a", row=1, col=1,
                      annotation_text=f"OR high {or_high:,.2f}",
                      annotation_position="top left")
    if or_low is not None and math.isfinite(or_low):
        fig.add_hline(y=or_low, line_dash="dash", line_color="#dc2626", row=1, col=1,
                      annotation_text=f"OR low {or_low:,.2f}",
                      annotation_position="bottom left")

    # Entry / exit levels from the signal
    if signal and signal.action in ("BUY", "SELL"):
        levels = [("Entry", signal.entry, "#6b7280"),
                  ("Target", signal.target, "#16a34a"),
                  ("Stop", signal.stop_loss, "#dc2626")]
        for label, value, color in levels:
            if value is not None:
                fig.add_hline(y=value, line_dash="dot", line_color=color, row=1, col=1,
                              annotation_text=f"{label} {value:,.2f}",
                              annotation_position="top right")

    # Volume, colored by candle direction
    if "Volume" in df.columns:
        up = (df["Close"] >= df["Open"]).fillna(True)
        colors = ["#16a34a" if u else "#dc2626" for u in up]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=colors, opacity=0.7, showlegend=False),
                      row=2, col=1)

    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                 line=dict(width=1.5, color="#0ea5e9")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="Signal",
                                 line=dict(width=1.2, color="#f59e0b")), row=3, col=1)
        hist = (df["MACD"] - df["MACDsig"]).fillna(0)
        fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                             marker_color="#94a3b8", opacity=0.6), row=3, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                 line=dict(width=1.5, color="#8b5cf6")), row=4, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#9ca3af", row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#9ca3af", row=4, col=1)

    fig.update_layout(
        height=780, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    )
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekend gaps
    return fig


def radar_history_figure(history: list) -> go.Figure:
    """Line chart of Direction Radar scores sampled over the session.

    history: list of (timestamp, score) tuples.
    """
    fig = go.Figure()
    if history:
        xs = [t for t, _ in history]
        ys = [s for _, s in history]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Radar score",
                                 line=dict(width=2, color="#0ea5e9")))
    for y, label in ((45, "+45 strong"), (20, "+20 leaning"),
                     (-20, "−20 leaning"), (-45, "−45 strong")):
        fig.add_hline(y=y, line_dash="dot", line_color="#9ca3af",
                      annotation_text=label, annotation_position="right")
    fig.add_hline(y=0, line_dash="solid", line_color="#6b7280")
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(range=[-100, 100], title="Score"),
                      showlegend=False)
    return fig


def forecast_figure(fc: dict) -> go.Figure:
    """Horizontal probability bars for the UP / DOWN / SIDEWAYS forecast."""
    labels = ["Up", "Sideways", "Down"]
    probs = [fc["pUp"] * 100, fc["pSide"] * 100, fc["pDown"] * 100]
    colors = ["#16a34a", "#9ca3af", "#dc2626"]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h",
        marker_color=colors, text=[f"{p:.0f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(range=[0, 100], title="Probability %"),
                      showlegend=False)
    return fig
