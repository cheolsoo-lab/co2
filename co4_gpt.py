
# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V2
- GitHub / Streamlit Community Cloud deployment oriented
- Mobile-first UI
- Confirmed-candle analysis (unfinished candle excluded)
- Rolling POC without look-ahead
- Cost-adjusted backtest
- Conservative same-bar TP/SL handling
- Rolling walk-forward validation
- Trend / Momentum / Structure / Volume / Volatility / Market-regime factors
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta


# ============================================================
# 0. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="🔥 Crypto Quant Dashboard V2",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TOP_MAJORS = {
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "AVAX", "DOT", "LINK",
    "SUI", "APT", "BCH", "NEAR", "DOGE",
}

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    spread_rate: float = 0.0002
    max_holding_bars: int = 15
    min_oos_trades: int = 5
    atr_window: int = 14
    train_bars: int = 180
    test_bars: int = 30
    step_bars: int = 30
    # WFO/OOS validation is a robustness modifier, never the primary signal engine.
    wfo_weight_max: float = 0.12
    wfo_min_trades: int = 5
    wfo_min_windows: int = 3
    account_size: float = 10000.0
    risk_per_trade: float = 0.0075
    max_portfolio_risk: float = 0.02
    max_position_weight: float = 0.40


# ============================================================
# 1. DATA ACCESS
# ============================================================

@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "spot"},
    })


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tickers(exchange_id: str) -> pd.DataFrame:
    ex = make_exchange(exchange_id)
    tickers = ex.fetch_tickers()
    rows = []
    for symbol, t in tickers.items():
        if not symbol.endswith("/USDT"):
            continue
        last = t.get("last")
        quote_volume = t.get("quoteVolume")
        pct = t.get("percentage")
        if last is None:
            continue
        rows.append({
            "symbol": symbol,
            "base": symbol.split("/")[0],
            "last": float(last),
            "change_pct": float(pct) if pct is not None else np.nan,
            "quote_volume": float(quote_volume) if quote_volume is not None else 0.0,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str = "1d",
                limit: int = 700) -> pd.DataFrame:
    ex = make_exchange(exchange_id)
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(
        raw,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    # IMPORTANT: last OHLCV candle can still be forming.
    # Confirmed-candle analysis always removes it.
    if len(df) > 2:
        df = df.iloc[:-1].copy()

    return add_indicators(df)


def fetch_ohlcv_fallback(symbol: str, timeframe: str = "1d",
                         limit: int = 700) -> tuple[pd.DataFrame, str]:
    errors = []
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            df = fetch_ohlcv(exchange_id, symbol, timeframe, limit)
            if len(df) >= 100:
                return df, exchange_id
        except Exception as e:
            errors.append(f"{exchange_id}: {e}")
    return pd.DataFrame(), ""


# ============================================================
# 2. INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    x = df.copy()

    x["EMA20"] = ta.trend.EMAIndicator(x["Close"], window=20).ema_indicator()
    x["EMA50"] = ta.trend.EMAIndicator(x["Close"], window=50).ema_indicator()
    x["EMA200"] = ta.trend.EMAIndicator(x["Close"], window=200).ema_indicator()

    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()

    x["RSI14"] = ta.momentum.RSIIndicator(x["Close"], window=14).rsi()
    x["ROC14"] = ta.momentum.ROCIndicator(x["Close"], window=14).roc()

    macd = ta.trend.MACD(x["Close"])
    x["MACD"] = macd.macd()
    x["MACD_SIGNAL"] = macd.macd_signal()
    x["MACD_HIST"] = macd.macd_diff()

    x["ADX14"] = ta.trend.ADXIndicator(
        x["High"], x["Low"], x["Close"], window=14
    ).adx()

    x["ATR14"] = ta.volatility.AverageTrueRange(
        x["High"], x["Low"], x["Close"], window=14
    ).average_true_range()
    x["ATR_PCT"] = x["ATR14"] / x["Close"] * 100

    bb = ta.volatility.BollingerBands(x["Close"], window=20, window_dev=2)
    x["BB_HIGH"] = bb.bollinger_hband()
    x["BB_LOW"] = bb.bollinger_lband()
    x["BB_WIDTH"] = (x["BB_HIGH"] - x["BB_LOW"]) / x["Close"] * 100

    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["REL_VOLUME"] = x["Volume"] / x["VOL_MA20"]
    vol_mean = x["Volume"].rolling(30).mean()
    vol_std = x["Volume"].rolling(30).std()
    x["VOL_Z"] = (x["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # OBV
    direction = np.sign(x["Close"].diff()).fillna(0)
    x["OBV"] = (direction * x["Volume"]).cumsum()
    x["OBV_EMA20"] = x["OBV"].ewm(span=20, adjust=False).mean()

    # Confirmed rolling structure
    x["SWING_HIGH"] = x["High"].rolling(10, center=False).max().shift(1)
    x["SWING_LOW"] = x["Low"].rolling(10, center=False).min().shift(1)

    # Rolling POC approximation. Volume is assigned to the candle-close bin.
    # Crucially, each row uses ONLY prior completed candles.
    x["POC_Price"] = rolling_poc(x, window=60, bins=20)

    return x


def fetch_mtf_context(symbol: str) -> dict:
    """Return confirmed-candle multi-timeframe context for 1D/4H/1H."""
    result = {}
    for tf, limit in [("1d", 320), ("4h", 320), ("1h", 320)]:
        try:
            df, ex = fetch_ohlcv_fallback(symbol, tf, limit)
            if len(df) < 220:
                continue
            r = df.iloc[-1]
            result[tf] = {
                "exchange": ex,
                "close": float(r["Close"]),
                "ema20": float(r["EMA20"]),
                "ema50": float(r["EMA50"]),
                "ema200": float(r["EMA200"]),
                "rsi": float(r["RSI14"]),
                "adx": float(r["ADX14"]),
                "atr_pct": float(r["ATR_PCT"]),
                "swing_high": float(r["SWING_HIGH"]) if pd.notna(r["SWING_HIGH"]) else np.nan,
                "swing_low": float(r["SWING_LOW"]) if pd.notna(r["SWING_LOW"]) else np.nan,
            }
        except Exception:
            continue
    return result


def mtf_direction_score(mtf: dict, direction: str) -> tuple[float, list[str]]:
    """Score alignment across 1D/4H/1H without using future candles."""
    if not mtf:
        return 50.0, ["MTF 데이터 부족"]

    long = direction == "LONG"
    score = 50.0
    details = []

    weights = {"1d": 0.50, "4h": 0.30, "1h": 0.20}
    for tf, w in weights.items():
        r = mtf.get(tf)
        if not r:
            continue
        local = 50.0
        if long:
            if r["close"] > r["ema20"] > r["ema50"]:
                local += 18
            elif r["close"] < r["ema20"] < r["ema50"]:
                local -= 18
            if r["ema50"] > r["ema200"]:
                local += 12
            elif r["ema50"] < r["ema200"]:
                local -= 12
            if r["rsi"] >= 50:
                local += 8
            else:
                local -= 8
        else:
            if r["close"] < r["ema20"] < r["ema50"]:
                local += 18
            elif r["close"] > r["ema20"] > r["ema50"]:
                local -= 18
            if r["ema50"] < r["ema200"]:
                local += 12
            elif r["ema50"] > r["ema200"]:
                local -= 12
            if r["rsi"] <= 50:
                local += 8
            else:
                local -= 8

        score += (local - 50) * w
        details.append(f"{tf} {'정렬' if local >= 55 else '불일치'}")

    return float(np.clip(score, 0, 100)), details


def classify_market_state(df: pd.DataFrame) -> dict:
    """Classify the current confirmed market state without using future data."""
    if len(df) < 220:
        return {"state": "UNKNOWN", "trend": 0.0, "reversal": 0.0, "details": []}
    r = df.iloc[-1]
    prev = df.iloc[-4:-1]
    atr = float(r["ATR14"])
    close = float(r["Close"])
    details = []
    trend = 50.0
    reversal = 50.0

    up = close > r["EMA20"] > r["EMA50"] and r["EMA50"] > r["EMA200"]
    down = close < r["EMA20"] < r["EMA50"] and r["EMA50"] < r["EMA200"]
    adx = float(r["ADX14"])
    rsi = float(r["RSI14"])
    atr_pct = float(r["ATR_PCT"])

    if up:
        trend += 25; details.append("상승 정렬")
    elif down:
        trend += 25; details.append("하락 정렬")
    else:
        trend -= 5; details.append("추세 혼재")

    if adx >= 25:
        trend += 15; details.append("추세 강함")
    elif adx < 18:
        trend -= 10; details.append("추세 약함")

    # Reversal requires extension + location + loss of momentum/structure.
    ext_long = rsi >= 72 or (close - float(r["EMA20"])) / max(atr, 1e-12) >= 2.0
    ext_short = rsi <= 28 or (float(r["EMA20"]) - close) / max(atr, 1e-12) >= 2.0
    fail_high = close < float(r["SWING_HIGH"]) if pd.notna(r["SWING_HIGH"]) else False
    fail_low = close > float(r["SWING_LOW"]) if pd.notna(r["SWING_LOW"]) else False
    macd_falling = len(prev) >= 3 and float(prev["MACD_HIST"].iloc[-1]) < float(prev["MACD_HIST"].iloc[0])
    macd_rising = len(prev) >= 3 and float(prev["MACD_HIST"].iloc[-1]) > float(prev["MACD_HIST"].iloc[0])
    vol_confirm = float(r["REL_VOLUME"]) >= 1.15

    rev_long = 50.0
    rev_short = 50.0
    if ext_short: rev_long += 18
    if fail_low: rev_long += 12
    if macd_rising: rev_long += 8
    if vol_confirm and ext_short: rev_long += 7
    if ext_long: rev_short += 18
    if fail_high: rev_short += 12
    if macd_falling: rev_short += 8
    if vol_confirm and ext_long: rev_short += 7
    reversal = max(rev_long, rev_short)

    if up and ext_long and fail_high and macd_falling:
        state = "EXHAUSTION_UP"
    elif down and ext_short and fail_low and macd_rising:
        state = "EXHAUSTION_DOWN"
    elif up and adx >= 25:
        state = "TREND_UP"
    elif down and adx >= 25:
        state = "TREND_DOWN"
    elif adx < 18:
        state = "RANGE"
    else:
        state = "TRANSITION"

    if atr_pct > 8:
        details.append("고변동성")
    return {"state": state, "trend": float(np.clip(trend, 0, 100)),
            "reversal": float(np.clip(reversal, 0, 100)), "details": details,
            "reversal_long": float(np.clip(rev_long, 0, 100)),
            "reversal_short": float(np.clip(rev_short, 0, 100))}


def select_optimal_tp(df: pd.DataFrame, direction: str, entry: float, sl: float,
                      atr: float, strategy: str, mtf: Optional[dict] = None) -> dict:
    """Select ONE structural TP using reachability + market geometry.

    The target is not simply the farthest resistance.  Candidates are built from
    confirmed structure, POC and ATR extensions, then scored by R:R, structural
    quality and a volatility-based reachability proxy.  All levels come from data
    available at the signal time.
    """
    recent = df.tail(160)
    levels = []

    if direction == "LONG":
        if recent["SWING_HIGH"].notna().any():
            levels.append((float(recent["SWING_HIGH"].dropna().iloc[-1]), 1.15, "swing"))
        levels.append((float(recent["High"].max()), 1.00, "recent_high"))
        if pd.notna(recent["POC_Price"].iloc[-1]):
            poc = float(recent["POC_Price"].iloc[-1])
            if poc > entry:
                levels.append((poc, 0.90, "poc"))
        for tf in ("1h", "4h", "1d"):
            m = (mtf or {}).get(tf)
            if m and np.isfinite(m.get("swing_high", np.nan)) and m["swing_high"] > entry:
                levels.append((float(m["swing_high"]), {"1h": 0.95, "4h": 1.15, "1d": 1.30}[tf], tf + "_swing"))
        atr_mults = [1.5, 2.0, 2.5, 3.5, 5.0]
        levels.extend((entry + m * atr, 0.70, f"atr{m}") for m in atr_mults)
    else:
        if recent["SWING_LOW"].notna().any():
            levels.append((float(recent["SWING_LOW"].dropna().iloc[-1]), 1.15, "swing"))
        levels.append((float(recent["Low"].min()), 1.00, "recent_low"))
        if pd.notna(recent["POC_Price"].iloc[-1]):
            poc = float(recent["POC_Price"].iloc[-1])
            if poc < entry:
                levels.append((poc, 0.90, "poc"))
        for tf in ("1h", "4h", "1d"):
            m = (mtf or {}).get(tf)
            if m and np.isfinite(m.get("swing_low", np.nan)) and m["swing_low"] < entry:
                levels.append((float(m["swing_low"]), {"1h": 0.95, "4h": 1.15, "1d": 1.30}[tf], tf + "_swing"))
        atr_mults = [1.5, 2.0, 2.5, 3.5, 5.0]
        levels.extend((entry - m * atr, 0.70, f"atr{m}") for m in atr_mults)

    candidates = []
    seen = set()
    for price, quality, source in levels:
        if not np.isfinite(price):
            continue
        if direction == "LONG" and price <= entry:
            continue
        if direction == "SHORT" and price >= entry:
            continue
        key = round(float(price), 8)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((float(price), float(quality), source))

    if not candidates:
        candidates = [(entry + 2.5 * atr if direction == "LONG" else entry - 2.5 * atr, 0.70, "atr2.5")]

    risk = abs(entry - sl)
    scored = []
    for tp, quality, source in candidates:
        reward = abs(tp - entry)
        rr = reward / max(risk, 1e-12)
        if rr < 1.25:
            continue
        dist_atr = reward / max(atr, 1e-12)
        # Reachability is a soft proxy, not a probability.  It favors targets
        # that are plausible within the current volatility regime.
        center = 2.2 if strategy == "REVERSAL" else 3.0
        scale = 1.8 if strategy == "REVERSAL" else 2.4
        reach = math.exp(-max(dist_atr - center, 0.0) / scale)
        near_bonus = math.exp(-abs(dist_atr - center) / (scale * 1.8))
        rr_quality = min(rr / 2.5, 1.6)
        ev_proxy = (0.45 * rr_quality + 0.30 * reach + 0.15 * near_bonus + 0.10 * quality)
        scored.append((ev_proxy, tp, rr, dist_atr, source))

    if not scored:
        # Keep the nearest valid target when every structural level fails the RR floor.
        tp = min(candidates, key=lambda z: abs(z[0] - entry))[0]
        return {"tp": float(tp), "rr": float(abs(tp-entry) / max(risk, 1e-12)),
                "tp_distance_atr": float(abs(tp-entry) / max(atr, 1e-12)), "tp_source": "fallback"}

    _, tp, rr, dist_atr, source = max(scored, key=lambda z: z[0])
    return {"tp": float(tp), "rr": float(rr), "tp_distance_atr": float(dist_atr), "tp_source": source}

def rolling_poc(df: pd.DataFrame, window: int = 60, bins: int = 20) -> pd.Series:
    values = np.full(len(df), np.nan)

    closes = df["Close"].to_numpy(float)
    volumes = df["Volume"].to_numpy(float)

    for i in range(window, len(df)):
        hist_close = closes[i - window:i]
        hist_vol = volumes[i - window:i]

        lo = np.nanmin(hist_close)
        hi = np.nanmax(hist_close)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(hist_close, edges) - 1, 0, bins - 1)
        vol_by_bin = np.bincount(idx, weights=hist_vol, minlength=bins)
        k = int(np.nanargmax(vol_by_bin))
        values[i] = (edges[k] + edges[k + 1]) / 2

    return pd.Series(values, index=df.index)


# ============================================================
# 3. MARKET REGIME
# ============================================================

def basic_regime_score(df: pd.DataFrame) -> dict:
    if len(df) < 220:
        return {"score": 50.0, "label": "데이터 부족", "details": []}

    r = df.iloc[-1]
    score = 50.0
    details = []

    if r["Close"] > r["EMA20"] > r["EMA50"]:
        score += 15
        details.append("단기·중기 상승 정렬")
    elif r["Close"] < r["EMA20"] < r["EMA50"]:
        score -= 15
        details.append("단기·중기 하락 정렬")

    if r["EMA50"] > r["EMA200"]:
        score += 15
        details.append("중장기 상승 추세")
    elif r["EMA50"] < r["EMA200"]:
        score -= 15
        details.append("중장기 하락 추세")

    if r["ADX14"] >= 25:
        score += 10
        details.append("강한 추세")
    elif r["ADX14"] < 15:
        details.append("추세 약함")

    if 45 <= r["RSI14"] <= 68:
        score += 5
        details.append("RSI 중립-강세")
    elif r["RSI14"] > 75:
        score -= 5
        details.append("RSI 과열")
    elif r["RSI14"] < 30:
        score += 2
        details.append("RSI 과매도")

    score = float(np.clip(score, 0, 100))
    if score >= 70:
        label = "🟢 RISK-ON"
    elif score <= 35:
        label = "🔴 RISK-OFF"
    else:
        label = "🟡 MIXED"

    return {"score": score, "label": label, "details": details}


def fetch_market_regime() -> dict:
    result = {
        "btc": None,
        "ethbtc": None,
        "btcd": None,
        "total3": None,
        "label": "데이터 부족",
        "score": 50.0,
        "details": [],
    }

    btc, _ = fetch_ohlcv_fallback("BTC/USDT", "1d", 300)
    if len(btc) >= 220:
        r = basic_regime_score(btc)
        result["btc"] = btc.iloc[-1]
        result["score"] = r["score"]
        result["label"] = r["label"]
        result["details"] = r["details"]

    # ETH/BTC is obtained directly when available.
    try:
        ethbtc, _ = fetch_ohlcv_fallback("ETH/BTC", "1d", 300)
        if len(ethbtc) >= 30:
            result["ethbtc"] = ethbtc.iloc[-1]
    except Exception:
        pass

    return result


# ============================================================
# 4. SIGNAL ENGINES
# ============================================================

def signal_components(row: pd.Series, direction: str) -> dict:
    long = direction == "LONG"

    trend = 0
    momentum = 0
    structure = 0
    volume = 0
    volatility = 0

    if long:
        trend += 10 if row["Close"] > row["EMA20"] else 0
        trend += 5 if row["EMA20"] > row["EMA50"] else 0
        trend += 5 if row["EMA50"] > row["EMA200"] else 0

        momentum += 7 if row["RSI14"] > 50 else 0
        momentum += 7 if row["MACD_HIST"] > 0 else 0
        momentum += 6 if row["ROC14"] > 0 else 0

        structure += 8 if row["Close"] > row["SWING_HIGH"] else 0
        structure += 7 if row["Close"] > row["POC_Price"] else 0

        volume += 8 if row["REL_VOLUME"] >= 1.1 else 0
        volume += 7 if row["VOL_Z"] >= 0.5 else 0

        volatility += 5 if 0.8 <= row["ATR_PCT"] <= 5.0 else 0
        volatility += 5 if row["ADX14"] >= 20 else 0
    else:
        trend += 10 if row["Close"] < row["EMA20"] else 0
        trend += 5 if row["EMA20"] < row["EMA50"] else 0
        trend += 5 if row["EMA50"] < row["EMA200"] else 0

        momentum += 7 if row["RSI14"] < 50 else 0
        momentum += 7 if row["MACD_HIST"] < 0 else 0
        momentum += 6 if row["ROC14"] < 0 else 0

        structure += 8 if row["Close"] < row["SWING_LOW"] else 0
        structure += 7 if row["Close"] < row["POC_Price"] else 0

        volume += 8 if row["REL_VOLUME"] >= 1.1 else 0
        volume += 7 if row["VOL_Z"] >= 0.5 else 0

        volatility += 5 if 0.8 <= row["ATR_PCT"] <= 5.0 else 0
        volatility += 5 if row["ADX14"] >= 20 else 0

    return {
        "trend": trend,
        "momentum": momentum,
        "structure": structure,
        "volume": volume,
        "volatility": volatility,
    }


def generate_signal(df: pd.DataFrame, direction: str, tp_atr: float = 2.0,
                    sl_atr: float = 1.0, mtf: Optional[dict] = None,
                    strategy: Optional[str] = None) -> Optional[dict]:
    if len(df) < 220:
        return None
    r = df.iloc[-1]
    required = ["EMA20", "EMA50", "EMA200", "RSI14", "MACD_HIST", "ROC14",
                "ADX14", "ATR14", "ATR_PCT", "REL_VOLUME", "VOL_Z",
                "SWING_HIGH", "SWING_LOW", "POC_Price"]
    if any(pd.isna(r.get(c)) for c in required):
        return None
    components = signal_components(r, direction)
    trend_score = float(np.clip(sum(components.values()), 0, 100))
    state = classify_market_state(df)
    mtf_score, mtf_details = mtf_direction_score(mtf or {}, direction)
    close = float(r["Close"]); atr = float(r["ATR14"])

    long = direction == "LONG"
    if strategy is None:
        if direction == "LONG" and state["reversal_long"] >= 68 and state["state"] in {"EXHAUSTION_DOWN", "RANGE", "TRANSITION"}:
            strategy = "REVERSAL"
        elif direction == "SHORT" and state["reversal_short"] >= 68 and state["state"] in {"EXHAUSTION_UP", "RANGE", "TRANSITION"}:
            strategy = "REVERSAL"
        else:
            strategy = "TREND"

    # Reversal score is independent from trend score; do not let trend indicators
    # veto a genuine exhaustion setup by themselves.
    if strategy == "REVERSAL":
        rev = state["reversal_long"] if long else state["reversal_short"]
        score = rev * 0.50 + (100 - mtf_score) * 0.20 + components["momentum"] / 20 * 15 + components["volume"] / 15 * 15
        # A reversal needs evidence of exhaustion, not merely an extreme oscillator.
        if (long and not (state["reversal_long"] >= 65)) or (not long and not (state["reversal_short"] >= 65)):
            return None
    else:
        score = trend_score * 0.65 + mtf_score * 0.25 + components["volume"] / 15 * 10
        if state["state"] == "RANGE":
            score -= 8
    score = float(np.clip(score, 0, 100))

    if direction == "LONG":
        entry = close
        structural_sl = float(r["SWING_LOW"]) - 0.25 * atr
        sl = min(entry - sl_atr * atr, structural_sl)
    else:
        entry = close
        structural_sl = float(r["SWING_HIGH"]) + 0.25 * atr
        sl = max(entry + sl_atr * atr, structural_sl)

    tp_info = select_optimal_tp(df, direction, entry, sl, atr, strategy, mtf=mtf)
    zone = 0.22 * atr if strategy == "TREND" else 0.18 * atr
    if direction == "LONG":
        entry_low, entry_high = entry - zone, entry + 0.08 * atr
    else:
        entry_low, entry_high = entry - 0.08 * atr, entry + zone

    return {
        "direction": direction, "strategy": strategy, "score": score,
        "entry_signal_close": entry, "entry_low": float(min(entry_low, entry_high)),
        "entry_high": float(max(entry_low, entry_high)), "tp": tp_info["tp"],
        "sl": float(sl), "rr": tp_info["rr"], "tp_distance_atr": tp_info["tp_distance_atr"],
        "mtf_score": mtf_score, "mtf_details": mtf_details, "market_state": state["state"],
        "state_details": state["details"], "trend_score": trend_score,
        "reversal_score": state["reversal_long"] if long else state["reversal_short"],
        "adx": float(r["ADX14"]), "rsi": float(r["RSI14"]), "atr_pct": float(r["ATR_PCT"]),
        "poc": float(r["POC_Price"]), "volume": float(r["REL_VOLUME"]), **components,
    }


# ============================================================
# 5. BACKTEST
# ============================================================

def execution_cost(cfg: BacktestConfig) -> float:
    # Entry + exit approximation.
    return 2 * (cfg.fee_rate + cfg.slippage_rate + cfg.spread_rate / 2)


def backtest_signals(
    df: pd.DataFrame,
    signals: list[tuple[int, str]],
    direction: str,
    tp_atr: float,
    sl_atr: float,
    cfg: BacktestConfig,
) -> dict:
    trades = []

    for idx, sig_dir in signals:
        if sig_dir != direction:
            continue
        if idx >= len(df) - 2:
            continue

        # Signal is known at confirmed close. Enter next candle open.
        entry_idx = idx + 1
        entry = float(df["Open"].iloc[entry_idx])
        atr = float(df["ATR14"].iloc[idx])
        if not np.isfinite(atr) or atr <= 0:
            continue

        if direction == "LONG":
            tp = entry + tp_atr * atr
            sl = entry - sl_atr * atr
        else:
            tp = entry - tp_atr * atr
            sl = entry + sl_atr * atr

        exit_price = None
        exit_idx = None
        outcome = "TIME"

        end = min(len(df), entry_idx + cfg.max_holding_bars + 1)

        for j in range(entry_idx, end):
            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])

            if direction == "LONG":
                hit_tp = high >= tp
                hit_sl = low <= sl
            else:
                hit_tp = low <= tp
                hit_sl = high >= sl

            # Conservative rule: if both are touched in one OHLC bar,
            # assume SL occurred first because intrabar order is unknown.
            if hit_tp and hit_sl:
                exit_price = sl
                exit_idx = j
                outcome = "SL_AMBIGUOUS"
                break
            if hit_sl:
                exit_price = sl
                exit_idx = j
                outcome = "SL"
                break
            if hit_tp:
                exit_price = tp
                exit_idx = j
                outcome = "TP"
                break

        if exit_price is None:
            exit_idx = end - 1
            exit_price = float(df["Close"].iloc[exit_idx])
            outcome = "TIME"

        if direction == "LONG":
            gross = exit_price / entry - 1
        else:
            gross = entry / exit_price - 1

        net = gross - execution_cost(cfg)

        trades.append({
            "signal_idx": idx,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry": entry,
            "exit": exit_price,
            "return": net,
            "outcome": outcome,
        })

    if not trades:
        return empty_metrics()

    tr = pd.DataFrame(trades)
    equity = (1 + tr["return"]).cumprod()
    total_return = float(equity.iloc[-1] - 1)

    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min())

    wins = tr.loc[tr["return"] > 0, "return"]
    losses = tr.loc[tr["return"] < 0, "return"]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Trade-level Sharpe is deliberately NOT annualized with sqrt(365).
    # It is shown as a standardized trade-return statistic only.
    std = float(tr["return"].std(ddof=1)) if len(tr) > 1 else np.nan
    sharpe = float(tr["return"].mean() / std) if std and np.isfinite(std) else np.nan

    return {
        "total_return": total_return,
        "win_rate": float((tr["return"] > 0).mean() * 100),
        "profit_factor": float(pf),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": int(len(tr)),
        "avg_trade": float(tr["return"].mean()),
        "trades_df": tr,
    }


def empty_metrics() -> dict:
    return {
        "total_return": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe": np.nan,
        "max_drawdown": 0.0,
        "trades": 0,
        "avg_trade": 0.0,
        "trades_df": pd.DataFrame(),
    }


# ============================================================
# 6. PATTERN SIGNALS
# ============================================================

def generate_breakout_signals(df: pd.DataFrame) -> list[tuple[int, str]]:
    signals = []
    for i in range(30, len(df) - 1):
        r = df.iloc[i]
        if pd.isna(r["SWING_HIGH"]) or pd.isna(r["SWING_LOW"]):
            continue

        if (
            r["Close"] > r["SWING_HIGH"]
            and r["ADX14"] >= 20
            and r["Close"] > r["EMA20"]
        ):
            signals.append((i, "LONG"))

        if (
            r["Close"] < r["SWING_LOW"]
            and r["ADX14"] >= 20
            and r["Close"] < r["EMA20"]
        ):
            signals.append((i, "SHORT"))

    return signals


# ============================================================
# 7. WALK-FORWARD
# ============================================================

def generate_strategy_signals(df: pd.DataFrame, direction: str, strategy: str) -> list[tuple[int, str]]:
    """Historical signal generator aligned with the live Trend/Reversal logic.

    This deliberately uses only information available at each confirmed bar.
    MTF is not reconstructed here; the WFO therefore validates the core 1D
    setup rather than pretending to validate unavailable future information.
    """
    signals = []
    if len(df) < 220:
        return signals
    for i in range(220, len(df) - 1):
        hist = df.iloc[:i + 1]
        r = hist.iloc[-1]
        try:
            comps = signal_components(r, direction)
            trend_score = float(np.clip(sum(comps.values()), 0, 100))
            state = classify_market_state(hist)
            if strategy == "TREND":
                if direction == "LONG":
                    ok = r["Close"] > r["EMA20"] > r["EMA50"] and r["EMA50"] > r["EMA200"] and r["MACD_HIST"] > 0 and r["ADX14"] >= 20
                else:
                    ok = r["Close"] < r["EMA20"] < r["EMA50"] and r["EMA50"] < r["EMA200"] and r["MACD_HIST"] < 0 and r["ADX14"] >= 20
                ok = bool(ok and trend_score >= 55 and r["REL_VOLUME"] >= 0.9)
            else:
                rev = state["reversal_long"] if direction == "LONG" else state["reversal_short"]
                ok = bool(rev >= 65)
            if ok:
                signals.append((i, direction))
        except Exception:
            continue
    return signals


def backtest_strategy_logic(df: pd.DataFrame, signals: list[tuple[int, str]],
                            direction: str, strategy: str, cfg: BacktestConfig,
                            min_idx: int = 0, max_idx: Optional[int] = None) -> dict:
    """Backtest the SAME structural Entry/SL/one-TP logic used by the live engine."""
    trades = []
    max_idx = len(df) if max_idx is None else min(max_idx, len(df))
    for idx, sig_dir in signals:
        if sig_dir != direction or idx < min_idx or idx >= max_idx - 2:
            continue
        entry_idx = idx + 1
        entry = float(df["Open"].iloc[entry_idx])
        atr = float(df["ATR14"].iloc[idx])
        if not np.isfinite(atr) or atr <= 0:
            continue
        r = df.iloc[idx]
        if direction == "LONG":
            structural_sl = float(r["SWING_LOW"]) - 0.25 * atr
            sl = min(entry - 1.0 * atr, structural_sl)
        else:
            structural_sl = float(r["SWING_HIGH"]) + 0.25 * atr
            sl = max(entry + 1.0 * atr, structural_sl)
        hist = df.iloc[:idx + 1]
        tp_info = select_optimal_tp(hist, direction, entry, sl, atr, strategy, mtf=None)
        tp = float(tp_info["tp"])
        if direction == "LONG" and not (sl < entry < tp):
            continue
        if direction == "SHORT" and not (tp < entry < sl):
            continue

        exit_price = None
        exit_idx = None
        outcome = "TIME"
        end = min(max_idx, entry_idx + cfg.max_holding_bars + 1)
        for j in range(entry_idx, end):
            high = float(df["High"].iloc[j]); low = float(df["Low"].iloc[j])
            if direction == "LONG":
                hit_tp, hit_sl = high >= tp, low <= sl
            else:
                hit_tp, hit_sl = low <= tp, high >= sl
            if hit_tp and hit_sl:
                exit_price, exit_idx, outcome = sl, j, "SL_AMBIGUOUS"; break
            if hit_sl:
                exit_price, exit_idx, outcome = sl, j, "SL"; break
            if hit_tp:
                exit_price, exit_idx, outcome = tp, j, "TP"; break
        if exit_price is None:
            exit_idx = end - 1
            exit_price = float(df["Close"].iloc[exit_idx])
        gross = (exit_price / entry - 1) if direction == "LONG" else (entry / exit_price - 1)
        net = gross - execution_cost(cfg)
        trades.append({"signal_idx": idx, "entry_idx": entry_idx, "exit_idx": exit_idx,
                       "entry": entry, "exit": exit_price, "return": net, "outcome": outcome})

    if not trades:
        return empty_metrics()
    tr = pd.DataFrame(trades)
    equity = (1 + tr["return"]).cumprod()
    peak = equity.cummax()
    wins = tr.loc[tr["return"] > 0, "return"]
    losses = tr.loc[tr["return"] < 0, "return"]
    std = float(tr["return"].std(ddof=1)) if len(tr) > 1 else np.nan
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "win_rate": float((tr["return"] > 0).mean() * 100),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.inf,
        "sharpe": float(tr["return"].mean() / std) if std and np.isfinite(std) else np.nan,
        "max_drawdown": float((equity / peak - 1).min()),
        "trades": int(len(tr)), "avg_trade": float(tr["return"].mean()), "trades_df": tr,
    }


def walk_forward(df: pd.DataFrame, direction: str, cfg: BacktestConfig, strategy: str = "TREND") -> dict:
    """Rolling OOS validation of the live structural strategy.

    Unlike the previous version, each OOS signal is generated with its preceding
    history attached, so 220-bar indicator warm-up does not erase the OOS sample.
    No WFO score is allowed to change LONG/SHORT/WAIT status.
    """
    if len(df) < cfg.train_bars + cfg.test_bars + 220:
        return {"windows": [], **empty_metrics()}
    all_trades, windows = [], []
    start = 0
    while start + cfg.train_bars + cfg.test_bars <= len(df):
        train_end = start + cfg.train_bars
        test_end = train_end + cfg.test_bars
        context_start = max(0, train_end - 240)
        context = df.iloc[context_start:test_end].copy()
        signals = generate_strategy_signals(context, direction, strategy)
        local_start = train_end - context_start
        oos = backtest_strategy_logic(context, signals, direction, strategy, cfg,
                                      min_idx=local_start, max_idx=len(context))
        if not oos["trades_df"].empty:
            all_trades.append(oos["trades_df"].copy())
        windows.append({
            "train_start": df.index[start], "train_end": df.index[train_end - 1],
            "test_start": df.index[train_end], "test_end": df.index[test_end - 1],
            "oos_return": oos["total_return"], "oos_win_rate": oos["win_rate"],
            "oos_trades": oos["trades"], "oos_mdd": oos["max_drawdown"],
        })
        start += cfg.step_bars
    if not all_trades:
        return {"windows": windows, **empty_metrics()}
    tr = pd.concat(all_trades, ignore_index=True)
    equity = (1 + tr["return"]).cumprod(); peak = equity.cummax()
    wins = tr.loc[tr["return"] > 0, "return"]; losses = tr.loc[tr["return"] < 0, "return"]
    std = float(tr["return"].std(ddof=1)) if len(tr) > 1 else np.nan
    return {
        "windows": windows, "total_return": float(equity.iloc[-1] - 1),
        "win_rate": float((tr["return"] > 0).mean() * 100),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.inf,
        "sharpe": float(tr["return"].mean() / std) if std and np.isfinite(std) else np.nan,
        "max_drawdown": float((equity / peak - 1).min()), "trades": int(len(tr)),
        "avg_trade": float(tr["return"].mean()), "trades_df": tr,
    }


# ============================================================
# 8. COIN ANALYSIS
# ============================================================

def dynamic_execution_gate(market_state: str, sig: dict) -> dict:
    """Return adaptive execution thresholds for the current market state.

    This is intentionally a gate, not a score boost.  The objective is to reduce
    false WAIT outcomes caused by one fixed threshold while preserving stricter
    confirmation when the strategy is exposed to trend/reversal-specific risks.
    """
    strategy = sig.get("strategy", "TREND")

    # Baseline: slightly less restrictive than the former fixed 68/1.30/8 gate.
    score_min, rr_min, gap_min, mtf_min, reversal_min = 65.0, 1.25, 6.0, 44.0, 65.0
    profile = "BALANCED"

    if market_state == "TREND_UP" or market_state == "TREND_DOWN":
        score_min, rr_min, gap_min, mtf_min = 67.0, 1.25, 7.0, 46.0
        profile = "TREND"
    elif market_state in {"EXHAUSTION_UP", "EXHAUSTION_DOWN"}:
        score_min, rr_min, gap_min, reversal_min = 64.0, 1.20, 5.0, 64.0
        profile = "EXHAUSTION"
    elif market_state == "TRANSITION":
        score_min, rr_min, gap_min = 63.0, 1.25, 5.0
        profile = "TRANSITION"
    elif market_state == "RANGE":
        # Range is still selective because directional continuation has less edge.
        score_min, rr_min, gap_min = 66.0, 1.35, 8.0
        profile = "RANGE"

    # Reversal entries need evidence, but should not be forced to meet a trend
    # MTF alignment requirement because the whole point is regime transition.
    if strategy == "REVERSAL":
        mtf_min = 35.0
        score_min = min(score_min, 65.0)
        reversal_min = max(reversal_min, 64.0)

    return {
        "score_min": float(score_min),
        "rr_min": float(rr_min),
        "gap_min": float(gap_min),
        "mtf_min": float(mtf_min),
        "reversal_min": float(reversal_min),
        "profile": profile,
    }


def analyze_symbol(symbol: str, cfg: BacktestConfig, use_mtf: bool = True) -> Optional[dict]:
    """Current-market decision engine. MTF can be deferred for speed."""
    try:
        df, exchange_id = fetch_ohlcv_fallback(symbol, "1d", 700)
        if len(df) < max(300, cfg.train_bars + cfg.test_bars + 30):
            return None
        latest = df.iloc[-1]
        mtf = fetch_mtf_context(symbol) if use_mtf else {}
        long_trend = generate_signal(df, "LONG", mtf=mtf, strategy="TREND")
        short_trend = generate_signal(df, "SHORT", mtf=mtf, strategy="TREND")
        long_rev = generate_signal(df, "LONG", mtf=mtf, strategy="REVERSAL")
        short_rev = generate_signal(df, "SHORT", mtf=mtf, strategy="REVERSAL")
        candidates = [x for x in (long_trend, short_trend, long_rev, short_rev) if x is not None]
        if not candidates:
            return None

        # Regime chooses which engine is allowed to compete, rather than WFO.
        state = classify_market_state(df)
        if state["state"] == "TREND_UP":
            allowed = [x for x in candidates if x["direction"] == "LONG" and x["strategy"] == "TREND"]
        elif state["state"] == "TREND_DOWN":
            allowed = [x for x in candidates if x["direction"] == "SHORT" and x["strategy"] == "TREND"]
        elif state["state"] == "EXHAUSTION_UP":
            allowed = [x for x in candidates if x["direction"] == "SHORT" and x["strategy"] == "REVERSAL"]
        elif state["state"] == "EXHAUSTION_DOWN":
            allowed = [x for x in candidates if x["direction"] == "LONG" and x["strategy"] == "REVERSAL"]
        else:
            allowed = candidates
        if not allowed:
            allowed = candidates
        best_sig = max(allowed, key=lambda x: (x["score"], x["rr"]))
        opposite = [x for x in candidates if x["direction"] != best_sig["direction"]]
        opp_score = max([x["score"] for x in opposite], default=0.0)
        score_gap = best_sig["score"] - opp_score

        reasons = [f"시장상태 {state['state']}", f"전략 {best_sig['strategy']}"]
        if best_sig["mtf_score"] >= 65: reasons.append("MTF 정렬")
        elif best_sig["mtf_score"] < 50: reasons.append("MTF 충돌")
        if best_sig["rr"] >= 1.8: reasons.append("기대 R:R 양호")
        elif best_sig["rr"] < 1.3: reasons.append("R:R 부족")

        # Dynamic current-market gate. WFO is NOT a hard gate.
        # The threshold adapts to regime/strategy so the engine does not become
        # artificially selective in transition/range markets, while strong trends
        # still require stronger confirmation.
        gate = dynamic_execution_gate(state["state"], best_sig)
        strong = (
            best_sig["score"] >= gate["score_min"]
            and best_sig["rr"] >= gate["rr_min"]
            and score_gap >= gate["gap_min"]
        )
        if best_sig["strategy"] == "REVERSAL":
            strong = strong and best_sig["reversal_score"] >= gate["reversal_min"]
        if best_sig["mtf_score"] < gate["mtf_min"] and best_sig["strategy"] == "TREND":
            strong = False
        status = best_sig["direction"] if strong else "WAIT"
        if status == "WAIT":
            if score_gap < gate["gap_min"]: wait_reason = f"방향 우세 부족({score_gap:.0f}<{gate['gap_min']:.0f})"
            elif best_sig["rr"] < gate["rr_min"]: wait_reason = f"기대 R:R 부족({best_sig['rr']:.2f}<{gate['rr_min']:.2f})"
            elif best_sig["strategy"] == "REVERSAL" and best_sig["reversal_score"] < gate["reversal_min"]: wait_reason = "역추세 반전 증거 부족"
            elif best_sig["strategy"] == "TREND" and best_sig["mtf_score"] < gate["mtf_min"]: wait_reason = "추세 MTF 확인 부족"
            else: wait_reason = "현재 구조의 실행 조건 미충족"
            reasons.insert(0, wait_reason)
        reasons.append(f"Gate {gate['profile']}")

        daily_ret = df["Close"].pct_change().dropna().tail(30)
        vol_30d_pct = float(daily_ret.std(ddof=1) * math.sqrt(30) * 100) if len(daily_ret) >= 10 else np.nan
        return {
            "symbol": symbol, "exchange": exchange_id, "price": float(latest["Close"]),
            "change_30d": float((latest["Close"] / df["Close"].iloc[-31] - 1) * 100),
            "vol_30d_pct": vol_30d_pct,
            "direction": status, "signal_direction": best_sig["direction"],
            "score": float(best_sig["score"]), "signal": best_sig,
            "wf": empty_metrics(), "rsi": float(latest["RSI14"]), "adx": float(latest["ADX14"]),
            "atr_pct": float(latest["ATR_PCT"]), "poc": float(latest["POC_Price"]),
            "rel_volume": float(latest["REL_VOLUME"]), "reason": " · ".join(reasons),
            "score_gap": float(score_gap), "status_pass": bool(status in {"LONG", "SHORT"}),
            "df": df, "cfg": cfg,
        }
    except Exception:
        return None


def wfo_robustness_score(wf: dict, strategy: str) -> float:
    """Convert OOS robustness into a bounded 0-100 *validation* score.

    This is not a probability and is intentionally capped as a small modifier
    to the live-market score. Positive-window consistency matters more than one
    spectacular OOS run.
    """
    trades = int(wf.get("trades", 0))
    windows = wf.get("windows", []) or []
    if trades < 5 or len(windows) < 3:
        return np.nan
    pf = float(wf.get("profit_factor", 0.0))
    ret = float(wf.get("total_return", 0.0))
    mdd = abs(float(wf.get("max_drawdown", 0.0)))
    positive = float(np.mean([w.get("oos_return", 0.0) > 0 for w in windows]))
    median_win = float(np.median([w.get("oos_return", 0.0) for w in windows]))
    active = [w for w in windows if w.get("oos_trades", 0) > 0]
    coverage = len(active) / max(len(windows), 1)
    score = 45.0
    score += coverage * 8.0
    score += min(max(pf - 1.0, 0.0), 2.0) * 12.0
    score += min(max(ret, -0.2), 0.5) * 25.0
    score += positive * 18.0
    score += np.clip(median_win, -0.10, 0.20) * 25.0
    score -= min(mdd, 0.30) * 35.0
    # Reversal deserves a small conservatism penalty because fewer setups and
    # regime dependence make its OOS sample more fragile.
    if strategy == "REVERSAL":
        score -= 3.0
    return float(np.clip(score, 0, 100))


def apply_adaptive_wfo(rows: list[dict], cfg: BacktestConfig) -> None:
    """Apply WFO as a capped robustness modifier, never as a hard gate.

    88% current-market evidence + up to 12% WFO robustness when enough OOS
    evidence exists. With insufficient WFO evidence the live score is untouched.
    """
    for r in rows:
        wf = r.get("wf", {})
        wscore = wfo_robustness_score(wf, r["signal"].get("strategy", "TREND"))
        r["wfo_score"] = None if not np.isfinite(wscore) else float(wscore)
        if np.isfinite(wscore):
            # Adaptive evidence weight: WFO earns influence only when its OOS
            # sample is large enough. It can never exceed 12% of the final score.
            trades = int(wf.get("trades", 0))
            windows = len(wf.get("windows", []) or [])
            pf = float(wf.get("profit_factor", 0.0))
            if trades >= max(15, cfg.wfo_min_trades * 3) and windows >= max(8, cfg.wfo_min_windows + 5) and pf >= 1.20:
                w = float(cfg.wfo_weight_max)
            elif trades >= cfg.wfo_min_trades * 2 and windows >= cfg.wfo_min_windows + 2:
                w = min(float(cfg.wfo_weight_max), 0.08)
            else:
                w = min(float(cfg.wfo_weight_max), 0.05)
            base = float(r["score"])
            r["score_raw"] = base
            r["score"] = float((1.0 - w) * base + w * wscore)
            r["wfo_weight"] = w
        else:
            r["score_raw"] = float(r["score"])
            r["wfo_weight"] = 0.0


def _candidate_return_series(row: dict, bars: int = 90) -> pd.Series:
    """Return aligned daily returns for portfolio-correlation checks."""
    df = row.get("df")
    if df is None or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna().tail(bars + 1)
    if len(close) < 20:
        return pd.Series(dtype=float)
    ret = np.log(close / close.shift(1)).dropna()
    ret.name = row.get("symbol", "")
    return ret


def _pair_corr(a: dict, b: dict) -> float:
    """Correlation of recent daily returns; NaN means insufficient evidence."""
    ra = _candidate_return_series(a)
    rb = _candidate_return_series(b)
    if ra.empty or rb.empty:
        return np.nan
    x = pd.concat([ra, rb], axis=1).dropna()
    if len(x) < 20:
        return np.nan
    c = float(x.iloc[:, 0].corr(x.iloc[:, 1]))
    return c if np.isfinite(c) else np.nan


def oos_calibrated_win_rate(row: dict) -> tuple[float, float, int]:
    """Estimate TP-before-SL probability from the OOS sample with shrinkage.

    This is an empirical OOS estimate, not a guaranteed future probability.
    A Beta(10, 10) prior pulls small samples toward 50%, while larger samples
    are allowed to move the estimate toward the observed OOS win rate.
    Returns (estimated_win_rate_pct, sample_confidence_pct, trades).
    """
    wf = row.get("wf", {}) or {}
    trades = int(wf.get("trades", 0) or 0)
    observed = float(wf.get("win_rate", np.nan))
    windows = wf.get("windows", []) or []
    if trades <= 0 or not np.isfinite(observed):
        return np.nan, 0.0, 0

    # Effective sample size discounts sparse OOS coverage and inconsistent windows.
    active = [w for w in windows if int(w.get("oos_trades", 0) or 0) > 0]
    coverage = len(active) / max(len(windows), 1)
    positive = np.mean([w.get("oos_return", 0.0) > 0 for w in active]) if active else 0.0
    consistency = 0.5 + 0.5 * float(positive)
    effective_n = max(1.0, trades * coverage * consistency)

    wins_eff = effective_n * np.clip(observed / 100.0, 0.0, 1.0)
    # Conservative prior centered at 50% for small samples.
    prior = 10.0
    p = (wins_eff + prior) / (effective_n + 2.0 * prior)
    confidence = 100.0 * (1.0 - math.exp(-effective_n / 12.0))
    return float(np.clip(p * 100.0, 0.0, 100.0)), float(np.clip(confidence, 0.0, 100.0)), trades


def _cross_sectional_percentile(values: list[float], value: float) -> float:
    """Percentile rank within the currently scanned universe (0-100)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 2 or not np.isfinite(value):
        return 50.0
    return float(np.clip((np.sum(arr <= value) - 1) / (arr.size - 1) * 100.0, 0.0, 100.0))


def add_market_opportunity_score(rows: list[dict]) -> None:
    """Score profit opportunity from liquidity, productive volatility and trend quality.

    This is a ranking overlay, not a probability. Liquidity uses cross-sectional
    quote-volume percentile; volatility favors a usable high-volatility zone and
    penalizes extreme ATR%; ADX measures directional trend strength. R:R and the
    empirical OOS TP-before-SL estimate are included so raw volatility cannot
    dominate the final decision.
    """
    if not rows:
        return

    volumes = [float(r.get("quote_volume", np.nan)) for r in rows]
    log_volumes = [math.log10(max(v, 1.0)) if np.isfinite(v) and v > 0 else np.nan for v in volumes]
    atr_values = [float(r.get("atr_pct", np.nan)) for r in rows]
    vol_30d_values = [float(r.get("vol_30d_pct", np.nan)) for r in rows]

    finite_atr = np.asarray([x for x in atr_values if np.isfinite(x) and x > 0], dtype=float)
    median_atr = float(np.median(finite_atr)) if finite_atr.size else 2.0
    target_atr = float(np.clip(median_atr * 1.8, 2.0, 5.0))

    for r, log_vol, atr, vol_30d in zip(rows, log_volumes, atr_values, vol_30d_values):
        sig = r.get("signal", {}) or {}
        adx = float(r.get("adx", sig.get("adx", 0.0)))
        rr = float(sig.get("rr", 0.0))

        liquidity = _cross_sectional_percentile(log_volumes, log_vol)

        if np.isfinite(atr) and atr > 0:
            # Log-distance keeps the score symmetric for low/high volatility.
            volatility = 100.0 * math.exp(-abs(math.log(max(atr, 0.25) / target_atr)) / 1.10)
            # Extreme ATR can be profitable but usually comes with worse execution
            # conditions, so reduce only the tail rather than hard-filtering it.
            if atr > 8.0:
                volatility *= max(0.55, 1.0 - (atr - 8.0) * 0.06)
            elif atr < 0.8:
                volatility *= 0.65
        else:
            volatility = 0.0

        vol_30d_score = _cross_sectional_percentile(vol_30d_values, vol_30d)
        adx_score = float(np.clip((adx - 15.0) / 20.0 * 100.0, 0.0, 100.0))
        rr_score = float(np.clip(50.0 + (rr - 1.0) * 35.0, 0.0, 100.0))
        oos_prob = float(r.get("oos_est_win_rate", np.nan))
        oos_conf = float(r.get("oos_confidence", 0.0))
        oos_score = 50.0 + (oos_prob - 50.0) * (oos_conf / 100.0) if np.isfinite(oos_prob) else 50.0

        # Profit-opportunity blend: liquidity + 30D realized volatility are the
        # primary universe-selection signals, while ATR/ADX/R:R/OOS keep the
        # ranking tied to tradability and actual setup quality.
        opportunity = (
            0.20 * liquidity
            + 0.20 * vol_30d_score
            + 0.10 * volatility
            + 0.15 * adx_score
            + 0.15 * rr_score
            + 0.20 * oos_score
        )

        r["liquidity_score"] = float(np.clip(liquidity, 0.0, 100.0))
        r["volatility_score"] = float(np.clip(volatility, 0.0, 100.0))
        r["vol_30d_score"] = float(np.clip(vol_30d_score, 0.0, 100.0))
        r["adx_score"] = float(np.clip(adx_score, 0.0, 100.0))
        r["rr_score"] = float(np.clip(rr_score, 0.0, 100.0))
        r["oos_opportunity_score"] = float(np.clip(oos_score, 0.0, 100.0))
        r["opportunity_score"] = float(np.clip(opportunity, 0.0, 100.0))
        r["atr_target_pct"] = target_atr


def execution_priority_score(row: dict) -> float:
    """Rank live candidates using signal quality plus profit-opportunity evidence.

    The result is an execution-priority index, NOT a probability. The opportunity
    overlay is capped so liquidity/volatility cannot overpower a weak trade setup.
    """
    sig = row.get("signal", {}) or {}
    base = float(row.get("score", 0.0))
    gap = float(row.get("score_gap", 0.0))
    mtf = float(sig.get("mtf_score", 0.0))
    strategy = sig.get("strategy", "TREND")
    reversal = float(sig.get("reversal_score", 0.0))
    opportunity = float(row.get("opportunity_score", 50.0))

    gap_component = float(np.clip(50.0 + gap * 3.0, 0.0, 100.0))
    mtf_component = float(np.clip(mtf, 0.0, 100.0))
    rev_component = float(np.clip(reversal, 0.0, 100.0)) if strategy == "REVERSAL" else 70.0

    # 70% live signal quality + 30% explicit profit-opportunity overlay.
    # OOS is 20% of the opportunity overlay (6% of total priority), while its
    # own confidence shrinkage prevents a small sample from dominating the rank.
    live_core = (
        0.58 * base
        + 0.22 * gap_component
        + 0.15 * mtf_component
        + 0.05 * rev_component
    )
    priority = 0.70 * live_core + 0.30 * opportunity
    return float(np.clip(priority, 0.0, 100.0))

def add_execution_priority(rows: list[dict]) -> None:
    """Attach OOS estimates, profit-opportunity score and execution rank."""
    for r in rows:
        r["oos_est_win_rate"], r["oos_confidence"], r["oos_sample_trades"] = oos_calibrated_win_rate(r)

    # Cross-sectional liquidity/volatility scoring must happen after OOS values
    # exist and before the final execution-priority rank is calculated.
    add_market_opportunity_score(rows)

    for r in rows:
        r["execution_priority"] = execution_priority_score(r)
        r["execution_rank"] = None

    ordered = sorted(
        rows,
        key=lambda r: (
            r.get("direction") in {"LONG", "SHORT"},
            float(r.get("execution_priority", 0.0)),
            float(r.get("score", 0.0)),
            float(r.get("signal", {}).get("rr", 0.0)),
        ),
        reverse=True,
    )
    rank = 0
    for r in ordered:
        if r.get("direction") in {"LONG", "SHORT"}:
            rank += 1
            r["execution_rank"] = rank


def select_portfolio_candidates(
    rows: list[dict],
    max_candidates: int = 3,
    max_same_direction: int = 2,
    corr_limit: float = 0.75,
) -> list[dict]:
    """Select a small diversified execution set from actionable signals.

    This is not a prediction layer. It reduces redundant exposure when several
    coins express essentially the same daily return stream. The first candidate
    is the strongest live-market signal; subsequent candidates must add enough
    diversification or be materially stronger than an already-selected signal.
    """
    candidates = [r for r in rows if r.get("direction") in {"LONG", "SHORT"}]
    for r in rows:
        r["portfolio_selected"] = False
        r["portfolio_rank"] = None
        r["portfolio_score"] = np.nan
        r["max_selected_corr"] = np.nan
        r["portfolio_reason"] = ""

    if not candidates:
        return []

    # Use the live execution-priority index first. QUANT remains the primary
    # underlying signal score, but the list is not presented as a probability
    # ranking. R:R and directional separation are tie-breakers.
    remaining = sorted(
        candidates,
        key=lambda r: (
            float(r.get("execution_priority", execution_priority_score(r))),
            float(r.get("score", 0.0)),
            float(r.get("signal", {}).get("rr", 0.0)),
            float(r.get("score_gap", 0.0)),
        ),
        reverse=True,
    )

    selected = []
    while remaining and len(selected) < max_candidates:
        best = None
        best_key = None
        for r in remaining:
            same_dir = sum(x.get("signal_direction") == r.get("signal_direction") for x in selected)
            if same_dir >= max_same_direction:
                continue

            corrs = [_pair_corr(r, x) for x in selected]
            valid_corrs = [abs(c) for c in corrs if np.isfinite(c)]
            max_corr = max(valid_corrs) if valid_corrs else 0.0

            # Penalize redundant exposure. A highly correlated candidate can still
            # enter if it is clearly stronger, but it pays a meaningful penalty.
            redundancy_penalty = max(0.0, max_corr - 0.35) * 28.0
            same_dir_penalty = 3.0 if selected and same_dir > 0 else 0.0
            pscore = float(r.get("execution_priority", execution_priority_score(r))) - redundancy_penalty - same_dir_penalty

            # Hard correlation cap unless this is materially stronger than all
            # currently selected alternatives. This prevents BTC/ETH/SOL-like
            # clusters from consuming the entire 3-slot execution list.
            blocked = bool(valid_corrs and max_corr >= corr_limit and pscore < float(r.get("score", 0.0)) - 4.0)
            if blocked:
                continue

            key = (pscore, float(r.get("signal", {}).get("rr", 0.0)), -max_corr)
            if best is None or key > best_key:
                best = (r, pscore, max_corr)
                best_key = key

        if best is None:
            break

        r, pscore, max_corr = best
        r["portfolio_selected"] = True
        r["portfolio_rank"] = len(selected) + 1
        r["portfolio_score"] = float(pscore)
        r["max_selected_corr"] = float(max_corr) if np.isfinite(max_corr) else np.nan
        if not selected:
            r["portfolio_reason"] = "최종 QUANT 1위 후보"
        elif np.isfinite(max_corr) and max_corr >= corr_limit:
            r["portfolio_reason"] = f"강한 상관에도 점수 우위(상관 {max_corr:.2f})"
        elif np.isfinite(max_corr):
            r["portfolio_reason"] = f"분산효과 확보(상관 {max_corr:.2f})"
        else:
            r["portfolio_reason"] = "상관 데이터 부족, 점수·R:R 기준"
        selected.append(r)
        remaining.remove(r)

    # Explain why strong actionable candidates were not selected.
    selected_set = {id(x) for x in selected}
    for r in candidates:
        if id(r) in selected_set:
            continue
        corr_to_selected = [_pair_corr(r, x) for x in selected]
        valid = [abs(c) for c in corr_to_selected if np.isfinite(c)]
        if valid and max(valid) >= corr_limit:
            r["portfolio_reason"] = f"중복노출 방지(상관 {max(valid):.2f})"
        else:
            r["portfolio_reason"] = "상위 3개 외 후보"
        r["max_selected_corr"] = max(valid) if valid else np.nan

    return selected


def apply_position_sizing(rows: list[dict], cfg: BacktestConfig) -> list[dict]:
    """Risk-based sizing for the selected execution portfolio.

    Size is determined from account risk and the actual Entry/SL distance, not
    from nominal coin price. Portfolio risk is capped across selected trades.
    This is a sizing layer only; it does not change signal direction.
    """
    selected = [r for r in rows if r.get("portfolio_selected")]
    if not selected or cfg.account_size <= 0:
        return rows

    total_budget = cfg.account_size * cfg.max_portfolio_risk
    base_budget = min(cfg.account_size * cfg.risk_per_trade, total_budget / max(len(selected), 1))
    remaining_budget = total_budget

    # Stronger portfolio candidates receive their normal risk budget first.
    selected = sorted(selected, key=lambda r: int(r.get("portfolio_rank") or 999))
    for r in selected:
        sig = r.get("signal", {})
        entry = float(sig.get("entry", r.get("price", np.nan)))
        sl = float(sig.get("sl", np.nan))
        tp = float(sig.get("tp", np.nan))
        if not (np.isfinite(entry) and np.isfinite(sl) and entry > 0 and sl > 0):
            continue

        stop_pct = abs(entry - sl) / entry
        if stop_pct <= 0:
            continue

        # Never let one position consume more than max_position_weight of equity.
        risk_budget = min(base_budget, remaining_budget)
        notional = min(risk_budget / stop_pct, cfg.account_size * cfg.max_position_weight)
        risk_used = notional * stop_pct

        r["risk_budget"] = float(risk_budget)
        r["stop_pct"] = float(stop_pct * 100.0)
        r["position_notional"] = float(notional)
        r["position_weight"] = float(notional / cfg.account_size * 100.0)
        r["risk_used"] = float(risk_used)
        r["tp_pct"] = float(abs(tp - entry) / entry * 100.0) if np.isfinite(tp) else np.nan
        r["portfolio_risk_after"] = float((total_budget - max(0.0, remaining_budget - risk_used)) / cfg.account_size * 100.0)
        remaining_budget = max(0.0, remaining_budget - risk_used)

    total_used = sum(float(r.get("risk_used", 0.0)) for r in selected)
    for r in rows:
        r["portfolio_risk_total"] = float(total_used / cfg.account_size * 100.0) if cfg.account_size > 0 else 0.0
    return rows


def attach_validation(rows: list[dict], cfg: BacktestConfig, max_validate: int = 5) -> pd.DataFrame:
    """Run OOS only on the best actionable/near-actionable candidates.

    The 1D dataframe produced by the first pass is retained until validation, so
    WFO does not issue another 1D download for the same symbol.
    """
    if not rows:
        return pd.DataFrame()

    rows = sorted(rows, key=lambda x: (x["status_pass"], x["score"], x["score_gap"]), reverse=True)

    # Prefer actual execution candidates, then the strongest WAIT candidates as
    # diagnostics. This avoids spending OOS time on weak rows.
    validation_pool = [r for r in rows if r.get("status_pass")]
    if len(validation_pool) < max_validate:
        validation_pool += [r for r in rows if not r.get("status_pass")]
    validation_pool = validation_pool[:max_validate]

    for r in validation_pool:
        try:
            df = r.get("df")
            if df is None or len(df) < 260:
                df, _ = fetch_ohlcv_fallback(r["symbol"], "1d", 700)
            strategy = r["signal"].get("strategy", "TREND")
            wf = walk_forward(df, r["signal_direction"], cfg, strategy) if len(df) >= 260 else empty_metrics()
            r["wf"] = wf
            r["wfo_strategy"] = strategy
            r["reason"] += f" · OOS({strategy}) PF {wf['profit_factor']:.2f}" if wf["trades"] else " · OOS 검증 표본 부족"
        except Exception:
            r["wf"] = empty_metrics()

    apply_adaptive_wfo(rows, cfg)
    # Rank only after WFO has finished updating the final QUANT score.
    add_execution_priority(rows)
    select_portfolio_candidates(rows, max_candidates=3, max_same_direction=2, corr_limit=0.75)
    apply_position_sizing(rows, cfg)

    # Remove large dataframes before DataFrame construction/session storage.
    for r in rows:
        r.pop("df", None)
        r.pop("cfg", None)
    return pd.DataFrame(rows)


def run_parallel(symbols: list[str], cfg: BacktestConfig, workers: int = 6, market_meta: Optional[dict] = None) -> pd.DataFrame:
    """Two-stage scan: cheap 1D prefilter, then MTF only for strongest candidates.

    This cuts the expensive 1D/4H/1H request fan-out substantially while keeping
    MTF in the final decision. Cached 1D data makes the second pass inexpensive.
    """
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_symbol, s, cfg, False) for s in symbols]
        for f in concurrent.futures.as_completed(futures):
            try:
                r = f.result()
                if r: rows.append(r)
            except Exception:
                pass

    # Only the strongest 12 current-market candidates receive the full MTF pass.
    # This is the main speed optimization for 30-coin scans (12 x 3 TF instead
    # of 30 x 3 TF), while preserving MTF for the candidates that can matter.
    shortlist_n = min(12, len(rows))
    shortlist = sorted(rows, key=lambda x: (x["status_pass"], x["score"], x["score_gap"]), reverse=True)[:shortlist_n]
    symbols_short = [r["symbol"] for r in shortlist]

    refined = []
    if symbols_short:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(analyze_symbol, s, cfg, True) for s in symbols_short]
            for f in concurrent.futures.as_completed(futures):
                try:
                    r = f.result()
                    if r: refined.append(r)
                except Exception:
                    pass

    refined_map = {r["symbol"]: r for r in refined}
    final_rows = [refined_map.get(r["symbol"], r) for r in rows]

    # Carry 24H quote volume into the final ranking layer. This avoids extra API
    # calls because the ticker universe is already fetched in main().
    meta = market_meta or {}
    for r in final_rows:
        r["quote_volume"] = float(meta.get(r["symbol"], {}).get("quote_volume", np.nan))
    return attach_validation(final_rows, cfg, max_validate=5)


# ============================================================
# 9. UI
# ============================================================

def fmt_price(x):
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:,.8f}"


def render_regime():
    regime = fetch_market_regime()

    label = str(regime.get("label", "Mixed"))
    score = float(regime.get("score", 50.0))
    if label == "Risk-On":
        direction = "상승 우세"
        long_env, short_env = "LONG 유리", "SHORT 불리"
        icon = "🟢"
    elif label == "Risk-Off":
        direction = "하락 우세"
        long_env, short_env = "LONG 불리", "SHORT 유리"
        icon = "🔴"
    else:
        label = "Mixed"
        direction = "방향 혼재"
        long_env, short_env = "LONG 선별", "SHORT 선별"
        icon = "🟡"

    # Main screen: keep the market regime to one compact, decision-oriented line.
    st.markdown(
        f"**🌐 시장상태**  {icon} **{label}** · {direction} · "
        f"{long_env} · {short_env} · 시장점수 **{score:.0f}/100**"
    )

    # Details are intentionally collapsed so the first screen stays compact.
    with st.expander("시장상태 상세보기", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("시장 레짐", f"{icon} {label}")
        with c2:
            st.metric("시장 방향", direction)
        with c3:
            st.metric("LONG 환경", long_env)
        with c4:
            st.metric("SHORT 환경", short_env)

        btc = regime.get("btc")
        btc_txt = fmt_price(float(btc["Close"])) if btc is not None else "-"
        detail_txt = " · ".join(regime.get("details", []))
        st.caption(f"BTC {btc_txt} · 시장점수 {score:.0f}/100" + (f" · {detail_txt}" if detail_txt else ""))
    return regime


def render_mobile_card(row: pd.Series):
    direction = row["direction"]
    icon = "🟢" if direction == "LONG" else ("🔴" if direction == "SHORT" else "⚪")
    sig = row["signal"]
    prio = float(row.get("execution_priority", row.get("score", 0.0)))
    opp = float(row.get("opportunity_score", 0.0))
    oos_p = row.get("oos_est_win_rate", np.nan)
    oos_c = float(row.get("oos_confidence", 0.0))
    oos_txt = f"OOS {oos_p:.1f}%/{oos_c:.0f}" if np.isfinite(oos_p) else "OOS -"

    with st.container(border=True):
        # Compact one-line decision view.
        st.markdown(
            f"**{icon} {row['symbol']} {direction}** · "
            f"우선순위 **{prio:.0f}** · 수익기회 **{opp:.0f}** · "
            f"R:R **1:{sig['rr']:.2f}** · MTF **{sig['mtf_score']:.0f}** · "
            f"ENTRY **{fmt_price(sig['entry_low'])}~{fmt_price(sig['entry_high'])}** · "
            f"SL **{fmt_price(sig['sl'])}** · TP **{fmt_price(sig['tp'])}**"
        )
        with st.expander("상세보기", expanded=False):
            a, b, c, d = st.columns(4)
            a.metric("실전 우선순위", f"{prio:.0f}")
            b.metric("수익기회", f"{opp:.0f}")
            c.metric("QUANT", f"{row['score']:.0f}")
            d.metric("현재가", fmt_price(row["price"]))
            a, b, c, d = st.columns(4)
            a.metric("R:R", f"1 : {sig['rr']:.2f}")
            b.metric("MTF", f"{sig['mtf_score']:.0f}")
            c.metric("방향차", f"{row['score_gap']:.1f}")
            d.metric("전략", sig["strategy"])
            a, b, c = st.columns(3)
            a.metric("ENTRY ZONE", f"{fmt_price(sig['entry_low'])} ~ {fmt_price(sig['entry_high'])}")
            b.metric("SL", fmt_price(sig["sl"]))
            c.metric("🎯 TP", fmt_price(sig["tp"]))
            st.caption(
                f"시장상태 {sig['market_state']} · {oos_txt} · "
                f"유동성 {float(row.get('liquidity_score', 0.0)):.0f} · "
                f"1M변동성 {float(row.get('vol_30d_pct', np.nan)):.1f}% · "
                f"변동성 {float(row.get('volatility_score', 0.0)):.0f} · "
                f"ADX {float(row.get('adx_score', 0.0)):.0f} · "
                f"24H 거래대금 {float(row.get('quote_volume', np.nan))/1_000_000:.1f}M USDT"
            )
            st.caption(
                f"RSI {row['rsi']:.1f} · ATR {row['atr_pct']:.2f}% · "
                f"POC {fmt_price(row['poc'])} · RVOL {row['rel_volume']:.2f}"
            )
            st.caption("판정: " + str(row.get("reason", "-")))


def render_portfolio_candidates(df: pd.DataFrame):
    selected = df[df.get("portfolio_selected", False) == True].copy() if "portfolio_selected" in df else pd.DataFrame()
    if selected.empty:
        actionable = df[df["direction"].isin(["LONG", "SHORT"])].copy() if "direction" in df else pd.DataFrame()
        if actionable.empty:
            st.warning("⏸️ 현재 즉시 실행 가능한 신호가 없습니다. WAIT 상태를 유지합니다.")
        else:
            st.info("현재 조건에서 포트폴리오 확정 후보가 없습니다. 상위 후보는 상세분석에서 확인하세요.")
        return

    selected = selected.sort_values("portfolio_rank")
    st.markdown("### 🚀 지금 실행할 후보")
    st.caption("한 줄 요약은 즉시 매매 판단용이며, 상세 수치와 근거는 각 후보의 '상세보기'에서 확인할 수 있습니다.")

    for _, row in selected.iterrows():
        sig = row["signal"]
        icon = "🟢" if row["direction"] == "LONG" else "🔴"
        pos = row.get("position_notional", np.nan)
        risk = row.get("risk_used", np.nan)
        pos_txt = f"{pos:,.0f} USDT" if np.isfinite(pos) else "-"
        risk_txt = f"{risk:,.2f} USDT" if np.isfinite(risk) else "-"
        prio = float(row.get("execution_priority", row.get("score", 0.0)))
        opp = float(row.get("opportunity_score", 0.0))
        oos_p = row.get("oos_est_win_rate", np.nan)
        oos_c = float(row.get("oos_confidence", 0.0))
        oos_txt = f"OOS {oos_p:.1f}%/{oos_c:.0f}" if np.isfinite(oos_p) else "OOS -"

        with st.container(border=True):
            # Same compact one-line format as the market/result cards.
            st.markdown(
                f"**{icon} #{int(row['portfolio_rank'])} {row['symbol']} {row['direction']}** · "
                f"우선순위 **{prio:.0f}** · 수익기회 **{opp:.0f}** · "
                f"R:R **1:{sig['rr']:.2f}** · MTF **{sig['mtf_score']:.0f}** · "
                f"ENTRY **{fmt_price(sig['entry_low'])}~{fmt_price(sig['entry_high'])}** · "
                f"SL **{fmt_price(sig['sl'])}** · TP **{fmt_price(sig['tp'])}**"
            )
            with st.expander("상세보기", expanded=False):
                a, b, c, d = st.columns(4)
                a.metric("실전 우선순위", f"{prio:.0f}")
                b.metric("수익기회", f"{opp:.0f}")
                c.metric("QUANT", f"{row['score']:.0f}")
                d.metric("R:R", f"1 : {sig['rr']:.2f}")
                a, b, c = st.columns(3)
                a.metric("ENTRY", f"{fmt_price(sig['entry_low'])} ~ {fmt_price(sig['entry_high'])}")
                b.metric("🔴 SL", fmt_price(sig["sl"]))
                c.metric("🎯 TP", fmt_price(sig["tp"]))
                a, b, c = st.columns(3)
                a.metric("포지션", pos_txt)
                b.metric("SL 위험", risk_txt)
                c.metric("전략", sig["strategy"])
                st.caption(
                    f"시장상태 {sig['market_state']} · {oos_txt} · "
                    f"유동성 {float(row.get('liquidity_score', 0.0)):.0f} · "
                    f"변동성 {float(row.get('volatility_score', 0.0)):.0f} · "
                    f"ADX {float(row.get('adx_score', 0.0)):.0f} · "
                    f"24H 거래대금 {float(row.get('quote_volume', np.nan))/1_000_000:.1f}M USDT"
                )
                st.caption(f"전략 {sig['strategy']} · 시장상태 {sig['market_state']} · {row.get('portfolio_reason', '')}")


def render_results(df: pd.DataFrame):
    if df.empty:
        st.warning("조건을 만족하는 검증 결과가 없습니다.")
        return

    with st.expander("🔎 후보 상세 / 전체 분석 결과", expanded=False):
        display = df.copy()
        display["우선순위"] = display["execution_priority"].round(1) if "execution_priority" in display else display["score"].round(1)
        display["수익기회"] = display["opportunity_score"].round(1) if "opportunity_score" in display else np.nan
        display["유동성"] = display["liquidity_score"].round(0) if "liquidity_score" in display else np.nan
        display["변동성"] = display["volatility_score"].round(0) if "volatility_score" in display else np.nan
        display["1M변동성"] = display["vol_30d_pct"].round(2) if "vol_30d_pct" in display else np.nan
        display["ADX"] = display["adx_score"].round(0) if "adx_score" in display else np.nan
        display["score"] = display["score"].round(1)
        display["price"] = display["price"].map(fmt_price)
        display["30D"] = display["change_30d"].round(2)
        display["OOS"] = (display["wf"].apply(lambda x: x["total_return"]) * 100).round(2)
        display["Win"] = display["wf"].apply(lambda x: x["win_rate"]).round(1)
        display["OOS추정승률"] = display["oos_est_win_rate"].round(1) if "oos_est_win_rate" in display else np.nan
        display["OOS신뢰도"] = display["oos_confidence"].round(0) if "oos_confidence" in display else 0
        display["PF"] = display["wf"].apply(lambda x: x["profit_factor"]).round(2)
        display["MDD"] = (display["wf"].apply(lambda x: x["max_drawdown"]) * 100).round(2)
        display["Trades"] = display["wf"].apply(lambda x: x["trades"])
        display["RR"] = display["signal"].apply(lambda x: x["rr"]).round(2)
        display["MTF"] = display["signal"].apply(lambda x: x["mtf_score"]).round(0)
        display["WFO"] = display["wfo_score"].round(0) if "wfo_score" in display else np.nan
        display["WFO비중"] = (display["wfo_weight"] * 100).round(0) if "wfo_weight" in display else 0
        display["포트폴리오"] = display["portfolio_rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "-") if "portfolio_rank" in display else "-"
        display["포지션"] = display["position_notional"].round(0) if "position_notional" in display else np.nan
        display["SL위험"] = display["risk_used"].round(0) if "risk_used" in display else np.nan
        display["Status"] = display["direction"]
        display["판정사유"] = display["reason"]
        cols = ["symbol", "Status", "포트폴리오", "우선순위", "수익기회", "유동성", "1M변동성", "변동성", "ADX", "score", "price", "RR", "MTF", "OOS추정승률", "OOS신뢰도", "WFO", "OOS", "Win", "PF", "MDD", "Trades", "포지션", "SL위험", "판정사유"]
        st.dataframe(display[cols], use_container_width=True, hide_index=True, column_config={
            "symbol":"종목", "Status":"최종판정", "포트폴리오":"실행순위", "우선순위":"실전 우선순위",
            "score":"QUANT", "price":"현재가", "RR":"R:R", "MTF":"MTF", "OOS추정승률":"OOS 추정 승률%", "OOS신뢰도":"OOS 신뢰도", "WFO":"WFO 견고성",
            "수익기회":"수익기회 점수", "유동성":"유동성", "1M변동성":"1개월 실현변동성%", "변동성":"단기 변동성", "ADX":"ADX",
            "OOS":"OOS%", "Win":"OOS 승률%", "PF":"PF", "MDD":"MDD%", "Trades":"거래수",
            "포지션":"권장 포지션(USDT)", "SL위험":"SL 위험금액", "판정사유":"판정사유",
        })

        st.markdown("### 📱 상세 신호")
        for _, row in df.head(10).iterrows():
            render_mobile_card(row)

    with st.expander("📈 점수 분포 / 진단", expanded=False):
        fig = go.Figure()
        ranked = df.sort_values("execution_priority", ascending=False) if "execution_priority" in df else df.sort_values("score", ascending=False)
        fig.add_trace(go.Bar(x=ranked.head(15)["symbol"], y=ranked.head(15)["score"], text=ranked.head(15)["score"].round(0), textposition="auto"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="QUANT SCORE", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key="score_chart_v2")
        st.caption("QUANT와 실전 우선순위는 예측확률이 아닙니다. OOS 승률은 과거 검증표본의 통계이며 미래 승률을 보장하지 않습니다.")



def main():
    st.title("🔥 Crypto Quant · 실전 매매판")
    st.caption("확정봉 · 1D/4H/1H MTF · Trend/Reverse · 단일 TP/SL · 거래대금×1개월 변동성×ATR×ADX×R:R×OOS 수익기회 점수")

    with st.sidebar:
        st.header("⚙️ 분석 설정")
        min_volume = st.number_input(
            "최소 24H 거래대금 (USDT)",
            min_value=100_000.0,
            max_value=100_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
        )
        min_oos_trades = st.number_input(
            "최소 OOS 거래수",
            min_value=2,
            max_value=30,
            value=5,
        )
        fee = st.number_input("편도 수수료", 0.0, 0.01, 0.0005, format="%.4f")
        slippage = st.number_input("편도 슬리피지", 0.0, 0.01, 0.0005, format="%.4f")
        spread = st.number_input("스프레드", 0.0, 0.01, 0.0002, format="%.4f")
        st.markdown("### 💰 리스크/포지션 설정")
        account_size = st.number_input("가정 계좌금액 (USDT)", 100.0, 10_000_000.0, 10_000.0, step=100.0)
        risk_per_trade = st.number_input("1회 거래 최대위험", 0.001, 0.03, 0.0075, step=0.001, format="%.3f")
        max_portfolio_risk = st.number_input("전체 포트폴리오 최대위험", 0.005, 0.10, 0.02, step=0.005, format="%.3f")
        max_position_weight = st.number_input("단일 포지션 최대 비중", 0.05, 1.0, 0.40, step=0.05, format="%.2f")

    regime = render_regime()

    st.divider()

    market = None
    for ex_id in DEFAULT_EXCHANGES:
        try:
            market = fetch_tickers(ex_id)
            if len(market):
                break
        except Exception:
            continue

    if market is None or market.empty:
        st.error("거래소 시세를 불러오지 못했습니다.")
        return

    market = market[market["quote_volume"] >= min_volume].copy()
    # Analyze the top 50 liquid USDT pairs. 24H quote volume defines the
    # tradable universe; 30D realized volatility is used later for ranking.
    TOP_N = 50
    universe = (
        market.sort_values("quote_volume", ascending=False)
        .drop_duplicates("symbol")
        .head(TOP_N)
    )
    symbols_top = universe["symbol"].tolist()
    market_meta = {
        row["symbol"]: {"quote_volume": float(row["quote_volume"])}
        for _, row in universe.iterrows()
    }

    st.markdown(
        f"**분석현황** · 분석가능 **{len(market)}종목** · 검토 **{len(symbols_top)}종목** · "
        f"대상 **거래대금 상위 {TOP_N} + 1개월 변동성 반영**"
    )

    cfg = BacktestConfig(
        fee_rate=fee,
        slippage_rate=slippage,
        spread_rate=spread,
        min_oos_trades=int(min_oos_trades),
        account_size=float(account_size),
        risk_per_trade=float(risk_per_trade),
        max_portfolio_risk=float(max_portfolio_risk),
        max_position_weight=float(max_position_weight),
    )

    st.subheader("🎯 분석 실행")

    # Mobile-friendly buttons
    quick = st.button("⚡ 빠른 분석", use_container_width=True)
    full = st.button("🔬 정밀 분석 · 상위 50", use_container_width=True)

    if quick or full:
        symbols = symbols_top[:10] if quick else symbols_top
        with st.spinner(f"{len(symbols)}개 종목 분석 중... (시장레짐 → 추세/역추세 → Entry/TP/SL)"):
            result = run_parallel(symbols, cfg, workers=6, market_meta=market_meta)
        st.session_state["quant_results"] = result
        st.session_state["quant_time"] = pd.Timestamp.now(tz="UTC")

    result = st.session_state.get("quant_results", pd.DataFrame())

    if not result.empty:
        qt = st.session_state.get("quant_time")
        if qt is not None:
            st.caption(f"마지막 분석: {qt.tz_convert('Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S')} KST · 실전 우선순위/ENTRY/SL/TP 기준으로 정렬")

        passed = result[result["direction"].isin(["LONG", "SHORT"])].copy()
        waits = result[result["direction"] == "WAIT"].copy()
        st.success(f"분석 완료 · {len(passed)}개 실행신호 · {len(waits)}개 WAIT")
        render_portfolio_candidates(result)
        render_results(result)


    st.divider()
    st.caption("확정봉 · 1D 주 분석 → 상위 후보만 1D/4H/1H MTF 정밀검증 · 구조적 단일 TP/SL · OOS 검증은 참고층 · Data source: CCXT-supported exchanges · Analysis is informational, not financial advice.")


if __name__ == "__main__":
    main()
