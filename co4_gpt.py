
# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V2.5
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
import json
import urllib.request
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
    page_title="🔥 Crypto Quant Dashboard V2.5",
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
    train_bars: int = 150
    test_bars: int = 30
    step_bars: int = 30
    min_train_trades: int = 3
    data_limit: int = 500
    min_signal_score: float = 55.0
    monte_carlo_runs: int = 500
    use_mtf: bool = True
    neutral_score_gap: float = 8.0
    min_direction_score: float = 55.0


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


def validate_ohlcv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    if df.empty:
        return False, ["OHLCV 데이터 없음"]
    issues = []
    if not df["timestamp"].is_monotonic_increasing:
        issues.append("시간순 정렬 오류")
    if df["timestamp"].duplicated().any():
        issues.append("중복 캔들")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        issues.append("0 이하 가격")
    if (df["High"] < df[["Open", "Close"]].max(axis=1)).any():
        issues.append("High 불일치")
    if (df["Low"] > df[["Open", "Close"]].min(axis=1)).any():
        issues.append("Low 불일치")
    return len(issues) == 0, issues


@st.cache_data(ttl=300, show_spinner=False)
def fetch_global_market_data() -> dict:
    url = "https://api.coingecko.com/api/v3/global"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CryptoQuantDashboard/2.2"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))["data"]
        total = float(data["total_market_cap"].get("usd", np.nan))
        pct = data.get("market_cap_percentage", {})
        btc_pct = float(pct.get("btc", np.nan))
        eth_pct = float(pct.get("eth", np.nan))
        btc_cap = total * btc_pct / 100 if np.isfinite(btc_pct) else np.nan
        eth_cap = total * eth_pct / 100 if np.isfinite(eth_pct) else np.nan
        total3 = total - btc_cap - eth_cap if np.isfinite(btc_cap) and np.isfinite(eth_cap) else np.nan
        return {"total_market_cap": total, "btc_dominance": btc_pct, "eth_dominance": eth_pct, "total3": total3}
    except Exception:
        return {"total_market_cap": np.nan, "btc_dominance": np.nan, "eth_dominance": np.nan, "total3": np.nan}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_derivatives_snapshot(symbol: str) -> dict:
    base = symbol.split("/")[0]
    out = {"funding": np.nan, "oi": np.nan, "oi_value": np.nan}
    # Derivatives are supplementary factors; failures never block spot analysis.
    for exchange_id in ["bybit", "binance"]:
        try:
            cls = getattr(ccxt, exchange_id)
            ex = cls({"enableRateLimit": True, "timeout": 10000, "options": {"defaultType": "swap"}})
            swap_symbol = f"{base}/USDT:USDT"
            if hasattr(ex, "fetch_funding_rate"):
                fr = ex.fetch_funding_rate(swap_symbol)
                if fr and fr.get("fundingRate") is not None:
                    out["funding"] = float(fr["fundingRate"])
            if hasattr(ex, "fetch_open_interest"):
                oi = ex.fetch_open_interest(swap_symbol)
                if oi:
                    val = oi.get("openInterestValue") or oi.get("quoteVolume")
                    amount = oi.get("openInterestAmount")
                    if val is not None:
                        out["oi_value"] = float(val)
                    if amount is not None:
                        out["oi"] = float(amount)
            if np.isfinite(out["funding"]) or np.isfinite(out["oi_value"]):
                return out
        except Exception:
            continue
    return out


def fetch_ohlcv_fallback(symbol: str, timeframe: str = "1d",
                         limit: int = 700) -> tuple[pd.DataFrame, str]:
    errors = []
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            df = fetch_ohlcv(exchange_id, symbol, timeframe, limit)
            ok, _ = validate_ohlcv(df)
            if ok and len(df) >= 100:
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
        "btc": None, "ethbtc": None, "btcd": np.nan, "total3": np.nan,
        "label": "데이터 부족", "score": 50.0, "details": [],
        "global": {},
    }

    btc, _ = fetch_ohlcv_fallback("BTC/USDT", "1d", 300)
    if len(btc) >= 220:
        r = basic_regime_score(btc)
        result["btc"] = btc.iloc[-1]
        result["score"] = r["score"]
        result["label"] = r["label"]
        result["details"] = list(r["details"])
        btc_ret_30 = float(btc["Close"].iloc[-1] / btc["Close"].iloc[-31] - 1)
        result["btc_ret_30"] = btc_ret_30
    else:
        result["btc_ret_30"] = np.nan

    try:
        ethbtc, _ = fetch_ohlcv_fallback("ETH/BTC", "1d", 300)
        if len(ethbtc) >= 31:
            result["ethbtc"] = ethbtc.iloc[-1]
            result["ethbtc_ret_30"] = float(ethbtc["Close"].iloc[-1] / ethbtc["Close"].iloc[-31] - 1)
    except Exception:
        result["ethbtc_ret_30"] = np.nan

    glob = fetch_global_market_data()
    result["global"] = glob
    result["btcd"] = glob.get("btc_dominance", np.nan)
    result["total3"] = glob.get("total3", np.nan)

    # BTC dominance is interpreted as concentration context, not a standalone direction call.
    if np.isfinite(result["btcd"]):
        result["details"].append(f"BTC Dominance {result['btcd']:.1f}%")
        if result["score"] >= 50 and result["btcd"] < 55:
            result["score"] = float(np.clip(result["score"] + 4, 0, 100))
            result["details"].append("BTC 강세 + 상대적 알트 공간")
        elif result["score"] >= 50 and result["btcd"] > 65:
            result["score"] = float(np.clip(result["score"] - 4, 0, 100))
            result["details"].append("BTC 강세 + BTC 집중도 높음")

    if np.isfinite(result.get("ethbtc_ret_30", np.nan)):
        result["details"].append(f"ETH/BTC 30D {result['ethbtc_ret_30']*100:+.1f}%")
        if result["ethbtc_ret_30"] > 0.03:
            result["score"] = float(np.clip(result["score"] + 3, 0, 100))
        elif result["ethbtc_ret_30"] < -0.03:
            result["score"] = float(np.clip(result["score"] - 3, 0, 100))

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
    derivatives = 0

    funding = row.get("FUNDING", np.nan)
    if np.isfinite(funding):
        # Use genuinely directional funding extremes; ordinary funding near
        # zero should not receive a directional bonus.
        if long and funding <= -0.0005:
            derivatives += 5
        if not long and funding >= 0.0005:
            derivatives += 5

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
        "derivatives": derivatives,
    }


def generate_signal(df: pd.DataFrame, direction: str,
                    tp_atr: float = 2.0, sl_atr: float = 1.0,
                    market_regime: Optional[dict] = None) -> Optional[dict]:
    if len(df) < 220:
        return None

    r = df.iloc[-1]
    required = [
        "EMA20", "EMA50", "EMA200", "RSI14", "MACD_HIST", "ROC14",
        "ADX14", "ATR14", "ATR_PCT", "REL_VOLUME", "VOL_Z",
        "SWING_HIGH", "SWING_LOW", "POC_Price"
    ]
    if any(pd.isna(r.get(c)) for c in required):
        return None

    components = signal_components(r, direction)
    raw = sum(components.values()) / 85 * 100  # 20+20+15+15+10+5 = 85

    # Market-quality adjustments.
    if r["ADX14"] < 15:
        raw -= 10
    if r["ATR_PCT"] > 8:
        raw -= 8

    score = float(np.clip(raw, 0, 100))
    close = float(r["Close"])
    atr = float(r["ATR14"])

    if direction == "LONG":
        entry = close
        tp = entry + tp_atr * atr
        sl = entry - sl_atr * atr
    else:
        entry = close
        tp = entry - tp_atr * atr
        sl = entry + sl_atr * atr

    rr = abs(tp - entry) / max(abs(entry - sl), 1e-12)

    return {
        "direction": direction,
        "score": score,
        "entry_signal_close": entry,
        "tp": tp,
        "sl": sl,
        "rr": rr,
        "adx": float(r["ADX14"]),
        "rsi": float(r["RSI14"]),
        "atr_pct": float(r["ATR_PCT"]),
        "poc": float(r["POC_Price"]),
        **components,
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
    """Cost-adjusted, non-overlapping trade simulation.

    A signal is generated on a confirmed close and entered at the next open.
    While a trade is open, later signals are ignored. This prevents accidental
    overlapping positions from overstating backtest performance.
    """
    trades = []
    next_free_idx = -1

    for idx, sig_dir in sorted(signals, key=lambda x: x[0]):
        if sig_dir != direction or idx < next_free_idx:
            continue
        if idx >= len(df) - 2:
            continue

        entry_idx = idx + 1
        entry = float(df["Open"].iloc[entry_idx])
        atr = float(df["ATR14"].iloc[idx])
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry) or entry <= 0:
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
        # Exactly max_holding_bars candles are eligible after entry.
        end = min(len(df), entry_idx + cfg.max_holding_bars)
        if end <= entry_idx:
            continue

        for j in range(entry_idx, end):
            open_j = float(df["Open"].iloc[j])
            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])

            # Gap-through handling: if the market opens beyond a stop/target,
            # execution occurs at the actual open, not at the stale trigger.
            if direction == "LONG":
                if open_j <= sl:
                    exit_price, exit_idx, outcome = open_j, j, "SL_GAP"
                    break
                if open_j >= tp:
                    exit_price, exit_idx, outcome = open_j, j, "TP_GAP"
                    break
                hit_tp, hit_sl = high >= tp, low <= sl
            else:
                if open_j >= sl:
                    exit_price, exit_idx, outcome = open_j, j, "SL_GAP"
                    break
                if open_j <= tp:
                    exit_price, exit_idx, outcome = open_j, j, "TP_GAP"
                    break
                hit_tp, hit_sl = low <= tp, high >= sl

            if hit_tp and hit_sl:
                # Without intrabar tick/path data, assume the adverse level
                # was reached first. This is deliberately conservative.
                exit_price, exit_idx, outcome = sl, j, "SL_AMBIGUOUS"
                break
            if hit_sl:
                exit_price, exit_idx, outcome = sl, j, "SL"
                break
            if hit_tp:
                exit_price, exit_idx, outcome = tp, j, "TP"
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
            "signal_time": df["timestamp"].iloc[idx] if "timestamp" in df.columns else pd.NaT,
            "entry_time": df["timestamp"].iloc[entry_idx] if "timestamp" in df.columns else pd.NaT,
            "exit_time": df["timestamp"].iloc[exit_idx] if "timestamp" in df.columns else pd.NaT,
            "entry": entry, "exit": exit_price, "return": net, "outcome": outcome,
        })
        next_free_idx = exit_idx + 1

    if not trades:
        return empty_metrics()

    tr = pd.DataFrame(trades)
    equity = (1 + tr["return"]).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())

    wins = tr.loc[tr["return"] > 0, "return"]
    losses = tr.loc[tr["return"] < 0, "return"]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    std = float(tr["return"].std(ddof=1)) if len(tr) > 1 else np.nan
    sharpe = float(tr["return"].mean() / std) if std > 0 and np.isfinite(std) else np.nan
    downside = tr.loc[tr["return"] < 0, "return"]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = float(tr["return"].mean() / downside_std) if downside_std > 0 and np.isfinite(downside_std) else np.nan
    calmar = float(total_return / abs(max_dd)) if max_dd < 0 else np.nan
    tm = time_based_metrics(tr, df)
    mc = monte_carlo_bootstrap(tr["return"], cfg.monte_carlo_runs)

    return {
        "total_return": total_return, "win_rate": float((tr["return"] > 0).mean() * 100),
        "profit_factor": float(pf), "sharpe": sharpe,
        "time_sharpe": tm["time_sharpe"], "sortino": sortino,
        "time_sortino": tm["time_sortino"], "calmar": calmar,
        "recovery_factor": calmar, "max_drawdown": max_dd,
        "trades": int(len(tr)), "avg_trade": float(tr["return"].mean()),
        **mc, "trades_df": tr,
    }

def empty_metrics() -> dict:
    return {
        "total_return": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe": np.nan,
        "time_sharpe": np.nan,
        "sortino": np.nan,
        "time_sortino": np.nan,
        "calmar": np.nan,
        "recovery_factor": np.nan,
        "max_drawdown": 0.0,
        "trades": 0,
        "avg_trade": 0.0,
        "mc_p05": np.nan, "mc_median": np.nan, "mc_p95": np.nan, "mc_mdd_q05": np.nan,
        "trades_df": pd.DataFrame(),
    }


def time_based_metrics(trades_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Time-based risk metrics on the actual chronological candle grid.

    Realized P&L is booked on exit timestamps. This is still not a full
    mark-to-market equity curve, but it preserves the true calendar order and
    avoids treating each trade as an equal time period.
    """
    if trades_df.empty or df.empty:
        return {"time_sharpe": np.nan, "time_sortino": np.nan}
    if "timestamp" not in df.columns or "exit_time" not in trades_df.columns:
        return {"time_sharpe": np.nan, "time_sortino": np.nan}
    idx = pd.DatetimeIndex(df["timestamp"])
    # Convert realized trade returns into equity-relative returns at the
    # actual exit candle. This preserves compounding instead of simply adding
    # independent trade returns.
    series = pd.Series(0.0, index=idx)
    equity = 1.0
    for _, t in trades_df.sort_values("exit_time").iterrows():
        ts = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
        if pd.notna(ts) and ts in series.index:
            r = float(t["return"])
            series.loc[ts] += equity * r
            equity *= (1.0 + r)
    # Normalize P&L increments by the starting equity. Non-trade candles remain
    # zero, so the series is a chronological realized-return stream.
    arr = series.to_numpy(float)
    if len(arr) < 2 or np.std(arr, ddof=1) <= 0:
        return {"time_sharpe": np.nan, "time_sortino": np.nan}
    ann = math.sqrt(365.0)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    downside = arr[arr < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else np.nan
    return {
        "time_sharpe": float(mean / std * ann),
        "time_sortino": float(mean / dstd * ann) if dstd > 0 and np.isfinite(dstd) else np.nan,
    }


def monte_carlo_bootstrap(returns: pd.Series, runs: int = 500, seed: int = 42) -> dict:
    """Bootstrap trade returns to quantify path uncertainty, not prediction."""
    arr = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(float)
    if len(arr) < 5:
        return {"mc_p05": np.nan, "mc_median": np.nan, "mc_p95": np.nan, "mc_mdd_q05": np.nan}
    rng = np.random.default_rng(seed)
    final_returns = np.empty(runs, dtype=float)
    mdds = np.empty(runs, dtype=float)
    for k in range(runs):
        sample = rng.choice(arr, size=len(arr), replace=True)
        eq = np.cumprod(1.0 + sample)
        final_returns[k] = eq[-1] - 1.0
        peak = np.maximum.accumulate(eq)
        mdds[k] = np.min(eq / peak - 1.0)
    return {
        "mc_p05": float(np.quantile(final_returns, 0.05)),
        "mc_median": float(np.quantile(final_returns, 0.50)),
        "mc_p95": float(np.quantile(final_returns, 0.95)),
        "mc_mdd_q05": float(np.quantile(mdds, 0.05)),
    }


# ============================================================
# 6. PATTERN SIGNALS
# ============================================================

def generate_breakout_signals(df: pd.DataFrame) -> list[tuple[int, str]]:
    signals = []
    for i in range(0, len(df) - 1):
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

def optimize_parameters(train: pd.DataFrame, direction: str,
                         cfg: BacktestConfig) -> Optional[dict]:
    signals = generate_breakout_signals(train)
    if not signals:
        return None

    best = None

    for tp_atr in [1.5, 2.0, 2.5]:
        for sl_atr in [0.75, 1.0, 1.25]:
            m = backtest_signals(
                train, signals, direction, tp_atr, sl_atr, cfg
            )
            if m["trades"] < cfg.min_train_trades:
                continue

            # Penalize unstable/high-drawdown parameter sets.
            score = (
                m["total_return"] * 100
                + min(m["profit_factor"], 5) * 5
                + (m["sortino"] if np.isfinite(m["sortino"]) else 0) * 5
                + m["max_drawdown"] * 20
            )

            candidate = {
                "score": score,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "metrics": m,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def walk_forward(df: pd.DataFrame, direction: str,
                 cfg: BacktestConfig) -> dict:
    if len(df) < cfg.train_bars + cfg.test_bars:
        return {"windows": [], **empty_metrics()}

    all_trades = []
    windows = []

    start = 0

    while start + cfg.train_bars + cfg.test_bars <= len(df):
        train_end = start + cfg.train_bars
        test_end = train_end + cfg.test_bars

        train = df.iloc[start:train_end].copy()
        test = df.iloc[train_end:test_end].copy()

        best = optimize_parameters(train, direction, cfg)

        if best is None:
            start += cfg.step_bars
            continue

        test_signals = generate_breakout_signals(test)
        oos = backtest_signals(
            test,
            test_signals,
            direction,
            best["tp_atr"],
            best["sl_atr"],
            cfg,
        )

        if not oos["trades_df"].empty:
            tr = oos["trades_df"].copy()
            all_trades.append(tr)

        windows.append({
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "tp_atr": best["tp_atr"],
            "sl_atr": best["sl_atr"],
            "oos_return": oos["total_return"],
            "oos_win_rate": oos["win_rate"],
            "oos_trades": oos["trades"],
            "oos_mdd": oos["max_drawdown"],
        })

        start += cfg.step_bars

    if not all_trades:
        return {"windows": windows, **empty_metrics()}

    tr = pd.concat(all_trades, ignore_index=True)
    equity = (1 + tr["return"]).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())

    wins = tr.loc[tr["return"] > 0, "return"]
    losses = tr.loc[tr["return"] < 0, "return"]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.inf
    std = float(tr["return"].std(ddof=1)) if len(tr) > 1 else np.nan
    sharpe = float(tr["return"].mean() / std) if std and np.isfinite(std) else np.nan
    downside = tr.loc[tr["return"] < 0, "return"]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = float(tr["return"].mean() / downside_std) if downside_std and np.isfinite(downside_std) else np.nan
    calmar = float(total_return / abs(max_dd)) if max_dd < 0 else np.nan
    tm = time_based_metrics(tr, df)
    mc = monte_carlo_bootstrap(tr["return"], cfg.monte_carlo_runs)
    if windows:
        counts = pd.Series([(w["tp_atr"], w["sl_atr"]) for w in windows]).value_counts()
        param_stability = float(counts.iloc[0] / len(windows))
    else:
        param_stability = np.nan

    return {
        "windows": windows,
        "total_return": total_return,
        "win_rate": float((tr["return"] > 0).mean() * 100),
        "profit_factor": pf,
        "sharpe": sharpe,
        "time_sharpe": tm["time_sharpe"],
        "sortino": sortino,
        "time_sortino": tm["time_sortino"],
        "calmar": calmar,
        "recovery_factor": calmar,
        "param_stability": param_stability,
        **mc,
        "max_drawdown": max_dd,
        "trades": int(len(tr)),
        "avg_trade": float(tr["return"].mean()),
        "trades_df": tr,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_mtf_confirmation(symbol: str) -> dict:
    """4H confirmation used only for the current signal, not historical WFO."""
    df4, _ = fetch_ohlcv_fallback(symbol, "4h", 260)
    if len(df4) < 220:
        return {"score": 50.0, "label": "MTF 부족"}
    r = df4.iloc[-1]
    score = 50.0
    if r["Close"] > r["EMA20"] > r["EMA50"]:
        score += 25
    elif r["Close"] < r["EMA20"] < r["EMA50"]:
        score -= 25
    if r["ADX14"] >= 20:
        score += 10 if r["Close"] > r["EMA20"] else -10
    return {"score": float(np.clip(score, 0, 100)), "label": "상승" if score >= 65 else ("하락" if score <= 35 else "혼조")}


# ============================================================
# 8. COIN ANALYSIS
# ============================================================

def analyze_symbol(symbol: str, cfg: BacktestConfig, market_regime: Optional[dict] = None,
                   enrich: bool = True) -> Optional[dict]:
    try:
        df, exchange_id = fetch_ohlcv_fallback(symbol, "1d", cfg.data_limit)
        if len(df) < max(260, cfg.train_bars + cfg.test_bars):
            return None

        long_wf = walk_forward(df, "LONG", cfg)
        short_wf = walk_forward(df, "SHORT", cfg)

        latest = df.iloc[-1]
        btc_ret_30 = float(market_regime.get("btc_ret_30", np.nan)) if market_regime else np.nan
        coin_ret_30 = float(latest["Close"] / df["Close"].iloc[-31] - 1)
        relative_strength_30 = coin_ret_30 - btc_ret_30 if np.isfinite(btc_ret_30) else np.nan
        deriv = fetch_derivatives_snapshot(symbol) if enrich else {"funding": np.nan, "oi_value": np.nan}
        mtf = fetch_mtf_confirmation(symbol) if (enrich and cfg.use_mtf) else {"score": 50.0, "label": "OFF"}
        df = df.copy()
        df["FUNDING"] = deriv.get("funding", np.nan)

        candidates = []
        for direction, wf in [("LONG", long_wf), ("SHORT", short_wf)]:
            if wf["trades"] < cfg.min_oos_trades:
                continue

            sig = generate_signal(df, direction, market_regime=market_regime)
            if sig is None:
                continue

            # OOS-focused score. MDD is a penalty, not a standalone ranking.
            pf_component = min(max(wf["profit_factor"], 0), 4) / 4 * 25
            ret_component = np.clip(wf["total_return"] * 100, -25, 25)
            win_component = np.clip(wf["win_rate"] - 40, 0, 40) / 40 * 20
            stability = (
                np.mean([w["oos_return"] > 0 for w in wf["windows"]])
                if wf["windows"] else 0
            )
            dd_penalty = min(abs(wf["max_drawdown"]) * 100, 30)
            rs_component = float(np.clip(relative_strength_30 * 100, -15, 15))
            funding = deriv.get("funding", np.nan)
            funding_component = 0.0
            if np.isfinite(funding):
                # Extreme positive funding penalizes LONG; extreme negative funding penalizes SHORT.
                funding_component = float(np.clip((-funding if direction == "LONG" else funding) * 10000, -5, 5))

                # MTF must be direction-aligned: a bullish 4H state helps LONG
            # but should penalize SHORT, and vice versa.
            mtf_alignment = (mtf["score"] - 50.0) if direction == "LONG" else (50.0 - mtf["score"])

            score = (
                sig["score"] * 0.45
                + pf_component * 0.20
                + ret_component * 0.10
                + win_component * 0.10
                + stability * 10
                - dd_penalty * 0.15
                + rs_component * 0.12
                + funding_component * 0.8
                + mtf_alignment * 0.10
                + (wf.get("param_stability", 0.0) * 5.0)
            )
            score = float(np.clip(score, 0, 100))
            # Keep a direction candidate for diagnostics. The final decision
            # layer will turn low scores into NEUTRAL instead of silently
            # deleting the coin from the result set.
            candidates.append({
                "direction": direction,
                "score": score,
                "signal": sig,
                "wf": wf,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        score_gap = best["score"] - second["score"] if second else best["score"]

        # Keep an explicit NEUTRAL decision when evidence is insufficient or
        # LONG/SHORT evidence is too close. This prevents the UI from silently
        # converting uncertainty into a directional trade idea.
        neutral_reason = None
        if best["score"] < cfg.min_direction_score:
            neutral_reason = "방향성 점수 미달"
        elif second is not None and score_gap < cfg.neutral_score_gap:
            neutral_reason = "LONG/SHORT 점수 차이 부족"

        if neutral_reason is not None:
            return {
                "symbol": symbol,
                "exchange": exchange_id,
                "price": float(latest["Close"]),
                "change_30d": float((latest["Close"] / df["Close"].iloc[-31] - 1) * 100),
                "direction": "NEUTRAL",
                "score": float(best["score"]),
                "score_gap": float(score_gap),
                "decision_reason": neutral_reason,
                "signal": best["signal"],
                "factor_positive": [],
                "factor_negative": [],
                "wf": best["wf"],
                "rsi": float(latest["RSI14"]),
                "adx": float(latest["ADX14"]),
                "atr_pct": float(latest["ATR_PCT"]),
                "poc": float(latest["POC_Price"]),
                "rel_volume": float(latest["REL_VOLUME"]),
                "relative_strength_30": relative_strength_30,
                "funding": deriv.get("funding", np.nan),
                "oi_value": deriv.get("oi_value", np.nan),
                "mtf_score": mtf["score"],
                "mtf_label": mtf["label"],
                "oos_time_sharpe": best["wf"].get("time_sharpe", np.nan),
                "param_stability": best["wf"].get("param_stability", np.nan),
                "mc_p05": best["wf"].get("mc_p05", np.nan),
                "mc_median": best["wf"].get("mc_median", np.nan),
                "mc_p95": best["wf"].get("mc_p95", np.nan),
                "mc_mdd_q05": best["wf"].get("mc_mdd_q05", np.nan),
            }

        # Explain the selected direction using auditable factor contributions.
        sig = best["signal"]
        comp = {k: float(v) for k, v in sig.items() if k in {"trend", "momentum", "structure", "volume", "volatility", "derivatives"}}
        positive = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)[:3]
        negative = sorted(comp.items(), key=lambda kv: kv[1])[:2]

        return {
            "symbol": symbol,
            "exchange": exchange_id,
            "price": float(latest["Close"]),
            "change_30d": float(
                (latest["Close"] / df["Close"].iloc[-31] - 1) * 100
            ),
            "direction": best["direction"],
            "score": best["score"],
            "decision_reason": "LONG/SHORT 점수차 및 최소 점수 기준 충족",
            "score_gap": float(score_gap),
            "signal": best["signal"],
            "factor_positive": positive,
            "factor_negative": negative,
            "wf": best["wf"],
            "rsi": float(latest["RSI14"]),
            "adx": float(latest["ADX14"]),
            "atr_pct": float(latest["ATR_PCT"]),
            "poc": float(latest["POC_Price"]),
            "rel_volume": float(latest["REL_VOLUME"]),
            "relative_strength_30": relative_strength_30,
            "funding": deriv.get("funding", np.nan),
            "oi_value": deriv.get("oi_value", np.nan),
            "mtf_score": mtf["score"],
            "mtf_label": mtf["label"],
            "oos_time_sharpe": best["wf"].get("time_sharpe", np.nan),
            "param_stability": best["wf"].get("param_stability", np.nan),
            "mc_p05": best["wf"].get("mc_p05", np.nan),
            "mc_median": best["wf"].get("mc_median", np.nan),
            "mc_p95": best["wf"].get("mc_p95", np.nan),
            "mc_mdd_q05": best["wf"].get("mc_mdd_q05", np.nan),
        }
    except Exception:
        return None



def fast_screen(symbol: str, market_regime: Optional[dict] = None, min_score: float = 42.0) -> Optional[dict]:
    """Cheap current-state screen. No WFO, derivatives, MTF or Monte Carlo."""
    try:
        df, exchange_id = fetch_ohlcv_fallback(symbol, "1d", 260)
        if len(df) < 220:
            return None
        r = df.iloc[-1]
        if any(pd.isna(r.get(c)) for c in ["EMA20","EMA50","EMA200","RSI14","MACD_HIST","ROC14","ADX14","ATR14","ATR_PCT","REL_VOLUME","VOL_Z","POC_Price"]):
            return None
        scores = {}
        for d in ("LONG", "SHORT"):
            sig = generate_signal(df, d, market_regime=market_regime)
            scores[d] = sig["score"] if sig else 0.0
        best_dir = max(scores, key=scores.get)
        best_score = float(scores[best_dir])
        if best_score < min_score:
            return None
        return {"symbol": symbol, "exchange": exchange_id, "screen_score": best_score, "direction": best_dir}
    except Exception:
        return None

def run_parallel(symbols: list[str], cfg: BacktestConfig,
                 workers: int = 6, market_regime: Optional[dict] = None,
                 enrich: bool = True) -> pd.DataFrame:
    rows = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_symbol, s, cfg, market_regime, enrich) for s in symbols]
        for f in concurrent.futures.as_completed(futures):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception:
                errors += 1

    # Keep the UI resilient; expose count for diagnostics instead of leaking API errors.

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


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

    st.subheader("🌐 시장 레짐")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("BTC", fmt_price(float(regime["btc"]["Close"])) if regime["btc"] is not None else "-")
    with c2:
        st.metric("BTC 레짐", regime["label"])
    with c3:
        st.metric("시장 점수", f"{regime['score']:.0f}/100")
    with c4:
        btcd = regime.get("btcd", np.nan)
        st.metric("BTC Dominance", f"{btcd:.1f}%" if np.isfinite(btcd) else "-")

    if regime["details"]:
        st.caption(" · ".join(regime["details"]))

    return regime


def render_mobile_card(row: pd.Series):
    direction = row["direction"]
    icon = "🟢" if direction == "LONG" else ("🔴" if direction == "SHORT" else "⚪")

    with st.container(border=True):
        st.markdown(f"### {icon} {row['symbol']} · {direction}")
        a, b, c = st.columns(3)
        a.metric("QUANT", f"{row['score']:.0f}")
        b.metric("현재가", fmt_price(row["price"]))
        c.metric("RR", f"{row['signal']['rr']:.2f}" if direction != "NEUTRAL" else "-")

        a, b, c = st.columns(3)
        a.metric("TP", fmt_price(row["signal"]["tp"]) if direction != "NEUTRAL" else "-")
        b.metric("SL", fmt_price(row["signal"]["sl"]) if direction != "NEUTRAL" else "-")
        c.metric("ADX", f"{row['adx']:.1f}")

        rs = row.get("relative_strength_30", np.nan)
        funding = row.get("funding", np.nan)
        rs_text = f"RS(BTC) {rs*100:+.2f}%" if np.isfinite(rs) else "RS(BTC) -"
        fund_text = f"Funding {funding*100:.4f}%" if np.isfinite(funding) else "Funding -"
        mtf_text = f"4H {row.get('mtf_label', '-') }"
        mc = row.get("mc_median", np.nan)
        mc_text = f"MC50 {mc*100:+.1f}%" if np.isfinite(mc) else "MC50 -"
        gap_text = f"Gap {row.get('score_gap', np.nan):.1f}" if np.isfinite(row.get('score_gap', np.nan)) else "Gap -"
        reason_text = row.get("decision_reason", "")
        st.caption(
            f"RSI {row['rsi']:.1f} · ATR {row['atr_pct']:.2f}% · "
            f"POC {fmt_price(row['poc'])} · RVOL {row['rel_volume']:.2f} · {rs_text} · {fund_text} · {mtf_text} · {mc_text} · {gap_text}"
        )
        if reason_text:
            st.caption(f"판정: {reason_text}")
        pos = row.get("factor_positive", [])
        neg = row.get("factor_negative", [])
        if pos:
            st.caption("근거: " + ", ".join(f"{k} {v:+.0f}" for k, v in pos))
        if neg:
            st.caption("주의: " + ", ".join(f"{k} {v:+.0f}" for k, v in neg))


def render_results(df: pd.DataFrame):
    if df.empty:
        st.warning("조건을 만족하는 검증 결과가 없습니다.")
        return

    display = df.copy()
    display["score"] = display["score"].round(1)
    display["price"] = display["price"].map(fmt_price)
    display["30D"] = display["change_30d"].round(2)
    display["OOS"] = (display["wf"].apply(lambda x: x["total_return"]) * 100).round(2)
    display["Win"] = display["wf"].apply(lambda x: x["win_rate"]).round(1)
    display["PF"] = display["wf"].apply(lambda x: x["profit_factor"]).round(2)
    display["MDD"] = (display["wf"].apply(lambda x: x["max_drawdown"]) * 100).round(2)
    display["Trades"] = display["wf"].apply(lambda x: x["trades"])
    display["TSharpe"] = display["wf"].apply(lambda x: x.get("time_sharpe", np.nan)).round(2)
    display["Stability"] = (display["wf"].apply(lambda x: x.get("param_stability", np.nan)) * 100).round(0)
    display["MC50"] = (display["wf"].apply(lambda x: x.get("mc_median", np.nan)) * 100).round(1)
    display["Gap"] = display["score_gap"].round(1)
    display["RS30"] = (display["relative_strength_30"] * 100).round(2)
    display["Funding"] = (display["funding"] * 100).round(4)

    cols = ["symbol", "direction", "score", "Gap", "price", "30D", "RS30", "Funding", "OOS", "Win", "PF", "MDD", "TSharpe", "Stability", "MC50", "Trades"]
    st.dataframe(
        display[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "종목",
            "direction": "신호",
            "score": "QUANT",
            "Gap": "LONG/SHORT 점수차",
            "price": "현재가",
            "30D": "30D%",
            "RS30": "BTC 대비 RS%",
            "Funding": "Funding%",
            "OOS": "OOS%",
            "Win": "승률%",
            "PF": "PF",
            "MDD": "MDD%",
            "Trades": "거래수",
            "TSharpe": "시간기반 Sharpe",
            "Stability": "파라미터 안정성%",
            "MC50": "MC 중앙값%",
        },
    )

    st.markdown("### 📱 상세 신호")
    for _, row in df.head(10).iterrows():
        render_mobile_card(row)


def main():
    st.title("🔥 Crypto Quant Dashboard V2.5")
    st.caption("확정봉 · Look-ahead 방지 · Rolling WFO · 비용반영 · BTC.D · ETH/BTC · RS · Derivatives · MTF · Monte Carlo · 모바일 최적화")

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
            value=3,
        )
        fee = st.number_input("편도 수수료", 0.0, 0.01, 0.0005, format="%.4f")
        slippage = st.number_input("편도 슬리피지", 0.0, 0.01, 0.0005, format="%.4f")
        spread = st.number_input("스프레드", 0.0, 0.01, 0.0002, format="%.4f")
        min_signal_score = st.slider("최소 신호 점수", 40.0, 80.0, 50.0, 1.0)
        neutral_score_gap = st.slider(
            "LONG/SHORT 중립 간격", 2.0, 20.0, 8.0, 1.0,
            help="LONG과 SHORT 점수 차이가 이 값보다 작으면 NEUTRAL로 판정합니다.",
        )
        use_mtf = st.checkbox("4H 멀티타임프레임 확인", value=True)
        mc_runs = st.select_slider("Monte Carlo 반복", options=[200, 500, 1000], value=500)

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
    breadth = float((market["change_pct"] > 0).mean() * 100) if len(market) else np.nan
    st.metric("시장 상승 종목 비율", f"{breadth:.1f}%" if np.isfinite(breadth) else "-")
    if np.isfinite(breadth):
        regime["breadth"] = breadth
        if breadth >= 60:
            regime["score"] = float(np.clip(regime["score"] + 3, 0, 100))
        elif breadth <= 40:
            regime["score"] = float(np.clip(regime["score"] - 3, 0, 100))
    majors = market[market["base"].isin(TOP_MAJORS)]["symbol"].tolist()
    others = (
        market[~market["base"].isin(TOP_MAJORS)]
        .sort_values("quote_volume", ascending=False)["symbol"]
        .head(25)
        .tolist()
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("분석 가능 종목", len(market))
    with c2:
        st.metric("메이저", len(majors))
    with c3:
        st.metric("후보군", len(others))

    cfg = BacktestConfig(
        fee_rate=fee,
        slippage_rate=slippage,
        spread_rate=spread,
        min_oos_trades=int(min_oos_trades),
        min_signal_score=float(min_signal_score),
        neutral_score_gap=float(neutral_score_gap),
        min_direction_score=float(min_signal_score),
        monte_carlo_runs=int(mc_runs),
        use_mtf=bool(use_mtf),
    )

    st.subheader("🎯 분석 실행")
    st.caption("1단계 빠른 스크리닝 → 2단계 WFO 정밀검증 → 최종 후보에만 MTF/파생지표 적용")

    quick = st.button("⚡ 빠른 분석 — 메이저 + 상위 10", use_container_width=True)
    full = st.button("🔬 정밀 분석 — 상위 25 + 메이저", use_container_width=True)

    if quick or full:
        symbols = list(dict.fromkeys(majors + others[:10])) if quick else list(dict.fromkeys(majors + others))
        screen_floor = max(40.0, float(min_signal_score) - 12.0)
        with st.spinner(f"1단계: {len(symbols)}개 종목 빠른 스크리닝 중..."):
            screened = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                fs = [pool.submit(fast_screen, s, regime, screen_floor) for s in symbols]
                for f in concurrent.futures.as_completed(fs):
                    try:
                        r = f.result()
                        if r:
                            screened.append(r)
                    except Exception:
                        pass
            screened = sorted(screened, key=lambda x: x["screen_score"], reverse=True)

        # Never run expensive WFO on the entire universe. Keep a meaningful
        # candidate pool so a weak market can still return NEUTRAL diagnostics.
        max_candidates = 12 if quick else 18
        candidate_symbols = [x["symbol"] for x in screened[:max_candidates]]
        st.info(f"1단계 통과 {len(screened)}개 → 2단계 정밀검증 {len(candidate_symbols)}개")

        if candidate_symbols:
            # Fast mode deliberately disables expensive live enrichment.
            # Full mode enriches only the already screened candidates.
            cfg_run = BacktestConfig(
                fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, spread_rate=cfg.spread_rate,
                max_holding_bars=cfg.max_holding_bars, min_oos_trades=cfg.min_oos_trades,
                atr_window=cfg.atr_window, train_bars=cfg.train_bars, test_bars=cfg.test_bars,
                step_bars=cfg.step_bars, min_train_trades=cfg.min_train_trades,
                min_signal_score=cfg.min_signal_score, monte_carlo_runs=(200 if quick else cfg.monte_carlo_runs),
                use_mtf=(cfg.use_mtf if full else False), neutral_score_gap=cfg.neutral_score_gap,
                min_direction_score=cfg.min_direction_score, data_limit=cfg.data_limit
            )
            with st.spinner(f"2단계: {len(candidate_symbols)}개 종목 WFO 정밀검증 중..."):
                result = run_parallel(candidate_symbols, cfg_run, workers=6, market_regime=regime, enrich=not quick)
        else:
            result = pd.DataFrame()

        st.session_state["quant_results"] = result
        st.session_state["quant_screened"] = len(screened)
        st.session_state["quant_time"] = pd.Timestamp.now(tz="UTC")

    result = st.session_state.get("quant_results", pd.DataFrame())

    if not result.empty:
        st.success(f"분석 완료 · {len(result)}개 결과")
        render_results(result)
    elif st.session_state.get("quant_screened") is not None:
        st.warning(
            f"정밀검증 결과 방향성 후보가 없습니다. 1단계 스크리닝 통과 종목: "
            f"{st.session_state.get('quant_screened', 0)}개. "
            "시장 방향성이 약하거나 OOS 거래수가 부족한 경우 정상적으로 발생할 수 있습니다."
        )

        st.markdown("### 📈 점수 분포")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=result.head(15)["symbol"],
            y=result.head(15)["score"],
            text=result.head(15)["score"].round(0),
            textposition="auto",
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="QUANT SCORE",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True, key="score_chart_v2")

        st.caption(
            "주의: QUANT SCORE는 예측확률이 아니라 추세·모멘텀·구조·거래량·변동성·"
            "검증성과를 종합한 분석 점수입니다."
        )

    st.divider()
    st.caption("Data source: CCXT-supported exchanges · Analysis is informational, not financial advice.")


if __name__ == "__main__":
    main()
