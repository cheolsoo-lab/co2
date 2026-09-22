# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V9
- Optimized Noise Filters (Sweet Spot tuning to ensure aggressive signals appear)
- Physical Separation of LONG vs SHORT inside Aggressive & Defensive Tabs
- Main Screen Macro Regime Bar
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
    page_title="🔥 Crypto Quant Dashboard V9",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    spread_rate: float = 0.0002
    max_holding_bars: int = 15
    min_oos_trades: int = 5
    account_size: float = 10000.0
    risk_per_trade: float = 0.0075
    max_portfolio_risk: float = 0.02
    max_position_weight: float = 0.40


# ============================================================
# 1. DATA ACCESS & MACRO REGIME ANALYSIS
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
def fetch_tickers_with_fallback() -> tuple[pd.DataFrame, str]:
    for exchange_id in DEFAULT_EXCHANGES:
        try:
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
            df = pd.DataFrame(rows)
            if not df.empty:
                return df, exchange_id
        except Exception:
            continue
    return pd.DataFrame(), ""


@st.cache_data(ttl=30, show_spinner=False)
def fetch_funding_rate(exchange_id: str, symbol: str) -> float:
    try:
        ex = make_exchange(exchange_id)
        if hasattr(ex, 'fetch_funding_rate'):
            fr = ex.fetch_funding_rate(symbol)
            rate = fr.get('fundingRate')
            if rate is not None:
                return float(rate)
    except Exception:
        pass
    return 0.0


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
    if len(df) > 2:
        df = df.iloc[:-1].copy()

    return add_indicators(df)


def fetch_ohlcv_fallback(symbol: str, timeframe: str = "1d",
                         limit: int = 700) -> tuple[pd.DataFrame, str]:
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            df = fetch_ohlcv(exchange_id, symbol, timeframe, limit)
            if len(df) >= 100:
                return df, exchange_id
        except Exception:
            continue
    return pd.DataFrame(), ""


def render_macro_regime_bar(market_df: pd.DataFrame):
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0
    
    total_vol = market_df["quote_volume"].sum()
    majors_vol = market_df[market_df["base"].isin(["BTC", "ETH"])]["quote_volume"].sum()
    btc_dominance_proxy = (majors_vol / total_vol) * 100 if total_vol > 0 else 50.0

    if btc_change > 2.0:
        market_env = "🟢 Risk-On (BTC 주도 상승장)"
    elif btc_change < -2.0:
        market_env = "🔴 Risk-Off (시장 하락/방어 필요)"
    else:
        market_env = "🟡 횡보/눈치보기 장세 (선별적 접근)"

    st.markdown(
        f"**🌐 글로벌 매크로 레짐** | 상태: **{market_env}** | "
        f"BTC 24H 변동: **{btc_change:+.2f}%** | "
        f"메이저 집중도(도미넌스 프록시): **{btc_dominance_proxy:.1f}%**"
    )


# ============================================================
# 2. INDICATORS & POC
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["EMA20"] = ta.trend.EMAIndicator(x["Close"], window=20).ema_indicator()
    x["EMA50"] = ta.trend.EMAIndicator(x["Close"], window=50).ema_indicator()
    x["EMA200"] = ta.trend.EMAIndicator(x["Close"], window=200).ema_indicator()
    x["RSI14"] = ta.momentum.RSIIndicator(x["Close"], window=14).rsi()
    x["ROC14"] = ta.momentum.ROCIndicator(x["Close"], window=14).roc()

    macd = ta.trend.MACD(x["Close"])
    x["MACD_HIST"] = macd.macd_diff()
    x["ADX14"] = ta.trend.ADXIndicator(x["High"], x["Low"], x["Close"], window=14).adx()
    x["ATR14"] = ta.volatility.AverageTrueRange(x["High"], x["Low"], x["Close"], window=14).average_true_range()
    x["ATR_PCT"] = x["ATR14"] / x["Close"] * 100

    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["REL_VOLUME"] = x["Volume"] / x["VOL_MA20"]
    x["SWING_HIGH"] = x["High"].rolling(10, center=False).max().shift(1)
    x["SWING_LOW"] = x["Low"].rolling(10, center=False).min().shift(1)
    
    candle_range = x["High"] - x["Low"]
    body_size = (x["Close"] - x["Open"]).abs()
    x["BODY_RATIO"] = np.where(candle_range > 0, body_size / candle_range, 0.0)

    x["POC_Price"] = rolling_poc(x, window=60, bins=20)
    return x


def rolling_poc(df: pd.DataFrame, window: int = 60, bins: int = 20) -> pd.Series:
    values = np.full(len(df), np.nan)
    closes = df["Close"].to_numpy(float)
    volumes = df["Volume"].to_numpy(float)

    for i in range(window, len(df)):
        hist_close = closes[i - window:i]
        hist_vol = volumes[i - window:i]
        lo, hi = np.nanmin(hist_close), np.nanmax(hist_close)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(hist_close, edges) - 1, 0, bins - 1)
        vol_by_bin = np.bincount(idx, weights=hist_vol, minlength=bins)
        k = int(np.nanargmax(vol_by_bin))
        values[i] = (edges[k] + edges[k + 1]) / 2
    return pd.Series(values, index=df.index)


# ============================================================
# 3. HIERARCHICAL MTF & MARKET STATE
# ============================================================

def fetch_mtf_context(symbol: str) -> dict:
    result = {}
    for tf, limit in [("1d", 320), ("4h", 320), ("1h", 320)]:
        try:
            df, ex = fetch_ohlcv_fallback(symbol, tf, limit)
            if len(df) < 220:
                continue
            r = df.iloc[-1]
            result[tf] = {
                "close": float(r["Close"]), "ema20": float(r["EMA20"]),
                "ema50": float(r["EMA50"]), "ema200": float(r["EMA200"]),
                "rsi": float(r["RSI14"]), "adx": float(r["ADX14"])
            }
        except Exception:
            continue
    return result


def mtf_direction_score(mtf: dict, direction: str) -> tuple[float, list[str]]:
    if not mtf:
        return 50.0, ["MTF 데이터 부족"]
    long = direction == "LONG"
    score = 50.0
    details = []
    weights = {"1d": 0.60, "4h": 0.25, "1h": 0.15}
    for tf, w in weights.items():
        r = mtf.get(tf)
        if not r:
            continue
        local = 50.0
        if long:
            if r["close"] > r["ema20"] > r["ema50"]: local += 20
            if r["ema50"] > r["ema200"]: local += 15
            if tf == "1h" and r["rsi"] < 45: local += 10
        else:
            if r["close"] < r["ema20"] < r["ema50"]: local += 20
            if r["ema50"] < r["ema200"]: local += 15
            if tf == "1h" and r["rsi"] > 55: local += 10
        score += (local - 50) * w
        details.append(f"{tf} {'계층정렬' if local >= 55 else '불일치'}")
    return float(np.clip(score, 0, 100)), details


def classify_market_state(df: pd.DataFrame) -> dict:
    if len(df) < 220:
        return {"state": "UNKNOWN", "adx": 0.0, "rsi": 0.0}
    r = df.iloc[-1]
    close = float(r["Close"])
    up = close > r["EMA20"] > r["EMA50"] and r["EMA50"] > r["EMA200"]
    down = close < r["EMA20"] < r["EMA50"] and r["EMA50"] < r["EMA200"]
    adx = float(r["ADX14"])
    rsi = float(r["RSI14"])

    if up and adx >= 22: state = "TREND_UP"
    elif down and adx >= 22: state = "TREND_DOWN"
    elif adx < 17: state = "RANGE"
    else: state = "TRANSITION"
    return {"state": state, "adx": adx, "rsi": rsi}


# ============================================================
# 4. DYNAMIC ATR TP & BALANCED FILTERS
# ============================================================

def select_optimal_tp(df: pd.DataFrame, direction: str, entry: float, sl: float,
                      atr: float, strategy: str, market_state: str) -> dict:
    if market_state in {"TREND_UP", "TREND_DOWN"} and strategy == "TREND":
        atr_mults = [3.0, 4.0, 5.0, 6.0]
    else:
        atr_mults = [1.5, 2.0, 2.5]

    risk = abs(entry - sl)
    best_tp = entry + atr_mults[-1] * atr if direction == "LONG" else entry - atr_mults[-1] * atr
    best_rr = abs(best_tp - entry) / max(risk, 1e-12)
    return {"tp": float(best_tp), "rr": float(best_rr), "tp_source": "dynamic_atr"}


def generate_signal(df: pd.DataFrame, direction: str, mtf: dict, strategy: str) -> Optional[dict]:
    if len(df) < 220:
        return None
    r = df.iloc[-1]
    
    # ⚖️ [V9 황금 밸런스 필터] 맹목적 노이즈는 차단하되, 진입 기회는 확보
    rel_vol = float(r["REL_VOLUME"]) if pd.notna(r["REL_VOLUME"]) else 1.0
    body_ratio = float(r["BODY_RATIO"]) if pd.notna(r["BODY_RATIO"]) else 1.0
    
    if rel_vol < 1.15 or body_ratio < 0.35:
        return None  

    close = float(r["Close"]); atr = float(r["ATR14"])
    state = classify_market_state(df)
    mtf_score, _ = mtf_direction_score(mtf, direction)

    score = (r["ADX14"] * 0.4) + (mtf_score * 0.4) + (20 if strategy == "TREND" else 10)
    score = float(np.clip(score, 0, 100))

    sl_atr = 1.0
    if direction == "LONG":
        sl = min(entry := close, float(r["SWING_LOW"]) - 0.25 * atr) if pd.notna(r["SWING_LOW"]) else close - sl_atr * atr
    else:
        sl = max(entry := close, float(r["SWING_HIGH"]) + 0.25 * atr) if pd.notna(r["SWING_HIGH"]) else close + sl_atr * atr

    tp_info = select_optimal_tp(df, direction, entry, sl, atr, strategy, state["state"])
    
    return {
        "direction": direction, "strategy": strategy, "score": score,
        "entry_low": entry * 0.998, "entry_high": entry * 1.002,
        "tp": tp_info["tp"], "sl": float(sl), "rr": tp_info["rr"],
        "mtf_score": mtf_score, "market_state": state["state"]
    }


# ============================================================
# 5. SYMBOL ANALYSIS & DUAL MODE CATEGORIZATION
# ============================================================

def analyze_symbol(symbol: str, cfg: BacktestConfig) -> Optional[dict]:
    try:
        df, exchange_id = fetch_ohlcv_fallback(symbol, "1d", 500)
        if len(df) < 220:
            return None
        funding_rate = fetch_funding_rate(exchange_id, symbol)
        mtf = fetch_mtf_context(symbol)
        state = classify_market_state(df)

        strategy = "TREND" if state["state"] in {"TREND_UP", "TREND_DOWN"} else "REVERSAL"
        direction = "LONG" if state["state"] in {"TREND_UP", "EXHAUSTION_DOWN"} else "SHORT"
        
        sig = generate_signal(df, direction, mtf, strategy)
        if not sig:
            return None

        # ⚖️ 공격형 진입 기준 완화 (R:R 2.5 -> 2.0 이상이면 공격형 수용)
        mode_type = "AGGRESSIVE" if strategy == "TREND" and sig["rr"] >= 2.0 else "DEFENSIVE"

        return {
            "symbol": symbol, "exchange": exchange_id, "price": float(df["Close"].iloc[-1]),
            "direction": direction, "mode_type": mode_type, "signal": sig,
            "funding_rate": funding_rate, "score": sig["score"],
            "reason": f"레짐 {state['state']} · 전략 {strategy} · 펀딩비 {funding_rate*100:.4f}%"
        }
    except Exception:
        return None


# ============================================================
# 6. STREAMLIT UI
# ============================================================

def fmt_price(x):
    if x >= 1000: return f"{x:,.2f}"
    if x >= 1: return f"{x:,.4f}"
    return f"{x:,.8f}"


def main():
    st.title("🔥 Crypto Quant Dashboard V9")
    st.caption("글로벌 매크로 레짐 분석 탑재 · 공격형/방어형 황금 밸런스 튜닝 완료")

    with st.sidebar:
        st.header("⚙️ 설정")
        min_volume = st.number_input("최소 거래대금 (USDT)", 100_000.0, 50_000_000.0, 1_000_000.0, 100_000.0)
        account_size = st.number_input("계좌 금액 (USDT)", 100.0, 10_000_000.0, 10_000.0, 100.0)

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("모든 지원 거래소에서 시세를 불러오지 못했습니다. 네트워크 상태를 확인해주세요.")
        return
    else:
        st.caption(f"데이터 연동 성공 거래소: **{active_exchange.upper()}**")

    render_macro_regime_bar(market)
    st.divider()

    universe = market[market["quote_volume"] >= min_volume].sort_values("quote_volume", ascending=False).head(20)
    symbols = universe["symbol"].tolist()

    if st.button("🚀 밸런스 멀티모드 분석 실행", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(analyze_symbol, s, BacktestConfig(account_size=account_size)) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v9_results"] = results

    results = st.session_state.get("v9_results", [])
    if results:
        df_res = pd.DataFrame(results)
        
        tab1, tab2 = st.tabs(["🔥 [공격형] 대세 추종 알파 모드 (High RR)", "🛡️ [방어형] 숏컷 헌터 모드 (High Win-Rate)"])

        with tab1:
            st.markdown("### 🔥 공격형 추종 포지션 (황금 밸런스 적용)")
            agg_rows = df_res[df_res["mode_type"] == "AGGRESSIVE"]
            
            if agg_rows.empty:
                st.info("현재 조건을 만족하는 공격형 종목이 없습니다.")
            else:
                agg_longs = agg_rows[agg_rows["direction"] == "LONG"]
                agg_shorts = agg_rows[agg_rows["direction"] == "SHORT"]

                st.markdown("#### 🟢 LONG (매수) 추천 리스트")
                if agg_longs.empty:
                    st.caption("조건을 만족하는 롱 종목이 없습니다.")
                for _, row in agg_longs.iterrows():
                    sig = row["signal"]
                    st.markdown(
                        f"🔥 **{row['symbol']}** | 🟢 **LONG** | 🎯 **TP**: {fmt_price(sig['tp'])} | "
                        f"🛑 **SL**: {fmt_price(sig['sl'])} | ⚡ **진입**: {fmt_price(sig['entry_low'])}~{fmt_price(sig['entry_high'])} | "
                        f"점수: {row['score']:.1f} | R:R: 1:{sig['rr']:.2f}"
                    )

                st.markdown("---")
                st.markdown("#### 🔴 SHORT (매도) 추천 리스트")
                if agg_shorts.empty:
                    st.caption("조건을 만족하는 숏 종목이 없습니다.")
                for _, row in agg_shorts.iterrows():
                    sig = row["signal"]
                    st.markdown(
                        f"🔥 **{row['symbol']}** | 🔴 **SHORT** | 🎯 **TP**: {fmt_price(sig['tp'])} | "
                        f"🛑 **SL**: {fmt_price(sig['sl'])} | ⚡ **진입**: {fmt_price(sig['entry_low'])}~{fmt_price(sig['entry_high'])} | "
                        f"점수: {row['score']:.1f} | R:R: 1:{sig['rr']:.2f}"
                    )

        with tab2:
            st.markdown("### 🛡️ 방어형 숏컷 헌터 포지션")
            def_rows = df_res[df_res["mode_type"] == "DEFENSIVE"]
            
            if def_rows.empty:
                st.info("현재 방어적 국면에 부합하는 종목이 없습니다.")
            else:
                def_longs = def_rows[def_rows["direction"] == "LONG"]
                def_shorts = def_rows[def_rows["direction"] == "SHORT"]

                st.markdown("#### 🟢 LONG (매수) 추천 리스트")
                if def_longs.empty:
                    st.caption("조건을 만족하는 롱 종목이 없습니다.")
                for _, row in def_longs.iterrows():
                    sig = row["signal"]
                    st.markdown(
                        f"🛡️ **{row['symbol']}** | 🟢 **LONG** | 🎯 **TP**: {fmt_price(sig['tp'])} | "
                        f"🛑 **SL**: {fmt_price(sig['sl'])} | ⚡ **진입**: {fmt_price(sig['entry_low'])}~{fmt_price(sig['entry_high'])} | "
                        f"점수: {row['score']:.1f} | 펀딩비: {row['funding_rate']*100:.4f}%"
                    )

                st.markdown("---")
                st.markdown("#### 🔴 SHORT (매도) 추천 리스트")
                if def_shorts.empty:
                    st.caption("조건을 만족하는 숏 종목이 없습니다.")
                for _, row in def_shorts.iterrows():
                    sig = row["signal"]
                    st.markdown(
                        f"🛡️ **{row['symbol']}** | 🔴 **SHORT** | 🎯 **TP**: {fmt_price(sig['tp'])} | "
                        f"🛑 **SL**: {fmt_price(sig['sl'])} | ⚡ **진입**: {fmt_price(sig['entry_low'])}~{fmt_price(sig['entry_high'])} | "
                        f"점수: {row['score']:.1f} | 펀딩비: {row['funding_rate']*100:.4f}%"
                    )


if __name__ == "__main__":
    main()