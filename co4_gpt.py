# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V13
- Macro Horizon Decision Engine (Market-Wide Short-Term vs Long-Term Direction)
- Clean, Unified Dashboard Layout
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import ta

# ============================================================
# 0. APP CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="🔥 Crypto Quant Dashboard V13",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .macro-box {
        background: linear-gradient(135deg, #1e2530 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .badge-long { background-color: #0d9488; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-short { background-color: #e11d48; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]


# ============================================================
# 1. DATA ACCESS & MARKET-WIDE HORIZON ENGINE
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str = "1d", limit: int = 300) -> pd.DataFrame:
    ex = make_exchange(exchange_id)
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    if len(df) > 2:
        df = df.iloc[:-1].copy()
    
    # 지표 계산
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["ATR14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["REL_VOLUME"] = df["Volume"] / df["VOL_MA20"]
    return df


def fetch_ohlcv_fallback(symbol: str, timeframe: str = "1d", limit: int = 300) -> tuple[pd.DataFrame, str]:
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            df = fetch_ohlcv(exchange_id, symbol, timeframe, limit)
            if len(df) >= 100:
                return df, exchange_id
        except Exception:
            continue
    return pd.DataFrame(), ""


def analyze_market_wide_horizon(market_df: pd.DataFrame) -> dict:
    """
    전체 시장의 자금 흐름(BTC, 알트 점유율, 변동성)을 분석하여
    시장 전체가 '장기 홀딩' 장세인지 '단기 스캘핑/회전' 장세인지 판정합니다.
    """
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0
    
    total_vol = market_df["quote_volume"].sum()
    btc_vol = market_df[market_df["base"] == "BTC"]["quote_volume"].sum()
    eth_vol = market_df[market_df["base"] == "ETH"]["quote_volume"].sum()
    alt_vol = total_vol - (btc_vol + eth_vol)
    
    btc_dom = (btc_vol / total_vol) * 100 if total_vol > 0 else 50.0
    alt_share = (alt_vol / total_vol) * 100 if total_vol > 0 else 50.0
    avg_change = market_df["change_pct"].mean()

    # 시장 국면 및 전체 방향성 판정
    if btc_dom < 48.0 and alt_share > 45.0 and avg_change > 0.5:
        market_phase = "🚀 대세 알트 불장 (Altcoin Season)"
        horizon_strategy = "LONG_HOLD"
        market_advice = "시장 전체로 자금이 밀려 들어오는 **강력한 장기 홀딩(Long-term Hold) 장세**입니다. 중간에 잔파도에 흔들리지 말고 주요 코인들을 목표가(TP)까지 길게 끌고 가며 수익을 극대화하세요."
        portfolio = {"BTC": 10, "Major Alt": 30, "Small/Mid Alt": 50, "Cash": 10}
    elif btc_dom >= 52.0 and btc_change > 1.0:
        market_phase = "⚡ 비트코인 독주장 (BTC Dominance Rally)"
        horizon_strategy = "LONG_HOLD_BTC"
        market_advice = "오직 비트코인만 수급을 독식하고 있습니다. 알트코인은 장기 보유를 금지하고, **비트코인 위주로만 홀딩**하거나 알트는 짧게 치고 빠지는 **단기 순환매**로만 대응하세요."
        portfolio = {"BTC": 70, "Major Alt": 15, "Small/Mid Alt": 5, "Cash": 10}
    elif btc_change < -1.5 or avg_change < -1.0:
        market_phase = "🩸 리스크오프 및 현금 대피장 (Risk-Off)"
        horizon_strategy = "DEFENSIVE_CASH"
        market_advice = "시장 전체가 하락 압력을 받으며 테더(현금)로 자금이 대피 중입니다. 모든 장기 홀딩을 중단하고 **현금화 및 철저한 단기 방어(또는 숏)** 관점으로만 임하세요."
        portfolio = {"BTC": 0, "Major Alt": 0, "Small/Mid Alt": 0, "Cash": 100}
    else:
        market_phase = "⚖️ 방향성 없는 횡보/눈치보기 장세 (Range Bound)"
        horizon_strategy = "SHORT_ROTATE"
        market_advice = "주도 세력이 부재하여 지루한 박스권이 이어지고 있습니다. 장기 홀딩은 시간 손실을 유발하므로, 목표가 도달 시 **즉시 익절하고 새로운 수급 종목으로 갈아타는 단기 순환매(Short-term Rotation)** 전략이 필수적입니다."
        portfolio = {"BTC": 30, "Major Alt": 30, "Small/Mid Alt": 10, "Cash": 30}

    return {
        "market_phase": market_phase,
        "horizon_strategy": horizon_strategy,
        "market_advice": market_advice,
        "btc_change": btc_change,
        "btc_dom": btc_dom,
        "alt_share": alt_share,
        "portfolio": portfolio
    }


def render_market_horizon_dashboard(market_df: pd.DataFrame):
    m = analyze_market_wide_horizon(market_df)
    
    st.markdown(f"""
    <div class="macro-box">
        <h2 style="margin-top:0; color:#38bdf8;">🌐 시장 전체 매크로 방향성 & 홀딩 가이드</h2>
        <hr style="border-color: #374151;">
        <h3 style="color: #f43f5e; margin-bottom: 10px;">현재 판정: {m['market_phase']}</h3>
        <p style="font-size: 16px; line-height: 1.6; color: #e2e8f0;">
            {m['market_advice']}
        </p>
        <hr style="border-color: #374151; margin: 15px 0;">
        <div style="display: flex; gap: 20px; font-size: 14px; color: #94a3b8;">
            <div>₿ BTC 24H: <b>{m['btc_change']:+.2f}%</b></div>
            <div>📊 BTC 도미넌스: <b>{m['btc_dom']:.1f}%</b></div>
            <div>🚀 알트 자금 점유율: <b>{m['alt_share']:.1f}%</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    p = m["portfolio"]
    st.markdown(f"💰 **권장 자금 배분:** BTC `{p['BTC']}%` | 메이저알트 `{p['Major Alt']}%` | 중소알트 `{p['Small/Mid Alt']}%` | 현금 `{p['Cash']}%`")
    st.divider()
    return m["horizon_strategy"]


# ============================================================
# 2. INDIVIDUAL COIN ANALYSIS (UNIFIED)
# ============================================================

def analyze_symbol_unified(symbol: str, horizon_strategy: str) -> Optional[dict]:
    try:
        df, ex_id = fetch_ohlcv_fallback(symbol, "1d", 200)
        if len(df) < 100:
            return None

        r = df.iloc[-1]
        close = float(r["Close"])
        atr = float(r["ATR14"])
        rsi = float(r["RSI14"])
        rel_vol = float(r["REL_VOLUME"]) if pd.notna(r["REL_VOLUME"]) else 1.0

        # 시장 전체 전략에 따른 매매 방향 결정
        if horizon_strategy in {"LONG_HOLD", "LONG_HOLD_BTC"}:
            direction = "LONG"
            tp = close + (4.0 * atr)  # 불장에서는 목표가를 멀게 잡고 장기 홀딩 유도
            sl = close - (1.5 * atr)
        elif horizon_strategy == "SHORT_ROTATE":
            direction = "LONG" if close > r["EMA20"] else "SHORT"
            tp = close + (2.0 * atr)  # 횡보장에서는 짧게 먹고 회전
            sl = close - (1.0 * atr)
        else:
            direction = "SHORT"
            tp = close - (2.0 * atr)
            sl = close + (1.0 * atr)

        score = float(np.clip(rel_vol * 15 + (50 if direction == "LONG" else 30), 0, 100))

        return {
            "symbol": symbol,
            "exchange": ex_id.upper(),
            "price": close,
            "direction": direction,
            "tp": tp,
            "sl": sl,
            "rsi": rsi,
            "score": score
        }
    except Exception:
        return None


# ============================================================
# 3. STREAMLIT UI RENDER
# ============================================================

def fmt_price(x):
    if x >= 1000: return f"${x:,.2f}"
    if x >= 1: return f"${x:,.4f}"
    return f"${x:,.8f}"


def main():
    st.title("🔥 Crypto Quant Dashboard V13")
    st.caption("시장 전체 거시 방향성 기반 [장기 홀딩 vs 단기 순환매] 통합 의사결정 시스템")

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("거래소 시세를 불러오지 못했습니다.")
        return

    # 1. 시장 전체 거시 방향성 및 홀딩 가이드 출력
    horizon_strategy = render_market_horizon_dashboard(market)

    # 2. 주요 종목 스캔
    universe = market[market["quote_volume"] >= 2_000_000].sort_values("quote_volume", ascending=False).head(15)
    symbols = universe["symbol"].tolist()

    if st.button("🔍 시장 맞춤형 추천 종목 스캔 실행", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(analyze_symbol_unified, s, horizon_strategy) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v13_results"] = results

    results = st.session_state.get("v13_results", [])
    if results:
        df_res = pd.DataFrame(results).sort_values("score", ascending=False)

        st.markdown("### 📋 시장 국면 맞춤형 추천 종목 리스트")
        st.caption("💡 현재 시장 전체 방향성에 맞춰 산출된 목표가(TP)와 손절가(SL)입니다.")

        for _, row in df_res.iterrows():
            badge = "badge-long" if row["direction"] == "LONG" else "badge-short"
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="{badge}">{row['direction']}</span> &nbsp;
                        <b style="font-size: 16px;">{row['symbol']}</b> 
                        <span style="font-size: 12px; color: #94a3b8;">({row['exchange']})</span>
                    </div>
                    <div>
                        <b>현재가:</b> {fmt_price(row['price'])} &nbsp;|&nbsp; <b>점수:</b> {row['score']:.1f}점
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 13px; color: #cbd5e1;">
                    🎯 <b>목표가(TP):</b> <code>{fmt_price(row['tp'])}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                    🛑 <b>손절가(SL):</b> <code>{fmt_price(row['sl'])}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                    📊 <b>RSI:</b> {row['rsi']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()