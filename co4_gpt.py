# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V16 (Long/Short Visual Clarity)
- Clean Light UI Design
- Clear LONG (Green) & SHORT (Red) Badge & Color Coding
- Swing High/Low Dynamic TP/SL Engine
"""

from __future__ import annotations

import concurrent.futures
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import ta

# ============================================================
# 0. APP CONFIG & STYLING (Clean Light Theme)
# ============================================================

st.set_page_config(
    page_title="🔥 Crypto Quant Dashboard V16",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .macro-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid #3b82f6;
        padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 20px;
    }
    /* 🟢 LONG 포지션 카드 스타일 (초록/그린 계열) */
    .coin-card-long {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #10b981;
        padding: 15px; border-radius: 10px; margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.06);
    }
    /* 🔴 SHORT 포지션 카드 스타일 (빨강/레드 계열) */
    .coin-card-short {
        background: #fff5f5; border: 1px solid #fed7d7; border-left: 5px solid #ef4444;
        padding: 15px; border-radius: 10px; margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.06);
    }
    .badge-long { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-short { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .stat-pill { background: #f1f5f9; padding: 8px 12px; border-radius: 8px; font-weight: 600; font-size: 13px; color: #475569; text-align: center; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]


# ============================================================
# 1. DATA ACCESS & ADVANCED MACRO ENGINE
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
    
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["ATR14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["REL_VOLUME"] = df["Volume"] / df["VOL_MA20"]
    df["SWING_HIGH"] = df["High"].rolling(10, center=False).max().shift(1)
    df["SWING_LOW"] = df["Low"].rolling(10, center=False).min().shift(1)
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
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0
    
    total_vol = market_df["quote_volume"].sum()
    btc_vol = market_df[market_df["base"] == "BTC"]["quote_volume"].sum()
    alt_vol = total_vol - btc_vol
    
    btc_dom = (btc_vol / total_vol) * 100 if total_vol > 0 else 50.0
    alt_share = (alt_vol / total_vol) * 100 if total_vol > 0 else 50.0
    avg_change = market_df["change_pct"].mean()

    if btc_dom < 48.0 and alt_share > 45.0 and avg_change > 0.5:
        phase = "🚀 대세 알트 불장"
        action_guide = "롱(LONG) 포지션 위주 트레이딩 강력 추천"
    elif btc_dom >= 52.0 and btc_change > 1.0:
        phase = "⚡ 비트코인 독주장"
        action_guide = "비트코인 롱 및 메이저 중심 대응"
    elif btc_change < -1.5 or avg_change < -1.0:
        phase = "🩸 하락 추세 (Risk-Off)"
        action_guide = "숏(SHORT) 베팅 또는 현금 방어 우선"
    else:
        phase = "⚖️ 박스권 횡보장"
        action_guide = "지지선 롱 / 저항선 숏 양방향 스윙 대응"

    return {
        "phase": phase, "action_guide": action_guide,
        "btc_change": btc_change, "btc_dom": btc_dom, "alt_share": alt_share
    }


def render_market_horizon_dashboard(market_df: pd.DataFrame):
    m = analyze_market_wide_horizon(market_df)
    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 시장 거시 진단 및 포지션 가이드</h3>
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {m['phase']}
            </span>
        </div>
        <p style="font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 15px;">
            💡 전략 권고: <span style="color: #2563eb;">{m['action_guide']}</span>
        </p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 12px 0;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H: <b>{m['btc_change']:+.2f}%</b></div>
            <div class="stat-pill">📊 BTC 도미넌스: <b>{m['btc_dom']:.1f}%</b></div>
            <div class="stat-pill">🚀 알트 자금 점유율: <b>{m['alt_share']:.1f}%</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


# ============================================================
# 2. ADVANCED SYMBOL ANALYSIS (LONG / SHORT CLASSIFICATION)
# ============================================================

def analyze_symbol_v16(symbol: str) -> Optional[dict]:
    try:
        df, ex_id = fetch_ohlcv_fallback(symbol, "1d", 200)
        if len(df) < 100:
            return None

        r = df.iloc[-1]
        close = float(r["Close"])
        atr = float(r["ATR14"])
        rsi = float(r["RSI14"])
        rel_vol = float(r["REL_VOLUME"]) if pd.notna(r["REL_VOLUME"]) else 1.0
        change_24h = float(df.iloc[-1]["Close"] - df.iloc[-2]["Close"]) / float(df.iloc[-2]["Close"]) * 100

        swing_high = float(r["SWING_HIGH"]) if pd.notna(r["SWING_HIGH"]) else close + (3.0 * atr)
        swing_low = float(r["SWING_LOW"]) if pd.notna(r["SWING_LOW"]) else close - (1.5 * atr)

        # 🟢 1. LONG 조건 (상승 돌파 또는 이평선 지지 반등)
        if (rel_vol >= 1.6 and change_24h > 1.5 and rsi < 75) or (close >= float(r["EMA20"]) and 42 <= rsi <= 62):
            position_type = "LONG"
            tp = min(swing_high, close + (3.5 * atr))
            sl = max(swing_low, close - (1.2 * atr))
            score = float(np.clip(rel_vol * 20 + change_24h * 5, 50, 100))
        
        # 🔴 2. SHORT 조건 (과매수 상태에서 거래량 동반 이탈 혹은 저항선 맞고 하락)
        elif (rsi >= 68 and change_24h < 0.5) or (close < float(r["EMA20"]) and rsi < 45 and rel_vol >= 1.4):
            position_type = "SHORT"
            tp = max(swing_low, close - (3.0 * atr))  # 숏의 목표가(익절가)는 아래쪽
            sl = min(swing_high, close + (1.2 * atr)) # 숏의 손절가는 위쪽 저항선
            score = float(np.clip(rel_vol * 20 + abs(change_24h) * 5, 50, 100))
        else:
            return None

        risk_per_share = abs(close - sl)
        recommended_allocation_pct = min(max(5.0, 15.0 / (risk_per_share / close * 100)), 25.0)

        return {
            "symbol": symbol,
            "exchange": ex_id.upper(),
            "price": close,
            "position_type": position_type,
            "tp": tp,
            "sl": sl,
            "rsi": rsi,
            "rel_vol": rel_vol,
            "score": score,
            "allocation": recommended_allocation_pct
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
    st.title("🔥 Crypto Quant Dashboard V16")
    st.caption("롱(LONG) / 숏(SHORT) 포지션 자동 분류 및 다이내믹 TP/SL 퀀트 시스템")

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("거래소 시세를 불러오지 못했습니다.")
        return

    render_market_horizon_dashboard(market)

    universe = market[market["quote_volume"] >= 2_000_000].sort_values("quote_volume", ascending=False).head(25)
    symbols = universe["symbol"].tolist()

    if st.button("🚀 롱 / 숏 추천 종목 정밀 스캔", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(analyze_symbol_v16, s) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v16_results"] = results

    results = st.session_state.get("v16_results", [])
    if results:
        df_res = pd.DataFrame(results)
        
        long_df = df_res[df_res["position_type"] == "LONG"].sort_values("score", ascending=False)
        short_df = df_res[df_res["position_type"] == "SHORT"].sort_values("score", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 추천 롱 (LONG) 포지션")
            st.caption("상승 추세 지지 또는 거래량 동반 상방 돌파 종목군입니다.")
            if long_df.empty:
                st.info("현재 조건에 부합하는 롱 종목이 없습니다.")
            else:
                for _, row in long_df.iterrows():
                    st.markdown(f"""
                    <div class="coin-card-long">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-long">🟢 LONG (매수)</span> &nbsp;
                                <b style="font-size: 15px; color: #0f172a;">{row['symbol']}</b> 
                                <span style="font-size: 11px; color: #64748b;">({row['exchange']})</span>
                            </div>
                            <div style="font-size: 13px; color: #334155;">
                                <b>현재가:</b> {fmt_price(row['price'])}
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #475569; background: #ffffff; padding: 6px 10px; border-radius: 6px;">
                            🎯 <b>TP(목표가):</b> <code style="color: #10b981;">{fmt_price(row['tp'])}</code> &nbsp;|&nbsp; 
                            🛑 <b>SL(손절가):</b> <code style="color: #ef4444;">{fmt_price(row['sl'])}</code><br>
                            📊 <b>RSI:</b> {row['rsi']:.1f} &nbsp;|&nbsp; <b>상대볼륨:</b> {row['rel_vol']:.1f}배 &nbsp;|&nbsp; <b>권장비중:</b> <b>{row['allocation']:.0f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔴 추천 숏 (SHORT) 포지션")
            st.caption("저항선 부근 과매수 이탈 또는 하방 압력이 우세한 종목군입니다.")
            if short_df.empty:
                st.info("현재 조건에 부합하는 숏 종목이 없습니다.")
            else:
                for _, row in short_df.iterrows():
                    st.markdown(f"""
                    <div class="coin-card-short">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-short">🔴 SHORT (매도)</span> &nbsp;
                                <b style="font-size: 15px; color: #0f172a;">{row['symbol']}</b> 
                                <span style="font-size: 11px; color: #64748b;">({row['exchange']})</span>
                            </div>
                            <div style="font-size: 13px; color: #334155;">
                                <b>현재가:</b> {fmt_price(row['price'])}
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #475569; background: #ffffff; padding: 6px 10px; border-radius: 6px;">
                            🎯 <b>TP(목표가):</b> <code style="color: #10b981;">{fmt_price(row['tp'])}</code> &nbsp;|&nbsp; 
                            🛑 <b>SL(손절가):</b> <code style="color: #ef4444;">{fmt_price(row['sl'])}</code><br>
                            📊 <b>RSI:</b> {row['rsi']:.1f} &nbsp;|&nbsp; <b>상대볼륨:</b> {row['rel_vol']:.1f}배 &nbsp;|&nbsp; <b>권장비중:</b> <b>{row['allocation']:.0f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()