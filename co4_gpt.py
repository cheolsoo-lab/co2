# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V17 (Aggressive/Stable Long & Short Split)
- Clean Light UI Design
- 3-Way Classification: Aggressive Long, Stable Long, Short
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
    page_title="🔥 Crypto Quant Dashboard V17",
    page_icon="⚡",
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
    /* 🚀 공격형 롱 카드 (주황/레드 계열 핫 모멘텀) */
    .card-agg-long {
        background: #fff1f2; border: 1px solid #fecdd3; border-left: 5px solid #e11d48;
        padding: 15px; border-radius: 10px; margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(225, 29, 72, 0.05);
    }
    /* 🛡️ 안정형 롱 카드 (초록 계열 눌림목) */
    .card-stable-long {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #10b981;
        padding: 15px; border-radius: 10px; margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.05);
    }
    /* 🔴 숏 카드 (진한 레드 계열 하방 베팅) */
    .card-short {
        background: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid #dc2626;
        padding: 15px; border-radius: 10px; margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(220, 38, 38, 0.05);
    }
    .badge-agg { background-color: #e11d48; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-stable { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-short { background-color: #dc2626; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .stat-pill { background: #f1f5f9; padding: 8px 12px; border-radius: 8px; font-weight: 600; font-size: 13px; color: #475569; text-align: center; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]


# ============================================================
# 1. DATA ACCESS & MACRO ENGINE
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
        action_guide = "공격형 및 안정형 롱(LONG) 포지션 위주 트레이딩 권장"
    elif btc_dom >= 52.0 and btc_change > 1.0:
        phase = "⚡ 비트코인 독주장"
        action_guide = "비트코인 및 메이저 롱 포지션 집중"
    elif btc_change < -1.5 or avg_change < -1.0:
        phase = "🩸 하락 추세 (Risk-Off)"
        action_guide = "숏(SHORT) 베팅 및 현금 방어 우선"
    else:
        phase = "⚖️ 박스권 횡보장"
        action_guide = "안정형 롱 및 저항선 숏 양방향 균형 대응"

    return {
        "phase": phase, "action_guide": action_guide,
        "btc_change": btc_change, "btc_dom": btc_dom, "alt_share": alt_share
    }


def render_market_horizon_dashboard(market_df: pd.DataFrame):
    m = analyze_market_wide_horizon(market_df)
    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 시장 거시 진단 및 전략 가이드</h3>
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {m['phase']}
            </span>
        </div>
        <p style="font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 15px;">
            💡 전략 가이드: <span style="color: #2563eb;">{m['action_guide']}</span>
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
# 2. 3-WAY STRATEGY ANALYSIS (Aggressive Long / Stable Long / Short)
# ============================================================

def analyze_symbol_v17(symbol: str) -> Optional[dict]:
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

        # 1. 🔥 공격형 롱 (고수익 돌파 모멘텀)
        if rel_vol >= 1.8 and change_24h > 2.0 and rsi < 75:
            strategy_type = "AGGRESSIVE_LONG"
            tp = min(swing_high, close + (3.5 * atr))
            sl = max(swing_low, close - (1.2 * atr))
            score = float(np.clip(rel_vol * 25 + change_24h * 5, 50, 100))

        # 2. 🛡️ 안정형 롱 (이평선 눌림목 반등)
        elif close >= float(r["EMA20"]) and 40 <= rsi <= 62:
            strategy_type = "STABLE_LONG"
            tp = min(swing_high, close + (3.0 * atr))
            sl = max(swing_low, close - (1.0 * atr))
            score = float(np.clip((62 - abs(rsi - 50)) * 1.5 + rel_vol * 15, 40, 95))

        # 3. 🔴 숏 베팅 (과매수 이탈 또는 하방 압력 우세)
        elif (rsi >= 68 and change_24h < 1.0) or (close < float(r["EMA20"]) and rsi < 45 and rel_vol >= 1.4):
            strategy_type = "SHORT"
            tp = max(swing_low, close - (3.0 * atr))
            sl = min(swing_high, close + (1.2 * atr))
            score = float(np.clip(rel_vol * 20 + abs(change_24h) * 5, 50, 100))
        else:
            return None

        risk_per_share = abs(close - sl)
        recommended_allocation_pct = min(max(5.0, 15.0 / (risk_per_share / close * 100)), 25.0)

        return {
            "symbol": symbol,
            "exchange": ex_id.upper(),
            "price": close,
            "strategy_type": strategy_type,
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
    st.title("🔥 Crypto Quant Dashboard V17")
    st.caption("공격형 롱 / 안정형 롱 / 숏 포지션 3분할 퀀트 분석 시스템")

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("거래소 시세를 불러오지 못했습니다.")
        return

    render_market_horizon_dashboard(market)

    universe = market[market["quote_volume"] >= 2_000_000].sort_values("quote_volume", ascending=False).head(25)
    symbols = universe["symbol"].tolist()

    if st.button("🚀 3원화 전략 종목 정밀 스캔 실행", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(analyze_symbol_v17, s) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v17_results"] = results

    results = st.session_state.get("v17_results", [])
    if results:
        df_res = pd.DataFrame(results)
        
        agg_long_df = df_res[df_res["strategy_type"] == "AGGRESSIVE_LONG"].sort_values("score", ascending=False)
        stable_long_df = df_res[df_res["strategy_type"] == "STABLE_LONG"].sort_values("score", ascending=False)
        short_df = df_res[df_res["strategy_type"] == "SHORT"].sort_values("score", ascending=False)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔥 공격형 롱")
            st.caption("거래량 폭증과 함께 고점을 돌파하는 고수익 모멘텀 종목")
            if agg_long_df.empty:
                st.info("조건에 맞는 종목 없음")
            else:
                for _, row in agg_long_df.iterrows():
                    st.markdown(f"""
                    <div class="card-agg-long">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-agg">AGG LONG</span> &nbsp;
                                <b style="font-size: 14px; color: #0f172a;">{row['symbol']}</b>
                            </div>
                            <div style="font-size: 12px; color: #334155;">
                                <b>{fmt_price(row['price'])}</b>
                            </div>
                        </div>
                        <div style="margin-top: 6px; font-size: 11px; color: #475569; background: #ffffff; padding: 6px; border-radius: 6px;">
                            🎯 TP: <code style="color: #e11d48;">{fmt_price(row['tp'])}</code> | 🛑 SL: <code style="color: #64748b;">{fmt_price(row['sl'])}</code><br>
                            📊 RSI: {row['rsi']:.1f} | 볼륨: {row['rel_vol']:.1f}배 | 비중: <b>{row['allocation']:.0f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🛡️ 안정형 롱")
            st.caption("이평선 지지를 받으며 안전하게 반등하는 스윙 매집 종목")
            if stable_long_df.empty:
                st.info("조건에 맞는 종목 없음")
            else:
                for _, row in stable_long_df.iterrows():
                    st.markdown(f"""
                    <div class="card-stable-long">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-stable">STABLE LONG</span> &nbsp;
                                <b style="font-size: 14px; color: #0f172a;">{row['symbol']}</b>
                            </div>
                            <div style="font-size: 12px; color: #334155;">
                                <b>{fmt_price(row['price'])}</b>
                            </div>
                        </div>
                        <div style="margin-top: 6px; font-size: 11px; color: #475569; background: #ffffff; padding: 6px; border-radius: 6px;">
                            🎯 TP: <code style="color: #10b981;">{fmt_price(row['tp'])}</code> | 🛑 SL: <code style="color: #64748b;">{fmt_price(row['sl'])}</code><br>
                            📊 RSI: {row['rsi']:.1f} | 볼륨: {row['rel_vol']:.1f}배 | 비중: <b>{row['allocation']:.0f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col3:
            st.markdown("### 🔴 숏 포지션")
            st.caption("과매수권 저항 이탈 또는 하방 압력이 우세한 하락 베팅 종목")
            if short_df.empty:
                st.info("조건에 맞는 종목 없음")
            else:
                for _, row in short_df.iterrows():
                    st.markdown(f"""
                    <div class="card-short">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-short">SHORT</span> &nbsp;
                                <b style="font-size: 14px; color: #0f172a;">{row['symbol']}</b>
                            </div>
                            <div style="font-size: 12px; color: #334155;">
                                <b>{fmt_price(row['price'])}</b>
                            </div>
                        </div>
                        <div style="margin-top: 6px; font-size: 11px; color: #475569; background: #ffffff; padding: 6px; border-radius: 6px;">
                            🎯 TP: <code style="color: #dc2626;">{fmt_price(row['tp'])}</code> | 🛑 SL: <code style="color: #64748b;">{fmt_price(row['sl'])}</code><br>
                            📊 RSI: {row['rsi']:.1f} | 볼륨: {row['rel_vol']:.1f}배 | 비중: <b>{row['allocation']:.0f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()