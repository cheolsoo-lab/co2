# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V14
- Clean, Modern Light-Tone UI Design
- Visual Card-Based Macro Horizon Matrix
- Streamlined Actionable Briefing
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
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
    page_title="🔥 Crypto Quant Dashboard V14",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Global Clean Theme */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Macro Header Card */
    .macro-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Symbol Card */
    .coin-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 10px;
    }
    
    /* Badges */
    .badge-long { 
        background-color: #10b981; 
        color: white; 
        padding: 4px 10px; 
        border-radius: 6px; 
        font-weight: 700; 
        font-size: 12px; 
    }
    .badge-short { 
        background-color: #ef4444; 
        color: white; 
        padding: 4px 10px; 
        border-radius: 6px; 
        font-weight: 700; 
        font-size: 12px; 
    }
    
    /* Stat Pills */
    .stat-pill {
        background: #f1f5f9;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        color: #475569;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]


# ============================================================
# 1. DATA ACCESS & INTUITIVE MACRO ENGINE
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
    eth_vol = market_df[market_df["base"] == "ETH"]["quote_volume"].sum()
    alt_vol = total_vol - (btc_vol + eth_vol)
    
    btc_dom = (btc_vol / total_vol) * 100 if total_vol > 0 else 50.0
    alt_share = (alt_vol / total_vol) * 100 if total_vol > 0 else 50.0
    avg_change = market_df["change_pct"].mean()

    # 핵심 진단 및 운영 지침 분류
    if btc_dom < 48.0 and alt_share > 45.0 and avg_change > 0.5:
        phase = "🚀 대세 알트 불장"
        horizon = "LONG_HOLD"
        action_guide = "목표가(TP)까지 길게 가져가는 **장기 홀딩** 전략 권장"
        portfolio = {"BTC": 10, "Major Alt": 30, "Small Alt": 50, "Cash": 10}
    elif btc_dom >= 52.0 and btc_change > 1.0:
        phase = "⚡ 비트코인 독주장"
        horizon = "LONG_HOLD_BTC"
        action_guide = "알트 배제, **비트코인 중심 홀딩** 또는 단기 순환매"
        portfolio = {"BTC": 70, "Major Alt": 15, "Small Alt": 5, "Cash": 10}
    elif btc_change < -1.5 or avg_change < -1.0:
        phase = "🩸 현금 대피장 (Risk-Off)"
        horizon = "DEFENSIVE_CASH"
        action_guide = "모든 홀딩 중단, **현금 100% 방어** 및 신규 진입 자제"
        portfolio = {"BTC": 0, "Major Alt": 0, "Small Alt": 0, "Cash": 100}
    else:
        phase = "⚖️ 박스권 횡보장"
        horizon = "SHORT_ROTATE"
        action_guide = "짧게 먹고 빠지는 **단기 순환매(회전)** 전략 필수"
        portfolio = {"BTC": 30, "Major Alt": 30, "Small Alt": 10, "Cash": 30}

    return {
        "phase": phase,
        "horizon": horizon,
        "action_guide": action_guide,
        "btc_change": btc_change,
        "btc_dom": btc_dom,
        "alt_share": alt_share,
        "portfolio": portfolio
    }


def render_market_horizon_dashboard(market_df: pd.DataFrame):
    m = analyze_market_wide_horizon(market_df)
    
    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 시장 거시 진단 및 대응 지침</h3>
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {m['phase']}
            </span>
        </div>
        <p style="font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 15px;">
            💡 핵심 전략: <span style="color: #2563eb;">{m['action_guide']}</span>
        </p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 12px 0;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H: <b>{m['btc_change']:+.2f}%</b></div>
            <div class="stat-pill">📊 BTC 도미넌스: <b>{m['btc_dom']:.1f}%</b></div>
            <div class="stat-pill">🚀 알트 자금 점유율: <b>{m['alt_share']:.1f}%</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    p = m["portfolio"]
    st.markdown(f"💰 **권장 자금 배분:** BTC `{p['BTC']}%` | 메이저알트 `{p['Major Alt']}%` | 중소알트 `{p['Small Alt']}%` | 현금 `{p['Cash']}%`")
    st.divider()
    return m["horizon"]


# ============================================================
# 2. INDIVIDUAL COIN ANALYSIS
# ============================================================

def analyze_symbol_unified(symbol: str, horizon: str) -> Optional[dict]:
    try:
        df, ex_id = fetch_ohlcv_fallback(symbol, "1d", 200)
        if len(df) < 100:
            return None

        r = df.iloc[-1]
        close = float(r["Close"])
        atr = float(r["ATR14"])
        rsi = float(r["RSI14"])
        rel_vol = float(r["REL_VOLUME"]) if pd.notna(r["REL_VOLUME"]) else 1.0

        if horizon in {"LONG_HOLD", "LONG_HOLD_BTC"}:
            direction = "LONG"
            tp = close + (4.0 * atr)
            sl = close - (1.5 * atr)
        elif horizon == "SHORT_ROTATE":
            direction = "LONG" if close > r["EMA20"] else "SHORT"
            tp = close + (2.0 * atr)
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
    st.title("🔥 Crypto Quant Dashboard V14")
    st.caption("클린 UI 디자인 및 직관적인 시장 맞춤형 트레이딩 대시보드")

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("거래소 시세를 불러오지 못했습니다.")
        return

    horizon = render_market_horizon_dashboard(market)

    universe = market[market["quote_volume"] >= 2_000_000].sort_values("quote_volume", ascending=False).head(15)
    symbols = universe["symbol"].tolist()

    if st.button("🚀 추천 종목 정밀 스캔 실행", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(analyze_symbol_unified, s, horizon) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v14_results"] = results

    results = st.session_state.get("v14_results", [])
    if results:
        df_res = pd.DataFrame(results).sort_values("score", ascending=False)

        st.markdown("### 📋 추천 종목 브리핑")
        st.caption("현재 시장 국면 지침에 최적화된 가격 라인입니다.")

        for _, row in df_res.iterrows():
            badge = "badge-long" if row["direction"] == "LONG" else "badge-short"
            st.markdown(f"""
            <div class="coin-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="{badge}">{row['direction']}</span> &nbsp;
                        <b style="font-size: 16px; color: #0f172a;">{row['symbol']}</b> 
                        <span style="font-size: 12px; color: #64748b;">({row['exchange']})</span>
                    </div>
                    <div style="font-size: 14px; color: #334155;">
                        <b>현재가:</b> {fmt_price(row['price'])} &nbsp;|&nbsp; <b>점수:</b> <b>{row['score']:.1f}점</b>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #475569; background: #f8fafc; padding: 8px 12px; border-radius: 6px;">
                    🎯 <b>목표가(TP):</b> <code style="color: #2563eb;">{fmt_price(row['tp'])}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                    🛑 <b>손절가(SL):</b> <code style="color: #dc2626;">{fmt_price(row['sl'])}</code> &nbsp;&nbsp;|&nbsp;&nbsp; 
                    📊 <b>RSI:</b> {row['rsi']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()