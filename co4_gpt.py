# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V20 (Ultimate 50-Coin Expansion & Hybrid TP/SL)
- Expanded Universe: Top 50 Coins by Volume
- Hybrid TP/SL Engine: ATR/Swing Dynamic + Safe Percentage Boundary Caps
- Multi-Timeframe (1D + 4H) + Orderflow (OI/Funding) + Relative Strength (RS)
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
# 0. APP CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="🔥 Crypto Quant Dashboard V20",
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
    .card-agg-long {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #10b981;
        padding: 12px; border-radius: 8px; margin-bottom: 10px;
    }
    .card-agg-short {
        background: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid #ef4444;
        padding: 12px; border-radius: 8px; margin-bottom: 10px;
    }
    .card-stable-long {
        background: #f0fdf4; border: 1px solid #d1fae5; border-left: 5px solid #059669;
        padding: 12px; border-radius: 8px; margin-bottom: 10px;
    }
    .card-stable-short {
        background: #fff1f2; border: 1px solid #fecdd3; border-left: 5px solid #e11d48;
        padding: 12px; border-radius: 8px; margin-bottom: 10px;
    }
    .badge-long { background-color: #10b981; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
    .badge-short { background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
    .stat-pill { background: #f1f5f9; padding: 8px 12px; border-radius: 8px; font-weight: 600; font-size: 13px; color: #475569; text-align: center; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["binance", "bybit"]


# ============================================================
# 1. DATA ACCESS & ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "swap"},
    })


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tickers_with_fallback() -> tuple[pd.DataFrame, str]:
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            ex = make_exchange(exchange_id)
            tickers = ex.fetch_tickers()
            rows = []
            for symbol, t in tickers.items():
                if not symbol.endswith("/USDT:USDT") and not symbol.endswith("/USDT"):
                    continue
                clean_symbol = symbol.split(":")[0] if ":" in symbol else symbol
                last = t.get("last")
                quote_volume = t.get("quoteVolume")
                pct = t.get("percentage")
                if last is None:
                    continue
                rows.append({
                    "symbol": clean_symbol,
                    "base": clean_symbol.split("/")[0],
                    "last": float(last),
                    "change_pct": float(pct) if pct is not None else np.nan,
                    "quote_volume": float(quote_volume) if quote_volume is not None else 0.0,
                })
            df = pd.DataFrame(rows).drop_duplicates("symbol")
            if not df.empty:
                return df, exchange_id
        except Exception:
            continue
    return pd.DataFrame(), ""


@st.cache_data(ttl=300, show_spinner=False)
def fetch_multi_timeframe_data(exchange_id: str, symbol: str) -> Optional[dict]:
    ex = make_exchange(exchange_id)
    raw_symbol = f"{symbol}:USDT" if exchange_id == "binance" else symbol
    
    try:
        raw_1d = ex.fetch_ohlcv(raw_symbol, timeframe="1d", limit=150)
        df_1d = pd.DataFrame(raw_1d, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_1d["EMA20"] = ta.trend.EMAIndicator(df_1d["Close"], window=20).ema_indicator()
        df_1d["RSI14"] = ta.momentum.RSIIndicator(df_1d["Close"], window=14).rsi()

        raw_4h = ex.fetch_ohlcv(raw_symbol, timeframe="4h", limit=150)
        df_4h = pd.DataFrame(raw_4h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_4h["EMA20"] = ta.trend.EMAIndicator(df_4h["Close"], window=20).ema_indicator()
        df_4h["RSI14"] = ta.momentum.RSIIndicator(df_4h["Close"], window=14).rsi()
        df_4h["ATR14"] = ta.volatility.AverageTrueRange(df_4h["High"], df_4h["Low"], df_4h["Close"], window=14).average_true_range()
        df_4h["VOL_MA20"] = df_4h["Volume"].rolling(20).mean()
        df_4h["REL_VOLUME"] = df_4h["Volume"] / df_4h["VOL_MA20"]
        df_4h["SWING_HIGH"] = df_4h["High"].rolling(12, center=False).max().shift(1)
        df_4h["SWING_LOW"] = df_4h["Low"].rolling(12, center=False).min().shift(1)

        oi_change, funding_rate = 1.0, 0.0
        try:
            oi_data = ex.fetch_open_interest(raw_symbol)
            if oi_data.get("openInterestAmount", 0) > 0: oi_change = 1.2
        except Exception:
            pass

        try:
            fr_data = ex.fetch_funding_rate(raw_symbol)
            funding_rate = float(fr_data.get("fundingRate", 0.0))
        except Exception:
            pass

        if len(df_1d) < 50 or len(df_4h) < 50:
            return None

        return {
            "df_1d": df_1d.iloc[:-1],
            "df_4h": df_4h.iloc[:-1],
            "oi_change": oi_change,
            "funding_rate": funding_rate,
            "exchange": exchange_id.upper()
        }
    except Exception:
        return None


def render_market_horizon_dashboard(market_df: pd.DataFrame):
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0
    avg_change = market_df["change_pct"].mean()

    if btc_change > 1.5 and avg_change > 0.8:
        phase = "🚀 강한 상승장 (Risk-On)"
    elif btc_change < -1.5 or avg_change < -1.0:
        phase = "🩸 하락 추세 (Risk-Off)"
    else:
        phase = "⚖️ 혼조세 및 횡보장"

    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 Top 50 코인 멀티 타임프레임 & 하이브리드 퀀트 진단</h3>
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {phase}
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H: <b>{btc_change:+.2f}%</b></div>
            <div class="stat-pill">🔍 스캔 유니버스: <b>상위 50개 알트·메이저 종목</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


# ============================================================
# 2. HYBRID TP/SL & QUANT ANALYSIS ENGINE (V20)
# ============================================================

def analyze_symbol_v20(symbol: str, market_avg_change: float) -> Optional[dict]:
    for ex_id in DEFAULT_EXCHANGES:
        data = fetch_multi_timeframe_data(ex_id, symbol)
        if not data:
            continue
        
        df_1d = data["df_1d"]
        df_4h = data["df_4h"]
        funding_rate = data["funding_rate"]

        r_1d = df_1d.iloc[-1]
        r_4h = df_4h.iloc[-1]

        close = float(r_4h["Close"])
        atr = float(r_4h["ATR14"]) if pd.notna(r_4h["ATR14"]) else close * 0.03
        rsi_4h = float(r_4h["RSI14"])
        rel_vol = float(r_4h["REL_VOLUME"]) if pd.notna(r_4h["REL_VOLUME"]) else 1.0
        
        symbol_change_24h = float((close - df_4h.iloc[-6]["Close"]) / df_4h.iloc[-6]["Close"] * 100)
        relative_strength = symbol_change_24h - market_avg_change

        group, pos_type = None, None

        # 조건 판별
        if r_1d["Close"] > r_1d["EMA20"] and rel_vol >= 1.5 and relative_strength > 0.8 and rsi_4h < 75:
            group, pos_type = "AGGRESSIVE", "LONG"
        elif r_1d["Close"] < r_1d["EMA20"] and rel_vol >= 1.5 and relative_strength < -0.8 and rsi_4h > 25:
            group, pos_type = "AGGRESSIVE", "SHORT"
        elif r_1d["Close"] >= r_1d["EMA20"] and close >= float(r_4h["EMA20"]) and 40 <= rsi_4h <= 60 and funding_rate <= 0.0006:
            group, pos_type = "STABLE", "LONG"
        elif r_1d["Close"] < r_1d["EMA20"] and rsi_4h >= 65 and funding_rate >= 0.0008:
            group, pos_type = "STABLE", "SHORT"
        else:
            continue

        # 🎯 하이브리드 TP/SL 산출 (ATR 변동성 반영 + 안전 퍼센트 캡 적용으로 비정상 가격 방지)
        if pos_type == "LONG":
            # TP: ATR 기반 목표가와 고정 +4%~+6% 사이를 조화 (최대 7% 안넘게 캡)
            raw_tp = close + (2.5 * atr)
            cap_tp = close * 1.05
            tp = min(raw_tp, cap_tp)
            
            # SL: ATR 기반 손절가와 고정 -1.8%~-2.5% 사이 캡
            raw_sl = close - (1.2 * atr)
            floor_sl = close * 0.982
            sl = max(raw_sl, floor_sl)
        else:
            # SHORT
            raw_tp = close - (2.5 * atr)
            cap_tp = close * 0.95
            tp = max(raw_tp, cap_tp)
            
            raw_sl = close + (1.2 * atr)
            floor_sl = close * 1.018
            sl = min(raw_sl, floor_sl)

        score = float(np.clip(rel_vol * 20 + abs(relative_strength) * 10 + (50 - abs(rsi_4h - 50)), 40, 100))
        allocation = 15.0

        return {
            "symbol": symbol, "exchange": data["exchange"], "price": close,
            "group": group, "pos_type": pos_type, "tp": tp, "sl": sl,
            "rsi": rsi_4h, "rel_vol": rel_vol, "rs": relative_strength,
            "score": score, "allocation": allocation
        }
    return None


# ============================================================
# 3. STREAMLIT UI
# ============================================================

def fmt_price(x):
    if x >= 1000: return f"${x:,.2f}"
    if x >= 1: return f"${x:,.4f}"
    return f"${x:,.8f}"


def main():
    st.title("🔥 Crypto Quant Dashboard V20")
    st.caption("Top 50 코인 확장 및 ATR 변동성 + 안전 퍼센트 캡이 결합된 하이브리드 TP/SL 시스템")

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("시세 데이터를 불러오지 못했습니다.")
        return

    render_market_horizon_dashboard(market)

    # 상위 50개 코인으로 유니버스 확장
    universe = market[market["quote_volume"] >= 1_000_000].sort_values("quote_volume", ascending=False).head(50)
    symbols = universe["symbol"].tolist()
    market_avg_change = float(market["change_pct"].mean())

    if st.button("🚀 Top 50 종목 하이브리드 퀀트 스캔 실행", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(analyze_symbol_v20, s, market_avg_change) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v20_results"] = results

    results = st.session_state.get("v20_results", [])
    if results:
        df_res = pd.DataFrame(results)
        agg_df = df_res[df_res["group"] == "AGGRESSIVE"].sort_values("score", ascending=False)
        stable_df = df_res[df_res["group"] == "STABLE"].sort_values("score", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔥 공격형 알파 트레이딩 (Top 50 돌파)")
            if agg_df.empty:
                st.info("조건에 부합하는 공격형 종목이 없습니다.")
            else:
                for _, row in agg_df.iterrows():
                    is_long = row["pos_type"] == "LONG"
                    card_cls = "card-agg-long" if is_long else "card-agg-short"
                    badge_html = '<span class="badge-long">🟢 AGG LONG</span>' if is_long else '<span class="badge-short">🔴 AGG SHORT</span>'
                    tp_color = "#10b981" if is_long else "#ef4444"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>{badge_html} &nbsp; <b style="font-size: 14px; color: #0f172a;">{row['symbol']}</b> <span style="font-size: 11px; color: #64748b;">({row['exchange']})</span></div>
                            <div style="font-size: 12px; color: #334155;"><b>{fmt_price(row['price'])}</b></div>
                        </div>
                        <div style="margin-top: 6px; font-size: 11px; color: #475569; background: #ffffff; padding: 6px; border-radius: 6px;">
                            🎯 TP: <code style="color: {tp_color};">{fmt_price(row['tp'])}</code> | 🛑 SL: <code style="color: #64748b;">{fmt_price(row['sl'])}</code><br>
                            📊 4H RSI: {row['rsi']:.1f} | 볼륨: {row['rel_vol']:.1f}배 | 상대강도: <b style="color: #2563eb;">{row['rs']:+.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🛡️ 안정형 스윙 트레이딩 (Top 50 눌림목)")
            if stable_df.empty:
                st.info("조건에 부합하는 안정형 종목이 없습니다.")
            else:
                for _, row in stable_df.iterrows():
                    is_long = row["pos_type"] == "LONG"
                    card_cls = "card-stable-long" if is_long else "card-stable-short"
                    badge_html = '<span class="badge-long">🟢 STABLE LONG</span>' if is_long else '<span class="badge-short">🔴 STABLE SHORT</span>'
                    tp_color = "#10b981" if is_long else "#e11d48"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>{badge_html} &nbsp; <b style="font-size: 14px; color: #0f172a;">{row['symbol']}</b> <span style="font-size: 11px; color: #64748b;">({row['exchange']})</span></div>
                            <div style="font-size: 12px; color: #334155;"><b>{fmt_price(row['price'])}</b></div>
                        </div>
                        <div style="margin-top: 6px; font-size: 11px; color: #475569; background: #ffffff; padding: 6px; border-radius: 6px;">
                            🎯 TP: <code style="color: {tp_color};">{fmt_price(row['tp'])}</code> | 🛑 SL: <code style="color: #64748b;">{fmt_price(row['sl'])}</code><br>
                            📊 4H RSI: {row['rsi']:.1f} | 볼륨: {row['rel_vol']:.1f}배 | 상대강도: <b style="color: #2563eb;">{row['rs']:+.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()