# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V23 (Aggressive Alpha Sensitivity Boosted)
- Lowered Vol/RS Thresholds for More Frequent Breakout Signals
- Restored Macro Comprehensive Analysis & Top 50 Hybrid TP/SL Engine
- Robust Multi-Exchange Fallback (Binance, Bybit, OKX)
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
    page_title="🔥 Crypto Quant Dashboard V23",
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

DEFAULT_EXCHANGES = ["binance", "bybit", "okx"]


# ============================================================
# 1. ROBUST DATA ACCESS & ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"},
    })


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tickers_safe() -> tuple[pd.DataFrame, str]:
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            ex = make_exchange(exchange_id)
            ex.load_markets()
            tickers = ex.fetch_tickers()
            rows = []
            for symbol, t in tickers.items():
                if not symbol.endswith("USDT") and not "/USDT" in symbol:
                    continue
                
                clean_symbol = symbol.split(":")[0] if ":" in symbol else symbol
                if not clean_symbol.endswith("/USDT"):
                    if clean_symbol.endswith("USDT") and "/" not in clean_symbol:
                        base = clean_symbol[:-4]
                        clean_symbol = f"{base}/USDT"
                    else:
                        continue

                last = t.get("last")
                quote_volume = t.get("quoteVolume") or t.get("baseVolume") or 0.0
                pct = t.get("percentage")
                
                if last is None:
                    continue
                
                rows.append({
                    "symbol": clean_symbol,
                    "base": clean_symbol.split("/")[0],
                    "last": float(last),
                    "change_pct": float(pct) if pct is not None else 0.0,
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
    try:
        ex = make_exchange(exchange_id)
        raw_symbol = symbol if exchange_id != "binance" else f"{symbol.replace('/','')} :USDT" if ":" not in symbol else symbol
        if exchange_id == "binance" and ":" not in raw_symbol:
            raw_symbol = f"{symbol.split('/')[0]}/USDT:USDT"

        raw_1d = ex.fetch_ohlcv(raw_symbol, timeframe="1d", limit=100)
        df_1d = pd.DataFrame(raw_1d, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_1d["EMA20"] = ta.trend.EMAIndicator(df_1d["Close"], window=20).ema_indicator()
        df_1d["RSI14"] = ta.momentum.RSIIndicator(df_1d["Close"], window=14).rsi()

        raw_4h = ex.fetch_ohlcv(raw_symbol, timeframe="4h", limit=100)
        df_4h = pd.DataFrame(raw_4h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_4h["EMA20"] = ta.trend.EMAIndicator(df_4h["Close"], window=20).ema_indicator()
        df_4h["RSI14"] = ta.momentum.RSIIndicator(df_4h["Close"], window=14).rsi()
        df_4h["ATR14"] = ta.volatility.AverageTrueRange(df_4h["High"], df_4h["Low"], df_4h["Close"], window=14).average_true_range()
        df_4h["VOL_MA20"] = df_4h["Volume"].rolling(20).mean()
        df_4h["REL_VOLUME"] = df_4h["Volume"] / df_4h["VOL_MA20"]

        oi_change, funding_rate = 1.0, 0.0
        try:
            oi_data = ex.fetch_open_interest(raw_symbol)
            if oi_data.get("openInterestAmount", 0) > 0: 
                oi_change = 1.2
        except Exception:
            pass

        try:
            fr_data = ex.fetch_funding_rate(raw_symbol)
            funding_rate = float(fr_data.get("fundingRate", 0.0))
        except Exception:
            pass

        if len(df_1d) < 30 or len(df_4h) < 30:
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


# ============================================================
# 2. MACRO COMPREHENSIVE ANALYSIS ENGINE
# ============================================================

def analyze_market_wide_horizon(market_df: pd.DataFrame) -> dict:
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0
    avg_change = float(market_df["change_pct"].mean())

    if btc_change > 1.0 and avg_change > 0.3:
        phase = "🚀 강한 상승장 (Risk-On)"
        action_guide = "공격형 돌파 및 주도주 중심의 적극적인 롱(LONG) 포지션 공략 권장"
    elif btc_change < -1.0 or avg_change < -0.5:
        phase = "🩸 하락 추세 (Risk-Off)"
        action_guide = "공격형/안정형 숏(SHORT) 베팅 및 하락 돌파 종목 집중 대응"
    else:
        phase = "⚖️ 혼조세 및 박스권 횡보장"
        action_guide = "돌파 실패에 유의하며 완화된 조건으로 선별된 모멘텀 종목 매매"

    return {
        "phase": phase,
        "action_guide": action_guide,
        "btc_change": btc_change,
        "avg_change": avg_change
    }


def render_market_horizon_dashboard(market_df: pd.DataFrame):
    m = analyze_market_wide_horizon(market_df)
    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 거시적 종합분석 & 마켓 센티먼트 진단</h3>
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {m['phase']}
            </span>
        </div>
        <p style="font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 15px;">
            💡 실전 대응 가이드: <span style="color: #2563eb;">{m['action_guide']}</span>
        </p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 12px 0;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H 변동률: <b>{m['btc_change']:+.2f}%</b></div>
            <div class="stat-pill">📊 시장 평균 변동률: <b>{m['avg_change']:+.2f}%</b></div>
            <div class="stat-pill">⚡ 엔진 V23: <b>공격형 감도 상향 & 하이브리드 TP/SL</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


# ============================================================
# 3. QUANT ANALYSIS & UI (V23: 공격형 필터 완화)
# ============================================================

def analyze_symbol_v23(symbol: str, exchange_id: str, market_avg_change: float) -> Optional[dict]:
    data = fetch_multi_timeframe_data(exchange_id.lower(), symbol)
    if not data:
        return None
    
    df_1d = data["df_1d"]
    df_4h = data["df_4h"]
    funding_rate = data["funding_rate"]

    r_1d = df_1d.iloc[-1]
    r_4h = df_4h.iloc[-1]

    close = float(r_4h["Close"])
    atr = float(r_4h["ATR14"]) if pd.notna(r_4h["ATR14"]) else close * 0.03
    rsi_4h = float(r_4h["RSI14"])
    rel_vol = float(r_4h["REL_VOLUME"]) if pd.notna(r_4h["REL_VOLUME"]) else 1.0
    
    if len(df_4h) >= 6:
        symbol_change_24h = float((close - df_4h.iloc[-6]["Close"]) / df_4h.iloc[-6]["Close"] * 100)
    else:
        symbol_change_24h = 0.0
        
    relative_strength = symbol_change_24h - market_avg_change

    group, pos_type = None, None

    # 🚀 [V23 핵심 변경] 공격형 조건 완화 (거래량 1.3배 -> 1.05배, 상대강도 0.5% -> 0.1%로 낮춰 잘 잡히도록 개선)
    if r_1d["Close"] >= r_1d["EMA20"] * 0.995 and rel_vol >= 1.05 and relative_strength > 0.1 and rsi_4h < 78:
        group, pos_type = "AGGRESSIVE", "LONG"
    elif r_1d["Close"] <= r_1d["EMA20"] * 1.005 and rel_vol >= 1.05 and relative_strength < -0.1 and rsi_4h > 22:
        group, pos_type = "AGGRESSIVE", "SHORT"
    elif close >= float(r_4h["EMA20"]) and 38 <= rsi_4h <= 68 and funding_rate <= 0.001:
        group, pos_type = "STABLE", "LONG"
    elif rsi_4h >= 58 and funding_rate >= 0.0005:
        group, pos_type = "STABLE", "SHORT"
    else:
        return None

    # 하이브리드 TP/SL 산출 (안전 캡 적용)
    if pos_type == "LONG":
        raw_tp = close + (2.2 * atr)
        cap_tp = close * 1.05
        tp = min(raw_tp, cap_tp)
        
        raw_sl = close - (1.1 * atr)
        floor_sl = close * 0.98
        sl = max(raw_sl, floor_sl)
    else:
        raw_tp = close - (2.2 * atr)
        cap_tp = close * 0.95
        tp = max(raw_tp, cap_tp)
        
        raw_sl = close + (1.1 * atr)
        floor_sl = close * 1.02
        sl = min(raw_sl, floor_sl)

    score = float(np.clip(rel_vol * 20 + abs(relative_strength) * 10 + (50 - abs(rsi_4h - 50)), 40, 100))
    
    return {
        "symbol": symbol, "exchange": data["exchange"], "price": close,
        "group": group, "pos_type": pos_type, "tp": tp, "sl": sl,
        "rsi": rsi_4h, "rel_vol": rel_vol, "rs": relative_strength,
        "score": score, "allocation": 15.0
    }


def fmt_price(x):
    if x >= 1000: return f"${x:,.2f}"
    if x >= 1: return f"${x:,.4f}"
    return f"${x:,.8f}"


def main():
    st.title("🔥 Crypto Quant Dashboard V23")
    st.caption("공격형 돌파 감도 상향 및 거시적 종합분석 탑재 스캐너")

    with st.spinner("시세 데이터를 안전하게 불러오는 중입니다..."):
        market, active_exchange = fetch_tickers_safe()

    if market.empty:
        st.error("⚠️ 거래소 데이터 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.")
        return

    st.success(f"✅ 연결 성공: [{active_exchange.upper}] 거래소 데이터 연동 완료 (총 {len(market)}개 심볼 감지)")
    
    render_market_horizon_dashboard(market)

    universe = market.sort_values("quote_volume", ascending=False).head(50)
    symbols = universe["symbol"].tolist()
    market_avg_change = float(market["change_pct"].mean())

    if st.button("🚀 Top 50 종목 하이브리드 퀀트 스캔 실행 (공격형 감도 UP)", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        total_symbols = len(symbols)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(analyze_symbol_v23, s, active_exchange, market_avg_change): s for s in symbols}
            completed = 0
            for f in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total_symbols)
                r = f.result()
                if r: 
                    results.append(r)
        
        progress_bar.empty()
        st.session_state["v23_results"] = results

    results = st.session_state.get("v23_results", [])
    if results:
        df_res = pd.DataFrame(results)
        agg_df = df_res[df_res["group"] == "AGGRESSIVE"].sort_values("score", ascending=False)
        stable_df = df_res[df_res["group"] == "STABLE"].sort_values("score", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔥 공격형 알파 트레이딩 (돌파)")
            if agg_df.empty:
                st.info("조건을 완화했음에도 현재 장세에서 부합하는 공격형 종목이 없습니다.")
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
            st.markdown("### 🛡️ 안정형 스윙 트레이딩 (눌림목)")
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