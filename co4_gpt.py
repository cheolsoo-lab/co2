# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V12
- UI/UX Refined & Clean Formatting
- Dual-Horizon Engine (Short-Term Momentum vs Long-Term Macro Trend)
- Position Lifetime Decision Matrix (Hold vs Rotate Guide)
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
    page_title="🔥 Crypto Quant Dashboard V12",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean UI cards and layout
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .badge-long { background-color: #0d9488; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-short { background-color: #e11d48; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["binance", "bybit", "mexc", "gateio"]

@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    account_size: float = 10000.0


# ============================================================
# 1. DATA ACCESS & 4D INTERMARKET MACRO ENGINE
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
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str = "1d", limit: int = 700) -> pd.DataFrame:
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
    return add_indicators(df)


def fetch_ohlcv_fallback(symbol: str, timeframe: str = "1d", limit: int = 700) -> tuple[pd.DataFrame, str]:
    for exchange_id in DEFAULT_EXCHANGES:
        try:
            df = fetch_ohlcv(exchange_id, symbol, timeframe, limit)
            if len(df) >= 100:
                return df, exchange_id
        except Exception:
            continue
    return pd.DataFrame(), ""


def analyze_4d_macro_matrix(market_df: pd.DataFrame) -> dict:
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0
    
    total_vol = market_df["quote_volume"].sum()
    btc_vol = market_df[market_df["base"] == "BTC"]["quote_volume"].sum()
    eth_vol = market_df[market_df["base"] == "ETH"]["quote_volume"].sum()
    alt_vol = total_vol - (btc_vol + eth_vol)
    
    btc_dom = (btc_vol / total_vol) * 100 if total_vol > 0 else 50.0
    alt_share = (alt_vol / total_vol) * 100 if total_vol > 0 else 50.0
    avg_change = market_df["change_pct"].mean()

    if btc_dom < 48.0 and alt_share > 45.0 and avg_change > 0.5:
        pattern = "🚀 본질적 알트 불장 (Altcoin Season)"
        bias = "LONG"
        portfolio = {"BTC": 10, "Major Alt": 30, "Small/Mid Alt": 50, "Cash": 10}
        strategy_desc = "알트로 자금 대이동 발생. 순수 알트 홀딩 강력 추천 (목표가까지 장기 보유 유리)"
    elif btc_dom >= 52.0 and btc_change > 1.0:
        pattern = "⚡ 비트코인 독주장 (BTC Dominance)"
        bias = "LONG"
        portfolio = {"BTC": 70, "Major Alt": 15, "Small/Mid Alt": 5, "Cash": 10}
        strategy_desc = "오직 비트만 상승. 알트는 단기 스캘핑 외 장기 홀딩 금지"
    elif btc_change < -1.5 or avg_change < -1.0:
        pattern = "🩸 현금 대피장 (Risk-Off)"
        bias = "SHORT"
        portfolio = {"BTC": 0, "Major Alt": 0, "Small/Mid Alt": 0, "Cash": 100}
        strategy_desc = "자금 이탈 중. 신규 진입 자제 및 현금 100% 방어"
    else:
        pattern = "⚖️ 방향성 탐색 횡보장"
        bias = "NEUTRAL"
        portfolio = {"BTC": 30, "Major Alt": 30, "Small/Mid Alt": 10, "Cash": 30}
        strategy_desc = "확실한 주도 세력 부재. 짧게 먹고 빠지는 단기 관점만 유효"

    return {
        "pattern": pattern, "bias": bias, "btc_change": btc_change,
        "btc_dom": btc_dom, "alt_share": alt_share,
        "portfolio": portfolio, "strategy_desc": strategy_desc
    }


def render_macro_dashboard(market_df: pd.DataFrame):
    macro = analyze_4d_macro_matrix(market_df)
    
    st.markdown("### 🌐 4차원 인터마켓 매크로 매트릭스")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("시장 판정 국면", macro["pattern"].split(" ")[1])
    with col2:
        st.metric("BTC 24H 변동", f"{macro['btc_change']:+.2f}%")
    with col3:
        st.metric("BTC 도미넌스(추정)", f"{macro['btc_dom']:.1f}%")
    with col4:
        st.metric("알트 자금 점유율", f"{macro['alt_share']:.1f}%")
        
    st.info(f"💡 **매크로 트레이딩 가이드:** {macro['strategy_desc']}")
    
    # Capital Allocation Visual
    p = macro["portfolio"]
    st.markdown(f"💰 **추천 자금 배분:** BTC `{p['BTC']}%` | 메이저알트 `{p['Major Alt']}%` | 중소알트 `{p['Small/Mid Alt']}%` | 현금 `{p['Cash']}%`")
    st.divider()
    return macro["bias"], macro["pattern"]


# ============================================================
# 2. INDICATORS & DUAL HORIZON ENGINE
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["EMA20"] = ta.trend.EMAIndicator(x["Close"], window=20).ema_indicator()
    x["EMA50"] = ta.trend.EMAIndicator(x["Close"], window=50).ema_indicator()
    x["EMA200"] = ta.trend.EMAIndicator(x["Close"], window=200).ema_indicator()
    x["RSI14"] = ta.momentum.RSIIndicator(x["Close"], window=14).rsi()
    x["ADX14"] = ta.trend.ADXIndicator(x["High"], x["Low"], x["Close"], window=14).adx()
    x["ATR14"] = ta.volatility.AverageTrueRange(x["High"], x["Low"], x["Close"], window=14).average_true_range()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["REL_VOLUME"] = x["Volume"] / x["VOL_MA20"]
    x["SWING_HIGH"] = x["High"].rolling(10, center=False).max().shift(1)
    x["SWING_LOW"] = x["Low"].rolling(10, center=False).min().shift(1)
    
    candle_range = x["High"] - x["Low"]
    body_size = (x["Close"] - x["Open"]).abs()
    x["BODY_RATIO"] = np.where(candle_range > 0, body_size / candle_range, 0.0)
    return x


def analyze_symbol_dual_horizon(symbol: str, macro_bias: str, macro_pattern: str) -> Optional[dict]:
    try:
        # 단기용 4시간봉, 장기용 1일봉 데이터 동시 수집
        df_1d, ex_id = fetch_ohlcv_fallback(symbol, "1d", 300)
        df_4h, _ = fetch_ohlcv_fallback(symbol, "4h", 300)
        if len(df_1d) < 100 or len(df_4h) < 100:
            return None

        r_1d = df_1d.iloc[-1]
        r_4h = df_4h.iloc[-1]
        
        close = float(r_1d["Close"])
        atr_1d = float(r_1d["ATR14"])
        
        # --- 1. 단기 관점 분석 (4시간봉 모멘텀 스윙) ---
        short_rsi = float(r_4h["RSI14"])
        short_rel_vol = float(r_4h["REL_VOLUME"]) if pd.notna(r_4h["REL_VOLUME"]) else 1.0
        short_dir = "LONG" if r_4h["Close"] > r_4h["EMA20"] else "SHORT"
        short_tp = close + (2.0 * atr_1d) if short_dir == "LONG" else close - (2.0 * atr_1d)
        short_sl = close - (1.0 * atr_1d) if short_dir == "LONG" else close + (1.0 * atr_1d)

        # --- 2. 장기 관점 분석 (1일봉 매크로 트렌드) ---
        long_trend_up = close > r_1d["EMA50"] and r_1d["EMA50"] > r_1d["EMA200"]
        long_dir = "LONG" if long_trend_up else "SHORT"
        long_tp = close + (5.0 * atr_1d) if long_dir == "LONG" else close - (5.0 * atr_1d)
        long_sl = close - (2.0 * atr_1d) if long_dir == "LONG" else close + (2.0 * atr_1d)

        # --- 3. 홀딩 가이드 (오래 들고 갈까 vs 회전시킬까) ---
        # 매크로 패턴이 불장이거나 장기 이평선 정배열이면 '장기 홀딩' 추천
        if "불장" in macro_pattern and long_trend_up:
            holding_advice = "🚀 **장기 홀딩 추천** (대세 상승장: TP 상단까지 분할 익절하며 길게 가져가세요)"
            action_type = "LONG_HOLD"
        elif "비트코인 독주" in macro_pattern and "BTC" in symbol:
            holding_advice = "⚡ **비트 장기 홀딩** (독주장 수혜주: 비트코인 중심 메인 홀딩)"
            action_type = "LONG_HOLD"
        elif "횡보" in macro_pattern:
            holding_advice = "🔄 **단기 회전(스캘핑) 추천** (횡보장: 목표가 도달 시 즉시 익절 후 순환 코인 교체)"
            action_type = "SHORT_ROTATE"
        else:
            holding_advice = "⚖️ **목표가 도달 시 절반 익절 후 추적 관찰**"
            action_type = "FLEXIBLE"

        score = float(np.clip(r_1d["ADX14"] * 0.7 + short_rel_vol * 10, 0, 100))

        return {
            "symbol": symbol,
            "exchange": ex_id.upper(),
            "price": close,
            "score": score,
            "action_type": action_type,
            "holding_advice": holding_advice,
            # 단기 데이터
            "short_dir": short_dir,
            "short_tp": short_tp,
            "short_sl": short_sl,
            "short_rsi": short_rsi,
            # 장기 데이터
            "long_dir": long_dir,
            "long_tp": long_tp,
            "long_sl": long_sl,
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
    st.title("🔥 Crypto Quant Dashboard V12")
    st.caption("단기 스윙 vs 장기 홀딩 전략 및 자금 회전 가이드 시스템")

    market, active_exchange = fetch_tickers_with_fallback()
    if market.empty:
        st.error("거래소 시세 데이터를 불러오지 못했습니다.")
        return

    # 4D 매크로 분석
    macro_bias, macro_pattern = render_macro_dashboard(market)

    # 상위 종목 필터링
    universe = market[market["quote_volume"] >= 2_000_000].sort_values("quote_volume", ascending=False).head(15)
    symbols = universe["symbol"].tolist()

    if st.button("🔍 듀얼 호라이즌(단기/장기) 정밀 분석 시작", use_container_width=True):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(analyze_symbol_dual_horizon, s, macro_bias, macro_pattern) for s in symbols]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r: results.append(r)
        st.session_state["v12_results"] = results

    results = st.session_state.get("v12_results", [])
    if results:
        df_res = pd.DataFrame(results).sort_values("score", ascending=False)

        st.markdown("### 📋 추천 코인 종합 분석 브리핑 (카드형 뷰)")
        st.caption("💡 각 종목카드를 통해 **단기 관점(빠른 익절 후 순환)**과 **장기 관점(목표 TP까지 홀딩)**을 한눈에 비교할 수 있습니다.")

        for _, row in df_res.iterrows():
            badge_class = "badge-long" if row["short_dir"] == "LONG" else "badge-short"
            
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4><b>{row['symbol']}</b> <span style="font-size:12px; color:#a0aec0;">({row['exchange']})</span> &nbsp; <span class="{badge_class}">{row['short_dir']}</span> &nbsp; <b>현재가:</b> {fmt_price(row['price'])} &nbsp; <b>퀀트점수:</b> {row['score']:.1f}점</h4>
                    <hr style="margin: 5px 0 10px 0; border-color: #2d3748;">
                    <b>⚡ [단기 관점 - 4시간봉 스윙]</b><br>
                    &nbsp;&nbsp;• 목표가(TP): <code>{fmt_price(row['short_tp'])}</code> &nbsp;|&nbsp; 손절가(SL): <code>{fmt_price(row['short_sl'])}</code> &nbsp;|&nbsp; RSI: <code>{row['short_rsi']:.1f}</code><br><br>
                    <b>📈 [장기 관점 - 1일봉 매크로 트렌드]</b><br>
                    &nbsp;&nbsp;• 대세 목표가(TP): <code>{fmt_price(row['long_tp'])}</code> &nbsp;|&nbsp; 대세 손절가(SL): <code>{fmt_price(row['long_sl'])}</code><br><br>
                    📌 <b>운영 가이드:</b> {row['holding_advice']}
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()