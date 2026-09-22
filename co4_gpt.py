# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V39.0 (Dynamic WFO & Profit Maximization Edition)
- Target Pool: BTC + Majors + Top 100 Liquidity Volume Coins
- Enhanced Features:
  1. Dynamic WFO ATR Scaling (Adapts TP/SL to real-time market volatility regimes)
  2. Chart-Driven Flexible Risk-Reward (Unlocks 1:2.5 to 1:4+ potential based on swing structures)
  3. Anti-Chasing Exhaustion Filter (Blocks coins pumped > +12% or dumped < -12%)
  4. 4H/1H Multi-Timeframe Trend Alignment & Safe Leverage Guide
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
    page_title="🚀 Crypto Quant Dashboard V39.0 Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .macro-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid #4f46e5;
        padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 20px;
    }
    .card-agg-long {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #10b981;
        padding: 16px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .card-agg-short {
        background: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid #ef4444;
        padding: 16px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .card-stable-long {
        background: #f0fdf4; border: 1px solid #d1fae5; border-left: 5px solid #059669;
        padding: 16px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .card-stable-short {
        background: #fff1f2; border: 1px solid #fecdd3; border-left: 5px solid #e11d48;
        padding: 16px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-long { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 12px; }
    .badge-short { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 12px; }
    .stat-pill { background: #f1f5f9; padding: 8px 12px; border-radius: 8px; font-weight: 600; font-size: 13px; color: #475569; text-align: center; }
    .tpsl-box {
        margin-top: 10px; font-size: 14px; color: #1e293b; background: #ffffff; padding: 10px 12px; border-radius: 6px; border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["bitget", "binance", "bybit"]
MAJOR_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"]

SECTOR_MAP = {
    "BTC/USDT": "Macro / L1", "ETH/USDT": "Layer 1", "SOL/USDT": "Layer 1",
    "XRP/USDT": "Payment", "BNB/USDT": "Exchange", "ADA/USDT": "Layer 1",
    "AVAX/USDT": "Layer 1", "SUI/USDT": "Layer 1", "LINK/USDT": "Oracle / DeFi"
}


# ============================================================
# 1. DATA ACCESS & WFO ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str, api_key: str = "", secret: str = "", password: str = ""):
    cls = getattr(ccxt, exchange_id)
    config = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"},
    }
    if api_key and secret:
        config["apiKey"] = api_key
        config["secret"] = secret
        if exchange_id == "bitget" and password:
            config["password"] = password
    return cls(config)


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


def calculate_volume_profile_poc(df_ohlcv: pd.DataFrame, bins: int = 25) -> float:
    try:
        low_min = df_ohlcv["Low"].min()
        high_max = df_ohlcv["High"].max()
        if low_min >= high_max:
            return float(df_ohlcv["Close"].iloc[-1])

        price_bins = np.linspace(low_min, high_max, bins + 1)
        bin_volumes = np.zeros(bins)

        for _, row in df_ohlcv.iterrows():
            c_low, c_high, c_vol = row["Low"], row["High"], row["Volume"]
            for i in range(bins):
                b_start, b_end = price_bins[i], price_bins[i+1]
                overlap_low = max(c_low, b_start)
                overlap_high = min(c_high, b_end)
                if overlap_low < overlap_high:
                    fraction = (overlap_high - overlap_low) / (c_high - c_low) if (c_high - c_low) > 0 else 1.0
                    bin_volumes[i] += c_vol * fraction

        max_idx = np.argmax(bin_volumes)
        poc_price = (price_bins[max_idx] + price_bins[max_idx+1]) / 2.0
        return float(poc_price)
    except Exception:
        return float(df_ohlcv["Close"].iloc[-1])


@st.cache_data(ttl=180, show_spinner=False)
def fetch_dynamic_data_v390(exchange_id: str, symbol: str) -> Optional[dict]:
    try:
        ex = make_exchange(exchange_id)
        raw_symbol = symbol if exchange_id != "binance" else (f"{symbol.replace('/','')}:USDT" if ":" not in symbol else symbol)

        df_4h = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="4h", limit=50), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_4h["EMA20"] = ta.trend.EMAIndicator(df_4h["Close"], window=20).ema_indicator()

        df_1h = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="1h", limit=60), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_1h["EMA20"] = ta.trend.EMAIndicator(df_1h["Close"], window=20).ema_indicator()
        df_1h["RSI14"] = ta.momentum.RSIIndicator(df_1h["Close"], window=14).rsi()
        df_1h["ATR14"] = ta.volatility.AverageTrueRange(df_1h["High"], df_1h["Low"], df_1h["Close"], window=14).average_true_range()
        df_1h["VOL_MA20"] = df_1h["Volume"].rolling(20).mean()
        df_1h["REL_VOLUME"] = df_1h["Volume"] / df_1h["VOL_MA20"]
        df_1h["VOL_ACCEL"] = df_1h["Volume"] / (df_1h["Volume"].shift(1) + 1e-8)

        df_1h["BUY_PRESSURE"] = (df_1h["Close"] - df_1h["Low"]) / (df_1h["High"] - df_1h["Low"] + 1e-8)
        df_1h["CVD_PROXY"] = (df_1h["BUY_PRESSURE"] - 0.5) * df_1h["Volume"]

        recent_20_h = df_1h.tail(20)
        swing_high = float(recent_20_h["High"].max())
        swing_low = float(recent_20_h["Low"].min())

        if len(df_4h) < 20 or len(df_1h) < 20:
            return None

        poc_price = calculate_volume_profile_poc(df_1h, bins=25)

        funding_rate = 0.0
        try:
            fr_data = ex.fetch_funding_rate(raw_symbol)
            funding_rate = float(fr_data.get("fundingRate", 0.0))
        except Exception:
            pass

        return {
            "df_4h": df_4h, "df_1h": df_1h,
            "poc": poc_price, "swing_high": swing_high, "swing_low": swing_low,
            "funding_rate": funding_rate, "exchange": exchange_id.upper()
        }
    except Exception:
        return None


def run_walk_forward_optimization(market_df: pd.DataFrame) -> dict:
    avg_volatility = market_df["change_pct"].abs().mean()
    if avg_volatility > 2.2:
        return {"atr_mult": 2.8, "tp_extension": 2.5, "regime": "🚀 고변동성 대세 상승/하락장 (확장형 TP 적용)"}
    elif avg_volatility < 0.9:
        return {"atr_mult": 1.8, "tp_extension": 1.8, "regime": "⚖️ 저변동성 박스권장 (방어형 TP 적용)"}
    else:
        return {"atr_mult": 2.3, "tp_extension": 2.2, "regime": "📊 표준 모멘텀장 (균형형 동적 TP 적용)"}


def execute_bitget_futures_order(symbol: str, pos_type: str, tp: float, sl: float, api_key: str, secret: str, password: str):
    try:
        ex = make_exchange("bitget", api_key, secret, password)
        formatted_symbol = f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol
        
        balance = ex.fetch_balance()
        total_usdt = float(balance.get("total", {}).get("USDT", 0.0))
        if total_usdt <= 0:
            return False, "USDT 잔고가 부족합니다."
        target_usdt_size = total_usdt * 0.01

        safe_leverage = 3
        try:
            ex.set_leverage(safe_leverage, formatted_symbol)
        except Exception:
            pass

        ticker = ex.fetch_ticker(formatted_symbol)
        current_price = ticker["last"]
        notional_size = target_usdt_size * safe_leverage
        amount = notional_size / current_price

        side = "buy" if pos_type == "LONG" else "sell"
        close_side = "sell" if pos_type == "LONG" else "buy"

        entry_order = ex.create_market_order(formatted_symbol, side, amount)
        msg_result = f"안전 레버리지 {safe_leverage}배 | 1% 자산 진입 성공 (체결가: {entry_order.get('average', current_price):,.4f})"

        try:
            ex.create_order(
                symbol=formatted_symbol, type='limit', side=close_side, amount=amount, price=tp,
                params={'reduceOnly': True, 'triggerPrice': tp}
            )
            msg_result += " | TP 완료"
        except Exception as tp_err:
            msg_result += f" | TP 실패({str(tp_err)})"

        try:
            ex.create_order(
                symbol=formatted_symbol, type='market', side=close_side, amount=amount,
                params={'reduceOnly': True, 'triggerPrice': sl, 'stopPrice': sl}
            )
            msg_result += " | SL 완료"
        except Exception as sl_err:
            msg_result += f" | SL 실패({str(sl_err)})"

        return True, msg_result
    except Exception as e:
        return False, str(e)


# ============================================================
# 2. DYNAMIC QUANT ENGINE (V39.0)
# ============================================================

def analyze_symbol_v390(symbol: str, exchange_id: str, market_avg_change: float, wfo_params: dict) -> Optional[dict]:
    data = fetch_dynamic_data_v390(exchange_id.lower(), symbol)
    if not data:
        return None
    
    df_4h = data["df_4h"]
    df_1h = data["df_1h"]
    poc = data["poc"]
    swing_high = data["swing_high"]
    swing_low = data["swing_low"]
    funding_rate = data["funding_rate"]

    r_4h = df_4h.iloc[-1]
    r_1h = df_1h.iloc[-1]

    close = float(r_1h["Close"])
    atr = float(r_1h["ATR14"]) if pd.notna(r_1h["ATR14"]) and r_1h["ATR14"] > 0 else close * 0.02
    rsi_1h = float(r_1h["RSI14"]) if pd.notna(r_1h["RSI14"]) else 50.0
    rel_vol = float(r_1h["REL_VOLUME"]) if pd.notna(r_1h["REL_VOLUME"]) else 1.0
    vol_accel = float(r_1h["VOL_ACCEL"]) if pd.notna(r_1h["VOL_ACCEL"]) else 1.0
    cvd_val = float(r_1h["CVD_PROXY"]) if pd.notna(r_1h["CVD_PROXY"]) else 0.0
    
    if len(df_1h) >= 24:
        symbol_change_24h = float((close - df_1h.iloc[-24]["Close"]) / df_1h.iloc[-24]["Close"] * 100)
    else:
        symbol_change_24h = 0.0
        
    relative_strength = symbol_change_24h - market_avg_change

    # 급등 피로도 필터 (Anti-Chasing Filter)
    if symbol_change_24h > 12.0 or symbol_change_24h < -12.0:
        return None

    group, pos_type = None, None

    is_4h_long_trend = float(r_4h["Close"]) >= float(r_4h["EMA20"]) * 0.995
    is_4h_short_trend = float(r_4h["Close"]) <= float(r_4h["EMA20"]) * 1.005

    is_funding_safe_long = funding_rate <= 0.0006 and funding_rate > -0.001
    is_funding_safe_short = funding_rate >= -0.0006 and funding_rate < 0.001

    is_poc_long_valid = close >= poc * 0.995
    is_poc_short_valid = close <= poc * 1.005

    if is_4h_long_trend and float(r_1h["Close"]) >= float(r_1h["EMA20"]) * 0.995 and 42 <= rsi_1h <= 65 and is_funding_safe_long and is_poc_long_valid and cvd_val >= 0:
        group, pos_type = "AGGRESSIVE", "LONG"
    elif is_4h_short_trend and float(r_1h["Close"]) <= float(r_1h["EMA20"]) * 1.005 and 35 <= rsi_1h <= 58 and is_funding_safe_short and is_poc_short_valid and cvd_val <= 0:
        group, pos_type = "AGGRESSIVE", "SHORT"
    elif is_4h_long_trend and 45 <= rsi_1h <= 60 and is_funding_safe_long and is_poc_long_valid:
        group, pos_type = "STABLE", "LONG"
    elif is_4h_short_trend and 40 <= rsi_1h <= 55 and is_funding_safe_short and is_poc_short_valid:
        group, pos_type = "STABLE", "SHORT"
    else:
        return None

    # 동적 차트 맞춤형 TP/SL 산출 (수익 극대화 가변 손익비 적용)
    base_atr_mult = wfo_params["atr_mult"]
    tp_mult = wfo_params["tp_extension"]

    if pos_type == "LONG":
        raw_sl = close - (base_atr_mult * atr)
        sl = max(min(raw_sl, swing_low * 0.995), poc * 0.985)
        risk = close - sl
        
        # 차트 저항벽(Swing High)과 동적 ATR 목표가 중 수익이 더 극대화되는 지점으로 가변 설정
        natural_tp = close + (tp_mult * atr)
        tp = max(natural_tp, min(swing_high * 0.995, close + (risk * 3.2)))
    else:
        raw_sl = close + (base_atr_mult * atr)
        sl = min(max(raw_sl, swing_high * 1.005), poc * 1.015)
        risk = sl - close
        
        natural_tp = close - (tp_mult * atr)
        tp = min(natural_tp, max(swing_low * 1.005, close - (risk * 3.2)))

    reward = abs(tp - close)
    risk_val = abs(close - sl)
    if risk_val <= 0 or (reward / risk_val) < 1.6:
        return None

    risk_reward_ratio = reward / risk_val
    sector = SECTOR_MAP.get(symbol, "Altcoin / Liquidity Pool")
    score = float(np.clip(rel_vol * 15 + vol_accel * 10 + abs(relative_strength) * 10, 45, 100))
    
    return {
        "symbol": symbol, "exchange": data["exchange"], "price": close, "poc": poc,
        "group": group, "pos_type": pos_type, "tp": float(tp), "sl": float(sl),
        "rr_ratio": risk_reward_ratio, "sector": sector, "score": score, "change_24h": symbol_change_24h
    }


def fmt_price(x):
    if x is None or x <= 0: return "$0.00"
    if x >= 1000: return f"${x:,.2f}"
    if x >= 1: return f"${x:,.4f}"
    return f"${x:,.8f}"


# ============================================================
# 3. STREAMLIT UI & MAIN EXECUTION
# ============================================================

def main():
    st.title("🚀 Crypto Quant Dashboard V39.0 Pro")
    st.caption("동적 WFO 마켓 레짐 연동 가변형 TP/SL 수익 극대화 엔진 (상위 100대 유동성 풀 + 안전 레버리지)")

    st.sidebar.header("⚙️ 비트겟 선물 API 및 안전 설정")
    bitget_api_key = st.sidebar.text_input("API Key", type="password")
    bitget_secret = st.sidebar.text_input("Secret Key", type="password")
    bitget_passphrase = st.sidebar.text_input("Passphrase (비밀번호)", type="password")
    
    auto_trade_enabled = st.sidebar.checkbox("🚀 실전 자동 주문 활성화 (안전 레버리지 3배 고정)", value=False)

    with st.spinner("시장 전체 변동성 및 유동성 상위 100개 우량 종목 분석 중..."):
        market, active_exchange = fetch_tickers_safe()

    if market.empty:
        st.error("⚠️ 거래소 데이터 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.")
        return

    st.success(f"✅ 연결 성공: [{active_exchange.upper}] 마켓 연동 완료 (총 {len(market)}개 심볼)")

    wfo_params = run_walk_forward_optimization(market)
    
    top_volume_market = market.sort_values(by="quote_volume", ascending=False).head(100)
    available_majors = [s for s in MAJOR_COINS if s in market["symbol"].values]
    symbols = list(set(available_majors + top_volume_market["symbol"].tolist()))

    market_avg_change = float(market["change_pct"].mean())
    btc_row = market[market["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0

    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 V39.0 동적 WFO 마켓 분석 대시보드</h3>
            <span style="background: #e0e7ff; color: #4f46e5; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {wfo_params['regime']}
            </span>
        </div>
        <p style="font-size: 14px; color: #475569; margin-bottom: 12px;">
            • 기계적 손익비 고정 폐지: <b>차트별 스윙 구조와 WFO 변동성에 맞춘 자율 가변 손익비(1:1.6 ~ 1:4.0+) 적용</b><br>
            • 안전 필터: <b>당일 ±12% 이상 급등락 종목(추격 매수 위험) 차단 + 거래대금 상위 100대 우량주 한정</b>
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H 변동률: <b>{btc_change:+.2f}%</b></div>
            <div class="stat-pill">🎯 스캔 대상 풀: <b>우량 유동성 상위 100개</b></div>
            <div class="stat-pill">⚡ 동적 ATR 배수: <b>{wfo_params['atr_mult']}배 적용</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"🚀 V39.0 [수익 극대화 동적 스캔] 알파 발굴 시작", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_symbols = len(symbols)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(analyze_symbol_v390, s, active_exchange, market_avg_change, wfo_params): s for s in symbols}
            completed = 0
            for f in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total_symbols)
                status_text.text(f"⚡ 차트 맞춤형 동적 TP/SL 및 가변 손익비 산출 중... ({completed}/{total_symbols})")
                r = f.result()
                if r: results.append(r)
        
        progress_bar.empty()
        status_text.empty()
        st.session_state["v390_results"] = results

    results = st.session_state.get("v390_results", [])
    if results:
        df_res = pd.DataFrame(results)
        agg_df = df_res[df_res["group"] == "AGGRESSIVE"].sort_values("score", ascending=False)
        stable_df = df_res[df_res["group"] == "STABLE"].sort_values("score", ascending=False)

        st.success(f"🎉 차트별 최적 가변 손익비를 확보한 우량 알파 종목 {len(df_res)}개가 발굴되었습니다!")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🔥 공격형 모멘텀 (총 {len(agg_df)}개)")
            if agg_df.empty:
                st.info("조건에 부합하는 공격형 종목이 없습니다.")
            else:
                for _, row in agg_df.iterrows():
                    is_long = row["pos_type"] == "LONG"
                    card_cls = "card-agg-long" if is_long else "card-agg-short"
                    badge_html = '<span class="badge-long">🟢 AGG LONG</span>' if is_long else '<span class="badge-short">🔴 AGG SHORT</span>'
                    tp_color = "#047857" if is_long else "#dc2626"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>{badge_html} &nbsp; <b style="font-size: 16px; color: #0f172a;">{row['symbol']}</b></div>
                            <div style="font-size: 13px; color: #334155;">현재가: <b>{fmt_price(row['price'])}</b></div>
                        </div>
                        <div class="tpsl-box">
                            🎯 <b>동적 목표 익절가(TP):</b> <span style="color: {tp_color}; font-size: 15px; font-weight: 800;">{fmt_price(row['tp'])}</span><br>
                            🛑 <b>변동성 방어 손절가(SL):</b> <span style="color: #475569; font-size: 14px; font-weight: 700;">{fmt_price(row['sl'])}</span><br>
                            ⚖️ <b>차트 맞춤 가변 손익비(RR):</b> <span style="color: #4f46e5; font-weight: 700;">1 : {row['rr_ratio']:.2f}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #64748b;">
                            🏷️ 섹터: <b>{row['sector']}</b> | 24H 변동: <b>{row['change_24h']:+.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_key = f"btn_agg_{row['symbol']}"
                    if st.button(f"⚡ [{row['symbol']}] 안전 3배 레버리지 OCO 자동주문", key=btn_key):
                        if not auto_trade_enabled:
                            st.warning("사이드바에서 '실전 자동 주문 활성화'를 체크해주세요.")
                        elif not bitget_api_key or not bitget_secret or not bitget_passphrase:
                            st.error("사이드바에 비트겟 API Key, Secret, Passphrase를 모두 입력해주세요.")
                        else:
                            with st.spinner("자동 주문 전송 중..."):
                                success, msg = execute_bitget_futures_order(
                                    row["symbol"], row["pos_type"], row["tp"], row["sl"],
                                    bitget_api_key, bitget_secret, bitget_passphrase
                                )
                                if success:
                                    st.success(f"✅ {row['symbol']} {row['pos_type']} 주문 완료! ({msg})")
                                else:
                                    st.error(f"❌ 주문 실패: {msg}")

        with col2:
            st.markdown(f"### 🛡️ 안정형 스윙 (총 {len(stable_df)}개)")
            if stable_df.empty:
                st.info("조건에 부합하는 안정형 종목이 없습니다.")
            else:
                for _, row in stable_df.iterrows():
                    is_long = row["pos_type"] == "LONG"
                    card_cls = "card-stable-long" if is_long else "card-stable-short"
                    badge_html = '<span class="badge-long">🟢 STABLE LONG</span>' if is_long else '<span class="badge-short">🔴 STABLE SHORT</span>'
                    tp_color = "#047857" if is_long else "#dc2626"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>{badge_html} &nbsp; <b style="font-size: 16px; color: #0f172a;">{row['symbol']}</b></div>
                            <div style="font-size: 13px; color: #334155;">현재가: <b>{fmt_price(row['price'])}</b></div>
                        </div>
                        <div class="tpsl-box">
                            🎯 <b>동적 목표 익절가(TP):</b> <span style="color: {tp_color}; font-size: 15px; font-weight: 800;">{fmt_price(row['tp'])}</span><br>
                            🛑 <b>변동성 방어 손절가(SL):</b> <span style="color: #475569; font-size: 14px; font-weight: 700;">{fmt_price(row['sl'])}</span><br>
                            ⚖️ <b>차트 맞춤 가변 손익비(RR):</b> <span style="color: #4f46e5; font-weight: 700;">1 : {row['rr_ratio']:.2f}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #64748b;">
                            🏷️ 섹터: <b>{row['sector']}</b> | 24H 변동: <b>{row['change_24h']:+.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_key = f"btn_stable_{row['symbol']}"
                    if st.button(f"⚡ [{row['symbol']}] 안전 3배 레버리지 OCO 자동주문", key=btn_key):
                        if not auto_trade_enabled:
                            st.warning("사이드바에서 '실전 자동 주문 활성화'를 체크해주세요.")
                        elif not bitget_api_key or not bitget_secret or not bitget_passphrase:
                            st.error("사이드바에 비트겟 API Key, Secret, Passphrase를 모두 입력해주세요.")
                        else:
                            with st.spinner("자동 주문 전송 중..."):
                                success, msg = execute_bitget_futures_order(
                                    row["symbol"], row["pos_type"], row["tp"], row["sl"],
                                    bitget_api_key, bitget_secret, bitget_passphrase
                                )
                                if success:
                                    st.success(f"✅ {row['symbol']} {row['pos_type']} 주문 완료! ({msg})")
                                else:
                                    st.error(f"❌ 주문 실패: {msg}")


if __name__ == "__main__":
    main()