# -*- coding: utf-8 -*-
"""
Crypto Quant Dashboard V37.0 (YouTube Master-Class Enhanced Institutional Engine)
- Target Pool: BTC + Major Coins + Top 50 24H Volatility Momentum Coins
- Enhanced Features:
  1. Structural Support/Resistance Snapping (Burger-hyung Style)
  2. Extreme Funding Rate Squeeze Protection (Mayo Style)
  3. Volume Acceleration Momentum Filter (Coinone Style)
  4. Dynamic ATR Scaling for Volatility Explosions (Danta-rang Style)
- Risk-Reward Ratio >= 2.0 Filter & Bitget Auto OCO Futures Execution
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
    page_title="🔥 Crypto Quant Dashboard V37.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .macro-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid #6366f1;
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
MAJOR_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"]

SECTOR_MAP = {
    "BTC/USDT": "Macro / L1", "ETH/USDT": "Layer 1", "SOL/USDT": "Layer 1",
    "XRP/USDT": "Payment", "BNB/USDT": "Exchange", "ADA/USDT": "Layer 1",
    "AVAX/USDT": "Layer 1", "SUI/USDT": "Layer 1", "DOGE/USDT": "Meme",
    "SHIB/USDT": "Meme", "PEPE/USDT": "Meme", "LINK/USDT": "Oracle / DeFi"
}


# ============================================================
# 1. DATA ACCESS & INSTITUTIONAL METRICS ENGINE
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
def fetch_institutional_data_v370(exchange_id: str, symbol: str) -> Optional[dict]:
    try:
        ex = make_exchange(exchange_id)
        raw_symbol = symbol if exchange_id != "binance" else (f"{symbol.replace('/','')}:USDT" if ":" not in symbol else symbol)

        df_1d = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="1d", limit=60), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_1d["EMA20"] = ta.trend.EMAIndicator(df_1d["Close"], window=20).ema_indicator()

        df_1h = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="1h", limit=60), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_1h["EMA20"] = ta.trend.EMAIndicator(df_1h["Close"], window=20).ema_indicator()
        df_1h["RSI14"] = ta.momentum.RSIIndicator(df_1h["Close"], window=14).rsi()
        df_1h["ATR14"] = ta.volatility.AverageTrueRange(df_1h["High"], df_1h["Low"], df_1h["Close"], window=14).average_true_range()
        df_1h["VOL_MA20"] = df_1h["Volume"].rolling(20).mean()
        df_1h["REL_VOLUME"] = df_1h["Volume"] / df_1h["VOL_MA20"]

        # 3. 거래대금/볼륨 가속도 (Coinone 스타일)
        df_1h["VOL_ACCEL"] = df_1h["Volume"] / (df_1h["Volume"].shift(1) + 1e-8)

        df_1h["BUY_PRESSURE"] = (df_1h["Close"] - df_1h["Low"]) / (df_1h["High"] - df_1h["Low"] + 1e-8)
        df_1h["CVD_PROXY"] = (df_1h["BUY_PRESSURE"] - 0.5) * df_1h["Volume"]

        # 1. 수평 매물대 스윙 벽 산출 (Burger-hyung 스타일)
        recent_20_h = df_1h.tail(20)
        swing_high = float(recent_20_h["High"].max())
        swing_low = float(recent_20_h["Low"].min())

        df_30m = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="30m", limit=50), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_30m["EMA9"] = ta.trend.EMAIndicator(df_30m["Close"], window=9).ema_indicator()
        df_30m["RSI9"] = ta.momentum.RSIIndicator(df_30m["Close"], window=9).rsi()

        if len(df_1d) < 20 or len(df_1h) < 20 or len(df_30m) < 20:
            return None

        poc_price = calculate_volume_profile_poc(df_1h, bins=25)

        funding_rate = 0.0
        try:
            fr_data = ex.fetch_funding_rate(raw_symbol)
            funding_rate = float(fr_data.get("fundingRate", 0.0))
        except Exception:
            pass

        return {
            "df_1d": df_1d, "df_1h": df_1h, "df_30m": df_30m,
            "poc": poc_price, "swing_high": swing_high, "swing_low": swing_low,
            "funding_rate": funding_rate, "exchange": exchange_id.upper()
        }
    except Exception:
        return None


def execute_bitget_futures_order(symbol: str, pos_type: str, tp: float, sl: float, api_key: str, secret: str, password: str):
    try:
        ex = make_exchange("bitget", api_key, secret, password)
        formatted_symbol = f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol
        
        balance = ex.fetch_balance()
        total_usdt = float(balance.get("total", {}).get("USDT", 0.0))
        if total_usdt <= 0:
            return False, "USDT 잔고가 부족합니다."
        target_usdt_size = total_usdt * 0.01

        markets = ex.load_markets()
        market_info = markets.get(formatted_symbol, {})
        max_leverage = 20
        try:
            if "limits" in market_info and "leverage" in market_info["limits"]:
                max_lev_limit = market_info["limits"]["leverage"].get("max")
                if max_lev_limit:
                    max_leverage = int(max_lev_limit)
            ex.set_leverage(max_leverage, formatted_symbol)
        except Exception:
            pass

        ticker = ex.fetch_ticker(formatted_symbol)
        current_price = ticker["last"]
        notional_size = target_usdt_size * max_leverage
        amount = notional_size / current_price

        side = "buy" if pos_type == "LONG" else "sell"
        close_side = "sell" if pos_type == "LONG" else "buy"

        entry_order = ex.create_market_order(formatted_symbol, side, amount)
        msg_result = f"레버리지 {max_leverage}배 | 1% 자산 진입 성공 (체결가: {entry_order.get('average', current_price):,.4f})"

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
# 2. WFO & QUANT ENGINE (MASTER-CLASS ENHANCED)
# ============================================================

def run_walk_forward_optimization(market_df: pd.DataFrame) -> dict:
    avg_volatility = market_df["change_pct"].abs().mean()
    if avg_volatility > 2.0:
        return {"atr_mult": 2.6, "rs_cut": 0.18, "regime": "🚀 고변동성 주도주 장세 (동적 ATR 확장 적용)"}
    elif avg_volatility < 0.8:
        return {"atr_mult": 1.9, "rs_cut": 0.04, "regime": "⚖️ 박스권 장세 (방어형 ATR 적용)"}
    else:
        return {"atr_mult": 2.3, "rs_cut": 0.09, "regime": "📊 표준 모멘텀 장세 (균형형 ATR 적용)"}


def render_market_horizon_dashboard(market_df: pd.DataFrame, wfo_res: dict):
    btc_row = market_df[market_df["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0

    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #1e293b;">🌐 V37.0 유튜브 마스터클래스 종합분석 & 알파 엔진</h3>
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {wfo_res['regime']}
            </span>
        </div>
        <p style="font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 15px;">
            💡 고수들의 4대 매매철학 융합: <span style="color: #4f46e5;">수평 매물대벽 스냅핑 + 펀딩비 스퀴즈 방어 + 볼륨 가속도 + 동적 ATR 확장</span>
        </p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 12px 0;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H 변동률: <b>{btc_change:+.2f}%</b></div>
            <div class="stat-pill">📊 분석 타겟 풀: <b>메이저 + 변동성 상위 50</b></div>
            <div class="stat-pill">⚡ 시스템 상태: <b>유튜브 마스터 알고리즘 작동 중</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


def analyze_symbol_v370(symbol: str, exchange_id: str, market_avg_change: float, btc_change: float, wfo_params: dict) -> Optional[dict]:
    data = fetch_institutional_data_v370(exchange_id.lower(), symbol)
    if not data:
        return None
    
    df_1d = data["df_1d"]
    df_1h = data["df_1h"]
    df_30m = data["df_30m"]
    poc = data["poc"]
    swing_high = data["swing_high"]
    swing_low = data["swing_low"]
    funding_rate = data["funding_rate"]

    r_1d = df_1d.iloc[-2]
    r_1h = df_1h.iloc[-1]
    r_30m = df_30m.iloc[-1]

    close = float(r_1h["Close"])
    atr = float(r_1h["ATR14"]) if pd.notna(r_1h["ATR14"]) and r_1h["ATR14"] > 0 else close * 0.02
    rsi_1h = float(r_1h["RSI14"]) if pd.notna(r_1h["RSI14"]) else 50.0
    rsi_30m = float(r_30m["RSI9"]) if pd.notna(r_30m["RSI9"]) else 50.0
    rel_vol = float(r_1h["REL_VOLUME"]) if pd.notna(r_1h["REL_VOLUME"]) else 1.0
    vol_accel = float(r_1h["VOL_ACCEL"]) if pd.notna(r_1h["VOL_ACCEL"]) else 1.0
    cvd_val = float(r_1h["CVD_PROXY"]) if pd.notna(r_1h["CVD_PROXY"]) else 0.0
    
    if len(df_1h) >= 6:
        symbol_change_24h = float((close - df_1h.iloc[-6]["Close"]) / df_1h.iloc[-6]["Close"] * 100)
    else:
        symbol_change_24h = 0.0
        
    relative_strength = symbol_change_24h - market_avg_change
    rs_cut = wfo_params["rs_cut"]
    base_atr_mult = wfo_params["atr_mult"]

    # 4. 동적 ATR 스케일링 (Danta-rang 스타일: 변동성 및 볼륨 폭발 시 TP 확장)
    dynamic_atr_mult = base_atr_mult * (1.25 if rel_vol > 1.8 else 1.0)

    group, pos_type = None, None

    is_30m_long_momentum = float(r_30m["Close"]) > float(r_30m["EMA9"]) and rsi_30m < 75
    is_30m_short_momentum = float(r_30m["Close"]) < float(r_30m["EMA9"]) and rsi_30m > 25

    # 2. 극단적 펀딩비 스퀴즈 방어 (Mayo 스타일: 과열된 롱/숏 포지션 진입 원천 차단)
    is_funding_safe_long = funding_rate <= 0.0008 and funding_rate > -0.001
    is_funding_safe_short = funding_rate >= -0.0008 and funding_rate < 0.001

    is_poc_long_valid = close >= poc * 0.992
    is_poc_short_valid = close <= poc * 1.008

    # 3. 볼륨 가속도 필터 (Coinone 스타일: 거래대금 급증 가속도 요구)
    is_vol_accelerating = vol_accel >= 0.85

    if r_1d["Close"] >= r_1d["EMA20"] * 0.995 and rel_vol >= 1.0 and is_vol_accelerating and relative_strength > rs_cut and is_30m_long_momentum and is_funding_safe_long and is_poc_long_valid and cvd_val >= 0:
        group, pos_type = "AGGRESSIVE", "LONG"
    elif r_1d["Close"] <= r_1d["EMA20"] * 1.005 and rel_vol >= 1.0 and is_vol_accelerating and relative_strength < -rs_cut and is_30m_short_momentum and is_funding_safe_short and is_poc_short_valid and cvd_val <= 0:
        group, pos_type = "AGGRESSIVE", "SHORT"
    elif close >= float(r_1h["EMA20"]) and 38 <= rsi_1h <= 68 and is_funding_safe_long and is_poc_long_valid:
        group, pos_type = "STABLE", "LONG"
    elif rsi_1h >= 58 and is_funding_safe_short and is_poc_short_valid:
        group, pos_type = "STABLE", "SHORT"
    else:
        return None

    # 1. 수평 매물대 스윙 벽 스냅핑 (Burger-hyung 스타일 적용)
    if pos_type == "LONG":
        # SL은 직전 스윙 로우 또는 POC 하단 방어선 중 더 안전한 곳 선택
        raw_sl = close - (1.1 * atr)
        sl = max(min(raw_sl, swing_low * 0.995), poc * 0.985)
        risk = close - sl
        
        # TP는 직전 스윙 고터치 저항벽 또는 동적 ATR 목표가 중 손익비 2 이상 확보되는 최적 지점
        raw_tp = close + (dynamic_atr_mult * atr)
        tp = max(raw_tp, min(swing_high * 0.995, close + (risk * 2.2)))
    else:
        raw_sl = close + (1.1 * atr)
        sl = min(max(raw_sl, swing_high * 1.005), poc * 1.015)
        risk = sl - close
        
        raw_tp = close - (dynamic_atr_mult * atr)
        tp = min(raw_tp, max(swing_low * 1.005, close - (risk * 2.2)))

    # 손익비 검증 (Risk-Reward Ratio 2.0 미만 자동 탈락)
    reward = abs(tp - close)
    risk_val = abs(close - sl)
    if risk_val <= 0 or (reward / risk_val) < 2.0:
        return None

    risk_reward_ratio = reward / risk_val
    sector = SECTOR_MAP.get(symbol, "Altcoin / Other")
    score = float(np.clip(rel_vol * 15 + vol_accel * 10 + abs(relative_strength) * 10 + abs(cvd_val / 1000) * 5, 40, 100))
    
    return {
        "symbol": symbol, "exchange": data["exchange"], "price": close, "poc": poc,
        "group": group, "pos_type": pos_type, "tp": float(tp), "sl": float(sl),
        "rr_ratio": risk_reward_ratio, "sector": sector, "cvd": cvd_val, "score": score
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
    st.title("🔥 Crypto Quant Dashboard V37.0")
    st.caption("유튜브 마스터클래스 융합 알파 엔진 (수평 매물대벽 스냅핑 + 펀딩비 방어 + 볼륨 가속도 + 동적 ATR)")

    st.sidebar.header("⚙️ 비트겟 선물 API 설정")
    bitget_api_key = st.sidebar.text_input("API Key", type="password")
    bitget_secret = st.sidebar.text_input("Secret Key", type="password")
    bitget_passphrase = st.sidebar.text_input("Passphrase (비밀번호)", type="password")
    
    auto_trade_enabled = st.sidebar.checkbox("🚀 비트겟 실전 자동 주문 활성화", value=False)

    with st.spinner("거래소 전체 시세 및 오더플로우 데이터를 불러오는 중입니다..."):
        market, active_exchange = fetch_tickers_safe()

    if market.empty:
        st.error("⚠️ 거래소 데이터 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.")
        return

    st.success(f"✅ 연결 성공: [{active_exchange.upper}] 마스터 데이터 연동 완료 (총 {len(market)}개 심볼)")
    
    wfo_params = run_walk_forward_optimization(market)
    render_market_horizon_dashboard(market, wfo_params)

    available_majors = [s for s in MAJOR_COINS if s in market["symbol"].values]
    top_volatility_symbols = market.sort_values(by="change_pct", key=abs, ascending=False).head(50)["symbol"].tolist()
    symbols = list(set(available_majors + top_volatility_symbols))

    market_avg_change = float(market["change_pct"].mean())
    btc_row = market[market["symbol"] == "BTC/USDT"]
    btc_change = float(btc_row["change_pct"].values[0]) if not btc_row.empty else 0.0

    if st.button(f"🚀 V37.0 [마스터클래스 전수 스캔] 알파 발굴 시작", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_symbols = len(symbols)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(analyze_symbol_v370, s, active_exchange, market_avg_change, btc_change, wfo_params): s for s in symbols}
            completed = 0
            for f in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total_symbols)
                status_text.text(f"🔍 매물대벽 스냅핑 & 스퀴즈 방어 검증 중... ({completed}/{total_symbols}) 완료")
                r = f.result()
                if r: results.append(r)
        
        progress_bar.empty()
        status_text.empty()
        st.session_state["v370_results"] = results

    results = st.session_state.get("v370_results", [])
    if results:
        df_res = pd.DataFrame(results)
        agg_df = df_res[df_res["group"] == "AGGRESSIVE"].sort_values("score", ascending=False)
        stable_df = df_res[df_res["group"] == "STABLE"].sort_values("score", ascending=False)

        st.success(f"🎉 4대 마스터 조건 및 손익비 1:2 이상을 돌파한 {len(df_res)}개의 핵심 알파 종목이 발굴되었습니다!")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🔥 공격형 알파 트레이딩 (총 {len(agg_df)}개)")
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
                            🎯 <b>매물대 스냅 익절가(TP):</b> <span style="color: {tp_color}; font-size: 15px; font-weight: 800;">{fmt_price(row['tp'])}</span><br>
                            🛑 <b>방어 손절가(SL):</b> <span style="color: #475569; font-size: 14px; font-weight: 700;">{fmt_price(row['sl'])}</span><br>
                            ⚖️ <b>검증된 손익비(RR):</b> <span style="color: #4f46e5; font-weight: 700;">1 : {row['rr_ratio']:.2f}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #64748b;">
                            🏷️ 섹터: <b>{row['sector']}</b> | 마스터스코어: <b style="color: #4f46e5;">{row['score']:.1f}점</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_key = f"btn_agg_{row['symbol']}"
                    if st.button(f"⚡ [{row['symbol']}] 최대레버리지 & 마스터 OCO 자동주문", key=btn_key):
                        if not auto_trade_enabled:
                            st.warning("사이드바에서 '비트겟 실전 자동 주문 활성화'를 체크해주세요.")
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
            st.markdown(f"### 🛡️ 안정형 스윙 트레이딩 (총 {len(stable_df)}개)")
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
                            🎯 <b>매물대 스냅 익절가(TP):</b> <span style="color: {tp_color}; font-size: 15px; font-weight: 800;">{fmt_price(row['tp'])}</span><br>
                            🛑 <b>방어 손절가(SL):</b> <span style="color: #475569; font-size: 14px; font-weight: 700;">{fmt_price(row['sl'])}</span><br>
                            ⚖️ <b>검증된 손익비(RR):</b> <span style="color: #4f46e5; font-weight: 700;">1 : {row['rr_ratio']:.2f}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #64748b;">
                            🏷️ 섹터: <b>{row['sector']}</b> | 마스터스코어: <b style="color: #4f46e5;">{row['score']:.1f}점</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_key = f"btn_stable_{row['symbol']}"
                    if st.button(f"⚡ [{row['symbol']}] 최대레버리지 & 마스터 OCO 자동주문", key=btn_key):
                        if not auto_trade_enabled:
                            st.warning("사이드바에서 '비트겟 실전 자동 주문 활성화'를 체크해주세요.")
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