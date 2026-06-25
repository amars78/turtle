import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

# --- pykrx 예외 처리 및 로드 ---
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# --- 페이지 설정 ---
st.set_page_config(page_title="CAN SLIM x 터틀 실전 매니저", layout="wide")
st.title("🦅 CAN SLIM x 🐢 터틀 트레이딩 실전 자산 매니저")
st.markdown("""
* **국내 주식**: `.KS`를 붙일 필요 없이 **6자리 숫자**만 입력하고 빈 곳을 누르세요. (예: `005930`, `000660`)
* **해외 주식**: 기존처럼 영문 티커를 입력하세요. (예: `AAPL`, `TSLA`)
""")

# =========================================================
# 티커 정제 및 종목명 조회 함수
# =========================================================
@st.cache_data(ttl=86400)
def resolve_ticker_and_name(raw_ticker: str) -> tuple:
    """
    - 6자리 숫자: KRX/yfinance로 종목명 + .KS/.KQ 접미사 반환
    - 그 외: yfinance로 영문 종목명 반환
    반환: (sanitized_ticker, display_name)
    """
    raw_ticker = str(raw_ticker).strip().upper()
    if not raw_ticker or raw_ticker in ("NAN", ""):
        return "", ""

    # 국내 주식 (6자리 숫자)
    if raw_ticker.isdigit() and len(raw_ticker) == 6:
        if PYKRX_AVAILABLE:
            try:
                name = krx.get_market_ticker_name(raw_ticker)
                if name and name.strip() and name != raw_ticker:
                    try:
                        kospi_list = krx.get_market_ticker_list(market="KOSPI")
                        suffix = ".KS" if raw_ticker in kospi_list else ".KQ"
                    except Exception:
                        suffix = ".KS"
                    return f"{raw_ticker}{suffix}", name
            except Exception:
                pass

        # pykrx 실패 시 yfinance 폴백
        for suffix in [".KS", ".KQ"]:
            test_ticker = f"{raw_ticker}{suffix}"
            try:
                tk = yf.Ticker(test_ticker)
                info = tk.fast_info  # fast_info: 불필요한 API 호출 최소화
                # fast_info에는 shortName이 없으므로 info 사용, 단 타임아웃 방어
                full_info = tk.info
                name = full_info.get("shortName") or full_info.get("longName")
                if name:
                    return test_ticker, name
            except Exception:
                pass
        return f"{raw_ticker}.KS", raw_ticker

    # 이미 접미사가 붙은 국내 주식 (.KS / .KQ)
    if raw_ticker.endswith(".KS") or raw_ticker.endswith(".KQ"):
        pure_code = raw_ticker.split(".")[0]
        if PYKRX_AVAILABLE:
            try:
                name = krx.get_market_ticker_name(pure_code)
                if name and name.strip() and name != pure_code:
                    return raw_ticker, name
            except Exception:
                pass
        try:
            tk = yf.Ticker(raw_ticker)
            info = tk.info
            name = info.get("shortName") or info.get("longName")
            if name:
                return raw_ticker, name
        except Exception:
            pass
        return raw_ticker, pure_code

    # 해외 주식 (영문 티커)
    try:
        tk = yf.Ticker(raw_ticker)
        info = tk.info
        name = info.get("shortName") or info.get("longName")
        if name:
            return raw_ticker, name
    except Exception:
        pass
    return raw_ticker, raw_ticker


# =========================================================
# 금융 데이터 연산 함수들
# =========================================================
def get_benchmark_ticker(ticker: str) -> str:
    if ticker.endswith(".KS"):
        return "^KS11"
    elif ticker.endswith(".KQ"):
        return "^KQ11"
    return "^GSPC"


@st.cache_data(ttl=3600)
def load_price_history(ticker: str, days: int = 500) -> Optional[pd.DataFrame]:
    start = datetime.today() - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start, end=datetime.today(), progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_market_trend(benchmark_ticker: str) -> dict:
    df = load_price_history(benchmark_ticker, days=400)
    if df is None or len(df) < 200:
        return {"status": "확인불가", "detail": "데이터 부족"}
    close = df["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200_series = close.rolling(200).mean()
    sma200 = sma200_series.iloc[-1]
    sma200_prev = sma200_series.iloc[-20] if len(df) > 220 else sma200
    cur = close.iloc[-1]

    if cur > sma50 and cur > sma200 and sma200 > sma200_prev:
        return {"status": "🟢 상승추세", "detail": "지수가 50/200일선 위, 200일선 상승 중"}
    elif cur < sma50 and cur < sma200 and sma200 < sma200_prev:
        return {"status": "🔴 하락추세", "detail": "지수가 이평선 아래, 200일선 하락 중"}
    return {"status": "🟡 중립/전환구간", "detail": "이평선 신호 혼재"}


def compute_rs_raw(df: pd.DataFrame) -> Optional[float]:
    close = df["Close"].dropna()
    n = len(close)
    if n < 253:
        return None
    # 인덱스 범위 안전 처리
    def safe_r(e_idx: int, s_idx: int) -> float:
        e = close.iloc[e_idx]
        s = close.iloc[s_idx]
        return float((e / s) - 1.0) if s != 0 else 0.0

    try:
        return 2 * safe_r(-1, -64) + safe_r(-64, -127) + safe_r(-127, -190) + safe_r(-190, -253)
    except IndexError:
        return None


def rs_rating_from_raw(raw_scores: dict) -> dict:
    valid = {k: v for k, v in raw_scores.items() if v is not None}
    if len(valid) <= 1:
        return {k: (50 if v is not None else None) for k, v in raw_scores.items()}
    values = sorted(valid.values())
    result = {}
    for k, v in raw_scores.items():
        if v is None:
            result[k] = None
        else:
            percentile = sum(1 for x in values if x <= v) / len(values)
            result[k] = max(1, min(99, round(percentile * 98) + 1))
    return result


def compute_volume_signal(df: pd.DataFrame) -> dict:
    recent = df.tail(50).copy()
    recent["change"] = recent["Close"].diff()
    up_vol = recent.loc[recent["change"] > 0, "Volume"].sum()
    down_vol = recent.loc[recent["change"] < 0, "Volume"].sum()
    ratio = float(up_vol / down_vol) if down_vol > 0 else 1.0
    if ratio >= 1.2:
        signal = "🟢 매집"
    elif ratio <= 0.8:
        signal = "🔴 분산"
    else:
        signal = "🟡 중립"
    return {"ratio": round(ratio, 2), "signal": signal}


@st.cache_data(ttl=3600)
def load_and_process_data(ticker: str, entry_window: int, exit_window: int) -> Optional[pd.DataFrame]:
    df = load_price_history(ticker, days=500)
    if df is None or df.empty:
        return None
    try:
        df = df.copy()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_150"] = df["Close"].rolling(150).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        df["52W_High"] = df["High"].rolling(252).max()
        df["52W_Low"] = df["Low"].rolling(252).min()
        df["SMA_200_Trend"] = df["SMA_200"] > df["SMA_200"].shift(20)

        # 터틀 진입/청산선: 전일 기준 (shift(1) 적용)
        df["Entry_High"] = df["High"].rolling(entry_window).max().shift(1)
        df["Exit_Low"] = df["Low"].rolling(exit_window).min().shift(1)

        # ATR(20) 계산
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift(1)).abs()
        lc = (df["Low"] - df["Close"].shift(1)).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(20).mean()

        return df
    except Exception as e:
        st.warning(f"데이터 처리 오류 ({ticker}): {e}")
        return None


# =========================================================
# 사이드바 설정
# =========================================================
st.sidebar.header("⚙️ 시스템 및 자금 관리 설정")
system_type = st.sidebar.radio("터틀 시스템 선택", ("시스템 1 (20일 돌파)", "시스템 2 (55일 돌파)"))
entry_window, exit_window = (20, 10) if system_type == "시스템 1 (20일 돌파)" else (55, 20)
account_size = st.sidebar.number_input("총 투자 자본금", value=10_000_000, step=1_000_000, min_value=1_000_000)
risk_per_trade = st.sidebar.slider("1유닛 리스크 비율 (%)", 0.5, 5.0, 1.0, 0.1) / 100
max_units = st.sidebar.slider("최대 피라미딩 유닛 수", 1, 4, 4)
apply_market_filter = st.sidebar.checkbox("시장이 하락추세면 매수등급 자동 하향", value=True)


# =========================================================
# 세션 상태 초기화
# =========================================================
if "active_positions" not in st.session_state:
    st.session_state.active_positions = pd.DataFrame([
        {"티커": "005930.KS", "종목명": "삼성전자", "실제최초매수가": 75000.0, "현재보유유닛": 1},
        {"티커": "000660.KS", "종목명": "SK하이닉스", "실제최초매수가": 180000.0, "현재보유유닛": 2},
        {"티커": "066570.KS", "종목명": "LG전자", "실제최초매수가": 100000.0, "현재보유유닛": 1},
    ])

# 종목명 캐시: 세션 전역 딕셔너리로 관리 (API 중복 호출 방지)
if "ticker_name_cache" not in st.session_state:
    st.session_state.ticker_name_cache = {}


def get_cached_name(raw_ticker: str) -> tuple:
    """세션 캐시 → @cache_data 순서로 조회"""
    key = str(raw_ticker).strip().upper()
    if key in st.session_state.ticker_name_cache:
        return st.session_state.ticker_name_cache[key]
    result = resolve_ticker_and_name(key)
    st.session_state.ticker_name_cache[key] = result
    return result


# =========================================================
# 탭 구성
# =========================================================
tab0, tab1, tab2 = st.tabs([
    "🔥 1. 실전 보유 포지션 관리",
    "📊 2. CAN SLIM 관심종목 스캐너",
    "📈 3. 개별 종목 융합 차트",
])

# =========================================================
# 탭 0: 실전 보유 포지션 관리
# =========================================================
with tab0:
    st.subheader("🛠 보유 포지션 입력 및 편집")
    st.caption("💡 국내 주식은 숫자(6자리)만 입력 후 다른 셀을 클릭하면 자동으로 티커 및 종목명이 완성됩니다.")

    # ── data_editor ──────────────────────────────────────
    edited_df = st.data_editor(
        st.session_state.active_positions,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "종목명": st.column_config.TextColumn("종목명 (자동 완성)", disabled=True),
            "티커": st.column_config.TextColumn("티커 (숫자6자리 또는 해외영문)", required=True),
            "실제최초매수가": st.column_config.NumberColumn("최초 매수가", required=True, min_value=0.0, format="%.2f"),
            "현재보유유닛": st.column_config.NumberColumn("현재 유닛 수", required=True, min_value=1, max_value=4, default=1),
        },
        key="position_editor",
    )

    # ── 티커 자동 정제 & 종목명 채우기 ──────────────────
    # 변경 감지: 편집 전후 티커 목록 비교
    needs_rerun = False
    updated_rows = []

    for idx, row in edited_df.iterrows():
        raw_t = str(row.get("티커", "")).strip().upper()
        current_name = str(row.get("종목명", "")).strip()

        if not raw_t or raw_t == "NAN":
            updated_rows.append(row.to_dict())
            continue

        # 재조회 필요 조건:
        #   1) 순수 6자리 숫자 (접미사 없음) → 아직 변환 안 된 상태
        #   2) 종목명이 비어 있거나 티커와 동일 (이름 미조회 상태)
        needs_resolve = (
            (raw_t.isdigit() and len(raw_t) == 6)
            or (not current_name or current_name == raw_t)
        )

        if needs_resolve:
            sanitized_t, fetched_name = get_cached_name(raw_t)
            new_row = row.to_dict()
            if sanitized_t:
                new_row["티커"] = sanitized_t
            if fetched_name:
                new_row["종목명"] = fetched_name
            if new_row["티커"] != row.get("티커") or new_row["종목명"] != current_name:
                needs_rerun = True
            updated_rows.append(new_row)
        else:
            updated_rows.append(row.to_dict())

    final_df = pd.DataFrame(updated_rows)

    # 컬럼 순서 & 타입 보정
    for col in ["티커", "종목명"]:
        if col not in final_df.columns:
            final_df[col] = ""
    if "실제최초매수가" not in final_df.columns:
        final_df["실제최초매수가"] = 0.0
    if "현재보유유닛" not in final_df.columns:
        final_df["현재보유유닛"] = 1

    final_df["현재보유유닛"] = pd.to_numeric(final_df["현재보유유닛"], errors="coerce").fillna(1).astype(int)
    final_df["실제최초매수가"] = pd.to_numeric(final_df["실제최초매수가"], errors="coerce").fillna(0.0)

    # 세션 저장 후 rerun (한 번만)
    st.session_state.active_positions = final_df[["티커", "종목명", "실제최초매수가", "현재보유유닛"]]

    if needs_rerun:
        st.rerun()

    # ── 실시간 보유 포지션 대응 알림판 ──────────────────
    st.divider()
    st.subheader("🚨 실시간 보유 포지션 대응 알림판")

    position_df = st.session_state.active_positions.copy()
    real_management_data = []

    with st.spinner("포지션 데이터 분석 중..."):
        for _, row in position_df.iterrows():
            ticker = str(row.get("티커", "")).strip().upper()
            init_price = row.get("실제최초매수가")
            held_units = row.get("현재보유유닛")
            stock_name = str(row.get("종목명", "")).strip() or ticker

            # 유효성 검사
            if (
                not ticker
                or ticker == "NAN"
                or pd.isna(init_price)
                or pd.isna(held_units)
                or float(init_price) <= 0
            ):
                continue

            df_pos = load_and_process_data(ticker, entry_window, exit_window)
            if df_pos is None or df_pos.empty:
                st.warning(f"⚠️ {ticker} 데이터를 불러오지 못했습니다.")
                continue

            latest = df_pos.iloc[-1]
            c_price = float(latest["Close"])
            atr = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0.0
            exit_channel = float(latest["Exit_Low"]) if not pd.isna(latest["Exit_Low"]) else 0.0

            if atr <= 0:
                st.warning(f"⚠️ {ticker} ATR 계산 불가 (데이터 부족).")
                continue

            # 피라미딩 계획
            # 유닛N 추가매수가 = 최초매수가 + 0.5 * ATR * (N-1)
            # 최신 유닛 추가가 = 최초매수가 + 0.5 * ATR * (현재유닛-1)
            latest_unit_entry = float(init_price) + 0.5 * atr * (int(held_units) - 1)
            actual_stop_loss = latest_unit_entry - 2 * atr
            next_pyramid_price = float(init_price) + 0.5 * atr * int(held_units)  # 다음 유닛 진입가

            risk_amount = account_size * risk_per_trade
            unit_shares = int(risk_amount / atr)

            # 대응 가이드 결정
            if c_price <= actual_stop_loss:
                action_guide = "🚨 즉시 매도 (2N 실전 손절선 탈락!)"
                status_color = "🔴"
            elif c_price <= exit_channel:
                action_guide = "🚨 즉시 매도 (채널 청산선 탈락!)"
                status_color = "🔴"
            elif int(held_units) < max_units and c_price >= next_pyramid_price:
                action_guide = f"➕ 증액 추천 (+1유닛 추가 매수 기준가: {round(next_pyramid_price, 2):,})"
                status_color = "🔵"
            else:
                action_guide = "🟢 정상 보유 (추세 유지 중)"
                status_color = "🟢"

            pnl_pct = ((c_price - float(init_price)) / float(init_price)) * 100

            real_management_data.append({
                "상태": status_color,
                "종목명": stock_name,
                "티커": ticker,
                "현재가": f"{c_price:,.0f}",
                "ATR(20N)": f"{atr:,.0f}",
                "추천 1유닛 수량": f"{unit_shares:,} 주",
                "최초 매수가": f"{float(init_price):,.0f}",
                "보유 유닛": f"{held_units} / {max_units}",
                "수익률": f"{pnl_pct:+.2f}%",
                "실전 손절가(2N)": f"{actual_stop_loss:,.0f}",
                "채널 청산선": f"{exit_channel:,.0f}",
                "다음 증액 목표가": f"{next_pyramid_price:,.0f}" if int(held_units) < max_units else "최대 유닛 도달",
                "실시간 대응 가이드": action_guide,
            })

    if real_management_data:
        res_df = pd.DataFrame(real_management_data)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        csv_data = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 포지션 대응 알림판 CSV 다운로드",
            data=csv_data,
            file_name=f"Position_Alerts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_positions",
        )
    else:
        st.info("유효한 포지션을 입력하시면 실시간 대응 분석표가 이곳에 출력됩니다.")


# =========================================================
# 탭 1: CAN SLIM 관심종목 스캐너
# =========================================================
with tab1:
    st.subheader("🔍 관심종목 발굴 스캐너")
    tickers_input = st.text_input(
        "스캔할 관심 종목 리스트 (쉼표 구분, 국내는 6자리 숫자 가능)",
        "005930, 000660, 066570",
    )

    scan_tickers_raw = [t.strip() for t in tickers_input.split(",") if t.strip()]
    scan_tickers: list[str] = []
    ticker_to_name: dict[str, str] = {}

    with st.spinner("종목 정보 조회 중..."):
        for t in scan_tickers_raw:
            sanitized_t, name = get_cached_name(t)
            if sanitized_t:
                scan_tickers.append(sanitized_t)
                ticker_to_name[sanitized_t] = name

    # 시장 추세 조회
    unique_benchmarks = set(get_benchmark_ticker(t) for t in scan_tickers)
    market_trend_map = {b: get_market_trend(b) for b in unique_benchmarks}

    # RS Raw Score 계산
    raw_scores: dict[str, Optional[float]] = {}
    processed_dfs: dict[str, Optional[pd.DataFrame]] = {}

    with st.spinner("종목 분석 중..."):
        for t in scan_tickers:
            df_t = load_and_process_data(t, entry_window, exit_window)
            processed_dfs[t] = df_t
            raw_scores[t] = compute_rs_raw(df_t) if df_t is not None else None

    rs_ratings = rs_rating_from_raw(raw_scores)

    scan_data = []
    for ticker in scan_tickers:
        df = processed_dfs.get(ticker)
        if df is None or df.empty:
            continue

        latest = df.iloc[-1]
        c_price = float(latest["Close"])
        scan_atr = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0.0
        entry_high = float(latest["Entry_High"]) if not pd.isna(latest["Entry_High"]) else 0.0
        exit_low = float(latest["Exit_Low"]) if not pd.isna(latest["Exit_Low"]) else 0.0

        # CAN SLIM 미네르비니 추세 조건 (7가지)
        sma50 = float(latest["SMA_50"]) if not pd.isna(latest["SMA_50"]) else 0.0
        sma150 = float(latest["SMA_150"]) if not pd.isna(latest["SMA_150"]) else 0.0
        sma200 = float(latest["SMA_200"]) if not pd.isna(latest["SMA_200"]) else 0.0
        w52_high = float(latest["52W_High"]) if not pd.isna(latest["52W_High"]) else 0.0
        w52_low = float(latest["52W_Low"]) if not pd.isna(latest["52W_Low"]) else 0.0
        sma200_trend = bool(latest["SMA_200_Trend"]) if not pd.isna(latest["SMA_200_Trend"]) else False

        cond1 = c_price > sma150 and c_price > sma200
        cond2 = sma150 > sma200
        cond3 = sma200_trend
        cond4 = sma50 > sma150 and sma50 > sma200
        cond5 = c_price > sma50
        cond6 = w52_high > 0 and c_price >= w52_high * 0.75
        cond7 = w52_low > 0 and c_price >= w52_low * 1.30
        trend_score = sum([cond1, cond2, cond3, cond4, cond5, cond6, cond7])

        rs_val = rs_ratings.get(ticker)
        cond_rs = rs_val is not None and rs_val >= 70
        vol_info = compute_volume_signal(df)
        cond_vol = vol_info["signal"] == "🟢 매집"

        canslim_score = trend_score + int(cond_rs) + int(cond_vol)  # 최대 9점

        bench = get_benchmark_ticker(ticker)
        market_bearish = "하락추세" in market_trend_map.get(bench, {}).get("status", "")

        if canslim_score >= 7:
            base_status, base_emoji = "강력 매수 고려", "🔥"
        elif canslim_score >= 5:
            base_status, base_emoji = "관망/대기", "🟡"
        else:
            base_status, base_emoji = "추세 약함", "❄️"

        if apply_market_filter and market_bearish and base_status == "강력 매수 고려":
            status_text = f"🟠 시장약세로 보류 ({canslim_score}/9)"
        else:
            status_text = f"{base_emoji} {base_status} ({canslim_score}/9)"

        scan_data.append({
            "종목명": ticker_to_name.get(ticker, ticker),
            "티커": ticker,
            "CAN SLIM 점수": status_text,
            "RS Rating": rs_val if rs_val is not None else "-",
            "수급 신호": vol_info["signal"],
            "현재가": f"{c_price:,.0f}",
            "ATR": f"{scan_atr:,.0f}",
            "터틀 진입선": f"{entry_high:,.0f}",
            "터틀 청산선": f"{exit_low:,.0f}",
        })

    if scan_data:
        scan_df = pd.DataFrame(scan_data)
        st.dataframe(scan_df, use_container_width=True, hide_index=True)

        csv_data_scan = scan_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 관심종목 스캐너 CSV 다운로드",
            data=csv_data_scan,
            file_name=f"Scanner_Results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_scanner",
        )
    else:
        st.info("스캔 결과가 없습니다. 종목 코드를 확인해 주세요.")


# =========================================================
# 탭 2: 개별 종목 융합 차트 및 세부 계획
# =========================================================
with tab2:
    st.subheader("📈 개별 종목 정밀 융합 차트")

    # 탭0 + 탭1 종목 통합 (중복 제거)
    position_tickers = [
        str(r.get("티커", "")).strip().upper()
        for _, r in st.session_state.active_positions.iterrows()
        if r.get("티커") and str(r.get("티커")).strip().upper() not in ("", "NAN")
    ]
    all_known_tickers = list(dict.fromkeys(scan_tickers + position_tickers))  # 순서 유지 중복 제거

    if all_known_tickers:
        def format_ticker_label(t: str) -> str:
            name = ticker_to_name.get(t) or st.session_state.ticker_name_cache.get(t, ("", t))[1]
            return f"{name} ({t})" if name and name != t else t

        selected_ticker = st.selectbox(
            "분석할 종목을 선택하세요",
            options=all_known_tickers,
            format_func=format_ticker_label,
        )

        df_chart = load_and_process_data(selected_ticker, entry_window, exit_window)

        if df_chart is not None and not df_chart.empty:
            df_plot = df_chart.tail(200).copy()

            # 돌파/이탈 시그널
            df_plot["Buy_Signal"] = (
                (df_plot["Close"] > df_plot["Entry_High"])
                & (df_plot["Close"].shift(1) <= df_plot["Entry_High"].shift(1))
            )
            df_plot["Sell_Signal"] = (
                (df_plot["Close"] < df_plot["Exit_Low"])
                & (df_plot["Close"].shift(1) >= df_plot["Exit_Low"].shift(1))
            )

            # 차트 생성
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_plot.index,
                open=df_plot["Open"], high=df_plot["High"],
                low=df_plot["Low"], close=df_plot["Close"],
                name="가격", increasing_line_color="#ef5350", decreasing_line_color="#1976d2",
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["SMA_50"],
                line=dict(color="orange", width=1.5), name="50일선",
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["SMA_200"],
                line=dict(color="purple", width=2), name="200일선",
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Entry_High"],
                line=dict(color="royalblue", dash="dot", width=1.5), name=f"터틀 진입선({entry_window}일)",
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Exit_Low"],
                line=dict(color="crimson", dash="dot", width=1.5), name=f"터틀 청산선({exit_window}일)",
            ))

            buy_signals = df_plot[df_plot["Buy_Signal"]]
            sell_signals = df_plot[df_plot["Sell_Signal"]]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index, y=buy_signals["Close"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="lime", size=13, line=dict(color="green", width=1)),
                    name="터틀 돌파 매수",
                ))
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index, y=sell_signals["Close"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="red", size=13, line=dict(color="darkred", width=1)),
                    name="터틀 이탈 청산",
                ))

            stock_label = format_ticker_label(selected_ticker)
            fig.update_layout(
                title=f"{stock_label} — 분석 차트",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 볼륨 차트 (서브플롯 대신 별도 expander)
            with st.expander("📊 거래량 차트 보기"):
                vol_fig = go.Figure()
                colors = ["#ef5350" if c >= o else "#1976d2"
                          for c, o in zip(df_plot["Close"], df_plot["Open"])]
                vol_fig.add_trace(go.Bar(
                    x=df_plot.index, y=df_plot["Volume"],
                    marker_color=colors, name="거래량",
                ))
                vol_fig.update_layout(height=250, template="plotly_white", showlegend=False)
                st.plotly_chart(vol_fig, use_container_width=True)

            # 터틀 피라미딩 계획
            st.subheader("🐢 터틀 피라미딩 & 손절 세부 계획")
            latest_chart = df_chart.iloc[-1]
            atr_chart = float(latest_chart["ATR"]) if not pd.isna(latest_chart["ATR"]) else 0.0
            entry_price = float(latest_chart["Entry_High"]) if not pd.isna(latest_chart["Entry_High"]) else 0.0

            if atr_chart > 0 and entry_price > 0:
                risk_amount = account_size * risk_per_trade
                unit_shares = int(risk_amount / atr_chart)

                col_info1, col_info2, col_info3 = st.columns(3)
                col_info1.metric("현재 ATR(20N)", f"{atr_chart:,.0f}")
                col_info2.metric("터틀 진입선", f"{entry_price:,.0f}")
                col_info3.metric("추천 1유닛 수량", f"{unit_shares:,} 주")

                plan_rows = []
                for unit_n in range(1, max_units + 1):
                    add_price = entry_price + 0.5 * atr_chart * (unit_n - 1)
                    stop_price = add_price - 2 * atr_chart
                    plan_rows.append({
                        "유닛 단계": f"{unit_n}유닛",
                        "추가 매수 기준가": f"{add_price:,.0f}",
                        "손절가 (2N)": f"{stop_price:,.0f}",
                        "유닛당 매수량": f"{unit_shares:,} 주",
                        "누적 총 물량": f"{unit_shares * unit_n:,} 주",
                        "누적 투자금액": f"{unit_shares * unit_n * add_price:,.0f} 원",
                    })
                st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
            else:
                st.warning("ATR 또는 진입선 데이터가 부족합니다. 더 많은 데이터가 누적되면 계획이 표시됩니다.")

            # CAN SLIM 추세 조건 상세
            with st.expander("📋 CAN SLIM 미네르비니 추세 조건 상세"):
                latest_c = df_chart.iloc[-1]
                cp = float(latest_c["Close"])
                conditions = {
                    "① 현재가 > 150일선 & 200일선": cp > float(latest_c["SMA_150"]) and cp > float(latest_c["SMA_200"]),
                    "② 150일선 > 200일선": float(latest_c["SMA_150"]) > float(latest_c["SMA_200"]),
                    "③ 200일선 20일 이상 상승 중": bool(latest_c["SMA_200_Trend"]),
                    "④ 50일선 > 150일선 & 200일선": float(latest_c["SMA_50"]) > float(latest_c["SMA_150"]) and float(latest_c["SMA_50"]) > float(latest_c["SMA_200"]),
                    "⑤ 현재가 > 50일선": cp > float(latest_c["SMA_50"]),
                    "⑥ 52주 고점 대비 75% 이상": float(latest_c["52W_High"]) > 0 and cp >= float(latest_c["52W_High"]) * 0.75,
                    "⑦ 52주 저점 대비 130% 이상": float(latest_c["52W_Low"]) > 0 and cp >= float(latest_c["52W_Low"]) * 1.30,
                }
                for cond_name, cond_val in conditions.items():
                    icon = "✅" if cond_val else "❌"
                    st.write(f"{icon} {cond_name}")

        else:
            st.warning("해당 종목의 차트 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
    else:
        st.info("스캐너 혹은 포지션 관리 테이블에 종목을 입력하시면 차트 탭이 활성화됩니다.")
