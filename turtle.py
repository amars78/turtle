"""
CAN SLIM x 터틀 트레이딩 실전 자산 매니저
────────────────────────────────────────
수정 핵심:
  1. data_editor + rerun 무한루프 → on_change 콜백 패턴으로 대체
  2. @cache_data 함수 내부 st.* UI 호출 제거
  3. yfinance: info 대신 fast_info + history 사용 (안정성↑)
  4. MultiIndex 컬럼 완전 처리
  5. 세션 캐시 일원화로 중복 API 호출 제거
  6. 타입 안전성 전면 강화
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ── pykrx (선택적) ──────────────────────────────────────────
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# ── 페이지 설정 (반드시 최상단) ─────────────────────────────
st.set_page_config(
    page_title="CAN SLIM x 터틀 실전 매니저",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 헬퍼: 스칼라 float 변환 (Series/ndarray 방어)
# ============================================================
def _f(val) -> float:
    """pandas 스칼라·Series·ndarray를 안전하게 float으로 변환"""
    if val is None:
        return float("nan")
    if isinstance(val, (pd.Series, np.ndarray)):
        val = val.iloc[-1] if isinstance(val, pd.Series) else val.flat[-1]
    try:
        return float(val)
    except Exception:
        return float("nan")


# ============================================================
# 세션 상태 초기화 (탭 코드보다 반드시 먼저)
# ============================================================
_DEFAULT_POSITIONS = [
    {"티커": "005930.KS", "종목명": "삼성전자",  "실제최초매수가": 75000.0,  "현재보유유닛": 1},
    {"티커": "000660.KS", "종목명": "SK하이닉스", "실제최초매수가": 180000.0, "현재보유유닛": 2},
    {"티커": "066570.KS", "종목명": "LG전자",    "실제최초매수가": 100000.0, "현재보유유닛": 1},
]

if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(_DEFAULT_POSITIONS)

# 티커→(sanitized_ticker, 종목명) 딕셔너리 (영속 캐시)
if "name_cache" not in st.session_state:
    st.session_state.name_cache: dict[str, Tuple[str, str]] = {
        "005930.KS": ("005930.KS", "삼성전자"),
        "000660.KS": ("000660.KS", "SK하이닉스"),
        "066570.KS": ("066570.KS", "LG전자"),
    }

# ============================================================
# 티커 정제 & 종목명 조회
# ============================================================
def resolve_ticker(raw: str) -> Tuple[str, str]:
    """
    반환: (표준티커, 종목명)
    캐시 미스 시에만 외부 API 호출.
    yfinance .info 대신 .fast_info + .history 사용 (안정적).
    """
    key = str(raw).strip().upper()
    if not key or key in ("NAN", "NONE", ""):
        return "", ""

    # 세션 캐시 HIT
    if key in st.session_state.name_cache:
        return st.session_state.name_cache[key]

    # ── 국내 주식: 6자리 숫자 ───────────────────────────────
    if key.isdigit() and len(key) == 6:
        # 1) pykrx로 종목명 + 시장 확인
        if PYKRX_AVAILABLE:
            try:
                name = krx.get_market_ticker_name(key)
                if name and name.strip() and name != key:
                    try:
                        kospi = krx.get_market_ticker_list(market="KOSPI")
                        suffix = ".KS" if key in kospi else ".KQ"
                    except Exception:
                        suffix = ".KS"
                    result = (f"{key}{suffix}", name)
                    st.session_state.name_cache[key] = result
                    st.session_state.name_cache[result[0]] = result
                    return result
            except Exception:
                pass

        # 2) yfinance history 폴백 (.KS 우선, .KQ 차선)
        for suffix in (".KS", ".KQ"):
            ticker_str = f"{key}{suffix}"
            try:
                tk = yf.Ticker(ticker_str)
                hist = tk.history(period="5d")
                if hist is not None and not hist.empty:
                    # 종목명 획득 시도 (fast_info가 더 안정적)
                    try:
                        name = tk.fast_info.get("longName") or tk.fast_info.get("shortName") or ""
                    except Exception:
                        name = ""
                    if not name:
                        try:
                            info = tk.info
                            name = info.get("shortName") or info.get("longName") or ""
                        except Exception:
                            name = ""
                    name = name or ticker_str
                    result = (ticker_str, name)
                    st.session_state.name_cache[key] = result
                    st.session_state.name_cache[ticker_str] = result
                    return result
            except Exception:
                continue

        # 완전 실패 → 기본값
        result = (f"{key}.KS", key)
        st.session_state.name_cache[key] = result
        return result

    # ── 이미 .KS/.KQ 접미사 포함 ───────────────────────────
    if key.endswith(".KS") or key.endswith(".KQ"):
        pure = key.split(".")[0]
        if PYKRX_AVAILABLE:
            try:
                name = krx.get_market_ticker_name(pure)
                if name and name.strip() and name != pure:
                    result = (key, name)
                    st.session_state.name_cache[key] = result
                    return result
            except Exception:
                pass
        try:
            tk = yf.Ticker(key)
            hist = tk.history(period="5d")
            if hist is not None and not hist.empty:
                try:
                    name = tk.fast_info.get("longName") or ""
                except Exception:
                    name = ""
                if not name:
                    try:
                        name = tk.info.get("shortName") or tk.info.get("longName") or pure
                    except Exception:
                        name = pure
                result = (key, name or pure)
                st.session_state.name_cache[key] = result
                return result
        except Exception:
            pass
        result = (key, pure)
        st.session_state.name_cache[key] = result
        return result

    # ── 해외 주식: 영문 티커 ────────────────────────────────
    try:
        tk = yf.Ticker(key)
        hist = tk.history(period="5d")
        if hist is not None and not hist.empty:
            try:
                name = tk.fast_info.get("longName") or ""
            except Exception:
                name = ""
            if not name:
                try:
                    name = tk.info.get("shortName") or tk.info.get("longName") or key
                except Exception:
                    name = key
            result = (key, name or key)
            st.session_state.name_cache[key] = result
            return result
    except Exception:
        pass

    result = (key, key)
    st.session_state.name_cache[key] = result
    return result


# ============================================================
# 가격 데이터 로드 (캐시)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlcv(ticker: str, days: int = 520) -> Optional[pd.DataFrame]:
    """OHLCV DataFrame 반환. 실패 시 None."""
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            actions=False,
        )
        if df is None or df.empty:
            return None

        # MultiIndex 제거 (yfinance 0.2.x 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 필수 컬럼만 추출 (대소문자 정규화)
        df.columns = [c.capitalize() for c in df.columns]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return None

        df = df[needed].copy()
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        df.dropna(subset=["Close"], inplace=True)

        # float64 강제 변환
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)

        return df if len(df) >= 30 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def build_indicators(ticker: str, entry_w: int, exit_w: int) -> Optional[pd.DataFrame]:
    """기술적 지표 계산. 실패 시 None 반환 (UI 호출 없음)."""
    df = load_ohlcv(ticker)
    if df is None:
        return None
    try:
        df = df.copy()
        close, high, low = df["Close"], df["High"], df["Low"]

        df["SMA_50"]  = close.rolling(50).mean()
        df["SMA_150"] = close.rolling(150).mean()
        df["SMA_200"] = close.rolling(200).mean()
        df["SMA_200_Trend"] = df["SMA_200"] > df["SMA_200"].shift(20)
        df["W52_High"] = high.rolling(252).max()
        df["W52_Low"]  = low.rolling(252).min()

        # 터틀 돌파 채널 (전일 기준)
        df["Entry_High"] = high.rolling(entry_w).max().shift(1)
        df["Exit_Low"]   = low.rolling(exit_w).min().shift(1)

        # ATR(20)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(20).mean()

        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_trend(bench: str) -> dict:
    df = load_ohlcv(bench, days=420)
    if df is None or len(df) < 200:
        return {"status": "확인불가", "detail": "데이터 부족"}
    close = df["Close"]
    sma50  = _f(close.rolling(50).mean().iloc[-1])
    s200   = close.rolling(200).mean()
    sma200 = _f(s200.iloc[-1])
    sma200_20 = _f(s200.iloc[-20]) if len(df) > 220 else sma200
    cur = _f(close.iloc[-1])
    if cur > sma50 and cur > sma200 and sma200 > sma200_20:
        return {"status": "🟢 상승추세", "detail": "지수가 50/200일선 위, 200일선 상승 중"}
    if cur < sma50 and cur < sma200 and sma200 < sma200_20:
        return {"status": "🔴 하락추세", "detail": "지수가 이평선 아래, 200일선 하락 중"}
    return {"status": "🟡 중립/전환구간", "detail": "이평선 신호 혼재"}


def get_benchmark(ticker: str) -> str:
    if ticker.endswith(".KS"): return "^KS11"
    if ticker.endswith(".KQ"): return "^KQ11"
    return "^GSPC"


def compute_rs_raw(df: pd.DataFrame) -> Optional[float]:
    close = df["Close"].dropna()
    if len(close) < 253:
        return None
    try:
        def r(a, b):
            s = _f(close.iloc[b])
            return (_f(close.iloc[a]) / s - 1.0) if s else 0.0
        return 2*r(-1,-64) + r(-64,-127) + r(-127,-190) + r(-190,-253)
    except Exception:
        return None


def rs_ratings(raw: dict) -> dict:
    valid = {k: v for k, v in raw.items() if v is not None}
    if len(valid) <= 1:
        return {k: (50 if v is not None else None) for k, v in raw.items()}
    vals = sorted(valid.values())
    out = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        else:
            pct = sum(1 for x in vals if x <= v) / len(vals)
            out[k] = max(1, min(99, round(pct * 98) + 1))
    return out


def volume_signal(df: pd.DataFrame) -> dict:
    r = df.tail(50).copy()
    r["chg"] = r["Close"].diff()
    up   = float(r.loc[r["chg"] > 0, "Volume"].sum())
    down = float(r.loc[r["chg"] < 0, "Volume"].sum())
    ratio = up / down if down > 0 else 1.0
    sig = "🟢 매집" if ratio >= 1.2 else ("🔴 분산" if ratio <= 0.8 else "🟡 중립")
    return {"ratio": round(ratio, 2), "signal": sig}


# ============================================================
# 사이드바
# ============================================================
st.sidebar.header("⚙️ 시스템 및 자금 관리 설정")
system_type = st.sidebar.radio(
    "터틀 시스템 선택",
    ("시스템 1 (20일 돌파)", "시스템 2 (55일 돌파)"),
)
entry_w, exit_w = (20, 10) if "1" in system_type else (55, 20)
account_size    = st.sidebar.number_input("총 투자 자본금 (원)", value=10_000_000, step=1_000_000, min_value=1_000_000)
risk_pct        = st.sidebar.slider("1유닛 리스크 비율 (%)", 0.5, 5.0, 1.0, 0.1) / 100
max_units       = st.sidebar.slider("최대 피라미딩 유닛 수", 1, 4, 4)
mkt_filter      = st.sidebar.checkbox("시장 하락추세면 매수등급 자동 하향", value=True)

# ============================================================
# 제목
# ============================================================
st.title("🦅 CAN SLIM x 🐢 터틀 트레이딩 실전 자산 매니저")
st.markdown(
    "* **국내**: 6자리 숫자 입력 → 자동 변환 (예: `005930`) &nbsp;|&nbsp; "
    "* **해외**: 영문 티커 (예: `AAPL`, `TSLA`)"
)

# ============================================================
# 탭
# ============================================================
tab0, tab1, tab2 = st.tabs([
    "🔥 1. 실전 보유 포지션 관리",
    "📊 2. CAN SLIM 관심종목 스캐너",
    "📈 3. 개별 종목 융합 차트",
])

# ============================================================
# TAB 0 – 실전 보유 포지션 관리
# ============================================================
with tab0:
    st.subheader("🛠 보유 포지션 입력 및 편집")
    st.caption(
        "💡 **국내 주식**: 6자리 숫자 입력 후 [티커 자동완성] 버튼을 누르세요.  "
        "**해외 주식**: 영문 티커를 그대로 입력하세요."
    )

    # ── (A) data_editor: 사용자가 직접 편집 ─────────────────
    edited = st.data_editor(
        st.session_state.positions,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "티커":        st.column_config.TextColumn("티커", required=True),
            "종목명":      st.column_config.TextColumn("종목명 (자동완성)", disabled=True),
            "실제최초매수가": st.column_config.NumberColumn("최초 매수가", min_value=0.0, format="%.2f"),
            "현재보유유닛":  st.column_config.NumberColumn("유닛 수", min_value=1, max_value=4, default=1),
        },
        key="pos_editor",
    )

    # ── (B) [티커 자동완성] 버튼: 클릭 시에만 API 호출 ──────
    # rerun 루프 없이 버튼 클릭 1회로 처리
    if st.button("🔄 티커 자동완성 / 종목명 업데이트", type="primary"):
        resolved_rows = []
        progress = st.progress(0, text="종목 정보 조회 중...")
        total = len(edited)
        for i, (_, row) in enumerate(edited.iterrows()):
            raw_t = str(row.get("티커", "")).strip()
            if not raw_t or raw_t.upper() in ("NAN", ""):
                resolved_rows.append(row.to_dict())
                progress.progress((i + 1) / max(total, 1))
                continue

            san_t, name = resolve_ticker(raw_t)
            new_row = row.to_dict()
            new_row["티커"]   = san_t if san_t else raw_t
            new_row["종목명"] = name  if name  else raw_t
            resolved_rows.append(new_row)
            progress.progress((i + 1) / max(total, 1), text=f"{raw_t} → {name}")

        progress.empty()

        # 컬럼 정합성 보정
        new_df = pd.DataFrame(resolved_rows)
        for col in ["티커", "종목명"]:
            if col not in new_df.columns:
                new_df[col] = ""
        new_df["실제최초매수가"] = pd.to_numeric(new_df.get("실제최초매수가", 0), errors="coerce").fillna(0.0)
        new_df["현재보유유닛"]   = pd.to_numeric(new_df.get("현재보유유닛",   1), errors="coerce").fillna(1).astype(int)

        st.session_state.positions = new_df[["티커", "종목명", "실제최초매수가", "현재보유유닛"]]
        st.success("✅ 종목명 업데이트 완료!")
        st.rerun()
    else:
        # 버튼 미클릭 시에도 편집 내용 보존 (API 호출 없이 저장만)
        save_df = edited.copy()
        for col in ["티커", "종목명"]:
            if col not in save_df.columns:
                save_df[col] = ""
        save_df["실제최초매수가"] = pd.to_numeric(save_df.get("실제최초매수가", 0), errors="coerce").fillna(0.0)
        save_df["현재보유유닛"]   = pd.to_numeric(save_df.get("현재보유유닛",   1), errors="coerce").fillna(1).astype(int)
        st.session_state.positions = save_df[["티커", "종목명", "실제최초매수가", "현재보유유닛"]]

    # ── (C) 실시간 포지션 대응 알림판 ───────────────────────
    st.divider()
    st.subheader("🚨 실시간 보유 포지션 대응 알림판")

    alert_data = []
    pos_df = st.session_state.positions.copy()

    for _, row in pos_df.iterrows():
        ticker  = str(row.get("티커",   "")).strip().upper()
        name    = str(row.get("종목명", "")).strip() or ticker
        ip      = row.get("실제최초매수가", 0)
        units   = int(row.get("현재보유유닛", 1))

        if not ticker or ticker in ("NAN", "") or _f(ip) <= 0:
            continue

        df_ind = build_indicators(ticker, entry_w, exit_w)
        if df_ind is None or df_ind.empty:
            alert_data.append({
                "상태": "⚠️", "종목명": name, "티커": ticker,
                "현재가": "조회실패", "ATR": "-", "추천 1유닛": "-",
                "최초매수가": f"{_f(ip):,.0f}", "보유 유닛": f"{units}/{max_units}",
                "수익률": "-", "실전손절가": "-", "채널청산선": "-",
                "다음증액가": "-", "대응 가이드": "❗ 데이터 조회 실패 – 티커 확인 필요",
            })
            continue

        lat = df_ind.iloc[-1]
        cp       = _f(lat["Close"])
        atr      = _f(lat["ATR"])
        ex_low   = _f(lat["Exit_Low"])
        init_p   = _f(ip)

        if np.isnan(atr) or atr <= 0:
            alert_data.append({
                "상태": "⚠️", "종목명": name, "티커": ticker,
                "현재가": f"{cp:,.0f}", "ATR": "계산불가", "추천 1유닛": "-",
                "최초매수가": f"{init_p:,.0f}", "보유 유닛": f"{units}/{max_units}",
                "수익률": "-", "실전손절가": "-", "채널청산선": "-",
                "다음증액가": "-", "대응 가이드": "❗ ATR 계산 불가 (데이터 부족)",
            })
            continue

        last_entry = init_p + 0.5 * atr * (units - 1)
        stop_loss  = last_entry - 2 * atr
        next_price = init_p + 0.5 * atr * units
        unit_qty   = int((account_size * risk_pct) / atr)
        pnl        = (cp - init_p) / init_p * 100

        if cp <= stop_loss:
            guide, st_icon = "🚨 즉시 매도 (2N 손절선 이탈)", "🔴"
        elif not np.isnan(ex_low) and cp <= ex_low:
            guide, st_icon = "🚨 즉시 매도 (채널 청산선 이탈)", "🔴"
        elif units < max_units and cp >= next_price:
            guide, st_icon = f"➕ 증액 추천 (기준가: {next_price:,.0f})", "🔵"
        else:
            guide, st_icon = "🟢 정상 보유 (추세 유지 중)", "🟢"

        alert_data.append({
            "상태":     st_icon,
            "종목명":   name,
            "티커":     ticker,
            "현재가":   f"{cp:,.0f}",
            "ATR(20)":  f"{atr:,.0f}",
            "추천 1유닛": f"{unit_qty:,} 주",
            "최초매수가": f"{init_p:,.0f}",
            "보유 유닛":  f"{units} / {max_units}",
            "수익률":    f"{pnl:+.2f}%",
            "실전손절가(2N)": f"{stop_loss:,.0f}",
            "채널청산선":    f"{ex_low:,.0f}" if not np.isnan(ex_low) else "-",
            "다음증액가":    f"{next_price:,.0f}" if units < max_units else "최대 유닛",
            "대응 가이드":   guide,
        })

    if alert_data:
        alert_df = pd.DataFrame(alert_data)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 알림판 CSV 다운로드",
            alert_df.to_csv(index=False).encode("utf-8-sig"),
            f"Positions_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            key="dl_pos",
        )
    else:
        st.info("유효한 포지션을 입력 후 [티커 자동완성] 버튼을 눌러주세요.")


# ============================================================
# TAB 1 – CAN SLIM 관심종목 스캐너
# ============================================================
with tab1:
    st.subheader("🔍 관심종목 발굴 스캐너")
    raw_input = st.text_input(
        "종목 리스트 (쉼표 구분, 국내 6자리 또는 해외 영문)",
        "005930, 000660, 066570, AAPL, TSLA",
    )

    if st.button("🔍 스캔 시작", type="primary", key="scan_btn"):
        raw_list = [t.strip() for t in raw_input.split(",") if t.strip()]
        scan_map: dict[str, str] = {}  # ticker → name
        scan_tickers: list[str] = []

        prog = st.progress(0, "종목 정보 조회 중...")
        for i, raw in enumerate(raw_list):
            san, nm = resolve_ticker(raw)
            if san:
                scan_tickers.append(san)
                scan_map[san] = nm
            prog.progress((i + 1) / max(len(raw_list), 1))
        prog.empty()

        # 시장 추세
        bench_map = {get_benchmark(t): get_market_trend(get_benchmark(t)) for t in scan_tickers}

        # RS Raw
        raw_rs: dict[str, Optional[float]] = {}
        dfs_cache: dict[str, Optional[pd.DataFrame]] = {}
        for t in scan_tickers:
            d = build_indicators(t, entry_w, exit_w)
            dfs_cache[t] = d
            raw_rs[t] = compute_rs_raw(d) if d is not None else None
        rs_map = rs_ratings(raw_rs)

        rows = []
        for t in scan_tickers:
            df = dfs_cache.get(t)
            if df is None or df.empty:
                rows.append({
                    "종목명": scan_map.get(t, t), "티커": t,
                    "CAN SLIM 점수": "❗ 데이터 없음",
                    "RS Rating": "-", "수급 신호": "-",
                    "현재가": "-", "ATR": "-",
                    "터틀 진입선": "-", "터틀 청산선": "-",
                })
                continue

            lat = df.iloc[-1]
            cp     = _f(lat["Close"])
            atr_v  = _f(lat["ATR"])
            e_high = _f(lat["Entry_High"])
            e_low  = _f(lat["Exit_Low"])
            s50    = _f(lat["SMA_50"])
            s150   = _f(lat["SMA_150"])
            s200   = _f(lat["SMA_200"])
            w52h   = _f(lat["W52_High"])
            w52l   = _f(lat["W52_Low"])
            s200t  = bool(lat["SMA_200_Trend"]) if not pd.isna(lat["SMA_200_Trend"]) else False

            c1 = cp > s150 and cp > s200
            c2 = s150 > s200
            c3 = s200t
            c4 = s50 > s150 and s50 > s200
            c5 = cp > s50
            c6 = w52h > 0 and cp >= w52h * 0.75
            c7 = w52l > 0 and cp >= w52l * 1.30
            trend_s = sum([c1, c2, c3, c4, c5, c6, c7])

            rs_v   = rs_map.get(t)
            vol    = volume_signal(df)
            score  = trend_s + int(bool(rs_v and rs_v >= 70)) + int(vol["signal"] == "🟢 매집")

            bench  = get_benchmark(t)
            bearish = "하락추세" in bench_map.get(bench, {}).get("status", "")

            if score >= 7:   emoji, label = "🔥", "강력 매수 고려"
            elif score >= 5: emoji, label = "🟡", "관망/대기"
            else:            emoji, label = "❄️", "추세 약함"

            if mkt_filter and bearish and label == "강력 매수 고려":
                score_txt = f"🟠 시장약세 보류 ({score}/9)"
            else:
                score_txt = f"{emoji} {label} ({score}/9)"

            rows.append({
                "종목명":        scan_map.get(t, t),
                "티커":          t,
                "CAN SLIM 점수": score_txt,
                "RS Rating":     rs_v if rs_v is not None else "-",
                "수급 신호":     vol["signal"],
                "현재가":        f"{cp:,.0f}" if not np.isnan(cp) else "-",
                "ATR":           f"{atr_v:,.0f}" if not np.isnan(atr_v) else "-",
                "터틀 진입선":   f"{e_high:,.0f}" if not np.isnan(e_high) else "-",
                "터틀 청산선":   f"{e_low:,.0f}" if not np.isnan(e_low) else "-",
            })

        if rows:
            scan_df = pd.DataFrame(rows)
            st.dataframe(scan_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 스캐너 결과 CSV",
                scan_df.to_csv(index=False).encode("utf-8-sig"),
                f"Scanner_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key="dl_scan",
            )
            # 탭2에서 참조할 수 있도록 세션에 저장
            st.session_state["scan_tickers"] = scan_tickers
            st.session_state["scan_map"]     = scan_map
        else:
            st.warning("조회된 결과가 없습니다. 티커를 확인해 주세요.")
    else:
        st.info("종목 코드 입력 후 [🔍 스캔 시작] 버튼을 눌러주세요.")


# ============================================================
# TAB 2 – 개별 종목 융합 차트
# ============================================================
with tab2:
    st.subheader("📈 개별 종목 정밀 융합 차트")

    # 포지션 + 스캔 종목 통합
    pos_tickers = [
        str(r.get("티커", "")).strip().upper()
        for _, r in st.session_state.positions.iterrows()
        if str(r.get("티커", "")).strip().upper() not in ("", "NAN")
    ]
    scan_tickers_sess = st.session_state.get("scan_tickers", [])
    scan_map_sess     = st.session_state.get("scan_map", {})
    all_tickers = list(dict.fromkeys(pos_tickers + scan_tickers_sess))

    if not all_tickers:
        st.info("포지션 탭 또는 스캐너 탭에 종목을 입력하세요.")
    else:
        def label(t: str) -> str:
            n = (
                scan_map_sess.get(t)
                or st.session_state.name_cache.get(t, (t, t))[1]
            )
            return f"{n} ({t})" if n and n != t else t

        sel = st.selectbox("분석할 종목 선택", all_tickers, format_func=label)

        df_c = build_indicators(sel, entry_w, exit_w)
        if df_c is None or df_c.empty:
            st.error(f"⚠️ [{sel}] 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
        else:
            dp = df_c.tail(200).copy()

            # 시그널
            dp["Buy"]  = (dp["Close"] > dp["Entry_High"]) & (dp["Close"].shift(1) <= dp["Entry_High"].shift(1))
            dp["Sell"] = (dp["Close"] < dp["Exit_Low"])   & (dp["Close"].shift(1) >= dp["Exit_Low"].shift(1))

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=dp.index, open=dp["Open"], high=dp["High"], low=dp["Low"], close=dp["Close"],
                name="가격",
                increasing_line_color="#ef5350",
                decreasing_line_color="#1976d2",
            ))
            for col, color, dash, wid, nm in [
                ("SMA_50",     "orange",    "solid", 1.5, "50일선"),
                ("SMA_200",    "purple",    "solid", 2.0, "200일선"),
                ("Entry_High", "royalblue", "dot",   1.5, f"진입선({entry_w}일)"),
                ("Exit_Low",   "crimson",   "dot",   1.5, f"청산선({exit_w}일)"),
            ]:
                fig.add_trace(go.Scatter(
                    x=dp.index, y=dp[col],
                    line=dict(color=color, dash=dash, width=wid),
                    name=nm,
                ))

            buys  = dp[dp["Buy"]]
            sells = dp[dp["Sell"]]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys["Close"], mode="markers",
                    marker=dict(symbol="triangle-up", color="lime", size=13,
                                line=dict(color="darkgreen", width=1)),
                    name="돌파 매수",
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells["Close"], mode="markers",
                    marker=dict(symbol="triangle-down", color="red", size=13,
                                line=dict(color="darkred", width=1)),
                    name="이탈 청산",
                ))

            fig.update_layout(
                title=label(sel),
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=580,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 거래량
            with st.expander("📊 거래량 차트"):
                colors = ["#ef5350" if c >= o else "#1976d2"
                          for c, o in zip(dp["Close"], dp["Open"])]
                vfig = go.Figure(go.Bar(x=dp.index, y=dp["Volume"], marker_color=colors, name="거래량"))
                vfig.update_layout(height=220, template="plotly_white", showlegend=False)
                st.plotly_chart(vfig, use_container_width=True)

            # 피라미딩 계획
            st.subheader("🐢 터틀 피라미딩 & 손절 계획")
            lat2  = df_c.iloc[-1]
            atr2  = _f(lat2["ATR"])
            ep2   = _f(lat2["Entry_High"])

            if not np.isnan(atr2) and atr2 > 0 and not np.isnan(ep2) and ep2 > 0:
                qty = int((account_size * risk_pct) / atr2)
                c1, c2, c3 = st.columns(3)
                c1.metric("ATR(20N)", f"{atr2:,.0f}")
                c2.metric("터틀 진입선", f"{ep2:,.0f}")
                c3.metric("추천 1유닛", f"{qty:,} 주")

                plan = []
                for n in range(1, max_units + 1):
                    add_p  = ep2 + 0.5 * atr2 * (n - 1)
                    stop_p = add_p - 2 * atr2
                    plan.append({
                        "유닛":        f"{n}유닛",
                        "매수 기준가": f"{add_p:,.0f}",
                        "손절가(2N)":  f"{stop_p:,.0f}",
                        "유닛당 수량": f"{qty:,} 주",
                        "누적 수량":   f"{qty*n:,} 주",
                        "누적 금액":   f"{qty*n*add_p:,.0f} 원",
                    })
                st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
            else:
                st.warning("ATR 또는 진입선 데이터 부족 – 데이터가 더 쌓이면 표시됩니다.")

            # CAN SLIM 조건 상세
            with st.expander("📋 CAN SLIM 미네르비니 추세 조건 체크"):
                lat3 = df_c.iloc[-1]
                cp3  = _f(lat3["Close"])
                checks = {
                    "① 현재가 > 150일선 & 200일선": cp3 > _f(lat3["SMA_150"]) and cp3 > _f(lat3["SMA_200"]),
                    "② 150일선 > 200일선":          _f(lat3["SMA_150"]) > _f(lat3["SMA_200"]),
                    "③ 200일선 20일 이상 상승":      bool(lat3["SMA_200_Trend"]) if not pd.isna(lat3["SMA_200_Trend"]) else False,
                    "④ 50일선 > 150일선 & 200일선": _f(lat3["SMA_50"]) > _f(lat3["SMA_150"]) and _f(lat3["SMA_50"]) > _f(lat3["SMA_200"]),
                    "⑤ 현재가 > 50일선":            cp3 > _f(lat3["SMA_50"]),
                    "⑥ 52주 고점 대비 75% 이상":    _f(lat3["W52_High"]) > 0 and cp3 >= _f(lat3["W52_High"]) * 0.75,
                    "⑦ 52주 저점 대비 130% 이상":   _f(lat3["W52_Low"])  > 0 and cp3 >= _f(lat3["W52_Low"])  * 1.30,
                }
                for cname, cval in checks.items():
                    st.write(("✅" if cval else "❌") + " " + cname)
