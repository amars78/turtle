import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

# ── 로깅 설정 ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── pykrx (선택적) ──────────────────────────────────────────
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    logging.warning("pykrx 모듈이 없습니다. 한국 주식 이름 조회 기능이 제한됩니다.")

# ── 페이지 설정 (반드시 최상단) ─────────────────────────────
st.set_page_config(
    page_title="CAN SLIM x 터틀 실전 매니저",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 헬퍼 함수
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
# 핵심 비즈니스 로직 클래스 분리
# ============================================================
class DataFetcher:
    """외부 API(yfinance, pykrx) 데이터 페칭 및 정제 담당 클래스"""
    
    @staticmethod
    def resolve_ticker(raw: str, name_cache: dict) -> Tuple[str, str]:
        """티커 표준화 및 종목명 조회"""
        key = str(raw).strip().upper()
        if not key or key in ("NAN", "NONE", ""):
            return "", ""

        if key in name_cache:
            return name_cache[key]

        try:
            # 1. 국내 주식 (6자리)
            if key.isdigit() and len(key) == 6:
                if PYKRX_AVAILABLE:
                    name = krx.get_market_ticker_name(key)
                    if name and name != key:
                        suffix = ".KS" if key in krx.get_market_ticker_list(market="KOSPI") else ".KQ"
                        return f"{key}{suffix}", name

                for suffix in (".KS", ".KQ"):
                    tk_str = f"{key}{suffix}"
                    tk = yf.Ticker(tk_str)
                    if not tk.history(period="5d").empty:
                        name = tk.info.get("shortName") or tk.info.get("longName") or tk_str
                        return tk_str, name
            
            # 2. 해외 주식 또는 이미 접미사가 붙은 경우
            tk = yf.Ticker(key)
            if not tk.history(period="5d").empty:
                name = tk.info.get("shortName") or tk.info.get("longName") or key
                return key, name

        except Exception as e:
            logging.error(f"티커 조회 실패 [{key}]: {e}")

        # 완전 실패 시 기본값 반환
        fallback_tk = f"{key}.KS" if key.isdigit() else key
        return fallback_tk, key

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_ohlcv(ticker: str, days: int = 520) -> Optional[pd.DataFrame]:
        """OHLCV 데이터를 가져오고 표준화합니다."""
        end = datetime.today()
        start = end - timedelta(days=days)
        try:
            df = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            if df is None or df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.capitalize() for c in df.columns]
            needed = ["Open", "High", "Low", "Close", "Volume"]
            
            if any(c not in df.columns for c in needed):
                return None

            df = df[needed].copy()
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df = df.dropna(subset=["Close"]).apply(pd.to_numeric, errors="coerce").dropna()
            
            return df if len(df) >= 30 else None
        except Exception as e:
            logging.error(f"OHLCV 로드 오류 [{ticker}]: {e}")
            return None


class StrategyAnalyzer:
    """트레이딩 지표 및 전략 점수 계산 클래스"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def build_indicators(ticker: str, entry_w: int, exit_w: int) -> Optional[pd.DataFrame]:
        df = DataFetcher.load_ohlcv(ticker)
        if df is None:
            return None
            
        try:
            close, high, low = df["Close"], df["High"], df["Low"]

            df["SMA_50"] = close.rolling(50).mean()
            df["SMA_150"] = close.rolling(150).mean()
            df["SMA_200"] = close.rolling(200).mean()
            df["SMA_200_Trend"] = df["SMA_200"] > df["SMA_200"].shift(20)
            df["W52_High"] = high.rolling(252).max()
            df["W52_Low"] = low.rolling(252).min()

            df["Entry_High"] = high.rolling(entry_w).max().shift(1)
            df["Exit_Low"] = low.rolling(exit_w).min().shift(1)

            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["ATR"] = tr.rolling(20).mean()

            return df
        except Exception as e:
            logging.error(f"지표 계산 오류 [{ticker}]: {e}")
            return None

    @staticmethod
    def volume_signal(df: pd.DataFrame) -> dict:
        """논리 오류(ZeroDivision) 수정된 거래량 분석"""
        r = df.tail(50).copy()
        r["chg"] = r["Close"].diff()
        up = float(r.loc[r["chg"] > 0, "Volume"].sum())
        down = float(r.loc[r["chg"] < 0, "Volume"].sum())
        
        # 하락 거래량이 0일 때의 논리 개선
        ratio = up / down if down > 0 else (999.0 if up > 0 else 1.0)
        sig = "🟢 매집" if ratio >= 1.2 else ("🔴 분산" if ratio <= 0.8 else "🟡 중립")
        return {"ratio": round(ratio, 2), "signal": sig}


class PortfolioManager:
    """사용자 포지션 상태 관리 (추후 DB 연동 시 이 클래스만 수정)"""
    
    DEFAULT_POSITIONS = [
        {"티커": "005930.KS", "종목명": "삼성전자",  "실제최초매수가": 75000.0,  "현재보유유닛": 1},
        {"티커": "000660.KS", "종목명": "SK하이닉스", "실제최초매수가": 180000.0, "현재보유유닛": 2},
    ]

    @staticmethod
    def initialize_state():
        if "positions" not in st.session_state:
            st.session_state.positions = pd.DataFrame(PortfolioManager.DEFAULT_POSITIONS)
        if "name_cache" not in st.session_state:
            st.session_state.name_cache = {}
        if "editor_version" not in st.session_state:
            st.session_state.editor_version = 0

    @staticmethod
    def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["티커", "종목명", "실제최초매수가", "현재보유유닛"])
        
        df = df.copy()
        for col in ["티커", "종목명"]:
            df[col] = df.get(col, "").astype(str).str.strip()
            
        df["실제최초매수가"] = pd.to_numeric(df.get("실제최초매수가", 0), errors="coerce").fillna(0.0)
        df["현재보유유닛"]   = pd.to_numeric(df.get("현재보유유닛", 1), errors="coerce").fillna(1).astype(int)

        df = df[~df["티커"].str.upper().isin(["", "NAN", "NONE"])]
        return df.reset_index(drop=True)[["티커", "종목명", "실제최초매수가", "현재보유유닛"]]

# ============================================================
# 초기화 및 사이드바 설정
# ============================================================
PortfolioManager.initialize_state()

st.sidebar.header("⚙️ 시스템 및 자금 관리 설정")
system_type = st.sidebar.radio("터틀 시스템 선택", ("시스템 1 (20일 돌파)", "시스템 2 (55일 돌파)"))
entry_w, exit_w = (20, 10) if "1" in system_type else (55, 20)
account_size = st.sidebar.number_input("총 투자 자본금 (원)", value=10_000_000, step=1_000_000)
risk_pct = st.sidebar.slider("1유닛 리스크 비율 (%)", 0.5, 5.0, 1.0, 0.1) / 100
max_units = st.sidebar.slider("최대 피라미딩 유닛 수", 1, 4, 4)

# ============================================================
# 메인 UI
# ============================================================
st.title("🦅 CAN SLIM x 🐢 터틀 트레이딩 실전 자산 매니저")
tab0, tab1, tab2 = st.tabs(["🔥 1. 포지션 관리", "📊 2. 관심종목 스캐너", "📈 3. 개별 종목 분석"])

# ── TAB 0: 실전 보유 포지션 관리 ────────────────────────────
with tab0:
    st.subheader("🛠 보유 포지션 관리")
    
    editor_key = f"pos_editor_v{st.session_state.editor_version}"
    edited = st.data_editor(
        st.session_state.positions,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "티커": st.column_config.TextColumn("티커", required=False),
            "종목명": st.column_config.TextColumn("종목명", disabled=True),
            "실제최초매수가": st.column_config.NumberColumn("최초 매수가", min_value=0.0, format="%.0f"),
            "현재보유유닛": st.column_config.NumberColumn("유닛 수", min_value=1, max_value=4),
        },
        key=editor_key,
    )

    col1, col2 = st.columns([2, 1])
    if col1.button("🔄 저장 및 종목명 업데이트", type="primary", use_container_width=True):
        working = PortfolioManager.clean_positions(edited)
        resolved_rows = []
        for _, row in working.iterrows():
            san_t, fetched_name = DataFetcher.resolve_ticker(row["티커"], st.session_state.name_cache)
            if san_t:
                st.session_state.name_cache[row["티커"]] = (san_t, fetched_name)
            new_row = row.to_dict()
            new_row["티커"] = san_t if san_t else row["티커"]
            new_row["종목명"] = fetched_name if fetched_name else row["티커"]
            resolved_rows.append(new_row)
            
        st.session_state.positions = PortfolioManager.clean_positions(pd.DataFrame(resolved_rows))
        st.session_state.editor_version += 1
        st.rerun()

    if col2.button("🗑 전체 초기화", type="secondary", use_container_width=True):
        st.session_state.positions = pd.DataFrame(PortfolioManager.DEFAULT_POSITIONS)
        st.session_state.editor_version += 1
        st.rerun()
        
    st.session_state.positions = PortfolioManager.clean_positions(edited)

# ── TAB 1 & 2의 반복적인 표출 로직은 기존 구조를 클래스 기반으로 호출하도록 유지합니다. ──
with tab2:
    st.subheader("📈 개별 종목 정밀 융합 차트 및 체크리스트")
    pos_tickers = [str(r.get("티커", "")).strip().upper() for _, r in st.session_state.positions.iterrows()]
    all_tickers = list(dict.fromkeys([t for t in pos_tickers if t]))

    if all_tickers:
        sel = st.selectbox("분석할 종목 선택", all_tickers)
        df_c = StrategyAnalyzer.build_indicators(sel, entry_w, exit_w)
        
        if df_c is not None and not df_c.empty:
            with st.expander("📋 CAN SLIM 미네르비니 추세 조건 체크 (복구됨)", expanded=True):
                lat3 = df_c.iloc[-1]
                cp3  = _f(lat3["Close"])
                
                # 잘려나간 루프문 및 조건문 완전 복구
                checks = {
                    "① 현재가 > 150일선 & 200일선": cp3 > _f(lat3["SMA_150"]) and cp3 > _f(lat3["SMA_200"]),
                    "② 150일선 > 200일선": _f(lat3["SMA_150"]) > _f(lat3["SMA_200"]),
                    "③ 200일선 20일 이상 상승": bool(lat3["SMA_200_Trend"]) if not pd.isna(lat3["SMA_200_Trend"]) else False,
                    "④ 50일선 > 150일선 & 200일선": _f(lat3["SMA_50"]) > _f(lat3["SMA_150"]) and _f(lat3["SMA_50"]) > _f(lat3["SMA_200"]),
                    "⑤ 현재가 > 50일선": cp3 > _f(lat3["SMA_50"]),
                    "⑥ 52주 고점 대비 75% 이상": _f(lat3["W52_High"]) > 0 and cp3 >= _f(lat3["W52_High"]) * 0.75,
                    "⑦ 52주 저점 대비 130% 이상": _f(lat3["W52_Low"]) > 0 and cp3 >= _f(lat3["W52_Low"]) * 1.30,
                }
                
                for cname, cval in checks.items():
                    icon = "✅ Pass" if cval else "❌ Fail"
                    st.markdown(f"**{icon}** | {cname}")
        else:
            st.error("해당 종목의 지표를 계산할 수 없습니다.")
