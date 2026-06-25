import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

# ── 로깅 설정 ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── pykrx (선택적) ──────────────────────────────────────────
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# ── 페이지 설정 (반드시 최상단) ─────────────────────────────
st.set_page_config(page_title="CAN SLIM x 터틀 시스템 매니저", layout="wide")

# ============================================================
# 헬퍼 함수
# ============================================================
def _f(val) -> float:
    if val is None: return float("nan")
    if isinstance(val, (pd.Series, np.ndarray)):
        val = val.iloc[-1] if isinstance(val, pd.Series) else val.flat[-1]
    try: return float(val)
    except: return float("nan")

# ============================================================
# 비즈니스 로직 클래스
# ============================================================
class DataFetcher:
    @staticmethod
    def resolve_ticker(raw: str, name_cache: dict) -> Tuple[str, str]:
        key = str(raw).strip().upper()
        if not key or key in ("NAN", "NONE", ""): return "", ""
        if key in name_cache: return name_cache[key]
        try:
            if key.isdigit() and len(key) == 6:
                if PYKRX_AVAILABLE:
                    name = krx.get_market_ticker_name(key)
                    if name and name != key:
                        suffix = ".KS" if key in krx.get_market_ticker_list(market="KOSPI") else ".KQ"
                        return f"{key}{suffix}", name
                for suffix in (".KS", ".KQ"):
                    tk_str = f"{key}{suffix}"
                    if not yf.Ticker(tk_str).history(period="1d").empty:
                        return tk_str, key
            tk = yf.Ticker(key)
            if not tk.history(period="1d").empty:
                name = tk.info.get("shortName") or tk.info.get("longName") or key
                return key, name
        except Exception:
            pass
        return f"{key}.KS" if key.isdigit() else key, key

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_ohlcv(ticker: str, days: int = 520) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, start=datetime.today() - timedelta(days=days), progress=False, auto_adjust=True)
            if df is None or df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.capitalize() for c in df.columns]
            needed = ["Open", "High", "Low", "Close", "Volume"]
            if any(c not in df.columns for c in needed): return None
            df = df[needed].copy().dropna(subset=["Close"]).apply(pd.to_numeric, errors="coerce").dropna()
            return df if len(df) >= 30 else None
        except:
            return None

class StrategyAnalyzer:
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def build_indicators(ticker: str, entry_w: int, exit_w: int) -> Optional[pd.DataFrame]:
        df = DataFetcher.load_ohlcv(ticker)
        if df is None: return None
        try:
            c, h, l = df["Close"], df["High"], df["Low"]
            df["SMA_50"] = c.rolling(50).mean()
            df["SMA_150"] = c.rolling(150).mean()
            df["SMA_200"] = c.rolling(200).mean()
            df["SMA_200_Trend"] = df["SMA_200"] > df["SMA_200"].shift(20)
            df["W52_High"] = h.rolling(252).max()
            df["W52_Low"] = l.rolling(252).min()
            df["Entry_High"] = h.rolling(entry_w).max().shift(1)
            df["Exit_Low"] = l.rolling(exit_w).min().shift(1)
            pc = c.shift(1)
            tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
            df["ATR"] = tr.rolling(20).mean()
            return df
        except:
            return None

    @staticmethod
    def volume_signal(df: pd.DataFrame) -> dict:
        r = df.tail(50).copy()
        r["chg"] = r["Close"].diff()
        up = float(r.loc[r["chg"] > 0, "Volume"].sum())
        dn = float(r.loc[r["chg"] < 0, "Volume"].sum())
        ratio = up / dn if dn > 0 else (999.0 if up > 0 else 1.0)
        sig = "🟢 매집" if ratio >= 1.2 else ("🔴 분산" if ratio <= 0.8 else "🟡 중립")
        return {"ratio": round(ratio, 2), "signal": sig}

class PortfolioManager:
    @staticmethod
    def init_state():
        if "pos" not in st.session_state:
            st.session_state.pos = pd.DataFrame([
                {"티커": "005930.KS", "종목명": "삼성전자", "최초매수가": 75000.0, "보유유닛": 1},
                {"티커": "AAPL", "종목명": "Apple", "최초매수가": 170.0, "보유유닛": 2}
            ])
        if "cache" not in st.session_state: st.session_state.cache = {}
        if "ev" not in st.session_state: st.session_state.ev = 0
        if "interest_input" not in st.session_state: st.session_state.interest_input = "000660, NVDA, TSLA, 035720"

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame(columns=["티커", "종목명", "최초매수가", "보유유닛"])
        df = df.copy()
        for c in ["티커", "종목명"]: df[c] = df.get(c, "").astype(str).str.strip()
        df["최초매수가"] = pd.to_numeric(df.get("최초매수가", 0), errors="coerce").fillna(0.0)
        df["보유유닛"] = pd.to_numeric(df.get("보유유닛", 1), errors="coerce").fillna(1).astype(int)
        return df[~df["티커"].str.upper().isin(["", "NAN", "NONE"])].reset_index(drop=True)

# ============================================================
# 초기화 & 사이드바
# ============================================================
PortfolioManager.init_state()

st.sidebar.header("⚙️ 시스템 설정")
sys_type = st.sidebar.radio("터틀 돌파 시스템", ("1 (20일/10일)", "2 (55일/20일)"))
entry_w, exit_w = (20, 10) if "1" in sys_type else (55, 20)
acc_size = st.sidebar.number_input("총 자본금", value=10000000, step=1000000)
risk_pct = st.sidebar.slider("1유닛 리스크(%)", 0.5, 5.0, 1.0, 0.1) / 100

# ============================================================
# 메인 UI
# ============================================================
st.title("🦅 CAN SLIM x 🐢 터틀 시스템 실전 매니저")
t0, t1, t2 = st.tabs(["🔥 포지션 관리 및 알림판", "📊 관심종목 스캐너", "📈 정밀 분석 및 차트"])

# ── TAB 0: 포지션 관리 및 알림판 (완전 복원) ───────────────────
with t0:
    st.subheader("🚨 실시간 보유 포지션 대응 알림판")
    
    # 보유 포지션에 대한 실시간 시그널 검출 및 요약 출력
    active_alerts = 0
    for _, row in st.session_state.pos.iterrows():
        tk = str(row.get("티커", "")).strip()
        if not tk: continue
        
        df = StrategyAnalyzer.build_indicators(tk, entry_w, exit_w)
        if df is not None and not df.empty:
            lat = df.iloc[-1]
            cp = _f(lat["Close"])
            entry_h = _f(lat["Entry_High"])
            exit_l = _f(lat["Exit_Low"])
            buy_p = float(row.get("최초매수가", 0))
            units = int(row.get("보유유닛", 0))
            
            # 수익률 계산
            pnl_pct = ((cp - buy_p) / buy_p * 100) if buy_p > 0 else 0.0
            
            # 돌파/청산 신호 판단
            if cp >= entry_h and units > 0:
                st.error(f"🔥 **{row['종목명']} ({tk})** 추가 매수 신호 발생! 현재가({cp:,.0f})가 {entry_w}일 최고점({entry_h:,.0f})을 돌파했습니다. (현재 손익: {pnl_pct:+.2f}%)")
                active_alerts += 1
            elif cp <= exit_l and units > 0:
                st.warning(f"⚠️ **{row['종목명']} ({tk})** 전량 청산(스톱) 신호 발생! 현재가({cp:,.0f})가 {exit_w}일 최저점({exit_l:,.0f})을 하향 돌파했습니다.")
                active_alerts += 1
                
    if active_alerts == 0:
        st.success("✅ 현재 특이 시그널이 발생한 보유 종목이 없습니다. 모든 포지션을 안정적으로 유지 중입니다.")
        
    st.markdown("---")
    st.subheader("🛠 보유 포지션 편집 및 저장")
    edited = st.data_editor(st.session_state.pos, num_rows="dynamic", use_container_width=True, key=f"ed_{st.session_state.ev}")
    
    c1, c2 = st.columns([2, 1])
    if c1.button("🔄 저장 및 종목명 업데이트", type="primary", use_container_width=True):
        wk = PortfolioManager.clean(edited)
        rows = []
        for _, r in wk.iterrows():
            t, n = DataFetcher.resolve_ticker(r["티커"], st.session_state.cache)
            if t: st.session_state.cache[r["티커"]] = (t, n)
            rows.append({"티커": t or r["티커"], "종목명": n or r["티커"], "최초매수가": r["최초매수가"], "보유유닛": r["보유유닛"]})
        st.session_state.pos = PortfolioManager.clean(pd.DataFrame(rows))
        st.session_state.ev += 1
        st.rerun()

# ── TAB 1: 관심종목 스캐너 (쉼표 구분 리스트 복원) ───────────────
with t1:
    st.subheader("📊 멀티 종목 실전 스캐너")
    
    # 사라졌던 쉼표 구분 관심종목 입력 필드 완전 복원
    interest_raw = st.text_area(
        "스캔할 관심 종목 리스트 (쉼표 구분 - 국내 주식은 숫자만 입력 가능)",
        value=st.session_state.interest_input,
        help="예시: 005930, 000660, AAPL, NVDA"
    )
    st.session_state.interest_input = interest_raw
    
    # 1. 텍스트 창 파싱 및 티커 표준화
    input_tickers = [t.strip() for t in interest_raw.split(",") if t.strip()]
    
    # 2. 포지션 탭에 있는 티커들도 함께 병합하여 전체 타겟 목록 구성
    pos_tickers = [str(r.get("티커", "")).strip() for _, r in st.session_state.pos.iterrows() if str(r.get("티커", ""))]
    all_target_raw = list(dict.fromkeys(input_tickers + pos_tickers))
    
    scan_data = []
    
    with st.spinner("🚀 실시간 시장 데이터 및 지표 분석 중..."):
        for raw_tk in all_target_raw:
            if not raw_tk: continue
            # 티커 해상 및 종목명 추출
            tk, name = DataFetcher.resolve_ticker(raw_tk, st.session_state.cache)
            if tk:
                st.session_state.cache[raw_tk] = (tk, name)
            else:
                tk = raw_tk
                name = raw_tk
                
            df = StrategyAnalyzer.build_indicators(tk, entry_w, exit_w)
            if df is not None and not df.empty:
                lat = df.iloc[-1]
                cp = _f(lat["Close"])
                atr = _f(lat["ATR"])
                vol = StrategyAnalyzer.volume_signal(df)
                
                # 터틀 1유닛 수량 계산
                unit_size = int((acc_size * risk_pct) / atr) if atr > 0 else 0
                
                # 돌파 상태 텍스트화
                entry_h = _f(lat["Entry_High"])
                exit_l = _f(lat["Exit_Low"])
                status = "🔥 진입 대기/돌파" if cp >= entry_h else ("⚠️ 위험/청산근접" if cp <= exit_l else "🟡 유지/관망")
                
                scan_data.append({
                    "티커": tk,
                    "종목명": name,
                    "현재가": round(cp, 2),
                    "ATR (N)": round(atr, 2),
                    "적정 1유닛": f"{unit_size} 주",
                    "수급상태": vol["signal"],
                    "돌파상태": status
                })
    
    if scan_data:
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True)
    else:
        st.info("상단 관심 종목 창 또는 포지션 탭에 종목을 입력하시면 스캔이 시작됩니다.")

# ── TAB 2: 차트 & 정밀 분석 ──────────────────────────────────
with t2:
    st.subheader("📈 개별 종목 분석 및 차트")
    # 포지션 및 텍스트창에 등장한 모든 유효 티커 추출
    tk_list = []
    for raw_tk in all_target_raw:
        tk, _ = DataFetcher.resolve_ticker(raw_tk, st.session_state.cache)
        if tk: tk_list.append(tk)
    tk_list = list(dict.fromkeys(tk_list))
    
    if tk_list:
        sel_tk = st.selectbox("정밀 분석할 종목 선택", tk_list)
        df_c = StrategyAnalyzer.build_indicators(sel_tk, entry_w, exit_w)
        
        if df_c is not None and not df_c.empty:
            lat = df_c.iloc[-1]
            cp = _f(lat["Close"])
            atr = _f(lat["ATR"])
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 📋 CAN SLIM 추세 필터")
                checks = {
                    "현재가 > 150/200일선": cp > _f(lat["SMA_150"]) and cp > _f(lat["SMA_200"]),
                    "150일선 > 200일선": _f(lat["SMA_150"]) > _f(lat["SMA_200"]),
                    "50일선 > 150/200일선": _f(lat["SMA_50"]) > _f(lat["SMA_150"]) and _f(lat["SMA_50"]) > _f(lat["SMA_200"]),
                    "52주 최고가 대비 -25% 이내": _f(lat["W52_High"]) > 0 and cp >= _f(lat["W52_High"]) * 0.75,
                }
                for k, v in checks.items():
                    st.markdown(f"{'✅' if v else '❌'} {k}")
            
            with c2:
                st.markdown("### 🐢 터틀 포지션 지표")
                unit_qty = int((acc_size * risk_pct) / atr) if atr > 0 else 0
                st.write(f"- **현재가**: {cp:,.0f} (원/달러)")
                st.write(f"- **현재 N (ATR)**: {atr:,.2f}")
                st.write(f"- **적정 1 Unit 크기**: {unit_qty} 주")
                st.write(f"- **최대 손절폭 (2N)**: {atr * 2:,.2f}")

            st.markdown("---")
            st.markdown("### 📊 정밀 융합 차트 (Plotly)")
            
            plot_df = df_c.tail(150)
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
                low=plot_df["Low"], close=plot_df["Close"], name="Price"
            ))
            
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_50"], line=dict(color="blue", width=1), name="50 SMA"))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_150"], line=dict(color="orange", width=1), name="150 SMA"))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_200"], line=dict(color="red", width=2), name="200 SMA"))
            
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Entry_High"], line=dict(color="green", width=2, dash="dash"), name=f"{entry_w}d High (진입)"))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Exit_Low"], line=dict(color="purple", width=2, dash="dash"), name=f"{exit_w}d Low (청산)"))
            
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("해당 종목의 데이터를 정상적으로 파싱할 수 없습니다.")
