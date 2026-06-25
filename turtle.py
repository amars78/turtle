import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ── 로깅 설정 ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── pykrx (선택적) ──────────────────────────────────────────
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# ── 페이지 설정 (반드시 최상단) ─────────────────────────────
st.set_page_config(page_title="CAN SLIM x 터틀 매니저", layout="wide")

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
                {"티커": "AAPL", "종목명": "Apple", "최초매수가": 150.0, "보유유닛": 0}
            ])
        if "cache" not in st.session_state: st.session_state.cache = {}
        if "ev" not in st.session_state: st.session_state.ev = 0

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
st.title("🦅 CAN SLIM x 🐢 터틀 시스템")
t0, t1, t2 = st.tabs(["🔥 포지션 관리", "📊 종목 스캐너", "📈 정밀 분석 및 차트"])

# ── TAB 0: 포지션 ───────────────────────────────────────────
with t0:
    st.subheader("🛠 보유 포지션 편집")
    edited = st.data_editor(st.session_state.pos, num_rows="dynamic", use_container_width=True, key=f"ed_{st.session_state.ev}")
    
    c1, c2 = st.columns([2, 1])
    if c1.button("🔄 저장 및 종목명 업데이트", type="primary"):
        wk = PortfolioManager.clean(edited)
        rows = []
        for _, r in wk.iterrows():
            t, n = DataFetcher.resolve_ticker(r["티커"], st.session_state.cache)
            if t: st.session_state.cache[r["티커"]] = (t, n)
            rows.append({"티커": t or r["티커"], "종목명": n or r["티커"], "최초매수가": r["최초매수가"], "보유유닛": r["보유유닛"]})
        st.session_state.pos = PortfolioManager.clean(pd.DataFrame(rows))
        st.session_state.ev += 1
        st.rerun()

# ── TAB 1: 스캐너 (복원됨) ──────────────────────────────────
with t1:
    st.subheader("📊 관심/보유 종목 스캔 결과")
    scan_data = []
    
    for _, row in st.session_state.pos.iterrows():
        tk = str(row.get("티커", "")).strip()
        if not tk: continue
        
        df = StrategyAnalyzer.build_indicators(tk, entry_w, exit_w)
        if df is not None and not df.empty:
            lat = df.iloc[-1]
            cp = _f(lat["Close"])
            atr = _f(lat["ATR"])
            vol = StrategyAnalyzer.volume_signal(df)
            
            # 터틀 1유닛 수량 계산 (계좌리스크 / 1주당 변동성 리스크)
            unit_size = int((acc_size * risk_pct) / atr) if atr > 0 else 0
            
            scan_data.append({
                "티커": tk,
                "종목명": row.get("종목명", tk),
                "현재가": round(cp, 2),
                "ATR (N)": round(atr, 2),
                "권장 1유닛": f"{unit_size} 주",
                "수급상태": vol["signal"]
            })
    
    if scan_data:
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True)
    else:
        st.info("포지션 탭에 분석할 종목을 추가해주세요.")

# ── TAB 2: 차트 & 정밀 분석 (복원됨) ────────────────────────
with t2:
    st.subheader("📈 개별 종목 분석 및 차트")
    tk_list = [str(r.get("티커", "")) for _, r in st.session_state.pos.iterrows() if str(r.get("티커", ""))]
    tk_list = list(dict.fromkeys(tk_list))
    
    if tk_list:
        sel_tk = st.selectbox("분석 종목 선택", tk_list)
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
            
            # Plotly 캔들스틱 및 지표 그리기
            plot_df = df_c.tail(150) # 최근 150일만 표시하여 가독성 확보
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
            st.error("해당 종목의 지표를 계산할 수 없습니다. 상장 폐지되었거나 데이터가 부족할 수 있습니다.")
