import math
import time
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# =========================
# Auto-refresh every 10 minutes
# =========================
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0

_ = st_autorefresh(interval=600_000, key="auto_refresh")

if "last_snapshot_time" not in st.session_state:
    st.session_state["last_snapshot_time"] = time.time()

def auto_snapshot_due():
    now = time.time()
    if now - st.session_state["last_snapshot_time"] >= 600:
        st.session_state["last_snapshot_time"] = now
        return True
    return False

refresh_btn = st.sidebar.button("Refresh Now")

# =========================
# Constants
# =========================
IST = pytz.timezone("Asia/Kolkata")
RISK_FREE = 0.06
DIV_YIELD = 0.00

# =========================
# NSE client
# =========================
class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive",
        })

    def _warm_cookies(self):
        try:
            self.session.get("https://www.nseindia.com/", timeout=10)
        except Exception:
            pass

    def get_json(self, url, max_tries=5, sleep_between=1.4):
        last_err = None
        for _ in range(max_tries):
            try:
                self._warm_cookies()
                resp = self.session.get(url, timeout=12)
                if resp.status_code in (401, 403):
                    last_err = Exception(f"HTTP {resp.status_code}")
                    time.sleep(sleep_between)
                    continue
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(sleep_between)
        st.error(f"NSE API error: {last_err}")
        return None

nse = NSEClient()

# =========================
# Helpers
# =========================
def format_inr(value: float) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= 1e7:
        return f"{v/1e7:.2f} Cr"
    elif v >= 1e5:
        return f"{v/1e5:.2f} Lakh"
    else:
        return f"{v:.0f}"

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1_d2(S, K, r, q, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return None, None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_delta_gamma(S, K, r, q, sigma, T, option_type: str):
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return 0.0, 0.0
    if option_type == "C":
        delta = math.exp(-q * T) * norm_cdf(d1)
    else:
        delta = -math.exp(-q * T) * norm_cdf(-d1)
    gamma = (math.exp(-q * T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))
    return delta, gamma

def nearest_index(sorted_list, x):
    if not sorted_list:
        return None
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))

def parse_expiry_to_dt(expiry_str: str):
    fmts = ["%d-%b-%Y", "%d-%b-%y"]
    dt_naive = None
    for f in fmts:
        try:
            dt_naive = datetime.strptime(expiry_str, f)
            break
        except Exception:
            continue
    if dt_naive is None:
        return None
    return IST.localize(dt_naive.replace(hour=15, minute=30, second=0, microsecond=0))

def time_to_expiry_years(expiry_str: str) -> float:
    expiry_dt = parse_expiry_to_dt(expiry_str)
    if expiry_dt is None:
        return 1.0 / 365.0
    now_ist = datetime.now(IST)
    sec = (expiry_dt - now_ist).total_seconds()
    if sec <= 0:
        return 1.0 / (365.0 * 24.0 * 3600.0)
    return sec / (365.0 * 24.0 * 3600.0)

def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

# =========================
# Compute Metrics
# =========================
def compute_metrics(data: dict, symbol: str, expiry: str):
    recs = (data or {}).get("records", {})
    rows = recs.get("data", []) or []
    spot = recs.get("underlyingValue")
    if spot is None:
        return None

    T = time_to_expiry_years(expiry)
    r = RISK_FREE
    q = DIV_YIELD

    ce_rows, pe_rows, strikes_all = [], [], []
    for item in rows:
        if item.get("expiryDate") != expiry:
            continue
        strike = item.get("strikePrice")
        ce, pe = item.get("CE"), item.get("PE")
        strikes_all.append(strike)
        if ce:
            ce_rows.append({
                "strike": strike,
                "iv": (ce.get("impliedVolatility", 0.0) or 0.0) / 100.0,
                "oi": ce.get("openInterest", 0) or 0,
                "bidQty": ce.get("bidQty", 0) or 0,
                "ltp": ce.get("lastPrice", 0.0) or 0.0,
            })
        if pe:
            pe_rows.append({
                "strike": strike,
                "iv": (pe.get("impliedVolatility", 0.0) or 0.0) / 100.0,
                "oi": pe.get("openInterest", 0) or 0,
                "bidQty": pe.get("bidQty", 0) or 0,
                "ltp": pe.get("lastPrice", 0.0) or 0.0,
            })

    strikes = sorted(set(strikes_all))
    if not strikes:
        return None

    # IV sums
    call_iv_sum = sum((r["iv"] * 100.0) for r in ce_rows)
    put_iv_sum  = sum((r["iv"] * 100.0) for r in pe_rows)

    # Minimum bid qty on each side
    def min_pos(rows):
        vals = [r["bidQty"] for r in rows if r["bidQty"] > 0]
        return min(vals) if vals else 1

    min_call_q = min_pos(ce_rows)
    min_put_q  = min_pos(pe_rows)

    tot_call_oi, tot_put_oi = 0, 0
    call_value_sum, put_value_sum = 0.0, 0.0
    dex_net, gex_net = 0.0, 0.0

    for r in ce_rows:
        tot_call_oi += r["oi"]
        call_value_sum += r["oi"] * min_call_q * r["ltp"]
        sigma = max(r["iv"], 1e-6)
        d, g = bs_delta_gamma(spot, r["strike"], RISK_FREE, DIV_YIELD, sigma, T, "C")
        dex_net += d * r["oi"] * min_call_q
        gex_net += g * r["oi"] * min_call_q * (spot**2)

    for r in pe_rows:
        tot_put_oi += r["oi"]
        put_value_sum += r["oi"] * min_put_q * r["ltp"]
        sigma = max(r["iv"], 1e-6)
        d, g = bs_delta_gamma(spot, r["strike"], RISK_FREE, DIV_YIELD, sigma, T, "P")
        dex_net += d * r["oi"] * min_put_q
        gex_net -= g * r["oi"] * min_put_q * (spot**2)

    pcr = (tot_put_oi / tot_call_oi) if tot_call_oi > 0 else None

    return {
        "spot": spot,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum":  put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum":  put_value_sum,
        "total_call_oi": tot_call_oi,
        "total_put_oi":  tot_put_oi,
        "pcr": pcr,
        "dex_net": dex_net,
        "gex_net": gex_net,
        "min_call_q": min_call_q,
        "min_put_q": min_put_q,
    }

# =========================
# Sidebar inputs
# =========================
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
data = fetch_option_chain(symbol)
if not data:
    st.stop()

expiries = data["records"].get("expiryDates", []) or []
if not expiries:
    st.error("No expiries found.")
    st.stop()

expiry = st.sidebar.selectbox("Select Expiry", expiries)

# =========================
# Compute
# =========================
m = compute_metrics(data, symbol, expiry)
if m is None:
    st.error("Unable to compute metrics.")
    st.stop()

prev = st.session_state.get("history", [])[-1] if st.session_state.get("history") else {}

# =========================
# Final Comment (simplified example here, can expand further)
# =========================
def final_comment(m, prev):
    notes = []
    if m["pcr"] is not None:
        if m["pcr"] > 1.1:
            notes.append("Put positioning supports upside")
        elif m["pcr"] < 0.9:
            notes.append("Call positioning adds resistance")

    if m["dex_net"] > 0:
        notes.append("Dealer hedging adds support")
    elif m["dex_net"] < 0:
        notes.append("Dealer hedging adds pressure")

    if m["gex_net"] > 0:
        notes.append("Positive gamma keeps moves contained")
    elif m["gex_net"] < 0:
        notes.append("Negative gamma can amplify moves")

    return " | ".join(notes) if notes else "Flows mixed — no clear edge."

st.subheader("Final Comment")
st.caption(final_comment(m, prev))

# =========================
# Snapshot History
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

if refresh_btn or auto_snapshot_due():
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,
        "Symbol": symbol,
        "Expiry": expiry,
        "Spot": m["spot"],
        "PCR": m["pcr"],
        "DEX Net": m["dex_net"],
        "GEX Net": m["gex_net"],
        "Total Call OI": m["total_call_oi"],
        "Total Put OI":  m["total_put_oi"],
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
else:
    st.info("Snapshots will appear here every ~10 minutes or on Refresh Now.")
