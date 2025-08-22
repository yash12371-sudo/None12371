import streamlit as st
import requests
import pandas as pd
import pytz
import time
from datetime import datetime

# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# --------------------------
# Auto-refresh every 10 minutes
# --------------------------
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    from streamlit import autorefresh as st_autorefresh

count = st_autorefresh(interval=600000, key="auto_refresh")  # 10 min

# --------------------------
# NSE client
# --------------------------
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

    def get_json(self, url, max_tries=5, sleep_between=1.5):
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

# --------------------------
# Helpers
# --------------------------
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

def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

def compute_metrics(data: dict, expiry: str):
    records = (data or {}).get("records", {})
    rows = records.get("data", []) or []

    ce_rows, pe_rows = [], []
    for item in rows:
        if item.get("expiryDate") != expiry:
            continue
        ce, pe, strike = item.get("CE"), item.get("PE"), item.get("strikePrice")
        if ce:
            ce_rows.append({
                "strike": strike,
                "iv": ce.get("impliedVolatility", 0) or 0.0,
                "oi": ce.get("openInterest", 0) or 0,
                "bidQty": ce.get("bidQty", 0) or 0,
                "ltp": ce.get("lastPrice", 0) or 0.0,
            })
        if pe:
            pe_rows.append({
                "strike": strike,
                "iv": pe.get("impliedVolatility", 0) or 0.0,
                "oi": pe.get("openInterest", 0) or 0,
                "bidQty": pe.get("bidQty", 0) or 0,
                "ltp": pe.get("lastPrice", 0) or 0.0,
            })

    call_iv_sum = sum(r["iv"] for r in ce_rows)
    put_iv_sum = sum(r["iv"] for r in pe_rows)

    def min_positive(rows_list):
        vals = [r["bidQty"] for r in rows_list if r["bidQty"] > 0]
        return min(vals) if vals else 0

    min_call_bidqty = min_positive(ce_rows)
    min_put_bidqty = min_positive(pe_rows)

    for r in ce_rows:
        r["value"] = r["oi"] * min_call_bidqty * r["ltp"]
    for r in pe_rows:
        r["value"] = r["oi"] * min_put_bidqty * r["ltp"]

    call_value_sum = sum(r["value"] for r in ce_rows)
    put_value_sum = sum(r["value"] for r in pe_rows)

    top_calls = sorted(ce_rows, key=lambda x: x["value"], reverse=True)[:2]
    top_puts = sorted(pe_rows, key=lambda x: x["value"], reverse=True)[:2]

    spot_price = records.get("underlyingValue", None)

    return {
        "spot": spot_price,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum": put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum": put_value_sum,
        "top_calls": top_calls,
        "top_puts": top_puts,
    }

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Controls")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

chain_data = fetch_option_chain(symbol)
if not chain_data or "records" not in chain_data:
    st.stop()

expiries = chain_data["records"].get("expiryDates", []) or []
if not expiries:
    st.error("No expiries found.")
    st.stop()

default_idx = 0
if "selected_expiry" in st.session_state and st.session_state["selected_expiry"] in expiries:
    default_idx = expiries.index(st.session_state["selected_expiry"])

expiry = st.sidebar.selectbox("Select Expiry", expiries, index=default_idx)
st.session_state["selected_expiry"] = expiry

refresh_btn = st.sidebar.button("Refresh Now")

# --------------------------
# Metrics
# --------------------------
metrics = compute_metrics(chain_data, expiry)

# Compare with last snapshot if available
prev_call_val, prev_put_val, prev_call_iv, prev_put_iv = None, None, None, None
if "history" in st.session_state and st.session_state["history"]:
    prev = st.session_state["history"][-1]
    prev_call_val = prev["Call Value"]
    prev_put_val = prev["Put Value"]
    prev_call_iv = prev["Call IV Sum"]
    prev_put_iv = prev["Put IV Sum"]

def diff_text(curr, prev, is_iv=False):
    if prev is None:
        return "–"
    diff = curr - prev
    if diff > 0:
        return f"▲ {round(diff,2) if is_iv else format_inr(diff)}"
    elif diff < 0:
        return f"▼ {round(abs(diff),2) if is_iv else format_inr(abs(diff))}"
    else:
        return "No change"

c0, c1, c2, c3, c4 = st.columns(5)
c0.metric("Spot Price", f"{metrics['spot']:.2f}" if metrics['spot'] else "-")
c1.metric("Call IV Sum", f"{metrics['call_iv_sum']:.2f}", diff_text(metrics["call_iv_sum"], prev_call_iv, True))
c2.metric("Put IV Sum", f"{metrics['put_iv_sum']:.2f}", diff_text(metrics["put_iv_sum"], prev_put_iv, True))
c3.metric("Call Value", format_inr(metrics["call_value_sum"]), diff_text(metrics["call_value_sum"], prev_call_val))
c4.metric("Put Value", format_inr(metrics["put_value_sum"]), diff_text(metrics["put_value_sum"], prev_put_val))

st.subheader("Top Strikes")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top 2 Calls**")
    for r in metrics["top_calls"]:
        st.write(f"Strike {r['strike']}: {format_inr(r['value'])}")
with col2:
    st.markdown("**Top 2 Puts**")
    for r in metrics["top_puts"]:
        st.write(f"Strike {r['strike']}: {format_inr(r['value'])}")

# --------------------------
# Snapshot History
# --------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if refresh_btn or count > 0:
    ist = pytz.timezone("Asia/Kolkata")
    ts = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,
        "Spot": metrics["spot"],
        "Call IV Sum": round(metrics["call_iv_sum"], 2),
        "Put IV Sum": round(metrics["put_iv_sum"], 2),
        "Call Value": metrics["call_value_sum"],
        "Put Value": metrics["put_value_sum"],
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    show_df = df.copy()
    show_df["Call Value"] = show_df["Call Value"].apply(format_inr)
    show_df["Put Value"] = show_df["Put Value"].apply(format_inr)
    st.dataframe(show_df, use_container_width=True)
else:
    st.info("Snapshots will appear here (auto every 10 min, or click Refresh Now).")
