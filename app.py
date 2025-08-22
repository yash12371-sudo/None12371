import time
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st

# =========================================================
# Page Setup
# =========================================================
st.set_page_config(page_title="Dashboard", layout="wide")

# ---------- Custom "Magical" CSS (neutral, no red/green) ----------
MAGICAL_CSS = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 20% 10%, #101524 0%, #0B0F1A 40%, #070A12 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(16,21,36,0.9) 0%, rgba(11,15,26,0.95) 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}

/* Headings */
h1, h2, h3, h4, h5 {
  font-weight: 700 !important;
  letter-spacing: 0.3px;
  color: #EDEBFF;
}

/* Glass cards */
.card {
  border-radius: 16px;
  padding: 18px 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* Metric title */
.card .title {
  font-size: 0.95rem;
  color: #C9C5FF;
  margin: 0 0 6px 0;
}

/* Metric value */
.card .value {
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: 0.3px;
  background: linear-gradient(90deg, #C9C5FF 0%, #A0E7E5 50%, #F9F871 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin: 0 0 4px 0;
}

/* Metric delta (neutral palette) */
.card .delta {
  font-size: 0.85rem;
  color: #B8C0FF; /* calm lavender */
}

/* Top box in sidebar */
.glowbox {
  border-radius: 14px;
  padding: 14px;
  margin-top: 8px;
  background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid rgba(180,170,255,0.25);
  box-shadow: 0 0 18px rgba(140,120,255,0.15);
}

/* Dataframe tweaks */
div[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

/* Subtle labels */
.muted {
  color: #9AA6BF;
  font-size: 0.9rem;
  margin-top: -8px;
  margin-bottom: 12px;
  text-align: center;
}
</style>
"""
st.markdown(MAGICAL_CSS, unsafe_allow_html=True)

# Header
st.markdown(
    "<h1 style='text-align:center; "
    "background: linear-gradient(90deg, #6A5ACD, #00CED1); "
    "-webkit-background-clip: text; color: transparent;'>"
    "Options Dashboard</h1>",
    unsafe_allow_html=True,
)

# =========================================================
# Auto-refresh (every 10 minutes)
# =========================================================
AUTO_INTERVAL_MS = 600_000  # 10 minutes

try:
    from streamlit_autorefresh import st_autorefresh
    auto_count = st_autorefresh(interval=AUTO_INTERVAL_MS, key="auto_refresh_key")
except Exception:
    # Fallback if package isn't available: no auto-refresh
    auto_count = 0
    st.caption("Auto-refresh disabled (install `streamlit-autorefresh` to enable).")

# For controlled snapshot on auto-refresh
if "auto_tick" not in st.session_state:
    st.session_state["auto_tick"] = auto_count

# Last updated display (IST)
IST = pytz.timezone("Asia/Kolkata")
last_updated = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<div class='muted'>⏳ Auto-refresh every 10 minutes &nbsp;|&nbsp; Last updated: {last_updated} IST</div>", unsafe_allow_html=True)

# =========================================================
# NSE Client (robust session with cookie warm-up & retries)
# =========================================================
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
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
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
                # Some times content-type is wrong; try json anyway
                try:
                    return resp.json()
                except Exception as e:
                    last_err = e
                    time.sleep(sleep_between)
                    continue
            except Exception as e:
                last_err = e
                time.sleep(sleep_between)
        st.error(f"NSE API error: {last_err}")
        return None

nse = NSEClient()

# =========================================================
# Helpers
# =========================================================
def format_inr(v: float) -> str:
    """Format a number in Cr/Lakh units for display."""
    if v is None:
        return "-"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if x >= 1e7:
        return f"{x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"{x/1e5:.2f} Lakh"
    return f"{x:.0f}"

def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

def compute_metrics(data: dict, expiry: str):
    """
    Compute:
      - Spot
      - Sum of Call IV / Put IV
      - Sum of Call Value / Put Value
        Value per strike = OI × (MIN BidQty across that side for this expiry) × LTP
      - Top 2 strikes by Value (Call & Put)
    """
    records = (data or {}).get("records", {})
    rows = records.get("data", []) or []

    ce_rows, pe_rows = [], []
    for item in rows:
        if item.get("expiryDate") != expiry:
            continue
        strike = item.get("strikePrice")
        ce, pe = item.get("CE"), item.get("PE")
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
    put_iv_sum  = sum(r["iv"] for r in pe_rows)

    def min_positive_bidqty(rows_list):
        vals = [r["bidQty"] for r in rows_list if r["bidQty"] and r["bidQty"] > 0]
        return min(vals) if vals else 0

    min_call_bq = min_positive_bidqty(ce_rows)
    min_put_bq  = min_positive_bidqty(pe_rows)

    for r in ce_rows:
        r["value"] = r["oi"] * min_call_bq * r["ltp"]
    for r in pe_rows:
        r["value"] = r["oi"] * min_put_bq * r["ltp"]

    call_value_sum = sum(r["value"] for r in ce_rows)
    put_value_sum  = sum(r["value"] for r in pe_rows)

    top_calls = sorted(ce_rows, key=lambda x: x["value"], reverse=True)[:2]
    top_puts  = sorted(pe_rows, key=lambda x: x["value"], reverse=True)[:2]

    spot = records.get("underlyingValue", None)

    return {
        "spot": spot,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum": put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum": put_value_sum,
        "top_calls": top_calls,
        "top_puts": top_puts,
    }

def delta_text(curr, prev, kind="num"):
    """
    Neutral delta string:
      kind="num" -> plain number delta with 2 decimals
      kind="inr" -> formatted using Cr/Lakh
    Uses unicode arrows in neutral style.
    """
    if prev is None:
        return "Δ –"
    d = curr - prev
    if kind == "inr":
        txt = format_inr(abs(d))
    else:
        txt = f"{abs(d):.2f}"
    if d > 0:
        return f"Δ ↑ {txt}"
    if d < 0:
        return f"Δ ↓ {txt}"
    return "Δ 0.00"

def card(title: str, value: str, delta: str = ""):
    """Render a glass card metric."""
    st.markdown(
        f"""
        <div class="card">
          <div class="title">{title}</div>
          <div class="value">{value}</div>
          <div class="delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Sidebar Controls
# =========================================================
st.sidebar.header("Controls")

symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

# Single API call; also fetch expiries from same response to reduce failures
chain_data = fetch_option_chain(symbol)
if not chain_data or "records" not in chain_data:
    st.stop()

expiries = chain_data["records"].get("expiryDates", []) or []
if not expiries:
    st.error("Could not load expiries. Please try again.")
    st.stop()

default_idx = 0
if "selected_expiry" in st.session_state and st.session_state["selected_expiry"] in expiries:
    default_idx = expiries.index(st.session_state["selected_expiry"])

expiry = st.sidebar.selectbox("Select Expiry", expiries, index=default_idx)
st.session_state["selected_expiry"] = expiry

refresh_btn = st.sidebar.button("Refresh Now")

# =========================================================
# Compute Metrics
# =========================================================
metrics = compute_metrics(chain_data, expiry)

# Previous snapshot (if any) for deltas
prev_call_val = prev_put_val = prev_call_iv = prev_put_iv = None
if "history" in st.session_state and st.session_state["history"]:
    prev = st.session_state["history"][-1]
    prev_call_val = prev["Call Value"]
    prev_put_val  = prev["Put Value"]
    prev_call_iv  = prev["Call IV Sum"]
    prev_put_iv   = prev["Put IV Sum"]

# =========================================================
# Metrics Row (glass cards)
# =========================================================
col0, col1, col2, col3, col4 = st.columns(5)

with col0:
    card("Spot Price",
         f"{metrics['spot']:.2f}" if metrics['spot'] is not None else "-",
         "")

with col1:
    card("Call IV Sum",
         f"{metrics['call_iv_sum']:.2f}",
         delta_text(metrics["call_iv_sum"], prev_call_iv, kind="num"))

with col2:
    card("Put IV Sum",
         f"{metrics['put_iv_sum']:.2f}",
         delta_text(metrics["put_iv_sum"], prev_put_iv, kind="num"))

with col3:
    card("Call Value",
         format_inr(metrics["call_value_sum"]),
         delta_text(metrics["call_value_sum"], prev_call_val, kind="inr"))

with col4:
    card("Put Value",
         format_inr(metrics["put_value_sum"]),
         delta_text(metrics["put_value_sum"], prev_put_val, kind="inr"))

# =========================================================
# Top Strikes (sidebar, glowing box)
# =========================================================
st.sidebar.subheader("Top Strikes by Value")
with st.sidebar.container():
    st.markdown("<div class='glowbox'><b>Calls (Top 2)</b><br/>", unsafe_allow_html=True)
    if metrics["top_calls"]:
        for r in metrics["top_calls"]:
            st.markdown(f"• Strike {r['strike']}: {format_inr(r['value'])}")
    else:
        st.markdown("• No call data")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glowbox'><b>Puts (Top 2)</b><br/>", unsafe_allow_html=True)
    if metrics["top_puts"]:
        for r in metrics["top_puts"]:
            st.markdown(f"• Strike {r['strike']}: {format_inr(r['value'])}")
    else:
        st.markdown("• No put data")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Snapshot History (append on manual or auto refresh)
# =========================================================
if "history" not in st.session_state:
    st.session_state["history"] = []

# Append snapshot on (1) manual click or (2) auto refresh tick changed
append_snapshot = False
if refresh_btn:
    append_snapshot = True
elif auto_count != st.session_state.get("auto_tick", 0):
    append_snapshot = True
    st.session_state["auto_tick"] = auto_count

if append_snapshot:
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,
        "Spot": metrics["spot"],
        "Call IV Sum": round(metrics["call_iv_sum"], 2),
        "Put IV Sum": round(metrics["put_iv_sum"], 2),
        "Call Value": float(metrics["call_value_sum"]),
        "Put Value": float(metrics["put_value_sum"]),
        "Symbol": symbol,
        "Expiry": expiry,
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    hist_df = pd.DataFrame(st.session_state["history"])
    # Display with formatted values
    show_df = hist_df.copy()
    show_df["Call Value"] = show_df["Call Value"].apply(format_inr)
    show_df["Put Value"]  = show_df["Put Value"].apply(format_inr)
    st.dataframe(show_df, use_container_width=True)

    # Download raw numeric CSV
    csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download History CSV", data=csv_bytes, file_name="history.csv", mime="text/csv")
else:
    st.info("Snapshots will appear here every 10 minutes or when you click “Refresh Now”.")
