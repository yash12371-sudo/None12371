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
# NSE client (robust session)
# --------------------------
class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        # Pretend to be a real browser
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
        # Hit homepage to set cookies (NSE requirement)
        try:
            self.session.get("https://www.nseindia.com/", timeout=10)
        except Exception:
            # Ignore; often still sets cookies
            pass

    def get_json(self, url, max_tries=5, sleep_between=1.2):
        """
        Robust getter:
        - warms cookies
        - retries on JSON decode / 401 / 403
        """
        last_err = None
        for _ in range(max_tries):
            try:
                self._warm_cookies()
                resp = self.session.get(url, timeout=12)
                # If blocked, retry
                if resp.status_code in (401, 403):
                    last_err = Exception(f"HTTP {resp.status_code}")
                    time.sleep(sleep_between)
                    continue
                # Some NSE edges: HTML body -> not JSON
                ctype = resp.headers.get("content-type", "").lower()
                if "application/json" not in ctype:
                    # Try anyway; else retry
                    try:
                        return resp.json()
                    except Exception:
                        last_err = Exception("Non-JSON response from NSE")
                        time.sleep(sleep_between)
                        continue
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(sleep_between)
        # Give a friendly error once all retries fail
        st.error(f"NSE API error: {last_err}")
        return None

nse = NSEClient()

# --------------------------
# Helpers
# --------------------------
def format_inr(value: float) -> str:
    """Format number in Cr/Lakh for readability."""
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
    """Fetch full option-chain JSON for the symbol (indices endpoint)."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

def compute_metrics(data: dict, expiry: str):
    """
    Compute:
      - Sum of Call IV / Put IV
      - Sum of Call Value / Put Value
        where Value = OI × (min BidQty across this expiry & side) × LTP
      - Top 2 strikes by Value (Call / Put)
    """
    records = (data or {}).get("records", {})
    rows = records.get("data", []) or []

    # Collect per-strike inputs first
    ce_rows, pe_rows = [], []
    for item in rows:
        if item.get("expiryDate") != expiry:
            continue
        ce = item.get("CE")
        pe = item.get("PE")
        strike = item.get("strikePrice")

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

    # Sum IVs
    call_iv_sum = sum(r["iv"] for r in ce_rows)
    put_iv_sum = sum(r["iv"] for r in pe_rows)

    # Determine MIN positive BidQty across the expiry (side-wise)
    def min_positive_bidqty(rows_list):
        positives = [r["bidQty"] for r in rows_list if r["bidQty"] and r["bidQty"] > 0]
        return min(positives) if positives else 0

    min_call_bidqty = min_positive_bidqty(ce_rows)
    min_put_bidqty  = min_positive_bidqty(pe_rows)

    # Compute Value per strike using MIN bidQty
    for r in ce_rows:
        r["value"] = r["oi"] * min_call_bidqty * r["ltp"]
    for r in pe_rows:
        r["value"] = r["oi"] * min_put_bidqty * r["ltp"]

    call_value_sum = sum(r["value"] for r in ce_rows)
    put_value_sum  = sum(r["value"] for r in pe_rows)

    # Top 2 strikes by Value
    top_calls = sorted(ce_rows, key=lambda x: x["value"], reverse=True)[:2]
    top_puts  = sorted(pe_rows, key=lambda x: x["value"], reverse=True)[:2]

    return {
        "call_iv_sum": call_iv_sum,
        "put_iv_sum": put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum": put_value_sum,
        "top_calls": top_calls,
        "top_puts": top_puts,
    }

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("Controls")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

# Make a single API call and extract expiries from it (reduces failure rate)
chain_data = fetch_option_chain(symbol)
if not chain_data or "records" not in chain_data:
    st.stop()

expiries = chain_data["records"].get("expiryDates", []) or []
if not expiries:
    st.error("Could not load expiries. Please try again.")
    st.stop()

# Preserve selected expiry across reruns
default_idx = 0
if "selected_expiry" in st.session_state and st.session_state["selected_expiry"] in expiries:
    default_idx = expiries.index(st.session_state["selected_expiry"])

expiry = st.sidebar.selectbox("Select Expiry", expiries, index=default_idx)
st.session_state["selected_expiry"] = expiry

refresh = st.sidebar.button("Refresh Now")

# --------------------------
# Metrics + Top Strikes (Main)
# --------------------------
metrics = compute_metrics(chain_data, expiry)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Call IV Sum", f"{metrics['call_iv_sum']:.2f}")
c2.metric("Put IV Sum", f"{metrics['put_iv_sum']:.2f}")
c3.metric("Call Value", format_inr(metrics["call_value_sum"]))
c4.metric("Put Value", format_inr(metrics["put_value_sum"]))

st.subheader("Top Strikes")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Top 2 Call Strikes**")
    if metrics["top_calls"]:
        for row in metrics["top_calls"]:
            st.write(f"Strike {row['strike']}: {format_inr(row['value'])}")
    else:
        st.write("No call data for this expiry.")

with col_right:
    st.markdown("**Top 2 Put Strikes**")
    if metrics["top_puts"]:
        for row in metrics["top_puts"]:
            st.write(f"Strike {row['strike']}: {format_inr(row['value'])}")
    else:
        st.write("No put data for this expiry.")

# --------------------------
# Snapshot History (manual save on Refresh)
# --------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if refresh:
    ist = pytz.timezone("Asia/Kolkata")
    ts = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,
        "Call IV Sum": round(metrics["call_iv_sum"], 2),
        "Put IV Sum": round(metrics["put_iv_sum"], 2),
        "Call Value": metrics["call_value_sum"],
        "Put Value": metrics["put_value_sum"],
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    hist_df = pd.DataFrame(st.session_state["history"])
    # Display with formatted values
    display_df = hist_df.copy()
    display_df["Call Value"] = display_df["Call Value"].apply(format_inr)
    display_df["Put Value"] = display_df["Put Value"].apply(format_inr)
    st.dataframe(display_df, use_container_width=True)

    # Download raw numeric CSV (good for analysis)
    csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download History CSV", data=csv_bytes, file_name="history.csv", mime="text/csv")
else:
    st.info("Click ‘Refresh Now’ to capture a snapshot.")
