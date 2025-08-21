
import os
import csv
import requests
import pandas as pd
from datetime import datetime
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "snapshots.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
}

IST = pytz.timezone("Asia/Kolkata")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    with requests.Session() as s:
        s.headers.update(HEADERS)
        try:
            s.get("https://www.nseindia.com/", timeout=8)
            s.get("https://www.nseindia.com/option-chain", timeout=8)
        except Exception:
            pass
        r = s.get(url, timeout=15)
        r.raise_for_status()
        return r.json()

def compute_iv_sums(chain_json: dict, expiry: str):
    if not chain_json or "records" not in chain_json:
        return None, None, 0
    rows = chain_json.get("records", {}).get("data", [])
    filtered = [row for row in rows if row.get("expiryDate") == expiry]
    call_sum = sum(row.get("CE", {}).get("impliedVolatility", 0) for row in filtered if row.get("CE"))
    put_sum  = sum(row.get("PE", {}).get("impliedVolatility", 0) for row in filtered if row.get("PE"))
    return float(call_sum), float(put_sum), len(filtered)

def load_history() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH)
        except Exception:
            return pd.DataFrame(columns=["Timestamp","Symbol","Expiry","Call IV","Put IV"])
    return pd.DataFrame(columns=["Timestamp","Symbol","Expiry","Call IV","Put IV"])

def append_history(ts: str, symbol: str, expiry: str, call_iv: float, put_iv: float):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Timestamp","Symbol","Expiry","Call IV","Put IV"])
        writer.writerow([ts, symbol, expiry, f"{call_iv:.6f}", f"{put_iv:.6f}"])

def get_previous_snapshot(df: pd.DataFrame, symbol: str, expiry: str):
    if df.empty:
        return None
    subset = df[(df["Symbol"] == symbol) & (df["Expiry"] == expiry)]
    if subset.empty:
        return None
    return subset.sort_values("Timestamp").iloc[-1]

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
    reset = st.button("🗑️ Reset History (local)")
    st.caption("Auto-refresh is fixed at 10 minutes. Use 'Refresh Now' to force an update.")

if reset and os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)
    st.sidebar.success("History cleared.")

# Fetch data
try:
    data = fetch_option_chain(symbol)
except Exception as e:
    st.error(f"Failed to fetch option chain: {e}")
    data = None

# Expiry dropdown
expiry = None
if data and "records" in data:
    expiries = data["records"].get("expiryDates", [])
    if expiries:
        expiry = st.selectbox("Select Expiry", expiries, index=0)

# Compute and display
if expiry:
    call_iv, put_iv, nrows = compute_iv_sums(data, expiry)
    if nrows == 0:
        st.warning("No rows for the selected expiry. Try another expiry.")
    else:
        hist_df = load_history()
        prev = get_previous_snapshot(hist_df, symbol, expiry)
        prev_call = float(prev["Call IV"]) if prev is not None else None
        prev_put  = float(prev["Put IV"]) if prev is not None else None

        col1, col2, col3 = st.columns(3)
        delta_call = None if prev_call is None else call_iv - prev_call
        delta_put  = None if prev_put  is None else put_iv  - prev_put

        col1.metric("Σ Call IV (selected expiry)", f"{call_iv:,.2f}", None if delta_call is None else f"{delta_call:+.2f}")
        col2.metric("Σ Put IV (selected expiry)",  f"{put_iv:,.2f}",  None if delta_put  is None else f"{delta_put:+.2f}")
        col3.metric("Rows counted", f"{nrows:,}")

        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last updated (IST): {now}")

        should_save = (prev is None) or (abs(delta_call) > 1e-9) or (abs(delta_put) > 1e-9)
        if should_save:
            append_history(now, symbol, expiry, call_iv, put_iv)
            hist_df = load_history()

        with st.expander("Snapshot History (persistent CSV)"):
            st.dataframe(hist_df.sort_values("Timestamp", ascending=False), use_container_width=True)
        st.sidebar.subheader("Recent Snapshots")
        st.sidebar.dataframe(hist_df.sort_values("Timestamp", ascending=False).head(25), height=350, use_container_width=True)
        st.sidebar.download_button("Download Full CSV", hist_df.to_csv(index=False).encode("utf-8"), file_name="snapshots.csv", mime="text/csv")

if st.button("🔄 Refresh Now"):
    fetch_option_chain.clear()
    st.rerun()

st_autorefresh(interval=600000, key="iv_refresh_key")
