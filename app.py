
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.title("Dashboard")

# Session state for snapshots
if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = []

# NSE API URL for NIFTY option chain
BASE_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def fetch_data():
    try:
        with requests.Session() as s:
            s.headers.update(HEADERS)
            data = s.get(BASE_URL).json()
            return data
    except Exception as e:
        st.error("Error fetching data: %s" % str(e))
        return None

# Dropdown for expiry selection
data = fetch_data()
if data and "records" in data:
    expiries = data["records"]["expiryDates"]
    expiry = st.selectbox("Select Expiry Date", expiries)
else:
    expiry = None

def calculate_iv_sums(expiry):
    if not data or "records" not in data:
        return None, None
    filtered = [i for i in data["records"]["data"] if i.get("expiryDate") == expiry]
    call_sum = sum(i["CE"]["impliedVolatility"] for i in filtered if "CE" in i)
    put_sum = sum(i["PE"]["impliedVolatility"] for i in filtered if "PE" in i)
    return call_sum, put_sum

# Manual refresh button
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.experimental_rerun()

if expiry:
    call_iv, put_iv = calculate_iv_sums(expiry)
    if call_iv is not None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.metric("Call IV Sum", f"{call_iv:.2f}")
        st.metric("Put IV Sum", f"{put_iv:.2f}")
        st.write(f"Last updated: {now}")

        # Store snapshot if it's new
        if not st.session_state["snapshots"] or st.session_state["snapshots"][-1]["Call IV"] != call_iv or st.session_state["snapshots"][-1]["Put IV"] != put_iv:
            st.session_state["snapshots"].append({
                "Timestamp": now,
                "Call IV": call_iv,
                "Put IV": put_iv
            })

        # Show history in sidebar
        st.sidebar.write("### Snapshot History")
        df = pd.DataFrame(st.session_state["snapshots"])
        st.sidebar.dataframe(df, height=400)
        csv = df.to_csv(index=False).encode()
        st.sidebar.download_button("Download CSV", csv, "snapshots.csv", "text/csv")

# Auto-refresh every 10 minutes (600s)
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=600000, key="iv_refresh")

