import streamlit as st
import requests
import pandas as pd
import pytz
from datetime import datetime

# --------------------------------
# Streamlit Page Setup
# --------------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📊 Options Dashboard")

# --------------------------------
# NSE Session Setup
# --------------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
})

def nse_get(url):
    """Safe NSE GET request with cookies & retry"""
    try:
        session.get("https://www.nseindia.com", timeout=10)
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"NSE API Error: {e}")
        return None

# --------------------------------
# Fetch Expiries
# --------------------------------
def get_expiries(symbol):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    data = nse_get(url)
    if data and "records" in data and "expiryDates" in data["records"]:
        return data["records"]["expiryDates"]
    return []

# --------------------------------
# Fetch Option Chain
# --------------------------------
def fetch_option_chain(symbol):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse_get(url)

# --------------------------------
# Format Helper
# --------------------------------
def format_in_crores_lakhs(value):
    if value >= 1e7:
        return f"{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"{value/1e5:.2f} L"
    else:
        return f"{value:.0f}"

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("Filters")

symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

expiries = get_expiries(symbol)
if not expiries:
    st.error("No expiry data available. Try again later.")
    st.stop()

expiry = st.sidebar.selectbox("Select Expiry", expiries)

refresh = st.sidebar.button("🔄 Refresh Now")

# --------------------------------
# Data Processing
# --------------------------------
data = fetch_option_chain(symbol)
if not data or "records" not in data:
    st.error("Failed to fetch option chain data.")
    st.stop()

records = data["records"]["data"]

call_iv_sum, put_iv_sum = 0, 0
call_value_sum, put_value_sum = 0, 0

top_calls, top_puts = [], []

for item in records:
    if item.get("expiryDate") != expiry:
        continue

    ce = item.get("CE")
    pe = item.get("PE")

    # Calls
    if ce:
        iv = ce.get("impliedVolatility", 0) or 0
        call_iv_sum += iv

        call_oi = ce.get("openInterest", 0)
        bid_qty = ce.get("bidQty", 1)
        ltp = ce.get("lastPrice", 0)
        call_value = call_oi * bid_qty * ltp
        call_value_sum += call_value

        top_calls.append((ce["strikePrice"], call_value))

    # Puts
    if pe:
        iv = pe.get("impliedVolatility", 0) or 0
        put_iv_sum += iv

        put_oi = pe.get("openInterest", 0)
        bid_qty = pe.get("bidQty", 1)
        ltp = pe.get("lastPrice", 0)
        put_value = put_oi * bid_qty * ltp
        put_value_sum += put_value

        top_puts.append((pe["strikePrice"], put_value))

# Sort Top Strikes
top_calls = sorted(top_calls, key=lambda x: x[1], reverse=True)[:2]
top_puts = sorted(top_puts, key=lambda x: x[1], reverse=True)[:2]

# --------------------------------
# Results Display
# --------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sum of Call IV", f"{call_iv_sum:.2f}")

with col2:
    st.metric("Sum of Put IV", f"{put_iv_sum:.2f}")

with col3:
    st.metric("Call Value", format_in_crores_lakhs(call_value_sum))

with col4:
    st.metric("Put Value", format_in_crores_lakhs(put_value_sum))

# IST Timestamp
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# --------------------------------
# Save Snapshot
# --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({
    "Time": now,
    "Expiry": expiry,
    "Call IV Sum": f"{call_iv_sum:.2f}",
    "Put IV Sum": f"{put_iv_sum:.2f}",
    "Call Value": format_in_crores_lakhs(call_value_sum),
    "Put Value": format_in_crores_lakhs(put_value_sum),
})

# --------------------------------
# Snapshot History
# --------------------------------
st.subheader("Snapshot History")
df = pd.DataFrame(st.session_state.history)
st.dataframe(df, use_container_width=True)

# --------------------------------
# Top Strikes
# --------------------------------
st.sidebar.subheader("Top Strikes (by Value)")

st.sidebar.write("**Top Calls:**")
for strike, val in top_calls:
    st.sidebar.write(f"Strike {strike} → {format_in_crores_lakhs(val)}")

st.sidebar.write("**Top Puts:**")
for strike, val in top_puts:
    st.sidebar.write(f"Strike {strike} → {format_in_crores_lakhs(val)}")
