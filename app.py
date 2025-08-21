import streamlit as st
import requests
import pandas as pd
import datetime
import pytz
import os

st.set_page_config(page_title="NSE Options Dashboard", layout="wide")

# File to store snapshots
SNAPSHOT_FILE = "snapshots.csv"

# Helper: format numbers in ₹ Cr / Lakh
def format_inr(value):
    try:
        value = float(value)
    except:
        return value
    if value >= 1e7:
        return f"₹ {value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹ {value/1e5:.2f} Lakh"
    else:
        return f"₹ {value:,.0f}"

# Fetch NSE option chain
@st.cache_data(ttl=600)
def fetch_data(symbol):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    resp = session.get("https://www.nseindia.com", headers=headers)
    resp = session.get(url, headers=headers)
    data = resp.json()
    return data

# Calculate sums
def calculate_sums(data, expiry):
    records = data["records"]["data"]
    lot_size = None
    sum_call_iv, sum_put_iv = 0, 0
    sum_call_oi_val, sum_put_oi_val = 0, 0

    for rec in records:
        if rec.get("expiryDate") == expiry:
            CE, PE = rec.get("CE"), rec.get("PE")

            if CE:
                sum_call_iv += CE.get("impliedVolatility", 0)
            if PE:
                sum_put_iv += PE.get("impliedVolatility", 0)

    # Get minimum lot size across strikes (from CE/PE)
    lot_sizes = []
    for rec in records:
        if rec.get("expiryDate") == expiry:
            CE, PE = rec.get("CE"), rec.get("PE")
            if CE: lot_sizes.append(CE.get("marketLot", 50))  # fallback
            if PE: lot_sizes.append(PE.get("marketLot", 50))
    lot_size = min(lot_sizes) if lot_sizes else 50

    # Compute OI Values
    for rec in records:
        if rec.get("expiryDate") == expiry:
            CE, PE = rec.get("CE"), rec.get("PE")
            if CE:
                sum_call_oi_val += CE.get("openInterest", 0) * CE.get("lastPrice", 0) * lot_size
            if PE:
                sum_put_oi_val += PE.get("openInterest", 0) * PE.get("lastPrice", 0) * lot_size

    return sum_call_iv, sum_put_iv, sum_call_oi_val, sum_put_oi_val, lot_size

# Sidebar
st.sidebar.header("Controls")
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
data = fetch_data(symbol)
expiries = sorted(list({rec["expiryDate"] for rec in data["records"]["data"]}))
expiry = st.sidebar.selectbox("Expiry", expiries)

# Calculate
sum_call_iv, sum_put_iv, sum_call_oi_val, sum_put_oi_val, lot_size = calculate_sums(data, expiry)

# IST timestamp
ist = pytz.timezone("Asia/Kolkata")
timestamp = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Σ Call IV", f"{sum_call_iv:.2f}")
col2.metric("Σ Put IV", f"{sum_put_iv:.2f}")
col3.metric("Σ Call OI Value", format_inr(sum_call_oi_val))
col4.metric("Σ Put OI Value", format_inr(sum_put_oi_val))

# Sidebar notes
st.sidebar.markdown(f"**Lot Size Used:** {lot_size}")

# Save snapshot
new_row = {
    "Time (IST)": timestamp,
    "Symbol": symbol,
    "Expiry": expiry,
    "Σ Call IV": round(sum_call_iv, 2),
    "Σ Put IV": round(sum_put_iv, 2),
    "Σ Call OI Value (₹)": sum_call_oi_val,
    "Σ Put OI Value (₹)": sum_put_oi_val,
}
if os.path.exists(SNAPSHOT_FILE):
    df = pd.read_csv(SNAPSHOT_FILE)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    df = pd.DataFrame([new_row])
df.to_csv(SNAPSHOT_FILE, index=False)

# Show table
st.subheader("Snapshot History")
st.dataframe(df.tail(10))

# Download & Reset
st.sidebar.download_button("Download CSV", df.to_csv(index=False), "snapshots.csv", "text/csv")
if st.sidebar.button("Reset History"):
    if os.path.exists(SNAPSHOT_FILE):
        os.remove(SNAPSHOT_FILE)
        st.sidebar.success("History cleared. Refresh the page.")
