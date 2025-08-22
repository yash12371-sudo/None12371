import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Option Chain Dashboard", layout="wide")

# --------------------------
# Helper Functions
# --------------------------

def format_value(num):
    """Format number into Cr/Lakh for display"""
    if num >= 1e7:
        return f"{num/1e7:.2f} Cr"
    elif num >= 1e5:
        return f"{num/1e5:.2f} Lakh"
    else:
        return f"{num:.2f}"

def get_expiries(symbol):
    """Fetch available expiries for given symbol safely from NSE"""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            st.error(f"Invalid response from NSE for {symbol}. Please retry.")
            return []

        if "records" in data and "expiryDates" in data["records"]:
            return data["records"]["expiryDates"]

        st.warning(f"No expiry data found for {symbol}")
        return []

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching expiries for {symbol}: {e}")
        return []

def fetch_option_chain(symbol, expiry):
    """Fetch option chain data for symbol and expiry"""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def calculate_values(data, expiry):
    """Calculate Call/Put IV sums and Values"""
    records = data.get("records", {}).get("data", [])
    underlying = data.get("records", {}).get("underlyingValue", 0)

    call_iv_sum, put_iv_sum = 0, 0
    call_value_sum, put_value_sum = 0, 0
    rows = []

    for item in records:
        if item.get("expiryDate") != expiry:
            continue

        strike = item.get("strikePrice")

        ce = item.get("CE")
        pe = item.get("PE")

        call_val = put_val = 0
        call_iv = put_iv = None

        if ce:
            OI = ce.get("openInterest", 0)
            bid_qty = ce.get("bidQty", 0)
            ltp = ce.get("lastPrice", 0)
            call_val = OI * bid_qty * ltp
            call_iv = ce.get("impliedVolatility", 0)
            call_iv_sum += call_iv
            call_value_sum += call_val

        if pe:
            OI = pe.get("openInterest", 0)
            bid_qty = pe.get("bidQty", 0)
            ltp = pe.get("lastPrice", 0)
            put_val = OI * bid_qty * ltp
            put_iv = pe.get("impliedVolatility", 0)
            put_iv_sum += put_iv
            put_value_sum += put_val

        rows.append({
            "strike": strike,
            "call_val": call_val,
            "put_val": put_val,
        })

    df = pd.DataFrame(rows)
    return call_iv_sum, put_iv_sum, call_value_sum, put_value_sum, underlying, df


# --------------------------
# Sidebar Settings
# --------------------------

st.sidebar.header("Dashboard Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

expiries = get_expiries(symbol)
expiry = st.sidebar.selectbox("Select Expiry", expiries) if expiries else None

if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# --------------------------
# Main Dashboard
# --------------------------

st.title("Dashboard")

if expiry:
    data = fetch_option_chain(symbol, expiry)

    if data:
        call_iv, put_iv, call_val, put_val, spot, df = calculate_values(data, expiry)

        # Top Strikes
        st.subheader("Top Strikes")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 2 Call Strikes**")
            if not df.empty:
                top_calls = df.sort_values("call_val", ascending=False).head(2)
                for _, row in top_calls.iterrows():
                    st.write(f"Strike {row['strike']}: {format_value(row['call_val'])}")

        with col2:
            st.markdown("**Top 2 Put Strikes**")
            if not df.empty:
                top_puts = df.sort_values("put_val", ascending=False).head(2)
                for _, row in top_puts.iterrows():
                    st.write(f"Strike {row['strike']}: {format_value(row['put_val'])}")

        # Snapshot history
        st.subheader("Snapshot History")

        if "history" not in st.session_state:
            st.session_state["history"] = []

        snapshot = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol,
            "Expiry": expiry,
            "Call IV": round(call_iv, 2),
            "Put IV": round(put_iv, 2),
            "Call Value": format_value(call_val),
            "Put Value": format_value(put_val),
        }
        st.session_state["history"].append(snapshot)

        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df)

        # CSV Download
        csv = pd.DataFrame(st.session_state["history"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", data=csv, file_name="snapshot_history.csv", mime="text/csv")
