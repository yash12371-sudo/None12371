import streamlit as st
import requests
import pandas as pd
import datetime
import pytz

# --------------------------
# Session + headers for NSE
# --------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/109.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/"
})

# --------------------------
# Helpers
# --------------------------
def format_inr_value(val):
    """Format numeric values into Cr/Lakh for readability."""
    if val >= 1e7:
        return f"{val/1e7:.2f} Cr"
    elif val >= 1e5:
        return f"{val/1e5:.2f} Lakh"
    else:
        return f"{val:.0f}"

def get_expiries(symbol):
    """Fetch available expiry dates for symbol (NIFTY/BANKNIFTY)."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    try:
        resp = session.get(url, timeout=10)
        data = resp.json()
        return data.get("records", {}).get("expiryDates", [])
    except Exception as e:
        st.error(f"Error fetching expiries for {symbol}: {e}")
        return []

def fetch_data(symbol, expiry):
    """Fetch option chain and compute sums + values."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    resp = session.get(url, timeout=10)
    data = resp.json()

    ce_iv, pe_iv = 0, 0
    ce_value, pe_value = 0, 0
    call_strikes, put_strikes = [], []

    for item in data.get("records", {}).get("data", []):
        strike = item.get("strikePrice")

        # ---- Call side ----
        ce = item.get("CE")
        if ce and ce.get("expiryDate") == expiry:
            ce_iv += ce.get("impliedVolatility", 0) or 0

            bid_qty = ce.get("bidQty", 0) or 0
            oi = ce.get("openInterest", 0) or 0
            ltp = ce.get("lastPrice", 0) or 0

            call_val = oi * bid_qty * ltp
            ce_value += call_val
            call_strikes.append((strike, call_val))

        # ---- Put side ----
        pe = item.get("PE")
        if pe and pe.get("expiryDate") == expiry:
            pe_iv += pe.get("impliedVolatility", 0) or 0

            bid_qty = pe.get("bidQty", 0) or 0
            oi = pe.get("openInterest", 0) or 0
            ltp = pe.get("lastPrice", 0) or 0

            put_val = oi * bid_qty * ltp
            pe_value += put_val
            put_strikes.append((strike, put_val))

    return ce_iv, pe_iv, ce_value, pe_value, call_strikes, put_strikes

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

expiries = get_expiries(symbol)
expiry = st.sidebar.selectbox("Select Expiry", expiries) if expiries else None

refresh = st.sidebar.button("Refresh Now")

# Snapshot history in session
if "history" not in st.session_state:
    st.session_state.history = []

if expiry:
    ce_iv, pe_iv, ce_value, pe_value, call_strikes, put_strikes = fetch_data(symbol, expiry)

    # Timestamp in IST
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    # Append to history
    if refresh:
        st.session_state.history.append({
            "Time": timestamp,
            "Call IV Sum": ce_iv,
            "Put IV Sum": pe_iv,
            "Call Value": ce_value,
            "Put Value": pe_value
        })

    # --------------------------
    # Metrics row
    # --------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Call IV Sum", f"{ce_iv:.2f}")
    col2.metric("Put IV Sum", f"{pe_iv:.2f}")
    col3.metric("Call Value", format_inr_value(ce_value))
    col4.metric("Put Value", format_inr_value(pe_value))

    # --------------------------
    # Top 2 strikes (sidebar)
    # --------------------------
    st.sidebar.subheader("Top Strikes by Value")

    if call_strikes:
        top_calls = sorted(call_strikes, key=lambda x: x[1], reverse=True)[:2]
        st.sidebar.write("**Calls:**")
        for strike, val in top_calls:
            st.sidebar.write(f"{strike}: {format_inr_value(val)}")

    if put_strikes:
        top_puts = sorted(put_strikes, key=lambda x: x[1], reverse=True)[:2]
        st.sidebar.write("**Puts:**")
        for strike, val in top_puts:
            st.sidebar.write(f"{strike}: {format_inr_value(val)}")

    # --------------------------
    # Snapshot history table
    # --------------------------
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        df["Call Value"] = df["Call Value"].apply(format_inr_value)
        df["Put Value"] = df["Put Value"].apply(format_inr_value)
        st.subheader("Snapshot History")
        st.dataframe(df, use_container_width=True)
