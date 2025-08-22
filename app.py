import streamlit as st
import pandas as pd
import requests
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ==============================
# Utility Functions
# ==============================

def fetch_option_chain(symbol: str, expiry: str):
    """Fetch option chain data for given symbol and expiry from NSE."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)
    data = response.json()

    # Filter by expiry
    records = data.get("records", {}).get("data", [])
    filtered = [r for r in records if r.get("expiryDate") == expiry]
    return filtered

def format_large_number(num: float) -> str:
    """Format number into Lakh / Crore for readability."""
    if num >= 1e7:
        return f"{num/1e7:.2f} Cr"
    elif num >= 1e5:
        return f"{num/1e5:.2f} L"
    else:
        return f"{num:.2f}"

def calculate_values(filtered_data, side: str):
    """Calculate sum of OI × min bid qty × LTP for given side (CE/PE)."""
    total_value = 0
    for record in filtered_data:
        option = record.get(side)
        if option:
            oi = option.get("openInterest", 0)
            bid_qtys = option.get("bidQty", [])
            min_bid_qty = min(bid_qtys) if bid_qtys else 0
            ltp = option.get("lastPrice", 0)
            total_value += oi * min_bid_qty * ltp
    return total_value

def calculate_iv_sums(filtered_data):
    """Calculate total Call IV and Put IV."""
    call_iv_sum, put_iv_sum, rows = 0, 0, 0
    for record in filtered_data:
        ce, pe = record.get("CE"), record.get("PE")
        if ce and "impliedVolatility" in ce:
            call_iv_sum += ce["impliedVolatility"]
        if pe and "impliedVolatility" in pe:
            put_iv_sum += pe["impliedVolatility"]
        if ce or pe:
            rows += 1
    return call_iv_sum, put_iv_sum, rows

def get_symbol_price(symbol: str):
    """Fetch spot price of NIFTY or BANKNIFTY."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)
    data = response.json()
    return data.get("records", {}).get("underlyingValue", 0)

# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📈 Dashboard")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
expiry = st.sidebar.text_input("Enter Expiry (e.g. 21-Aug-2025)")
refresh_button = st.sidebar.button("🔄 Refresh Now")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Formula Note:**\n\n"
    "Call/Put Value = OI × Min Bid Qty × LTP"
)

# Auto-refresh every 10 minutes
st_autorefresh(interval=600000, key="datarefresh")

if expiry:
    try:
        # Fetch option chain
        data = fetch_option_chain(symbol, expiry)

        # IV sums
        call_iv, put_iv, rows = calculate_iv_sums(data)

        # Call/Put values
        call_value = calculate_values(data, "CE")
        put_value = calculate_values(data, "PE")

        # Spot price
        spot_price = get_symbol_price(symbol)

        # Current IST timestamp
        ist = pytz.timezone("Asia/Kolkata")
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        # ====== Display ======
        col1, col2, col3 = st.columns(3)
        col1.metric("Σ Call IV", f"{call_iv:.2f}")
        col2.metric("Σ Put IV", f"{put_iv:.2f}")
        col3.metric("Rows Counted", rows)

        col4, col5, col6 = st.columns(3)
        col4.metric("Σ Call Value", format_large_number(call_value))
        col5.metric("Σ Put Value", format_large_number(put_value))
        col6.metric(f"{symbol} Spot", f"{spot_price:.2f}")

        # ====== Snapshot History ======
        snapshot_row = {
            "Timestamp": timestamp,
            "Symbol": symbol,
            "Expiry": expiry,
            "Call IV": round(call_iv, 2),
            "Put IV": round(put_iv, 2),
            "Call Value": call_value,
            "Put Value": put_value,
        }

        history_file = "history.csv"
        try:
            history_df = pd.read_csv(history_file)
        except FileNotFoundError:
            history_df = pd.DataFrame()

        history_df = pd.concat([history_df, pd.DataFrame([snapshot_row])], ignore_index=True)
        history_df.to_csv(history_file, index=False)

        st.subheader("Snapshot History")
        st.dataframe(history_df.tail(20))

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.info("Please enter an expiry date to fetch data.")
