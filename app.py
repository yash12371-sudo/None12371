import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ==============================
# Utility Functions
# ==============================

def format_inr(value: float) -> str:
    """Format number in crore/lakh style"""
    if value >= 1e7:
        return f"{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"{value/1e5:.2f} L"
    else:
        return f"{value:.0f}"

def get_expiries(symbol: str):
    """Fetch all available expiries from NSE"""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)
    data = response.json()
    return data.get("records", {}).get("expiryDates", [])

def fetch_data(symbol: str, expiry: str):
    """Fetch option chain and compute IV sums and Value sums"""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)
    data = response.json()

    records = data.get("records", {})
    filtered = [
        o for o in records.get("data", [])
        if o.get("expiryDate") == expiry
    ]

    # DataFrame for processing
    rows = []
    for o in filtered:
        strike = o.get("strikePrice")
        ce, pe = o.get("CE"), o.get("PE")

        ce_iv = ce.get("impliedVolatility") if ce else None
        ce_oi = ce.get("openInterest") if ce else None
        ce_bid_qty = ce.get("bidQty") if ce else None
        ce_ltp = ce.get("lastPrice") if ce else None

        pe_iv = pe.get("impliedVolatility") if pe else None
        pe_oi = pe.get("openInterest") if pe else None
        pe_bid_qty = pe.get("bidQty") if pe else None
        pe_ltp = pe.get("lastPrice") if pe else None

        rows.append({
            "strikePrice": strike,
            "CE_IV": ce_iv,
            "CE_OI": ce_oi,
            "CE_BidQty": ce_bid_qty,
            "CE_LTP": ce_ltp,
            "PE_IV": pe_iv,
            "PE_OI": pe_oi,
            "PE_BidQty": pe_bid_qty,
            "PE_LTP": pe_ltp,
        })

    df = pd.DataFrame(rows)

    # --- Compute values ---
    call_iv_sum = df["CE_IV"].dropna().sum()
    put_iv_sum = df["PE_IV"].dropna().sum()

    # Minimum bid quantity
    min_call_bid_qty = df["CE_BidQty"].dropna().min() if not df["CE_BidQty"].dropna().empty else 0
    min_put_bid_qty = df["PE_BidQty"].dropna().min() if not df["PE_BidQty"].dropna().empty else 0

    # Value calculations
    df["CallValue"] = df["CE_OI"].fillna(0) * min_call_bid_qty * df["CE_LTP"].fillna(0)
    df["PutValue"] = df["PE_OI"].fillna(0) * min_put_bid_qty * df["PE_LTP"].fillna(0)

    call_value_sum = df["CallValue"].sum()
    put_value_sum = df["PutValue"].sum()

    spot_price = records.get("underlyingValue", 0)

    return call_iv_sum, put_iv_sum, call_value_sum, put_value_sum, spot_price, len(df)

# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📈 Dashboard")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
expiries = get_expiries(symbol)
expiry = st.sidebar.selectbox("Select Expiry", expiries)
refresh_button = st.sidebar.button("🔄 Refresh Now")

st.sidebar.markdown("---")
st.sidebar.markdown("**Formula:** Call/Put Value = OI × Min Bid Qty × LTP")

# Auto-refresh every 10 minutes
st_autorefresh(interval=600000, key="datarefresh")

# --- Fetch data ---
try:
    call_iv, put_iv, call_val, put_val, spot, rows_count = fetch_data(symbol, expiry)

    # Display metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Σ Call IV", f"{call_iv:.2f}")
    c2.metric("Σ Put IV", f"{put_iv:.2f}")
    c3.metric("Rows Counted", rows_count)

    c4, c5, c6 = st.columns(3)
    c4.metric("Σ Call Value", format_inr(call_val))
    c5.metric("Σ Put Value", format_inr(put_val))
    c6.metric("Spot Price", f"{spot:.2f}")

    # Snapshot history
    tz = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    snapshot = {
        "Timestamp": timestamp,
        "Symbol": symbol,
        "Expiry": expiry,
        "Call IV": round(call_iv, 2),
        "Put IV": round(put_iv, 2),
        "Call Value": call_val,
        "Put Value": put_val,
    }

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if refresh_button or st.session_state.get("last_refresh") != timestamp:
        st.session_state["history"].append(snapshot)
        st.session_state["last_refresh"] = timestamp

    hist_df = pd.DataFrame(st.session_state["history"])
    st.subheader("📊 Snapshot History")
    st.dataframe(hist_df, use_container_width=True)

    # Download option
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download History CSV", csv, "history.csv", "text/csv")

except Exception as e:
    st.error("⚠️ Failed to fetch data. Please try again later.")
    st.exception(e)

