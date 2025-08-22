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

        rows.append({
            "strikePrice": strike,
            "CE_IV": ce.get("impliedVolatility") if ce else None,
            "CE_OI": ce.get("openInterest") if ce else None,
            "CE_BidQty": ce.get("bidQty") if ce else None,
            "CE_LTP": ce.get("lastPrice") if ce else None,
            "PE_IV": pe.get("impliedVolatility") if pe else None,
            "PE_OI": pe.get("openInterest") if pe else None,
            "PE_BidQty": pe.get("bidQty") if pe else None,
            "PE_LTP": pe.get("lastPrice") if pe else None,
        })

    df = pd.DataFrame(rows)

    # --- Compute values ---
    call_iv_sum = df["CE_IV"].dropna().sum()
    put_iv_sum = df["PE_IV"].dropna().sum()

    min_call_bid_qty = df["CE_BidQty"].dropna().min() if not df["CE_BidQty"].dropna().empty else 0
    min_put_bid_qty = df["PE_BidQty"].dropna().min() if not df["PE_BidQty"].dropna().empty else 0

    df["CallValue"] = df["CE_OI"].fillna(0) * min_call_bid_qty * df["CE_LTP"].fillna(0)
    df["PutValue"] = df["PE_OI"].fillna(0) * min_put_bid_qty * df["PE_LTP"].fillna(0)

    call_value_sum = df["CallValue"].sum()
    put_value_sum = df["PutValue"].sum()

    spot_price = records.get("underlyingValue", 0)

    return df, call_iv_sum, put_iv_sum, call_value_sum, put_value_sum, spot_price, len(df)

# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# Sidebar
st.sidebar.header("Dashboard Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
expiries = get_expiries(symbol)
expiry = st.sidebar.selectbox("Select Expiry", expiries)
refresh_button = st.sidebar.button("🔄 Refresh Now")

# Auto-refresh every 10 minutes
st_autorefresh(interval=600000, key="datarefresh")

# --- Fetch data ---
try:
    df, call_iv, put_iv, call_val, put_val, spot, rows_count = fetch_data(symbol, expiry)

    # Display metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Σ Call IV", f"{call_iv:.2f}")
    c2.metric("Σ Put IV", f"{put_iv:.2f}")
    c3.metric("Rows Counted", rows_count)

    c4, c5, c6 = st.columns(3)
    c4.metric("Σ Call Value", format_inr(call_val))
    c5.metric("Σ Put Value", format_inr(put_val))
    c6.metric("Spot Price", f"{spot:.2f}")

    # --- Top Strikes Section ---
    st.subheader("🔥 Top Strikes")
    top_calls = df.sort_values("CallValue", ascending=False).head(2)[["strikePrice", "CallValue"]]
    top_puts = df.sort_values("PutValue", ascending=False).head(2)[["strikePrice", "PutValue"]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 2 Call Strikes**")
        for _, row in top_calls.iterrows():
            st.write(f"Strike {row['strikePrice']}: {format_inr(row['CallValue'])}")

    with col2:
        st.markdown("**Top 2 Put Strikes**")
        for _, row in top_puts.iterrows():
            st.write(f"Strike {row['strikePrice']}: {format_inr(row['PutValue'])}")

    # --- Snapshot history ---
    tz = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    snapshot = {
        "Timestamp": timestamp,
        "Symbol": symbol,
        "Expiry": expiry,
        "Call IV": round(call_iv, 2),
        "Put IV": round(put_iv, 2),
        "Call Value": call_val,  # raw
        "Put Value": put_val,    # raw
    }

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if refresh_button:
        st.session_state["history"].append(snapshot)

    hist_df = pd.DataFrame(st.session_state["history"])
    if not hist_df.empty:
        hist_df_display = hist_df.copy()
        hist_df_display["Call Value"] = hist_df_display["Call Value"].apply(format_inr)
        hist_df_display["Put Value"] = hist_df_display["Put Value"].apply(format_inr)

        st.subheader("📜 Snapshot History")
        st.dataframe(hist_df_display, use_container_width=True)

        # Download option with raw values
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download History CSV", csv, "history.csv", "text/csv")

except Exception as e:
    st.error("⚠️ Failed to fetch data. Please try again later.")
    st.exception(e)
