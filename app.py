import streamlit as st
import pandas as pd
import requests
import datetime
import pytz

# ===============================
# Helper Functions
# ===============================

# Convert values to Lakh/Crore for readability
def format_in_crore_lakh(value):
    if value >= 1e7:  # 1 Crore = 1 Cr
        return f"{value/1e7:.2f} Cr"
    elif value >= 1e5:  # 1 Lakh = 1 L
        return f"{value/1e5:.2f} L"
    else:
        return f"{value:.2f}"

# Fetch option chain data from NSE
@st.cache_data(ttl=600)
def fetch_data(symbol, expiry):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    resp = session.get(url, headers=headers).json()
    records = resp["records"]["data"]

    df_rows = []
    for rec in records:
        if rec.get("expiryDate") == expiry:
            strike = rec.get("strikePrice")

            call_data = rec.get("CE", {})
            put_data = rec.get("PE", {})

            df_rows.append({
                "strikePrice": strike,
                "CE_OI": call_data.get("openInterest", 0),
                "CE_IV": call_data.get("impliedVolatility", 0),
                "CE_LTP": call_data.get("lastPrice", 0),
                "CE_bidQty": call_data.get("bidQty", 0),
                "PE_OI": put_data.get("openInterest", 0),
                "PE_IV": put_data.get("impliedVolatility", 0),
                "PE_LTP": put_data.get("lastPrice", 0),
                "PE_bidQty": put_data.get("bidQty", 0),
            })

    return pd.DataFrame(df_rows)

# Calculate summary stats
def calculate_summary(df):
    if df.empty:
        return {}, {}

    # Implied Volatility sums
    call_iv_sum = df["CE_IV"].sum()
    put_iv_sum = df["PE_IV"].sum()

    # Find minimum bid quantities
    min_call_bid = df["CE_bidQty"].min() if not df["CE_bidQty"].empty else 0
    min_put_bid = df["PE_bidQty"].min() if not df["PE_bidQty"].empty else 0

    # Call Value and Put Value
    call_value = (df["CE_OI"] * min_call_bid * df["CE_LTP"]).sum()
    put_value = (df["PE_OI"] * min_put_bid * df["PE_LTP"]).sum()

    iv_stats = {
        "Call IV": call_iv_sum,
        "Put IV": put_iv_sum,
        "Rows Counted": len(df)
    }

    value_stats = {
        "Call Value": call_value,
        "Put Value": put_value
    }

    return iv_stats, value_stats

# Fetch spot price
def fetch_spot_price(symbol):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    resp = session.get(url, headers=headers).json()
    return resp.get("priceInfo", {}).get("lastPrice", 0)

# Save snapshot to CSV
def save_snapshot(symbol, expiry, iv_stats, value_stats):
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    snapshot = {
        "Timestamp": timestamp,
        "Symbol": symbol,
        "Expiry": expiry,
        "Call IV": iv_stats["Call IV"],
        "Put IV": iv_stats["Put IV"],
        "Call Value": value_stats["Call Value"],
        "Put Value": value_stats["Put Value"]
    }

    try:
        history = pd.read_csv("snapshots.csv")
        history = pd.concat([history, pd.DataFrame([snapshot])], ignore_index=True)
    except FileNotFoundError:
        history = pd.DataFrame([snapshot])

    history.to_csv("snapshots.csv", index=False)
    return history

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📈 Dashboard")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
expiry = st.sidebar.text_input("Enter Expiry (e.g. 21-Aug-2025)")

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Manual Refresh")
st.sidebar.markdown("**Note:** Call/Put Value = OI × Min Bid Qty × LTP")

# Auto refresh every 10 min
st_autorefresh = st.experimental_rerun

if expiry:
    df = fetch_data(symbol, expiry)
    iv_stats, value_stats = calculate_summary(df)

    spot_price = fetch_spot_price(symbol)

    # Save snapshot
    history = save_snapshot(symbol, expiry, iv_stats, value_stats)

    # Row 1: IV stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Σ Call IV", f"{iv_stats['Call IV']:.2f}")
    col2.metric("Σ Put IV", f"{iv_stats['Put IV']:.2f}")
    col3.metric("Rows Counted", iv_stats["Rows Counted"])

    # Row 2: Values and Spot
    col4, col5, col6 = st.columns(3)
    col4.metric("Σ Call Value", format_in_crore_lakh(value_stats["Call Value"]))
    col5.metric("Σ Put Value", format_in_crore_lakh(value_stats["Put Value"]))
    col6.metric(f"{symbol} Spot Price", f"{spot_price:.2f}")

    # Snapshot history
    st.subheader("📜 Snapshot History")
    st.dataframe(history)

else:
    st.warning("Please enter expiry date to fetch data.")
