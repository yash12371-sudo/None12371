import streamlit as st
import requests
import pandas as pd
import datetime

st.set_page_config(page_title="NSE IV Dashboard", layout="wide")

st.title("📈 NSE Options IV Dashboard")

# Expiry input
expiry_date = st.text_input("Enter expiry date (dd-MMM-YYYY):", value=datetime.date.today().strftime("%d-%b-%Y"))

# Fetch data (placeholder API)
@st.cache_data(ttl=300)
def fetch_data(expiry):
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers)
        data = r.json()
        records = data.get("records", {}).get("data", [])
        return records
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

records = fetch_data(expiry_date)

if records:
    # Flatten option data
    calls, puts = [], []
    for row in records:
        ce = row.get("CE")
        pe = row.get("PE")
        if ce:
            calls.append(ce)
        if pe:
            puts.append(pe)

    import pandas as pd
    call_df = pd.DataFrame(calls)
    put_df = pd.DataFrame(puts)

    if not call_df.empty and not put_df.empty:
        sum_call_iv = call_df["impliedVolatility"].sum()
        sum_put_iv = put_df["impliedVolatility"].sum()

        st.metric("📊 Sum of Call IV", f"{sum_call_iv:.2f}")
        st.metric("📊 Sum of Put IV", f"{sum_put_iv:.2f}")

        # Show timestamp
        st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with st.expander("See raw data"):
            st.write(call_df[["strikePrice","impliedVolatility"]].head())
            st.write(put_df[["strikePrice","impliedVolatility"]].head())
    else:
        st.warning("No option chain data available for this expiry.")
else:
    st.warning("No data returned from NSE API.")
