import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import datetime

st.set_page_config(page_title="NSE Options IV Dashboard", layout="wide")

@st.cache_data(show_spinner=False, ttl=30)
def get_session_headers():
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
        "DNT": "1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }
    s = requests.Session()
    s.headers.update(base_headers)
    s.get("https://www.nseindia.com", timeout=10)
    s.get("https://www.nseindia.com/option-chain", timeout=10)
    return s

def fetch_option_chain(symbol: str) -> dict:
    s = get_session_headers()
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    r = s.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def parse_chain(json_data: dict) -> pd.DataFrame:
    records = json_data.get("records", {})
    data = records.get("data", [])
    rows = []
    for row in data:
        strike = row.get("strikePrice")
        expiry = row.get("expiryDate")
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        rows.append({
            "expiryDate": expiry,
            "strikePrice": strike,
            "CE_OI": ce.get("openInterest"),
            "CE_IV": ce.get("impliedVolatility"),
            "PE_OI": pe.get("openInterest"),
            "PE_IV": pe.get("impliedVolatility"),
            "Underlying": ce.get("underlying") or pe.get("underlying"),
            "UnderlyingValue": ce.get("underlyingValue") or pe.get("underlyingValue"),
        })
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("expiryDate","Underlying")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("strikePrice").reset_index(drop=True)
    return df

def compute_totals(df: pd.DataFrame) -> dict:
    totals = {
        "total_CE_OI": float(np.nansum(df["CE_OI"])),
        "total_PE_OI": float(np.nansum(df["PE_OI"])),
        "total_CE_IV": float(np.nansum(df["CE_IV"])),
        "total_PE_IV": float(np.nansum(df["PE_IV"])),
    }
    totals["PCR_OI"] = (totals["total_PE_OI"] / totals["total_CE_OI"]) if totals["total_CE_OI"] else np.nan
    return totals

st.title("📊 NSE Options IV Dashboard")
st.caption("Live option chain-based sums (OI & IV) with expiry filter and timestamp.")

with st.sidebar:
    st.header("⚙️ Controls")
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
    autorefresh_sec = st.slider("Auto-refresh (seconds)", 0, 60, 15)
    manual_expiry = st.text_input("Manual expiry (e.g., 28-Aug-2025). Leave blank to use dropdown.")

try:
    chain = fetch_option_chain(symbol)
except Exception as e:
    st.error(f"Failed to fetch option chain for {symbol}: {e}")
    st.stop()

records = chain.get("records", {})
expiry_list = records.get("expiryDates", []) or []
selected_expiry = manual_expiry.strip() if manual_expiry else st.selectbox("Available expiries (from NSE)", expiry_list, index=0 if expiry_list else None)

st.write(f"**Using expiry:** {selected_expiry}")

df_all = parse_chain(chain)
df = df_all[df_all["expiryDate"] == selected_expiry].copy()
if df.empty:
    st.warning("No rows found for the selected expiry.")
    st.stop()

totals = compute_totals(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Underlying", f'{df["UnderlyingValue"].dropna().iloc[0]:,.2f}' if df["UnderlyingValue"].notna().any() else "—")
col2.metric("Total CE OI", f'{totals["total_CE_OI"]:,.0f}')
col3.metric("Total PE OI", f'{totals["total_PE_OI"]:,.0f}')
col4.metric("Sum Call IV", f'{totals["total_CE_IV"]:,.2f}')
col5.metric("Sum Put IV", f'{totals["total_PE_IV"]:,.2f}')

st.info(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.subheader("Option Chain (selected expiry)")
st.dataframe(df, use_container_width=True)

if autorefresh_sec and autorefresh_sec > 0:
    st.caption(f"Auto-refreshing every {autorefresh_sec} seconds…")
    time.sleep(autorefresh_sec)
    st.rerun()
