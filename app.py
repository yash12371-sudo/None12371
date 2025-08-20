import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import datetime

st.set_page_config(page_title="None12371", layout="wide")

# ---- Helpers ----
@st.cache_data(show_spinner=False, ttl=45)
def get_session():
    """Warm up a session so NSE doesn't block us."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
        "DNT": "1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    # Warm up cookies
    s.get("https://www.nseindia.com", timeout=10)
    s.get("https://www.nseindia.com/option-chain", timeout=10)
    return s

def fetch_option_chain(symbol: str) -> dict:
    s = get_session()
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    r = s.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def to_df(chain_json: dict) -> pd.DataFrame:
    rows = []
    for row in chain_json.get("records", {}).get("data", []):
        strike = row.get("strikePrice")
        expiry = row.get("expiryDate")
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        rows.append({
            "expiryDate": expiry,
            "strikePrice": strike,
            "CE_IV": ce.get("impliedVolatility"),
            "PE_IV": pe.get("impliedVolatility"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Clean
    df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
    for c in ["CE_IV", "PE_IV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("strikePrice").reset_index(drop=True)

# ---- UI ----
st.title("None12371")
st.caption("Sums IV by the selected expiry and stores snapshots (timestamped) in-session.")

with st.sidebar:
    st.header("Controls & Snapshots")
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
    refresh = st.slider("Auto-refresh (seconds)", 0, 60, 0)
    manual_expiry = st.text_input("Manual expiry (e.g., 21-Aug-2025). Leave blank to use dropdown.")

# Fetch once
try:
    chain = fetch_option_chain(symbol)
except Exception as e:
    st.error(f"Failed to fetch option chain: {e}")
    st.stop()

expiry_list = chain.get("records", {}).get("expiryDates", []) or []
selected_expiry = manual_expiry.strip() if manual_expiry else st.selectbox("Available expiries (from NSE)", expiry_list, index=0 if expiry_list else None)
st.write(f"**Using expiry:** {selected_expiry}")

df_all = to_df(chain)
if df_all.empty:
    st.warning("No option rows received from NSE.")
    st.stop()

# Filter by expiry
df = df_all[df_all["expiryDate"] == selected_expiry].copy()
if df.empty:
    st.warning("No rows match the selected expiry. Try another expiry or clear the manual field.")
    st.stop()

sum_call_iv = float(np.nansum(df["CE_IV"]))  # sums only selected expiry
sum_put_iv  = float(np.nansum(df["PE_IV"]))

# Initialize snapshots store in session_state
if "snapshots" not in st.session_state:
    # structure: { expiry_str: [ {timestamp:..., call_iv:..., put_iv:...}, ... ] }
    st.session_state.snapshots = {}

# Append current snapshot for the selected expiry
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
snap = {"timestamp": now, "call_iv": sum_call_iv, "put_iv": sum_put_iv}

# Create list for expiry if missing
if selected_expiry not in st.session_state.snapshots:
    st.session_state.snapshots[selected_expiry] = []

# Append snapshot - only append if the last snapshot differs to avoid duplicates on rerun loops
last = st.session_state.snapshots[selected_expiry][-1] if st.session_state.snapshots[selected_expiry] else None
if (not last) or (last["call_iv"] != snap["call_iv"] or last["put_iv"] != snap["put_iv"]):
    st.session_state.snapshots[selected_expiry].append(snap)

# Top metrics
c1, c2, c3 = st.columns(3)
c1.metric("Σ Call IV (selected expiry)", f"{sum_call_iv:,.2f}")
c2.metric("Σ Put IV (selected expiry)",  f"{sum_put_iv:,.2f}")
c3.metric("Rows counted", f"{len(df):,}")

st.caption(f"Last calculated: {now}")

# Expandable raw data
with st.expander("Preview counted rows"):
    st.dataframe(df[["strikePrice", "CE_IV", "PE_IV"]], use_container_width=True)

# Sidebar: show snapshot history for selected expiry
with st.sidebar:
    st.subheader(f"Snapshots — {selected_expiry}")
    snaps = st.session_state.snapshots.get(selected_expiry, [])
    if snaps:
        snaps_df = pd.DataFrame(snaps)
        # Show latest first
        snaps_df = snaps_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(snaps_df, use_container_width=True)
        csv = snaps_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download snapshots CSV", csv, file_name=f"snapshots_{selected_expiry}.csv", mime="text/csv")
    else:
        st.write("No snapshots yet. They appear when values change or on refresh.")

# Auto-refresh handling
if refresh and refresh > 0:
    st.caption(f"Auto-refreshing every {refresh} seconds…")
    time.sleep(refresh)
    st.experimental_rerun()