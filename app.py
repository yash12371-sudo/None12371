import math
import time
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# =========================
# Auto-refresh (10 minutes)
# =========================
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0

_ = st_autorefresh(interval=600_000, key="auto_refresh")

if "last_snapshot_time" not in st.session_state:
    st.session_state["last_snapshot_time"] = time.time()

def auto_snapshot_due():
    now = time.time()
    if now - st.session_state["last_snapshot_time"] >= 600:
        st.session_state["last_snapshot_time"] = now
        return True
    return False

refresh_btn = st.sidebar.button("Refresh Now")

# =========================
# Constants
# =========================
IST = pytz.timezone("Asia/Kolkata")
RISK_FREE = 0.06
DIV_YIELD = 0.00

# =========================
# NSE client
# =========================
class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive",
        })

    def _warm_cookies(self):
        try:
            self.session.get("https://www.nseindia.com/", timeout=10)
        except Exception:
            pass

    def get_json(self, url, max_tries=5, sleep_between=1.4):
        last_err = None
        for _ in range(max_tries):
            try:
                self._warm_cookies()
                resp = self.session.get(url, timeout=12)
                if resp.status_code in (401, 403):
                    last_err = Exception(f"HTTP {resp.status_code}")
                    time.sleep(sleep_between)
                    continue
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(sleep_between)
        st.error(f"NSE API error: {last_err}")
        return None

nse = NSEClient()

# =========================
# Helpers
# =========================
def format_inr(value: float) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= 1e7:
        return f"{v/1e7:.2f} Cr"
    elif v >= 1e5:
        return f"{v/1e5:.2f} Lakh"
    else:
        return f"{v:.0f}"

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1_d2(S, K, r, q, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return None, None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_delta_gamma(S, K, r, q, sigma, T, option_type: str):
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return 0.0, 0.0
    if option_type == "C":
        delta = math.exp(-q * T) * norm_cdf(d1)
    else:
        delta = -math.exp(-q * T) * norm_cdf(-d1)
    gamma = (math.exp(-q * T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))
    return delta, gamma

def nearest_index(sorted_list, x):
    if not sorted_list:
        return None
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))

def parse_expiry_to_dt(expiry_str: str):
    fmts = ["%d-%b-%Y", "%d-%b-%y"]
    dt_naive = None
    for f in fmts:
        try:
            dt_naive = datetime.strptime(expiry_str, f)
            break
        except Exception:
            continue
    if dt_naive is None:
        return None
    return IST.localize(dt_naive.replace(hour=15, minute=30, second=0, microsecond=0))

def time_to_expiry_years(expiry_str: str) -> float:
    expiry_dt = parse_expiry_to_dt(expiry_str)
    if expiry_dt is None:
        return 1.0 / 365.0
    now_ist = datetime.now(IST)
    sec = (expiry_dt - now_ist).total_seconds()
    if sec <= 0:
        return 1.0 / (365.0 * 24.0 * 3600.0)
    return sec / (365.0 * 24.0 * 3600.0)

def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

# =========================
# Compute Metrics (scaled by min bid qty)
# =========================
def compute_metrics(data: dict, symbol: str, expiry: str):
    recs = (data or {}).get("records", {})
    rows = recs.get("data", []) or []
    spot = recs.get("underlyingValue")
    if spot is None:
        return None

    T = time_to_expiry_years(expiry)

    ce_rows, pe_rows, strikes_all = [], [], []
    for item in rows:
        if item.get("expiryDate") != expiry:
            continue
        strike = item.get("strikePrice")
        ce, pe = item.get("CE"), item.get("PE")
        strikes_all.append(strike)
        if ce:
            ce_rows.append({
                "strike": strike,
                "iv": (ce.get("impliedVolatility", 0.0) or 0.0) / 100.0,
                "oi": ce.get("openInterest", 0) or 0,
                "bidQty": ce.get("bidQty", 0) or 0,
                "ltp": ce.get("lastPrice", 0.0) or 0.0,
            })
        if pe:
            pe_rows.append({
                "strike": strike,
                "iv": (pe.get("impliedVolatility", 0.0) or 0.0) / 100.0,
                "oi": pe.get("openInterest", 0) or 0,
                "bidQty": pe.get("bidQty", 0) or 0,
                "ltp": pe.get("lastPrice", 0.0) or 0.0,
            })

    strikes = sorted(set(strikes_all))
    if not strikes:
        return None

    # Sums
    call_iv_sum = sum((r["iv"] * 100.0) for r in ce_rows)
    put_iv_sum  = sum((r["iv"] * 100.0) for r in pe_rows)

    # min bid qty scaling
    def min_pos(rows):
        vals = [r["bidQty"] for r in rows if r["bidQty"] > 0]
        return min(vals) if vals else 1
    min_call_q = min_pos(ce_rows)
    min_put_q  = min_pos(pe_rows)

    tot_call_oi, tot_put_oi = 0, 0
    dex_net, gex_net = 0.0, 0.0

    atm_idx = nearest_index(strikes, spot)
    atm_strike = strikes[atm_idx] if atm_idx is not None else None
    ce_map = {r["strike"]: r for r in ce_rows}
    pe_map = {r["strike"]: r for r in pe_rows}

    for s in strikes:
        ce = ce_map.get(s)
        pe = pe_map.get(s)
        if ce and ce["oi"] > 0:
            tot_call_oi += ce["oi"]
            d_c, g_c = bs_delta_gamma(spot, s, RISK_FREE, DIV_YIELD, max(ce["iv"],1e-6), T, "C")
            dex_net += d_c * ce["oi"] * min_call_q
            gex_net += g_c * ce["oi"] * min_call_q * (spot**2)
        if pe and pe["oi"] > 0:
            tot_put_oi += pe["oi"]
            d_p, g_p = bs_delta_gamma(spot, s, RISK_FREE, DIV_YIELD, max(pe["iv"],1e-6), T, "P")
            dex_net += d_p * pe["oi"] * min_put_q
            gex_net -= g_p * pe["oi"] * min_put_q * (spot**2)

    pcr = (tot_put_oi / tot_call_oi) if tot_call_oi else None

    # IV skew near ATM
    iv_skew, atm_iv_now = None, None
    if atm_strike is not None:
        idx = strikes.index(atm_strike)
        span = strikes[max(0, idx-1): idx+2]
        call_ivs = [ce_map[s]["iv"]*100.0 for s in span if s in ce_map]
        put_ivs  = [pe_map[s]["iv"]*100.0 for s in span if s in pe_map]
        if call_ivs and put_ivs:
            iv_skew = (sum(put_ivs)/len(put_ivs)) - (sum(call_ivs)/len(call_ivs))
            atm_iv_now = (sum(call_ivs)+sum(put_ivs))/(len(call_ivs)+len(put_ivs))

    # Max Pain
    def compute_max_pain():
        all_s = sorted(set([r["strike"] for r in ce_rows] + [r["strike"] for r in pe_rows]))
        call_map = {r["strike"]: r["oi"] for r in ce_rows}
        put_map  = {r["strike"]: r["oi"] for r in pe_rows}
        best, best_pay = None, None
        for s in all_s:
            pay = 0
            for k in all_s:
                if s > k: pay += call_map.get(k, 0) * (s - k)
                if k > s: pay += put_map.get(k, 0) * (k - s)
            if best_pay is None or pay < best_pay:
                best, best_pay = s, pay
        return best
    max_pain = compute_max_pain() if (ce_rows or pe_rows) else None

    return {
        "spot": spot, "expiry": expiry,
        "call_iv_sum": call_iv_sum, "put_iv_sum": put_iv_sum,
        "total_call_oi": tot_call_oi, "total_put_oi": tot_put_oi,
        "pcr": pcr, "iv_skew": iv_skew,
        "atm_iv_now": atm_iv_now, "atm_strike": atm_strike,
        "max_pain": max_pain,
        "dex_net": dex_net, "gex_net": gex_net,
        "min_call_q": min_call_q, "min_put_q": min_put_q
    }

# =========================
# Sidebar
# =========================
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY","BANKNIFTY"])
data = fetch_option_chain(symbol)
if not data: st.stop()
expiries = data["records"].get("expiryDates", []) or []
if not expiries: st.error("No expiries found."); st.stop()
expiry = st.sidebar.selectbox("Select Expiry", expiries)

m = compute_metrics(data, symbol, expiry)
if not m: st.error("Unable to compute metrics."); st.stop()
prev = st.session_state.get("history", [])[-1] if st.session_state.get("history") else {}

# =========================
# Final Comment (Institutional)
# =========================
def final_comment(m, prev):
    pos, neg, mods = [], [], []
    if m["pcr"]:
        if m["pcr"]>1.1: pos.append("put buildup supports upside")
        elif m["pcr"]<0.9: neg.append("call positioning adds resistance")
    if m["max_pain"] and m["spot"]:
        pull=(m["max_pain"]-m["spot"])/m["spot"]
        if pull>0.005: pos.append("expiry pull favors upticks")
        elif pull<-0.005: neg.append("expiry pull favors downticks")
    if m["iv_skew"]:
        if m["iv_skew"]>0.5: neg.append("downside skew signals caution")
        elif m["iv_skew"]<-0.5: pos.append("upside skew shows appetite")
    if prev:
        d_call=m["total_call_oi"]-(prev.get("Total Call OI",0) or 0)
        d_put=m["total_put_oi"]-(prev.get("Total Put OI",0) or 0)
        if d_put>d_call*1.1: pos.append("fresh put build-up aids bids")
        elif d_call>d_put*1.1: neg.append("fresh call build-up caps rallies")
    if m["dex_net"]>0: pos.append("dealer hedging adds support")
    elif m["dex_net"]<0: neg.append("dealer hedging adds pressure")
    prev_iv=prev.get("ATM IV (near)")
    if m["atm_iv_now"] and prev_iv:
        dv=m["atm_iv_now"]-prev_iv
        if dv>0.3: mods.append("rising IV fuels hedge demand")
        elif dv<-0.3: mods.append("IV easing reduces hedge demand")
    if m["atm_strike"] and m["spot"]:
        mods.append("time-decay drift leans up" if m["spot"]>=m["atm_strike"] else "time-decay drift leans down")
    if m["gex_net"]>0: mods.append("positive gamma keeps moves steady")
    elif m["gex_net"]<0: mods.append("negative gamma can amplify swings")
    if len(pos)>len(neg)+1: first="Bullish bias — "+", ".join(pos[:2])
    elif len(neg)>len(pos)+1: first="Bearish bias — "+", ".join(neg[:2])
    else:
        mix=pos[:1]+neg[:1]
        first="Flows mixed — "+", ".join(mix) if mix else "Flows mixed — positioning balanced"
    second=". "+ "; ".join(mods[:2]) if mods else ""
    return (first+second).strip()

st.subheader("Final Comment")
st.caption(final_comment(m, prev))

# =========================
# Snapshot History
# =========================
if "history" not in st.session_state: st.session_state["history"]=[]
if refresh_btn or auto_snapshot_due():
    ts=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,"Symbol":symbol,"Expiry":m["expiry"],"Spot":m["spot"],
        "PCR":m["pcr"],"Max Pain":m["max_pain"],"IV Skew":m["iv_skew"],
        "ATM IV (near)":m["atm_iv_now"],"DEX Net":m["dex_net"],"GEX Net":m["gex_net"],
        "Total Call OI":m["total_call_oi"],"Total Put OI":m["total_put_oi"],
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    df=pd.DataFrame(st.session_state["history"])
    st.dataframe(df,use_container_width=True)
    st.download_button("Download CSV",df.to_csv(index=False).encode(),"history.csv")
else:
    st.info("Snapshots will appear every ~10 minutes or on Refresh Now.")
