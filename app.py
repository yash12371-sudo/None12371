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
# Auto-refresh every 10 minutes (frontend)
# =========================
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0

_ = st_autorefresh(interval=600_000, key="auto_refresh")  # 10 minutes

# Backend guard so a snapshot is always logged ~every 10 minutes
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
RISK_FREE = 0.06   # annualized
DIV_YIELD = 0.00   # annualized

# =========================
# NSE client (robust to 401/403)
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
    else:  # Put
        delta = -math.exp(-q * T) * norm_cdf(-d1)
    gamma = (math.exp(-q * T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))
    return delta, gamma

def nearest_index(sorted_list, x):
    if not sorted_list:
        return None
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))

def parse_expiry_to_dt(expiry_str: str):
    # Typical NSE formats like "21-Aug-2025"
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
# Core computations
# =========================
def compute_metrics(data: dict, symbol: str, expiry: str):
    recs = (data or {}).get("records", {})
    rows = recs.get("data", []) or []
    spot = recs.get("underlyingValue")
    if spot is None:
        return None

    T = time_to_expiry_years(expiry)
    r = RISK_FREE
    q = DIV_YIELD

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

    # IV sums (display in %)
    call_iv_sum = sum((r["iv"] * 100.0) for r in ce_rows)
    put_iv_sum  = sum((r["iv"] * 100.0) for r in pe_rows)

    # Min positive bid qty per side (scaling used everywhere, as requested)
    def min_pos(rows):
        vals = [r["bidQty"] for r in rows if r["bidQty"] > 0]
        return min(vals) if vals else 1

    min_call_q = min_pos(ce_rows)
    min_put_q  = min_pos(pe_rows)

    # Totals & Greeks-based exposures (scaled by min bid quantity)
    tot_call_oi, tot_put_oi = 0, 0
    call_value_sum, put_value_sum = 0.0, 0.0
    dex_net, gex_net = 0.0, 0.0

    # For IV skew and Vanna/Charm we need ATM strike context
    atm_idx = nearest_index(strikes, spot)
    atm_strike = strikes[atm_idx] if atm_idx is not None else None

    # Maps for quick lookup
    ce_map = {r["strike"]: r for r in ce_rows}
    pe_map = {r["strike"]: r for r in pe_rows}

    # Compute exposures and your Value metric
    for s in strikes:
        ce = ce_map.get(s)
        pe = pe_map.get(s)

        if ce and ce["oi"] > 0:
            tot_call_oi += ce["oi"]
            call_value_sum += ce["oi"] * min_call_q * ce["ltp"]
            sigma_c = max(ce["iv"], 1e-6)
            d_c, g_c = bs_delta_gamma(spot, s, r, q, sigma_c, T, "C")
            dex_net += d_c * ce["oi"] * min_call_q
            gex_net += g_c * ce["oi"] * min_call_q * (spot ** 2)

        if pe and pe["oi"] > 0:
            tot_put_oi += pe["oi"]
            put_value_sum += pe["oi"] * min_put_q * pe["ltp"]
            sigma_p = max(pe["iv"], 1e-6)
            d_p, g_p = bs_delta_gamma(spot, s, r, q, sigma_p, T, "P")
            dex_net += d_p * pe["oi"] * min_put_q       # d_p is negative
            gex_net -= g_p * pe["oi"] * min_put_q * (spot ** 2)  # net convention: calls - puts

    pcr = (tot_put_oi / tot_call_oi) if tot_call_oi > 0 else None

    # --- IV Skew near ATM (Put - Call)
    def avg_iv_near_atm(window=1):
        if atm_strike is None:
            return None, None, None
        idx = strikes.index(atm_strike)
        span = strikes[max(0, idx - window): idx + window + 1]
        call_ivs = [ce_map[s]["iv"] * 100.0 for s in span if s in ce_map]
        put_ivs  = [pe_map[s]["iv"] * 100.0 for s in span if s in pe_map]
        if not call_ivs or not put_ivs:
            return None, None, None
        c_iv = sum(call_ivs) / len(call_ivs)
        p_iv = sum(put_ivs)  / len(put_ivs)
        return c_iv, p_iv, (p_iv - c_iv)

    call_iv_atm, put_iv_atm, iv_skew = avg_iv_near_atm(window=1)
    curr_atm_iv = None
    if call_iv_atm is not None and put_iv_atm is not None:
        curr_atm_iv = (call_iv_atm + put_iv_atm) / 2.0  # %

    # --- Max Pain (brute force over strikes present)
    def compute_max_pain():
        all_s = sorted(set([r["strike"] for r in ce_rows] + [r["strike"] for r in pe_rows]))
        call_oi_map = {r["strike"]: r["oi"] for r in ce_rows}
        put_oi_map  = {r["strike"]: r["oi"] for r in pe_rows}
        best, best_pay = None, None
        for s in all_s:
            pay = 0
            for k in all_s:
                if s > k:  # calls ITM at settlement
                    pay += call_oi_map.get(k, 0) * (s - k)
                if k > s:  # puts ITM
                    pay += put_oi_map.get(k, 0) * (k - s)
            if best_pay is None or pay < best_pay:
                best, best_pay = s, pay
        return best

    max_pain = compute_max_pain() if (ce_rows or pe_rows) else None

    return {
        "spot": spot,
        "expiry": expiry,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum":  put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum":  put_value_sum,
        "total_call_oi": tot_call_oi,
        "total_put_oi":  tot_put_oi,
        "pcr": pcr,
        "iv_skew": iv_skew,                 # % Put-Call near ATM
        "call_iv_atm": call_iv_atm,         # % (for display if needed)
        "put_iv_atm": put_iv_atm,           # %
        "atm_iv_now": curr_atm_iv,          # % (used for Vanna hint)
        "atm_strike": atm_strike,           # for Charm hint
        "max_pain": max_pain,
        "dex_net": dex_net,
        "gex_net": gex_net,
        "min_call_q": min_call_q,
        "min_put_q":  min_put_q,
    }

# =========================
# Sidebar inputs
# =========================
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
data = fetch_option_chain(symbol)
if not data:
    st.stop()

expiries = data["records"].get("expiryDates", []) or []
if not expiries:
    st.error("No expiries found.")
    st.stop()

expiry = st.sidebar.selectbox("Select Expiry", expiries)

# =========================
# Compute
# =========================
m = compute_metrics(data, symbol, expiry)
if m is None:
    st.error("Unable to compute metrics for the selected expiry.")
    st.stop()

# Previous snapshot (for diffs & Vanna)
prev = st.session_state.get("history", [])[-1] if st.session_state.get("history") else {}

# =========================
# Final Comment (Institutional Desk Summary; 1–2 lines)
# =========================
def final_comment(m, prev):
    phrases_pos = []
    phrases_neg = []
    modifiers   = []

    # --- PCR sentiment
    if m["pcr"] is not None:
        if m["pcr"] > 1.1:
            phrases_pos.append("put positioning supports upside")
        elif m["pcr"] < 0.9:
            phrases_neg.append("call positioning adds resistance")

    # --- Max Pain pull (only if meaningful distance >0.5%)
    if m["max_pain"] and m["spot"]:
        pull = (m["max_pain"] - m["spot"]) / m["spot"]
        if pull > 0.005:
            phrases_pos.append("expiry pull favors upticks")
        elif pull < -0.005:
            phrases_neg.append("expiry pull favors downticks")

    # --- IV Skew (Put - Call) near ATM
    if m["iv_skew"] is not None:
        if m["iv_skew"] > 0.5:
            phrases_neg.append("downside skew shows caution")
        elif m["iv_skew"] < -0.5:
            phrases_pos.append("upside skew shows appetite")

    # --- OI Flow Δ since last snapshot
    if prev:
        d_call = m["total_call_oi"] - (prev.get("Total Call OI", 0) or 0)
        d_put  = m["total_put_oi"]  - (prev.get("Total Put OI", 0)  or 0)
        if abs(d_call) > 0 or abs(d_put) > 0:
            if d_put > d_call * 1.1:
                phrases_pos.append("fresh put build-up aids bids")
            elif d_call > d_put * 1.1:
                phrases_neg.append("fresh call build-up caps rallies")

    # --- Dealer Delta Exposure
    if m["dex_net"] > 0:
        phrases_pos.append("dealer hedging adds support")
    elif m["dex_net"] < 0:
        phrases_neg.append("dealer hedging adds pressure")

    # --- Vanna (ATM IV change vs prev)
    prev_atm_iv = prev.get("ATM IV (near)")
    if m["atm_iv_now"] is not None and prev_atm_iv is not None:
        dv = m["atm_iv_now"] - prev_atm_iv
        if dv > 0.3:
            modifiers.append("rising IV fuels hedge buying")
        elif dv < -0.3:
            modifiers.append("IV ease reduces hedge demand")

    # --- Charm (moneyness proxy)
    if m["atm_strike"] is not None and m["spot"] is not None:
        if m["spot"] >= m["atm_strike"]:
            modifiers.append("time-decay drift leans up")
        else:
            modifiers.append("time-decay drift leans down")

    # --- Gamma regime
    if m["gex_net"] > 0:
        modifiers.append("positive gamma may keep moves controlled")
    elif m["gex_net"] < 0:
        modifiers.append("negative gamma can amplify moves")

    # Build a short, simple 1–2 line summary
    # Priority: dominant tilt sentence + regime/modifier sentence
    pos_score = len(phrases_pos)
    neg_score = len(phrases_neg)

    if pos_score > neg_score + 1:
        first = "Bullish bias — " + ", ".join(phrases_pos[:2])
    elif neg_score > pos_score + 1:
        first = "Bearish bias — " + ", ".join(phrases_neg[:2])
    else:
        # mixed or close
        mix = phrases_pos[:1] + phrases_neg[:1]
        first = "Flows mixed — " + ", ".join(mix) if mix else "Flows mixed — positioning balanced"

    second = ""
    if modifiers:
        second = ". " + "; ".join(modifiers[:2])

    comment = (first + second).strip()
    # Ensure one or two lines max: keep it concise
    return comment

st.subheader("Final Comment")
st.caption(final_comment(m, prev))

# =========================
# Snapshot History (auto + manual)
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

if refresh_btn or auto_snapshot_due():
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,
        "Symbol": symbol,
        "Expiry": m["expiry"],
        "Spot": m["spot"],
        "PCR": m["pcr"],
        "Max Pain": m["max_pain"],
        "IV Skew": m["iv_skew"],
        "ATM IV (near)": m["atm_iv_now"],
        "DEX Net": m["dex_net"],
        "GEX Net": m["gex_net"],
        "Total Call OI": m["total_call_oi"],
        "Total Put OI":  m["total_put_oi"],
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "history.csv")
else:
    st.info("Snapshots will appear here every ~10 minutes or on Refresh Now.")
