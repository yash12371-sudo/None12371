import math
import time
from datetime import datetime, timedelta

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
# Auto-refresh every 10 minutes (frontend trigger)
# =========================
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0

# 600,000 ms = 10 minutes
_ = st_autorefresh(interval=600_000, key="auto_refresh")

# =========================
# Backend guard for snapshots (every ~10 minutes)
# =========================
if "last_snapshot_time" not in st.session_state:
    st.session_state["last_snapshot_time"] = time.time()

def auto_snapshot_due():
    now = time.time()
    if now - st.session_state["last_snapshot_time"] >= 600:
        st.session_state["last_snapshot_time"] = now
        return True
    return False

# Manual refresh button
refresh_btn = st.sidebar.button("Refresh Now")

# =========================
# Constants / Config
# =========================
IST = pytz.timezone("Asia/Kolkata")

# Risk-free & dividend yield assumptions (annualized)
RISK_FREE = 0.06
DIV_YIELD = 0.00

call_dex = delta_c * ce["oi"] * min_call_q
call_gex = gamma_c * ce["oi"] * min_call_q * (spot ** 2)

put_dex  = delta_p * pe["oi"] * min_put_q
put_gex  = gamma_p * pe["oi"] * min_put_q * (spot ** 2)
}

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
    """
    Black–Scholes delta & gamma. option_type in {"C","P"}
    """
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return 0.0, 0.0
    # Delta
    if option_type == "C":
        delta = math.exp(-q * T) * norm_cdf(d1)
    else:  # Put
        delta = -math.exp(-q * T) * norm_cdf(-d1)
    # Gamma (same for C/P)
    gamma = (math.exp(-q * T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))
    return delta, gamma

def nearest_index(sorted_list, x):
    if not sorted_list:
        return None
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))

def parse_expiry_to_dt(expiry_str: str):
    """
    Convert NSE expiry string to a timezone-aware datetime at 15:30 IST.
    """
    # Common NSE format: "21-Aug-2025"
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
    # Set market close time 15:30 IST
    dt_ist = IST.localize(dt_naive.replace(hour=15, minute=30, second=0, microsecond=0))
    return dt_ist

def time_to_expiry_years(expiry_str: str) -> float:
    expiry_dt = parse_expiry_to_dt(expiry_str)
    if expiry_dt is None:
        return 1.0 / 365.0
    now_ist = datetime.now(IST)
    sec = (expiry_dt - now_ist).total_seconds()
    if sec <= 0:
        # Just a tiny positive time to avoid divide-by-zero
        return 1.0 / (365.0 * 24.0 * 3600.0)
    return sec / (365.0 * 24.0 * 3600.0)

def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

def compute_metrics(data: dict, symbol: str, expiry: str):
    recs = (data or {}).get("records", {})
    rows = recs.get("data", []) or []
    spot = recs.get("underlyingValue")

    if spot is None:
        return None

    lot = LOT_SIZES.get(symbol.upper(), 50)

    T = time_to_expiry_years(expiry)
    r = RISK_FREE
    q = DIV_YIELD

    # Collect rows filtered by expiry
    ce_rows, pe_rows = [], []
    strikes_all = []
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

    # IV sums
    call_iv_sum = sum((r["iv"] * 100.0) for r in ce_rows)  # back to % for display
    put_iv_sum = sum((r["iv"] * 100.0) for r in pe_rows)

    # Min positive bid qty (your custom "value" scaling)
    def min_pos(rows):
        vals = [r["bidQty"] for r in rows if r["bidQty"] > 0]
        return min(vals) if vals else 1

    min_call_q = min_pos(ce_rows)
    min_put_q = min_pos(pe_rows)

    # Compute "value" using your rule + Greeks for DEX/GEX
    call_value_sum, put_value_sum = 0.0, 0.0
    tot_call_oi, tot_put_oi = 0, 0

    # Per-strike exposures for Walls
    per_strike_gex = {}  # (call_gex, put_gex, net_gex)
    per_strike_dex = {}  # (call_dex, put_dex, net_dex)

    # Find ATM strike for skew/term-structure/Vanna
    atm_idx = nearest_index(strikes, spot)
    atm_strike = strikes[atm_idx] if atm_idx is not None else None

    # Build dict for quick lookups
    ce_map = {r["strike"]: r for r in ce_rows}
    pe_map = {r["strike"]: r for r in pe_rows}

    # Compute DEX/GEX + your "value"
    for s in strikes:
        ce = ce_map.get(s)
        pe = pe_map.get(s)

        call_dex = put_dex = 0.0
        call_gex = put_gex = 0.0

        if ce and ce["oi"] > 0:
            tot_call_oi += ce["oi"]
            call_value_sum += ce["oi"] * min_call_q * ce["ltp"]
            sigma = max(ce["iv"], 1e-6)  # decimal
            delta_c, gamma_c = bs_delta_gamma(spot, s, r, q, sigma, T, "C")
            call_dex = delta_c * ce["oi"] * lot
            call_gex = gamma_c * ce["oi"] * lot * (spot ** 2)

        if pe and pe["oi"] > 0:
            tot_put_oi += pe["oi"]
            put_value_sum += pe["oi"] * min_put_q * pe["ltp"]
            sigma = max(pe["iv"], 1e-6)  # decimal
            delta_p, gamma_p = bs_delta_gamma(spot, s, r, q, sigma, T, "P")
            put_dex = delta_p * pe["oi"] * lot  # delta_p is negative
            # Convention: many practitioners take puts as "negative" in net gamma exposure
            put_gex = gamma_p * pe["oi"] * lot * (spot ** 2)

        per_strike_dex[s] = (call_dex, put_dex, call_dex + put_dex)
        # Net GEX convention used here: call_gex - put_gex
        per_strike_gex[s] = (call_gex, put_gex, call_gex - put_gex)

    # Summaries
    pcr = (tot_put_oi / tot_call_oi) if tot_call_oi > 0 else None

    # Top 2 "value" strikes
    def collect_values(rows, mult):
        out = []
        for r in rows:
            out.append({"strike": r["strike"], "value": r["oi"] * mult * r["ltp"]})
        return out

    top_calls = sorted(collect_values(ce_rows, min_call_q), key=lambda x: x["value"], reverse=True)[:2]
    top_puts = sorted(collect_values(pe_rows, min_put_q), key=lambda x: x["value"], reverse=True)[:2]

    # Max Pain (payoff minimization)
    def compute_max_pain():
        all_s = sorted(set([r["strike"] for r in ce_rows] + [r["strike"] for r in pe_rows]))
        call_oi_map = {r["strike"]: r["oi"] for r in ce_rows}
        put_oi_map = {r["strike"]: r["oi"] for r in pe_rows}
        best, best_pay = None, None
        for s in all_s:
            pay = 0
            for k in all_s:
                # Calls in the money when s > k
                if s > k:
                    pay += call_oi_map.get(k, 0) * (s - k)
                # Puts in the money when k > s
                if k > s:
                    pay += put_oi_map.get(k, 0) * (k - s)
            if best_pay is None or pay < best_pay:
                best, best_pay = s, pay
        return best

    max_pain = compute_max_pain() if (ce_rows or pe_rows) else None

    # IV skew near ATM: average of call/put IV at/near ATM strikes
    def avg_iv_near_atm(window=1):
        if atm_strike is None:
            return None, None, None
        idx = strikes.index(atm_strike)
        span = strikes[max(0, idx - window): idx + window + 1]
        call_ivs = [ce_map[s]["iv"] * 100.0 for s in span if s in ce_map]  # as %
        put_ivs = [pe_map[s]["iv"] * 100.0 for s in span if s in pe_map]  # as %
        call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else None
        put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else None
        if call_iv is None or put_iv is None:
            return None, None, None
        return call_iv, put_iv, (put_iv - call_iv)

    call_iv_atm, put_iv_atm, iv_skew = avg_iv_near_atm(window=1)

    # GEX totals
    gex_call_total = sum(v[0] for v in per_strike_gex.values())
    gex_put_total = sum(v[1] for v in per_strike_gex.values())
    gex_net = gex_call_total - gex_put_total

    # DEX totals
    dex_net = sum(v[2] for v in per_strike_dex.values())

    # Gamma Walls: top 3 by |net_gex per strike|
    gamma_walls = sorted(
        [{"strike": s, "net_gex": v[2]} for s, v in per_strike_gex.items()],
        key=lambda x: abs(x["net_gex"]),
        reverse=True
    )[:3]

    # Delta Walls: top 3 by |net_dex per strike|
    delta_walls = sorted(
        [{"strike": s, "net_dex": v[2], "call_dex": per_strike_dex[s][0], "put_dex": per_strike_dex[s][1]}
         for s in per_strike_dex],
        key=lambda x: abs(x["net_dex"]),
        reverse=True
    )[:3]

    return {
        "spot": spot,
        "lot": lot,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum": put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum": put_value_sum,
        "total_call_oi": tot_call_oi,
        "total_put_oi": tot_put_oi,
        "pcr": pcr,
        "max_pain": max_pain,
        "iv_skew": iv_skew,
        "call_iv_atm": call_iv_atm,
        "put_iv_atm": put_iv_atm,
        "atm_strike": atm_strike,
        "gex_call_total": gex_call_total,
        "gex_put_total": gex_put_total,
        "gex_net": gex_net,
        "dex_net": dex_net,
        "gamma_walls": gamma_walls,
        "delta_walls": delta_walls,
        "top_calls": top_calls,
        "top_puts": top_puts,
    }

def term_structure_atm_iv(data: dict, symbol: str, expiries: list, spot: float):
    """
    Compare ATM IV (near) vs next expiry.
    Returns (near_iv, next_iv, ratio, diff) in %
    """
    if not expiries or spot is None:
        return None, None, None, None

    ce_by_exp = {}
    pe_by_exp = {}

    rows = (data or {}).get("records", {}).get("data", []) or []
    for item in rows:
        exp = item.get("expiryDate")
        strike = item.get("strikePrice")
        ce, pe = item.get("CE"), item.get("PE")
        if exp not in ce_by_exp:
            ce_by_exp[exp] = {}
            pe_by_exp[exp] = {}
        if ce:
            ce_by_exp[exp][strike] = (ce.get("impliedVolatility", 0.0) or 0.0) / 100.0
        if pe:
            pe_by_exp[exp][strike] = (pe.get("impliedVolatility", 0.0) or 0.0) / 100.0

    def atm_iv_for_exp(exp):
        if exp not in ce_by_exp or exp not in pe_by_exp:
            return None
        strikes = sorted(set(list(ce_by_exp[exp].keys()) + list(pe_by_exp[exp].keys())))
        if not strikes:
            return None
        idx = nearest_index(strikes, spot)
        if idx is None:
            return None
        atm_s = strikes[idx]
        c_iv = ce_by_exp[exp].get(atm_s)
        p_iv = pe_by_exp[exp].get(atm_s)
        if c_iv is None or p_iv is None:
            return None
        return (c_iv + p_iv) / 2.0  # decimal

    near = expiries[0]
    near_iv = atm_iv_for_exp(near)
    next_iv = atm_iv_for_exp(expiries[1]) if len(expiries) > 1 else None
    if near_iv is None or next_iv is None:
        return None, None, None, None
    near_iv_pct = near_iv * 100.0
    next_iv_pct = next_iv * 100.0
    ratio = (near_iv / next_iv) if next_iv > 0 else None
    diff = near_iv_pct - next_iv_pct
    return near_iv_pct, next_iv_pct, ratio, diff

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
if st.sidebar.button("Clear History"):
    st.session_state["history"] = []

# =========================
# Compute Metrics
# =========================
m = compute_metrics(data, symbol, expiry)
if m is None:
    st.error("Unable to compute metrics for the selected expiry.")
    st.stop()

# Previous snapshot (for diffs & Vanna hint)
prev = st.session_state.get("history", [])[-1] if st.session_state.get("history") else {}
def diff(curr, prevv):
    return "–" if (prevv is None) else f"{curr - prevv:+.2f}"

# =========================
# Headline Metrics
# =========================
st.subheader("Market Snapshot")
c0, c1, c2 = st.columns(3)
with c0:
    st.metric("Spot Price", f"{m['spot']:.2f}" if m['spot'] else "-")
with c1:
    st.metric("Call IV Sum", f"{m['call_iv_sum']:.2f}", diff(m['call_iv_sum'], prev.get("Call IV Sum")))
    st.metric("Call Value", format_inr(m['call_value_sum']), diff(m['call_value_sum'], prev.get("Call Value")))
with c2:
    st.metric("Put IV Sum", f"{m['put_iv_sum']:.2f}", diff(m['put_iv_sum'], prev.get("Put IV Sum")))
    st.metric("Put Value", format_inr(m['put_value_sum']), diff(m['put_value_sum'], prev.get("Put Value")))

# =========================
# Institutional Metrics
# =========================
st.subheader("Institutional Metrics")

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("PCR", f"{m['pcr']:.2f}" if m["pcr"] else "-")
    if m["pcr"] is not None:
        st.caption("Action: Buy (bullish) " if m["pcr"] > 1 else "Action: Sell (bearish)")

with d2:
    st.metric("Max Pain", f"{m['max_pain']}" if m["max_pain"] else "-")
    if m["max_pain"]:
        bias = "Buy (pull up)" if m["spot"] < m["max_pain"] else "Sell (pull down)"
        st.caption(f"Action: {bias}")

with d3:
    st.metric("IV Skew (Put - Call) %", f"{m['iv_skew']:.2f}" if m["iv_skew"] is not None else "-")
    if m["iv_skew"] is not None:
        st.caption("Action: Sell (downside fear)" if m["iv_skew"] > 0 else "Action: Buy (upside demand)")

with d4:
    delta_call = m["total_call_oi"] - prev.get("Total Call OI", 0) if prev else 0
    delta_put = m["total_put_oi"] - prev.get("Total Put OI", 0) if prev else 0
    st.metric("OI Flow Δ", f"C:{delta_call} / P:{delta_put}")
    if delta_call or delta_put:
        st.caption("Action: Sell (call writing)" if delta_call > delta_put else "Action: Buy (put writing)")
    else:
        st.caption("Action: Neutral (no fresh positioning)")

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.metric("GEX Net", format_inr(m["gex_net"]))
    st.caption("Action: Caution (range)" if m["gex_net"] > 0 else "Action: Trend (follow breakout)")

with g2:
    # Show top Gamma Walls
    if m["gamma_walls"]:
        walls = ", ".join([f"{w['strike']}" for w in m["gamma_walls"]])
        st.metric("Gamma Walls (Top)", walls)
        st.caption("Action: Caution near walls (pin/magnet likely)")
    else:
        st.metric("Gamma Walls (Top)", "-")

with g3:
    st.metric("DEX Net", format_inr(m["dex_net"]))
    st.caption("Action: Up-bias if +ve; Down-bias if -ve")

with g4:
    if m["delta_walls"]:
        notes = []
        for w in m["delta_walls"]:
            # Classify as support/resistance by dominance
            dom = "Resist" if abs(w["call_dex"]) >= abs(w["put_dex"]) else "Support"
            notes.append(f"{w['strike']} ({dom})")
        st.metric("Delta Walls (Top)", ", ".join(notes))
        st.caption("Action: Expect stall at walls (hedging)")
    else:
        st.metric("Delta Walls (Top)", "-")

# IV Term Structure
near_iv, next_iv, ts_ratio, ts_diff = term_structure_atm_iv(data, symbol, expiries, m["spot"])
t1, t2 = st.columns(2)
with t1:
    st.metric("Term Struct ATM IV (Near/Next) %", f"{near_iv:.2f}/{next_iv:.2f}" if (near_iv and next_iv) else "-")
with t2:
    if ts_ratio is not None:
        st.metric("IV Ratio (Near/Next)", f"{ts_ratio:.2f}")
        st.caption("Action: Caution (event risk)" if ts_ratio > 1.05 else "Action: IV calm (premium selling ok)")
    else:
        st.metric("IV Ratio (Near/Next)", "-")

# Vanna & Charm simple hints
v1, v2 = st.columns(2)
curr_atm_iv = None
if m["call_iv_atm"] is not None and m["put_iv_atm"] is not None:
    curr_atm_iv = (m["call_iv_atm"] + m["put_iv_atm"]) / 2.0

with v1:
    prev_atm_iv = prev.get("ATM IV (near)")
    if curr_atm_iv is not None:
        st.metric("Vanna hint (ATM IV change)",
                  f"{curr_atm_iv:.2f}%"
                  + (f" ({curr_atm_iv - prev_atm_iv:+.2f})" if prev_atm_iv is not None else ""))
        if prev_atm_iv is not None:
            st.caption("Action: Buy (IV↑ → hedge buying)" if curr_atm_iv > prev_atm_iv
                       else "Action: Sell (IV↓ → hedge selling)")
        else:
            st.caption("Action: Watch (first reading)")
    else:
        st.metric("Vanna hint (ATM IV change)", "-")

with v2:
    # Charm: simple moneyness proxy
    if m["atm_strike"] is not None:
        charm_bias = "Buy (bullish drift)" if m["spot"] >= m["atm_strike"] else "Sell (bearish drift)"
        st.metric("Charm hint (moneyness)", f"Spot vs ATM: {m['spot']:.0f} / {m['atm_strike']}")
        st.caption(f"Action: {charm_bias}")
    else:
        st.metric("Charm hint (moneyness)", "-")

# =========================
# Top Strikes by your Value
# =========================
st.subheader(f"Top Strikes by Value (Expiry {expiry})")
cc, pp = st.columns(2)
with cc:
    st.write("Top Calls")
    for r in m["top_calls"]:
        st.write(f"Strike {r['strike']}: {format_inr(r['value'])}")
with pp:
    st.write("Top Puts")
    for r in m["top_puts"]:
        st.write(f"Strike {r['strike']}: {format_inr(r['value'])}")

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
        "Expiry": expiry,
        "Spot": m["spot"],
        "Call IV Sum": m["call_iv_sum"],
        "Put IV Sum": m["put_iv_sum"],
        "Call Value": m["call_value_sum"],
        "Put Value": m["put_value_sum"],
        "PCR": m["pcr"],
        "Max Pain": m["max_pain"],
        "IV Skew": m["iv_skew"],
        "ATM IV (near)": curr_atm_iv if curr_atm_iv is not None else None,
        "GEX Net": m["gex_net"],
        "DEX Net": m["dex_net"],
        "Total Call OI": m["total_call_oi"],
        "Total Put OI": m["total_put_oi"],
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "history.csv")
else:
    st.info("Snapshots will appear here every ~10 minutes or on Refresh Now.")
